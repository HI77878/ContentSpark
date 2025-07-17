import logging
import numpy as np
import librosa
import os
import torch
import json
from typing import List, Dict, Any
from analyzers.base_analyzer import GPUBatchAnalyzer

# GPU Forcing
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

# FFmpeg pthread fix
os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

logger = logging.getLogger(__name__)

class GPUBatchSpeechFlow(GPUBatchAnalyzer):
    """
    Speech Flow Analyzer - Detects emphasized words, pauses, and rhythm
    """
    
    def __init__(self):
        super().__init__()
        self.device_type = "cpu"  # Audio processing on CPU
        self.emphasis_threshold = 1.3  # 30% louder than average
        self.pause_threshold = 0.3  # 300ms pause
        
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Analyze speech flow patterns including pitch and speed"""
        # FFmpeg Environment Fix
        import os
        os.environ['LD_LIBRARY_PATH'] = '/home/user/ffmpeg-install/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['PATH'] = '/home/user/ffmpeg-install/bin:' + os.environ.get('PATH', '')
        
        logger.info(f"[SpeechFlow] Analyzing {video_path}")
        
        try:
            # Load audio
            y, sr = librosa.load(video_path, sr=22050)
            
            # Extract pitch (F0) using librosa
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=512, fmin=50, fmax=500)
            
            # Get F0 values
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            # Calculate pitch statistics
            pitch_stats = {
                'mean_f0': float(np.mean(pitch_values)) if pitch_values else 0,
                'std_f0': float(np.std(pitch_values)) if pitch_values else 0,
                'min_f0': float(np.min(pitch_values)) if pitch_values else 0,
                'max_f0': float(np.max(pitch_values)) if pitch_values else 0,
                'range_f0': float(np.max(pitch_values) - np.min(pitch_values)) if pitch_values else 0
            }
            
            # Get transcription data
            transcription_path = video_path.replace('.mp4', '_transcription.json')
            transcription_data = None
            
            if os.path.exists(transcription_path):
                with open(transcription_path, 'r') as f:
                    transcription_data = json.load(f)
            else:
                # Try to get from speech transcription analyzer
                from analyzers.speech_transcription_ultimate import UltimateSpeechTranscription
                transcriber = UltimateSpeechTranscription()
                transcription_data = transcriber.analyze(video_path)
            
            if not transcription_data or 'segments' not in transcription_data:
                logger.warning("No transcription data available for speech flow analysis")
                # Still return pitch data
                return {
                    'segments': [],
                    'pitch_analysis': pitch_stats
                }
            
            # Extract RMS energy for emphasis detection
            hop_length = 512
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
            rms_times = np.arange(len(rms)) * (hop_length / sr)
            
            # Analyze each segment
            segments = []
            for seg in transcription_data['segments']:
                segment_result = self._analyze_segment_flow(
                    seg, 
                    rms, 
                    rms_times,
                    transcription_data.get('language', 'en')
                )
                if segment_result:
                    segments.append(segment_result)
            
            # Detect overall patterns
            flow_patterns = self._detect_flow_patterns(segments)
            
            return {
                'segments': segments,
                'flow_patterns': flow_patterns,
                'pitch_analysis': pitch_stats,
                'summary': self._generate_summary(segments, flow_patterns),
                'overall_stats': {
                    'avg_speech_rate': np.mean([s.get('speech_rate_wpm', 0) for s in segments]) if segments else 0,
                    'total_pauses': sum(len(s.get('pauses', [])) for s in segments),
                    'total_emphasized_words': sum(len(s.get('emphasized_words', [])) for s in segments),
                    'pitch_mean_hz': pitch_stats['mean_f0'],
                    'pitch_range_hz': pitch_stats['range_f0']
                }
            }
            
        except Exception as e:
            logger.error(f"Speech flow analysis failed: {e}")
            return {'segments': [], 'error': str(e)}
    
    def _analyze_segment_flow(self, segment, rms, rms_times, language):
        """Analyze flow within a segment including speech rate"""
        start_time = float(segment.get('start', segment.get('start_time', 0)))
        end_time = float(segment.get('end', segment.get('end_time', 0)))
        text = segment.get('text', '').strip()
        words = segment.get('words', [])
        
        # Calculate speech rate (words per minute)
        duration = end_time - start_time
        word_count = len(text.split()) if text else 0
        speech_rate = (word_count / duration * 60) if duration > 0 else 0
        
        if not words or not text:
            return None
        
        # Find RMS values for this segment
        segment_mask = (rms_times >= start_time) & (rms_times <= end_time)
        segment_rms = rms[segment_mask]
        
        if len(segment_rms) == 0:
            return None
        
        avg_rms = np.mean(segment_rms)
        
        # Detect emphasized words
        emphasized_words = []
        for word in words:
            word_start = float(word.get('start', 0))
            word_end = float(word.get('end', 0))
            word_text = word.get('word', '').strip()
            
            # Get RMS during word
            word_mask = (rms_times >= word_start) & (rms_times <= word_end)
            word_rms = rms[word_mask]
            
            if len(word_rms) > 0:
                word_energy = np.max(word_rms)
                if word_energy > avg_rms * self.emphasis_threshold:
                    emphasized_words.append({
                        'word': word_text,
                        'timestamp': word_start,
                        'emphasis_strength': float(word_energy / avg_rms),
                        'type': self._classify_emphasis(word_energy / avg_rms)
                    })
        
        # Detect pauses between words
        pauses = []
        for i in range(len(words) - 1):
            current_end = float(words[i].get('end', 0))
            next_start = float(words[i + 1].get('start', 0))
            pause_duration = next_start - current_end
            
            if pause_duration > self.pause_threshold:
                pause_type = self._classify_pause(pause_duration)
                pauses.append({
                    'timestamp': current_end,
                    'duration': round(pause_duration, 2),
                    'type': pause_type,
                    'after_word': words[i].get('word', ''),
                    'before_word': words[i + 1].get('word', '')
                })
        
        # Analyze rhythm
        rhythm = self._analyze_rhythm(words)
        
        return {
            'timestamp': start_time,
            'start_time': start_time,
            'end_time': end_time,
            'text': text,
            'speech_rate_wpm': round(speech_rate, 1),
            'word_count': word_count,
            'duration': round(duration, 2),
            'emphasized_words': emphasized_words,
            'pauses': pauses,
            'rhythm': rhythm,
            'flow_characteristics': self._determine_flow_characteristics(
                emphasized_words, pauses, rhythm
            ),
            'features': {
                'speech_rate': speech_rate,
                'emphasis_score': len(emphasized_words) / max(word_count, 1),
                'pause_ratio': len(pauses) / max(len(words) - 1, 1),
                'avg_rms': float(avg_rms)
            }
        }
    
    def _classify_emphasis(self, strength):
        """Classify emphasis type based on strength"""
        if strength > 2.0:
            return 'strong_emphasis'
        elif strength > 1.5:
            return 'moderate_emphasis'
        else:
            return 'slight_emphasis'
    
    def _classify_pause(self, duration):
        """Classify pause type based on duration"""
        if duration > 2.0:
            return 'long_pause'
        elif duration > 1.0:
            return 'dramatic_pause'
        elif duration > 0.5:
            return 'breath_pause'
        else:
            return 'brief_pause'
    
    def _analyze_rhythm(self, words):
        """Analyze speech rhythm from word timings"""
        if len(words) < 3:
            return {'pattern': 'insufficient_data'}
        
        # Calculate inter-word intervals
        intervals = []
        for i in range(1, len(words)):
            prev_end = float(words[i-1].get('end', 0))
            curr_start = float(words[i].get('start', 0))
            interval = curr_start - prev_end
            if interval >= 0:
                intervals.append(interval)
        
        if not intervals:
            return {'pattern': 'continuous'}
        
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Classify rhythm pattern
        if std_interval < 0.05:
            pattern = 'steady'
        elif std_interval < 0.15:
            pattern = 'regular'
        elif avg_interval > 0.3:
            pattern = 'deliberate'
        else:
            pattern = 'variable'
        
        return {
            'pattern': pattern,
            'avg_interval': round(avg_interval, 3),
            'variation': round(std_interval, 3),
            'tempo': 'fast' if avg_interval < 0.1 else 'moderate' if avg_interval < 0.3 else 'slow'
        }
    
    def _determine_flow_characteristics(self, emphasized_words, pauses, rhythm):
        """Determine overall flow characteristics"""
        characteristics = []
        
        # Emphasis patterns
        if len(emphasized_words) > 3:
            characteristics.append('highly_expressive')
        elif len(emphasized_words) > 0:
            characteristics.append('emphatic')
        
        # Pause patterns
        long_pauses = [p for p in pauses if p['type'] in ['long_pause', 'dramatic_pause']]
        if len(long_pauses) > 2:
            characteristics.append('thoughtful')
        elif len(pauses) > 5:
            characteristics.append('measured')
        
        # Rhythm patterns
        if rhythm['pattern'] == 'steady':
            characteristics.append('consistent_pacing')
        elif rhythm['pattern'] == 'variable':
            characteristics.append('dynamic_pacing')
        
        return characteristics
    
    def _detect_flow_patterns(self, segments):
        """Detect overall flow patterns across segments"""
        if not segments:
            return {}
        
        # Aggregate data
        all_emphasized = []
        all_pauses = []
        rhythms = []
        
        for seg in segments:
            all_emphasized.extend(seg.get('emphasized_words', []))
            all_pauses.extend(seg.get('pauses', []))
            rhythms.append(seg.get('rhythm', {}).get('pattern', 'unknown'))
        
        # Detect patterns
        patterns = {
            'emphasis_frequency': len(all_emphasized) / len(segments) if segments else 0,
            'pause_frequency': len(all_pauses) / len(segments) if segments else 0,
            'dominant_rhythm': max(set(rhythms), key=rhythms.count) if rhythms else 'unknown',
            'speech_style': self._determine_speech_style(all_emphasized, all_pauses, rhythms)
        }
        
        return patterns
    
    def _determine_speech_style(self, emphasized, pauses, rhythms):
        """Determine overall speech style"""
        emphasis_rate = len(emphasized)
        pause_rate = len(pauses)
        
        if emphasis_rate > 10 and pause_rate < 5:
            return 'energetic'
        elif pause_rate > 10:
            return 'contemplative'
        elif 'steady' in rhythms and emphasis_rate < 3:
            return 'monotone'
        elif emphasis_rate > 5:
            return 'expressive'
        else:
            return 'conversational'
    
    def _generate_summary(self, segments, flow_patterns):
        """Generate speech flow summary"""
        if not segments:
            return {}
        
        total_emphasized = sum(len(s.get('emphasized_words', [])) for s in segments)
        total_pauses = sum(len(s.get('pauses', [])) for s in segments)
        
        # Get most emphasized words
        all_emphasized = []
        for seg in segments:
            all_emphasized.extend(seg.get('emphasized_words', []))
        
        most_emphasized = sorted(
            all_emphasized, 
            key=lambda x: x['emphasis_strength'], 
            reverse=True
        )[:5]
        
        return {
            'total_emphasized_words': total_emphasized,
            'total_pauses': total_pauses,
            'speech_style': flow_patterns.get('speech_style', 'unknown'),
            'dominant_rhythm': flow_patterns.get('dominant_rhythm', 'unknown'),
            'most_emphasized_words': [
                {'word': w['word'], 'strength': round(w['emphasis_strength'], 2)}
                for w in most_emphasized
            ]
        }
    
    def process_batch_gpu(self, frames, frame_times):
        """Required by base class but not used"""
        return {'segments': []}