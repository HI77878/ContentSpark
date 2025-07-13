import logging
import re
import librosa
import numpy as np
import os
import torch
import cv2
import threading
import json
import tempfile
from typing import List, Dict, Any, Optional
from analyzers.base_analyzer import GPUBatchAnalyzer

# GPU Forcing
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

# FFmpeg pthread fix
os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

logger = logging.getLogger(__name__)

class GPUBatchSpeechRateEnhanced(GPUBatchAnalyzer):
    """
    Enhanced Speech Rate Analyzer with full pitch analysis
    """
    
    def __init__(self):
        super().__init__()
        self.device_type = "cpu"  # Audio processing on CPU
        
        # German and English filler words
        self.filler_words = {
            'de': ['äh', 'ähm', 'eh', 'ehm', 'ach', 'also', 'nun', 'ja', 'naja', 'hmm', 'mhm', 'quasi', 'halt', 'irgendwie'],
            'en': ['uh', 'um', 'er', 'ah', 'like', 'you know', 'so', 'well', 'actually', 'basically', 'literally']
        }
        
        # Pause detection threshold in seconds
        self.pause_threshold = 0.5
        self.emphasis_threshold = 1.3  # 30% louder than average
        
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Main analysis entry point with full audio processing"""
        logger.info(f"[SpeechRateEnhanced] Analyzing {video_path}")
        
        try:
            # Load audio for pitch analysis
            y, sr = librosa.load(video_path, sr=22050)
            
            # Extract pitch using pyin
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('C2'),  # 65 Hz
                fmax=librosa.note_to_hz('C7'),  # 2093 Hz
                sr=sr,
                frame_length=2048,
                hop_length=512
            )
            
            # Clean up NaN values in pitch
            f0_clean = np.nan_to_num(f0, nan=0.0)
            
            # Time vector for pitch data
            pitch_times = np.arange(len(f0)) * (512 / sr)
            
            # Extract intensity/energy for emphasis detection
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            rms_times = np.arange(len(rms)) * (512 / sr)
            
            # Get transcription data
            transcription_path = video_path.replace('.mp4', '_transcription.json')
            transcription_data = None
            
            # Try to load existing transcription
            if os.path.exists(transcription_path):
                with open(transcription_path, 'r') as f:
                    transcription_data = json.load(f)
            
            if not transcription_data:
                # Run transcription if needed
                from analyzers.speech_transcription_ultimate import UltimateSpeechTranscription
                transcriber = UltimateSpeechTranscription()
                transcription_data = transcriber.analyze(video_path)
            
            segments = []
            
            if transcription_data and 'segments' in transcription_data:
                # Analyze each segment
                for seg in transcription_data['segments']:
                    segment_result = self._analyze_segment_enhanced(
                        seg, 
                        transcription_data.get('language', 'en'),
                        f0_clean,
                        pitch_times,
                        rms,
                        rms_times
                    )
                    segments.append(segment_result)
            
            # Calculate overall statistics
            overall_stats = self._calculate_overall_stats_enhanced(segments, y, sr)
            
            return {
                'segments': segments,
                'overall_statistics': overall_stats,
                'metadata': {
                    'sample_rate': sr,
                    'total_duration': len(y) / sr,
                    'analysis_method': 'enhanced_with_pitch'
                }
            }
            
        except Exception as e:
            logger.error(f"Speech rate analysis failed: {e}")
            return {'segments': [], 'error': str(e)}
    
    def _analyze_segment_enhanced(self, segment: Dict, language: str, f0: np.ndarray, 
                                 pitch_times: np.ndarray, rms: np.ndarray, 
                                 rms_times: np.ndarray) -> Dict[str, Any]:
        """Enhanced segment analysis with pitch and emphasis"""
        
        start_time = float(segment.get('start', segment.get('start_time', 0.0)))
        end_time = float(segment.get('end', segment.get('end_time', 0.0)))
        text = segment.get('text', '').strip()
        words = segment.get('words', [])
        
        # Basic metrics
        word_count = len(text.split()) if text else 0
        duration = end_time - start_time
        
        # Calculate WPM
        if duration > 0:
            wpm = (word_count / duration) * 60.0
        else:
            wpm = 0.0
        
        # Extract pitch for this segment
        pitch_mask = (pitch_times >= start_time) & (pitch_times <= end_time)
        segment_pitch = f0[pitch_mask]
        segment_pitch_clean = segment_pitch[segment_pitch > 0]  # Only voiced parts
        
        # Calculate pitch statistics
        if len(segment_pitch_clean) > 0:
            pitch_mean = float(np.mean(segment_pitch_clean))
            pitch_std = float(np.std(segment_pitch_clean))
            pitch_min = float(np.min(segment_pitch_clean))
            pitch_max = float(np.max(segment_pitch_clean))
        else:
            pitch_mean = pitch_std = pitch_min = pitch_max = 0.0
        
        # Extract RMS energy for emphasis detection
        rms_mask = (rms_times >= start_time) & (rms_times <= end_time)
        segment_rms = rms[rms_mask]
        
        # Detect emphasized words
        emphasized_words = self._detect_emphasized_words(words, segment_rms, rms_times[rms_mask])
        
        # Detect pauses
        pauses = self._detect_pauses_enhanced(words)
        
        # Count filler words
        filler_words = self.filler_words.get(language, self.filler_words['en'])
        filler_count = self._count_filler_words(text, filler_words)
        
        # Detect speech rhythm pattern
        rhythm_pattern = self._analyze_rhythm(words)
        
        return {
            'timestamp': start_time,
            'start_time': start_time,
            'end_time': end_time,
            'text': text,
            'word_count': word_count,
            'duration': duration,
            'words_per_minute': round(wpm, 1),
            'average_pitch': round(pitch_mean, 1),
            'pitch_std': round(pitch_std, 1),
            'pitch_range': [round(pitch_min, 1), round(pitch_max, 1)],
            'emphasized_words': emphasized_words,
            'pauses': pauses,
            'filler_words': filler_count,
            'rhythm_pattern': rhythm_pattern
        }
    
    def _detect_emphasized_words(self, words: List[Dict], rms: np.ndarray, 
                                rms_times: np.ndarray) -> List[Dict]:
        """Detect emphasized words based on energy peaks"""
        if not words or len(rms) == 0:
            return []
        
        emphasized = []
        avg_rms = np.mean(rms) if len(rms) > 0 else 0
        
        for word in words:
            word_start = float(word.get('start', 0))
            word_end = float(word.get('end', 0))
            word_text = word.get('word', '').strip()
            
            # Find RMS values during this word
            word_mask = (rms_times >= word_start) & (rms_times <= word_end)
            word_rms = rms[np.where(word_mask)[0]] if np.any(word_mask) else []
            
            if len(word_rms) > 0:
                word_energy = np.max(word_rms)
                if word_energy > avg_rms * self.emphasis_threshold:
                    emphasized.append({
                        'timestamp': word_start,
                        'word': word_text,
                        'emphasis_strength': float(word_energy / avg_rms)
                    })
        
        return emphasized
    
    def _detect_pauses_enhanced(self, words: List[Dict]) -> List[Dict]:
        """Detect significant pauses between words"""
        if not words or len(words) < 2:
            return []
        
        pauses = []
        
        for i in range(len(words) - 1):
            current_end = float(words[i].get('end', 0))
            next_start = float(words[i + 1].get('start', 0))
            
            pause_duration = next_start - current_end
            
            if pause_duration > self.pause_threshold:
                # Determine pause context
                context = "breathing_pause" if pause_duration < 1.0 else "dramatic_pause"
                if pause_duration > 2.0:
                    context = "section_break"
                
                pauses.append({
                    'timestamp': current_end,
                    'duration': round(pause_duration, 2),
                    'context': context
                })
        
        return pauses
    
    def _count_filler_words(self, text: str, filler_words: List[str]) -> List[Dict]:
        """Count filler words in text"""
        text_lower = text.lower()
        found_fillers = []
        
        for filler in filler_words:
            count = text_lower.count(filler)
            if count > 0:
                found_fillers.append({
                    'word': filler,
                    'count': count
                })
        
        return found_fillers
    
    def _analyze_rhythm(self, words: List[Dict]) -> str:
        """Analyze speech rhythm pattern"""
        if not words or len(words) < 3:
            return "insufficient_data"
        
        # Calculate inter-word intervals
        intervals = []
        for i in range(1, len(words)):
            prev_end = float(words[i-1].get('end', 0))
            curr_start = float(words[i].get('start', 0))
            interval = curr_start - prev_end
            if interval >= 0:
                intervals.append(interval)
        
        if not intervals:
            return "continuous"
        
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Classify rhythm
        if std_interval < 0.1:
            return "steady_rhythm"
        elif std_interval < 0.3:
            return "moderate_variation"
        elif avg_interval > 0.5:
            return "slow_deliberate"
        else:
            return "variable_rhythm"
    
    def _calculate_overall_stats_enhanced(self, segments: List[Dict], 
                                        audio: np.ndarray, sr: int) -> Dict:
        """Calculate comprehensive overall statistics"""
        
        if not segments:
            return {}
        
        # Aggregate metrics
        total_words = sum(s.get('word_count', 0) for s in segments)
        total_duration = sum(s.get('duration', 0) for s in segments)
        
        # WPM statistics
        all_wpm = [s.get('words_per_minute', 0) for s in segments if s.get('words_per_minute', 0) > 0]
        avg_wpm = np.mean(all_wpm) if all_wpm else 0
        
        # Pitch statistics
        all_pitches = [s.get('average_pitch', 0) for s in segments if s.get('average_pitch', 0) > 0]
        if all_pitches:
            overall_pitch_mean = np.mean(all_pitches)
            overall_pitch_std = np.std(all_pitches)
            overall_pitch_range = [min(all_pitches), max(all_pitches)]
        else:
            overall_pitch_mean = overall_pitch_std = 0
            overall_pitch_range = [0, 0]
        
        # Aggregate pauses
        all_pauses = []
        for s in segments:
            all_pauses.extend(s.get('pauses', []))
        
        total_pause_time = sum(p['duration'] for p in all_pauses)
        
        # Aggregate emphasized words
        all_emphasized = []
        for s in segments:
            all_emphasized.extend(s.get('emphasized_words', []))
        
        return {
            'total_words': total_words,
            'total_speech_duration': round(total_duration, 2),
            'average_wpm': round(avg_wpm, 1),
            'wpm_range': [round(min(all_wpm), 1), round(max(all_wpm), 1)] if all_wpm else [0, 0],
            'average_pitch_hz': round(overall_pitch_mean, 1),
            'pitch_std_hz': round(overall_pitch_std, 1),
            'pitch_range_hz': [round(overall_pitch_range[0], 1), round(overall_pitch_range[1], 1)],
            'total_pauses': len(all_pauses),
            'total_pause_duration': round(total_pause_time, 2),
            'pause_percentage': round((total_pause_time / total_duration * 100), 1) if total_duration > 0 else 0,
            'total_emphasized_words': len(all_emphasized),
            'speech_to_silence_ratio': round(total_duration / (total_pause_time + 0.001), 2)
        }
    
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> Dict[str, Any]:
        """Required by base class but not used for audio analysis"""
        return {'segments': []}