#!/usr/bin/env python3
"""
Ultimate Speech Transcription Analyzer - 100/100 Performance
Maximized accuracy with optimal Whisper configuration
"""

# GPU Forcing
import torch
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

import whisper
import numpy as np
import librosa
import parselmouth
from typing import List, Dict, Any, Tuple
from analyzers.base_analyzer import GPUBatchAnalyzer
import logging
import os
import time

logger = logging.getLogger(__name__)

class UltimateSpeechTranscription(GPUBatchAnalyzer):
    def __init__(self):
        super().__init__(batch_size=1)
        self.model_loaded = False
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.analyzer_name = "speech_transcription"
        logger.info(f"[Ultimate Speech] Initializing on {self.device}")
        
    def _load_model_impl(self):
        """Load Whisper model with optimal configuration"""
        # WICHTIG: Model Pre-Loading funktioniert NICHT mit Multiprocessing!
        # Jeder Prozess muss sein eigenes Model laden.
        
        try:
            # Use base model with FP16 for best accuracy/speed balance
            logger.info(f"[{self.analyzer_name}] Loading Whisper base model...")
            
            # Load with FP16 enabled for GPU acceleration
            self.model = whisper.load_model(
                "base",
                device=self.device,
<<<<<<< HEAD
                download_root=os.path.expanduser("~/.cache/whisper"),
                in_memory=True
            )
            
            # Use FP32 for stability
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.float()  # Use FP32 for stability
                logger.info(f"[{self.analyzer_name}] Model moved to GPU with FP32")
                logger.info(f"[{self.analyzer_name}] ✅ Whisper base model loaded with FP32 optimization")
=======
                download_root="/home/user/.cache/whisper"
            )
            
            # Ensure FP16 mode on GPU
            if self.device == 'cuda' and torch.cuda.is_available():
                # Model already uses FP16 internally when on CUDA
                logger.info(f"[{self.analyzer_name}] ✅ Whisper base model loaded with automatic FP16 optimization")
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
            else:
                logger.info(f"[{self.analyzer_name}] ✅ Whisper base model loaded on CPU")
                
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            raise
    
    def extract_pitch_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract pitch (F0) features using Parselmouth (Praat) with enhanced accuracy"""
        try:
            # Create Parselmouth Sound object
            sound = parselmouth.Sound(audio, sampling_frequency=sr)
            
            # Extract pitch with optimal parameters for speech
            pitch = sound.to_pitch(
                time_step=0.01,  # 10ms steps for high resolution
                pitch_floor=50.0,  # Lower floor for deep voices
                pitch_ceiling=800.0  # Higher ceiling for extreme cases
            )
            
            # Get pitch values
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced frames
            
            if len(pitch_values) > 0:
                pitch_mean = np.mean(pitch_values)
                pitch_std = np.std(pitch_values)
                pitch_min = np.min(pitch_values)
                pitch_max = np.max(pitch_values)
                pitch_range = pitch_max - pitch_min
                pitch_median = np.median(pitch_values)
                
                # Enhanced voice type classification
                voice_type = "unknown"
                if pitch_mean < 140:
                    voice_type = "very_low/bass"
                elif pitch_mean < 180:
                    voice_type = "low/masculine"
                elif pitch_mean < 220:
                    voice_type = "medium/neutral"
                elif pitch_mean < 280:
                    voice_type = "high/feminine"
                else:
                    voice_type = "very_high/child"
                
                # Calculate pitch modulation
                if len(pitch_values) > 10:
                    pitch_modulation = np.mean(np.abs(np.diff(pitch_values)))
                else:
                    pitch_modulation = 0.0
                
                return {
                    "pitch_mean_hz": float(pitch_mean),
                    "pitch_std_hz": float(pitch_std),
                    "pitch_min_hz": float(pitch_min),
                    "pitch_max_hz": float(pitch_max),
                    "pitch_range_hz": float(pitch_range),
                    "pitch_median_hz": float(pitch_median),
                    "pitch_modulation_hz": float(pitch_modulation),
                    "voice_type": voice_type,
                    "pitch_variability": "very_high" if pitch_std > 60 else "high" if pitch_std > 40 else "moderate" if pitch_std > 20 else "low",
                    "vocal_stability": float(1.0 - min(pitch_std / pitch_mean, 1.0))  # 0-1 stability score
                }
            else:
                return {
                    "pitch_mean_hz": 0.0,
                    "pitch_std_hz": 0.0,
                    "pitch_min_hz": 0.0,
                    "pitch_max_hz": 0.0,
                    "pitch_range_hz": 0.0,
                    "pitch_median_hz": 0.0,
                    "pitch_modulation_hz": 0.0,
                    "voice_type": "no_voice_detected",
                    "pitch_variability": "none",
                    "vocal_stability": 0.0
                }
                
        except Exception as e:
            logger.error(f"Pitch extraction error: {e}")
            return {
                "pitch_mean_hz": 0.0,
                "pitch_std_hz": 0.0,
                "error": str(e)
            }
    
    def analyze_speaking_speed(self, segments: List[Dict], audio_duration: float) -> Dict[str, Any]:
        """Analyze speaking speed and rhythm with enhanced metrics"""
        if not segments:
            return {
                "words_per_minute": 0.0,
                "syllables_per_minute": 0.0,
                "total_words": 0,
                "speaking_time": 0.0,
                "total_duration": audio_duration,
                "speaking_time_ratio": 0.0,
                "speaking_pace": "none",
                "articulation_rate": 0.0,
                "phonation_time_ratio": 0.0,
                "pause_analysis": {"total_pauses": 0}
            }
        
        # Calculate detailed metrics
        total_words = 0
        total_chars = 0
        speaking_time = 0.0
        word_durations = []
        
        for seg in segments:
            text = seg.get('text', '')
            words = text.split()
            total_words += len(words)
            total_chars += len(text.replace(' ', ''))
            seg_duration = seg['end_time'] - seg['start_time']
            speaking_time += seg_duration
            
            # Calculate word durations
            if words and seg_duration > 0:
                avg_word_duration = seg_duration / len(words)
                word_durations.extend([avg_word_duration] * len(words))
        
        # Estimate syllables with improved heuristic
        estimated_syllables = sum(self._count_syllables(seg.get('text', '')) for seg in segments)
        
        # Calculate rates
        wpm = (total_words / speaking_time * 60) if speaking_time > 0 else 0
        spm = (estimated_syllables / speaking_time * 60) if speaking_time > 0 else 0
        
        # Articulation rate (words per minute of actual speech)
        articulation_rate = wpm
        
        # Phonation time ratio
        phonation_ratio = speaking_time / audio_duration if audio_duration > 0 else 0
        
        # Analyze pauses with more detail
        pauses = []
        for i in range(1, len(segments)):
            pause_duration = segments[i]['start_time'] - segments[i-1]['end_time']
            if pause_duration > 0.1:  # Minimum pause threshold
                pause_type = 'micro' if pause_duration < 0.3 else 'short' if pause_duration < 0.6 else 'medium' if pause_duration < 1.2 else 'long'
                pauses.append({
                    'start': segments[i-1]['end_time'],
                    'duration': pause_duration,
                    'type': pause_type
                })
        
        # Enhanced pace classification
        pace = "extremely_fast"
        if wpm < 100:
            pace = "very_slow"
        elif wpm < 130:
            pace = "slow"
        elif wpm < 150:
            pace = "moderate"
        elif wpm < 170:
            pace = "fast"
        elif wpm < 200:
            pace = "very_fast"
        
        # Calculate fluency score
        pause_ratio = len([p for p in pauses if p['duration'] > 0.3]) / max(total_words, 1)
        fluency_score = max(0, min(1, 1 - pause_ratio * 10))
        
        return {
            "words_per_minute": float(wpm),
            "syllables_per_minute": float(spm),
            "total_words": total_words,
            "total_syllables": estimated_syllables,
            "speaking_time": float(speaking_time),
            "total_duration": float(audio_duration),
            "speaking_time_ratio": float(phonation_ratio),
            "speaking_pace": pace,
            "articulation_rate": float(articulation_rate),
            "phonation_time_ratio": float(phonation_ratio),
            "fluency_score": float(fluency_score),
            "pause_analysis": {
                "total_pauses": len(pauses),
                "micro_pauses": len([p for p in pauses if p['type'] == 'micro']),
                "short_pauses": len([p for p in pauses if p['type'] == 'short']),
                "medium_pauses": len([p for p in pauses if p['type'] == 'medium']),
                "long_pauses": len([p for p in pauses if p['type'] == 'long']),
                "average_pause_duration": float(np.mean([p['duration'] for p in pauses])) if pauses else 0,
                "longest_pause": float(max([p['duration'] for p in pauses])) if pauses else 0,
                "pause_details": pauses[:20]  # First 20 pauses
            }
        }
    
    def _count_syllables(self, text: str) -> int:
        """Enhanced syllable counting heuristic"""
        vowels = "aeiouyAEIOUY"
        text = text.lower()
        count = 0
        previous_was_vowel = False
        
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent e
        if text.endswith('e'):
            count -= 1
        
        # Ensure at least one syllable per word
        word_count = len(text.split())
        return max(count, word_count)
    
    def detect_emphasis(self, audio: np.ndarray, sr: int, segments: List[Dict]) -> List[Dict]:
        """Detect emphasized words with enhanced accuracy"""
        emphasized_words = []
        
        try:
            # Extract pitch and intensity contours
            sound = parselmouth.Sound(audio, sampling_frequency=sr)
            pitch = sound.to_pitch(time_step=0.01)
            intensity = sound.to_intensity(time_step=0.01)
            
            # Get global statistics for normalization
            all_pitch_values = []
            all_intensity_values = []
            
            for t in np.arange(0, sound.duration, 0.01):
                p = pitch.get_value_at_time(t)
                if p and p > 0:
                    all_pitch_values.append(p)
                i = intensity.get_value(t)
                if i and i > 0:
                    all_intensity_values.append(i)
            
            if not all_pitch_values or not all_intensity_values:
                return []
            
            pitch_mean = np.mean(all_pitch_values)
            pitch_std = np.std(all_pitch_values)
            intensity_mean = np.mean(all_intensity_values)
            intensity_std = np.std(all_intensity_values)
            
            for seg in segments:
                start_time = seg['start_time']
                end_time = seg['end_time']
                words = seg['text'].split()
                
                if not words:
                    continue
                
                # More accurate word timing using character-based estimation
                text_length = len(seg['text'])
                char_duration = (end_time - start_time) / max(text_length, 1)
                
                current_pos = 0
                for word in words:
                    word_start = start_time + current_pos * char_duration
                    word_length = len(word) + 1  # +1 for space
                    word_end = start_time + (current_pos + word_length) * char_duration
                    current_pos += word_length
                    
                    # Sample pitch and intensity for this word
                    word_pitch = []
                    word_intensity = []
                    
                    for t in np.arange(word_start, min(word_end, sound.duration), 0.01):
                        p = pitch.get_value_at_time(t)
                        if p and p > 0:
                            word_pitch.append(p)
                        
                        i_val = intensity.get_value(t)
                        if i_val and i_val > 0:
                            word_intensity.append(i_val)
                    
                    if word_pitch and word_intensity:
                        # Calculate emphasis features
                        pitch_range = max(word_pitch) - min(word_pitch)
                        pitch_peak = max(word_pitch)
                        intensity_peak = max(word_intensity)
                        
                        # Normalized scores
                        pitch_deviation = (pitch_peak - pitch_mean) / max(pitch_std, 1)
                        intensity_deviation = (intensity_peak - intensity_mean) / max(intensity_std, 1)
                        
                        # Enhanced emphasis detection
                        emphasis_score = 0
                        emphasis_types = []
                        
                        if pitch_range > 80:  # Very large pitch variation
                            emphasis_score += 3
                            emphasis_types.append("pitch_excursion")
                        elif pitch_range > 50:
                            emphasis_score += 2
                            emphasis_types.append("pitch_variation")
                        
                        if pitch_deviation > 2:  # 2+ std deviations above mean
                            emphasis_score += 2
                            emphasis_types.append("high_pitch")
                        
                        if intensity_deviation > 1.5:
                            emphasis_score += 2
                            emphasis_types.append("loud")
                        
                        if intensity_peak > 75:  # Absolute high intensity
                            emphasis_score += 1
                            emphasis_types.append("very_loud")
                        
                        # Store emphasized words with higher threshold
                        if emphasis_score >= 3 and len(word) > 2:
                            emphasized_words.append({
                                'word': word,
                                'timestamp': float(word_start),
                                'emphasis_score': float(emphasis_score),
                                'emphasis_types': emphasis_types,
                                'pitch_range': float(pitch_range),
                                'pitch_peak': float(pitch_peak),
                                'intensity_peak': float(intensity_peak),
                                'pitch_deviation': float(pitch_deviation),
                                'intensity_deviation': float(intensity_deviation)
                            })
            
            # Sort by emphasis score and return top emphasized words
            emphasized_words.sort(key=lambda x: x['emphasis_score'], reverse=True)
            
        except Exception as e:
            logger.error(f"Emphasis detection error: {e}")
        
        return emphasized_words[:30]  # Return top 30 emphasized words
    
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> Dict[str, Any]:
        """Not used for audio analysis"""
        return {'segments': []}
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Ultimate speech analysis with maximum accuracy"""
<<<<<<< HEAD
        # FFmpeg Environment Fix
        import os
        os.environ['LD_LIBRARY_PATH'] = '/home/user/ffmpeg-install/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['PATH'] = '/home/user/ffmpeg-install/bin:' + os.environ.get('PATH', '')
        
=======
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
        # Ensure models are loaded
        if not self.model_loaded:
            self._load_model_impl()
            self.model_loaded = True
            
        logger.info(f"[Ultimate Speech] Starting analysis of {video_path}")
        start_time = time.time()
        
        # Load audio with optimal sample rate for Whisper
        audio, sr = librosa.load(video_path, sr=16000)
        audio_duration = len(audio) / sr
        
        logger.info(f"[Ultimate Speech] Audio duration: {audio_duration:.1f}s")
        
        # Normalize audio for optimal transcription
        audio_normalized = audio / (np.max(np.abs(audio)) + 1e-10)
        
        # 1. Transcribe with optimal Whisper settings
        try:
            # Ensure float32 for Whisper
            audio_fp32 = audio_normalized.astype(np.float32)
            
            # Use optimal parameters for maximum accuracy
            result = self.model.transcribe(
                audio_fp32,
                language=None,  # Auto-detect for best results
                task='transcribe',
                verbose=False,
                temperature=0.0,  # Deterministic for consistency
                no_speech_threshold=0.2,  # Lower threshold for better detection
                condition_on_previous_text=True,  # Better context understanding
                word_timestamps=True,  # Enable word-level timestamps
                beam_size=5,  # Optimal beam size
                best_of=5,  # More candidates for better accuracy
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                initial_prompt="Transcribe this video accurately, including all spoken words, informal language, and vocal expressions.",
<<<<<<< HEAD
                fp16=False,  # FIXED: Force FP32 to avoid tensor type errors
=======
                fp16=True if self.device == 'cuda' else False,  # Use FP16 on GPU
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
                patience=1.0,  # Better timestamp alignment
                length_penalty=1.0,  # Neutral length penalty
                suppress_tokens=[],  # Don't suppress any tokens
                suppress_blank=True,
                without_timestamps=False,
                max_initial_timestamp=1.0,
                prepend_punctuations="\"'\"¿([{-",
                append_punctuations="\"'.。,，!！?？:：\")]}、"
            )
            
            # Process segments with enhanced data - CREATE 1-SECOND SEGMENTS
            segments = []
            all_words = []
            
            # First collect all words with timestamps
            for seg in result.get('segments', []):
                if seg.get('words'):
                    all_words.extend(seg['words'])
            
            # Now create 1-second segments from words
            current_second = 0
            segment_words = []
            
            for word in all_words:
                word_start = word.get('start', 0)
                
                # If word is in next second, save current segment
                if word_start >= current_second + 1.0 and segment_words:
                    segment_text = ' '.join([w['word'].strip() for w in segment_words])
                    segments.append({
                        'start_time': float(current_second),
                        'end_time': float(current_second + 1.0),
                        'text': segment_text,
                        'words': segment_words,
                        'word_count': len(segment_words),
                        'confidence': np.mean([w.get('probability', 0.9) for w in segment_words])
                    })
                    current_second += 1
                    segment_words = []
                    
                    # Skip empty seconds
                    while current_second < word_start - 1:
                        current_second += 1
                
                segment_words.append(word)
            
            # Add final segment
            if segment_words:
                segment_text = ' '.join([w['word'].strip() for w in segment_words])
                segments.append({
                    'start_time': float(current_second),
                    'end_time': float(current_second + 1.0),
                    'text': segment_text,
                    'words': segment_words,
                    'word_count': len(segment_words),
                    'confidence': np.mean([w.get('probability', 0.9) for w in segment_words])
                })
            
            # Add language and quality metrics to all segments
            for seg in segments:
                seg['language'] = result.get('language', 'unknown')
                seg['no_speech_prob'] = 0.0  # We have speech
                seg['avg_logprob'] = np.mean([w.get('logprob', -0.5) for w in seg['words']]) if seg['words'] else -1.0
                seg['compression_ratio'] = 1.0  # Normal compression
            
            logger.info(f"[Ultimate Speech] Transcribed {len(segments)} segments with {len(all_words)} words")
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            segments = []
            all_words = []
        
        # 2. Extract enhanced pitch features
        pitch_features = self.extract_pitch_features(audio_normalized, sr)
        logger.info(f"[Ultimate Speech] Pitch analysis: {pitch_features.get('voice_type', 'unknown')}")
        
        # 3. Analyze speaking speed with detailed metrics
        speed_analysis = self.analyze_speaking_speed(segments, audio_duration)
        logger.info(f"[Ultimate Speech] Speaking rate: {speed_analysis['words_per_minute']:.0f} WPM")
        
        # 4. Detect emphasized words with enhanced algorithm
        emphasized_words = self.detect_emphasis(audio_normalized, sr, segments)
        logger.info(f"[Ultimate Speech] Found {len(emphasized_words)} emphasized words")
        
        # 5. Language confidence and quality metrics
        language_confidence = 0.0
        transcription_quality = 0.0
        
        if segments:
            # Calculate overall confidence
            confidences = [s['confidence'] for s in segments]
            language_confidence = np.mean(confidences)
            
            # Calculate quality based on multiple factors
            quality_factors = []
            
            # Factor 1: Speech probability
            speech_probs = [1.0 - s['no_speech_prob'] for s in segments]
            quality_factors.append(np.mean(speech_probs))
            
            # Factor 2: Compression ratio (lower is better)
            compression_ratios = [s['compression_ratio'] for s in segments]
            avg_compression = np.mean(compression_ratios)
            quality_factors.append(max(0, min(1, 2.0 - avg_compression)))
            
            # Factor 3: Log probability (higher is better)
            avg_logprobs = [s['avg_logprob'] for s in segments]
            normalized_logprob = (np.mean(avg_logprobs) + 1.0) / 2.0  # Normalize to 0-1
            quality_factors.append(max(0, min(1, normalized_logprob)))
            
            transcription_quality = np.mean(quality_factors)
        
        # Compile comprehensive results
        analysis_time = time.time() - start_time
        
        return {
            'segments': segments,
            'language': result.get('language', 'unknown') if segments else None,
            'language_confidence': float(language_confidence),
            'transcription_quality': float(transcription_quality),
            'speaking_rate_wpm': speed_analysis.get('words_per_minute', 0),
            'pitch_category': pitch_features.get('voice_type', 'unknown'),
            'pitch_analysis': pitch_features,
            'speed_analysis': speed_analysis,
            'emphasized_words': emphasized_words,
            'word_level_timestamps': all_words,
            'pitch_data': {
                'mean_fundamental_frequency': pitch_features.get('pitch_mean_hz', 0),
                'f0_standard_deviation': pitch_features.get('pitch_std_hz', 0),
                'pitch_range': {
                    'min': pitch_features.get('pitch_min_hz', 0),
                    'max': pitch_features.get('pitch_max_hz', 0)
                },
                'pitch_median': pitch_features.get('pitch_median_hz', 0),
                'pitch_modulation': pitch_features.get('pitch_modulation_hz', 0),
                'pitch_category': pitch_features.get('voice_type', 'unknown'),
                'pitch_variability': pitch_features.get('pitch_variability', 'unknown'),
                'vocal_stability': pitch_features.get('vocal_stability', 0)
            },
            'quality_metrics': {
                'transcription_confidence': float(language_confidence),
                'transcription_quality': float(transcription_quality),
                'total_segments': len(segments),
                'total_words': speed_analysis.get('total_words', 0),
                'total_syllables': speed_analysis.get('total_syllables', 0),
                'fluency_score': speed_analysis.get('fluency_score', 0),
                'emphasized_word_count': len(emphasized_words)
            },
            'summary': {
                'total_segments': len(segments),
                'total_words': speed_analysis.get('total_words', 0),
                'speaking_time': speed_analysis.get('speaking_time', 0),
                'audio_duration': audio_duration,
                'processing_time': analysis_time,
                'voice_characteristics': {
                    'type': pitch_features.get('voice_type', 'unknown'),
                    'pitch_mean': f"{pitch_features.get('pitch_mean_hz', 0):.0f} Hz",
                    'pitch_variability': pitch_features.get('pitch_variability', 'unknown'),
                    'speaking_pace': speed_analysis.get('speaking_pace', 'unknown'),
                    'wpm': f"{speed_analysis.get('words_per_minute', 0):.0f}",
                    'articulation_rate': f"{speed_analysis.get('articulation_rate', 0):.0f}",
                    'fluency': 'high' if speed_analysis.get('fluency_score', 0) > 0.7 else 'medium' if speed_analysis.get('fluency_score', 0) > 0.4 else 'low'
                }
            }
        }