#!/usr/bin/env python3
"""
Sound Effects Detector
Detects non-speech audio events like music, clicks, applause, etc.
"""

import numpy as np
import librosa
from typing import List, Dict, Any, Tuple
from analyzers.base_analyzer import GPUBatchAnalyzer
import logging
from scipy.signal import find_peaks
import torch

logger = logging.getLogger(__name__)

class SoundEffectsDetector(GPUBatchAnalyzer):
    def __init__(self):
        super().__init__(batch_size=1)
        self.analyzer_name = "sound_effects"
        self.sr = 22050  # Standard sample rate for audio analysis
        
    def _load_model_impl(self):
        """No model needed - using signal processing"""
        self.model = "signal_processing"
        print("✅ Sound Effects Detector initialized")
        
    def detect_onset_events(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Detect sudden audio events (clicks, hits, etc.)"""
        events = []
        
        # Compute onset strength
        onset_envelope = librosa.onset.onset_strength(y=audio, sr=sr)
        
        # Find peaks in onset strength
        peaks, properties = find_peaks(
            onset_envelope, 
            height=np.mean(onset_envelope) * 2,  # 2x average
            distance=int(sr * 0.1)  # Min 100ms between events
        )
        
        # Convert to time and classify
        hop_length = 512
        for idx, peak in enumerate(peaks):
            timestamp = librosa.frames_to_time(peak, sr=sr, hop_length=hop_length)
            strength = properties['peak_heights'][idx]
            
            # Extract short segment around event
            start_sample = max(0, int(timestamp * sr - 0.05 * sr))
            end_sample = min(len(audio), int(timestamp * sr + 0.05 * sr))
            segment = audio[start_sample:end_sample]
            
            # Analyze segment characteristics
            event_type = self.classify_onset_event(segment, sr, strength)
            
            events.append({
                'timestamp': float(timestamp),
                'type': event_type,
                'strength': float(strength),
                'duration': 0.1,  # Approximate
                'category': 'onset'
            })
        
        return events
    
    def classify_onset_event(self, segment: np.ndarray, sr: int, strength: float) -> str:
        """Classify type of onset event"""
        # Spectral centroid - brightness
        centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
        
        # Zero crossing rate - noisiness
        zcr = np.mean(librosa.feature.zero_crossing_rate(segment))
        
        # Spectral rolloff
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr))
        
        # Classification logic
        if centroid > 4000 and zcr > 0.1:
            return "click" if strength < 10 else "snap"
        elif centroid < 1000 and strength > 20:
            return "thud" if zcr < 0.05 else "knock"
        elif rolloff > 5000 and strength > 15:
            return "clap" if zcr > 0.08 else "hit"
        else:
            return "impact"
    
    def detect_continuous_sounds(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Detect continuous sounds like music, ambient noise"""
        sounds = []
        
        # Segment audio into chunks
        chunk_duration = 1.0  # 1 second chunks
        chunk_samples = int(chunk_duration * sr)
        
        for i in range(0, len(audio) - chunk_samples, chunk_samples // 2):  # 50% overlap
            chunk = audio[i:i + chunk_samples]
            timestamp = i / sr
            
            # Extract features
            features = self.extract_audio_features(chunk, sr)
            
            # Classify chunk
            sound_type = self.classify_continuous_sound(features)
            
            if sound_type != "silence":
                sounds.append({
                    'timestamp': float(timestamp),
                    'type': sound_type,
                    'duration': chunk_duration,
                    'category': 'continuous',
                    'confidence': features['confidence']
                })
        
        # Merge consecutive similar sounds
        merged_sounds = self.merge_consecutive_sounds(sounds)
        
        return merged_sounds
    
    def extract_audio_features(self, chunk: np.ndarray, sr: int) -> Dict:
        """Extract comprehensive audio features"""
        features = {}
        
        # Energy
        features['energy'] = float(np.mean(chunk ** 2))
        features['energy_db'] = float(librosa.amplitude_to_db(features['energy']))
        
        # Spectral features
        features['centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=chunk, sr=sr)))
        features['bandwidth'] = float(np.mean(librosa.feature.spectral_bandwidth(y=chunk, sr=sr)))
        features['rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(y=chunk, sr=sr)))
        features['zcr'] = float(np.mean(librosa.feature.zero_crossing_rate(chunk)))
        
        # Harmonic content
        harmonic, percussive = librosa.effects.hpss(chunk)
        features['harmonic_ratio'] = float(np.mean(harmonic ** 2) / (np.mean(chunk ** 2) + 1e-6))
        features['percussive_ratio'] = float(np.mean(percussive ** 2) / (np.mean(chunk ** 2) + 1e-6))
        
        # Tempo detection for music
        tempo, _ = librosa.beat.beat_track(y=chunk, sr=sr)
        features['tempo'] = float(tempo)
        
        # MFCC for texture
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = float(np.mean(mfcc))
        features['mfcc_std'] = float(np.std(mfcc))
        
        # Confidence based on energy
        features['confidence'] = min(1.0, features['energy'] * 100)
        
        return features
    
    def classify_continuous_sound(self, features: Dict) -> str:
        """Classify continuous sound based on features"""
        # Silence detection
        if features['energy_db'] < -40:
            return "silence"
        
        # Music detection
        if features['tempo'] > 60 and features['tempo'] < 200 and features['harmonic_ratio'] > 0.5:
            if features['centroid'] > 2000:
                return "music_upbeat"
            else:
                return "music_calm"
        
        # Ambient sounds
        if features['zcr'] < 0.05 and features['centroid'] < 1000:
            return "ambient_low"  # Like air conditioner, traffic
        elif features['zcr'] > 0.1 and features['bandwidth'] > 2000:
            return "ambient_noise"  # General background noise
        
        # Nature sounds
        if features['harmonic_ratio'] < 0.3 and features['percussive_ratio'] > 0.5:
            if features['centroid'] > 3000:
                return "nature_birds"
            else:
                return "nature_wind"
        
        # Crowd/people
        if features['mfcc_std'] > 5 and features['centroid'] > 1500 and features['centroid'] < 3000:
            return "crowd_noise"
        
        # Default
        return "background_sound"
    
    def merge_consecutive_sounds(self, sounds: List[Dict]) -> List[Dict]:
        """Merge consecutive sounds of the same type"""
        if not sounds:
            return []
        
        merged = []
        current = sounds[0].copy()
        
        for sound in sounds[1:]:
            # Check if same type and close in time
            if (sound['type'] == current['type'] and 
                sound['timestamp'] - (current['timestamp'] + current['duration']) < 0.5):
                # Extend current sound
                current['duration'] = sound['timestamp'] + sound['duration'] - current['timestamp']
                current['confidence'] = max(current['confidence'], sound['confidence'])
            else:
                # Save current and start new
                merged.append(current)
                current = sound.copy()
        
        merged.append(current)
        return merged
    
    def detect_special_effects(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Detect special audio effects like transitions, swooshes"""
        effects = []
        
        # Detect frequency sweeps (whoosh, transition sounds)
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Look for diagonal patterns in spectrogram (frequency sweeps)
        for i in range(magnitude.shape[1] - 10):
            # Check for rising or falling frequency pattern
            column_centroids = []
            for j in range(10):
                col = magnitude[:, i + j]
                if np.sum(col) > 0:
                    freqs = librosa.fft_frequencies(sr=sr)
                    centroid = np.sum(freqs * col) / np.sum(col)
                    column_centroids.append(centroid)
            
            if len(column_centroids) > 5:
                # Check for consistent rise or fall
                diffs = np.diff(column_centroids)
                if np.all(diffs > 100):  # Rising sweep
                    timestamp = librosa.frames_to_time(i, sr=sr)
                    effects.append({
                        'timestamp': float(timestamp),
                        'type': 'whoosh_up',
                        'duration': 0.5,
                        'category': 'effect'
                    })
                elif np.all(diffs < -100):  # Falling sweep
                    timestamp = librosa.frames_to_time(i, sr=sr)
                    effects.append({
                        'timestamp': float(timestamp),
                        'type': 'whoosh_down',
                        'duration': 0.5,
                        'category': 'effect'
                    })
        
        return effects
    
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> Dict[str, Any]:
        """Not used for audio analysis"""
        return {'segments': []}
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Detect all sound effects in video"""
        print(f"[Sound Effects] Starting analysis of {video_path}")
        
        # Load audio
        try:
            audio, sr = librosa.load(video_path, sr=self.sr)
            duration = len(audio) / sr
            print(f"[Sound Effects] Audio duration: {duration:.1f}s")
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return {'segments': [], 'error': str(e)}
        
        # Detect different types of sounds
        onset_events = self.detect_onset_events(audio, sr)
        continuous_sounds = self.detect_continuous_sounds(audio, sr)
        special_effects = self.detect_special_effects(audio, sr)
        
        # Combine all detections
        all_segments = onset_events + continuous_sounds + special_effects
        
        # Sort by timestamp
        all_segments.sort(key=lambda x: x['timestamp'])
        
        # Add function classification
        for segment in all_segments:
            segment['function'] = self.classify_function(segment)
        
        print(f"[Sound Effects] Detected {len(all_segments)} sound effects")
        
        # Create summary
        summary = {
            'total_effects': len(all_segments),
            'onset_events': len(onset_events),
            'continuous_sounds': len(continuous_sounds),
            'special_effects': len(special_effects),
            'duration': duration,
            'dominant_sounds': self.get_dominant_sounds(all_segments)
        }
        
        return {
            'segments': all_segments,
            'summary': summary
        }
    
    def classify_function(self, segment: Dict) -> str:
        """Classify the function/purpose of a sound effect"""
        sound_type = segment['type']
        
        # Transition effects
        if 'whoosh' in sound_type or 'sweep' in sound_type:
            return "transition"
        
        # Emphasis effects
        if sound_type in ['click', 'snap', 'clap', 'hit']:
            return "emphasis"
        
        # Atmosphere
        if 'ambient' in sound_type or 'nature' in sound_type or 'crowd' in sound_type:
            return "atmosphere"
        
        # Music
        if 'music' in sound_type:
            return "soundtrack"
        
        # Impact
        if sound_type in ['thud', 'knock', 'impact']:
            return "impact"
        
        return "background"
    
    def get_dominant_sounds(self, segments: List[Dict]) -> List[str]:
        """Get the most common sound types"""
        if not segments:
            return []
        
        sound_counts = {}
        for seg in segments:
            sound_type = seg['type']
            sound_counts[sound_type] = sound_counts.get(sound_type, 0) + seg.get('duration', 1)
        
        # Sort by total duration
        sorted_sounds = sorted(sound_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [sound[0] for sound in sorted_sounds[:5]]  # Top 5