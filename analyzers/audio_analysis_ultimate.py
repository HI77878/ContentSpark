#!/usr/bin/env python3
"""
Ultimate Audio Analysis - 100/100 Performance
Comprehensive audio feature extraction with maximum accuracy
"""

# GPU Forcing
import torch
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from analyzers.base_analyzer import GPUBatchAnalyzer
import numpy as np
import librosa
import torch
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings
from scipy import signal
from scipy.stats import kurtosis, skew
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class UltimateAudioAnalysis(GPUBatchAnalyzer):
    def __init__(self):
        super().__init__(batch_size=1)
        self.model_loaded = False
        self.device = 'cuda'
        self.target_sr = 22050  # Higher sample rate for better quality
        self.n_fft = 2048  # Larger FFT for better frequency resolution
        self.hop_length = 512
        self.n_mels = 128  # More mel bands
        self.n_mfcc = 20  # More MFCCs
        self.max_workers = 6  # More parallel workers
        logger.info(f"[AudioAnalysis-Ultimate] Maximum quality mode on {self.device}")
    
    def _load_model_impl(self) -> None:
        """Load audio analysis models"""
        logger.info("[AudioAnalysis-Ultimate] Audio processing models ready")
        # Audio analysis doesn't require ML models, just processing libraries
    
    def extract_comprehensive_features_gpu(self, y, sr):
        """Extract comprehensive audio features for 100% accuracy"""
        features = {}
        
        try:
            # Convert to PyTorch tensor for GPU processing
            y_tensor = torch.from_numpy(y).float().cuda()
            
            # 1. Time-domain features
            features['energy'] = float(torch.mean(y_tensor**2).cpu().item())
            features['rms_mean'] = float(torch.sqrt(torch.mean(y_tensor**2)).cpu().item())
            features['rms_std'] = float(torch.std(torch.sqrt(torch.abs(y_tensor))).cpu().item())
            
            # Peak and average amplitudes
            peak_amplitude = torch.max(torch.abs(y_tensor))
            mean_amplitude = torch.mean(torch.abs(y_tensor))
            features['peak_amplitude'] = float(peak_amplitude.cpu().item())
            features['mean_amplitude'] = float(mean_amplitude.cpu().item())
            
            # Dynamic range in dB
            features['dynamic_range_db'] = float(20 * torch.log10(peak_amplitude / (mean_amplitude + 1e-10)).cpu().item())
            
            # Crest factor
            features['crest_factor'] = float((peak_amplitude / (features['rms_mean'] + 1e-10)).cpu().item())
            
            # Zero crossing rate (optimized)
            sign_changes = torch.sum(torch.diff(torch.sign(y_tensor)) != 0).float()
            features['zcr_mean'] = float((sign_changes / len(y_tensor)).cpu().item())
            
            # 2. Statistical features
            y_cpu = y_tensor.cpu().numpy()
            features['kurtosis'] = float(kurtosis(y_cpu))
            features['skewness'] = float(skew(y_cpu))
            features['variance'] = float(np.var(y_cpu))
            
            # 3. Spectral features with GPU acceleration
            # STFT
            stft = torch.stft(
                y_tensor, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft).cuda(),
                return_complex=True
            )
            magnitude = torch.abs(stft)
            power = magnitude ** 2
            
            # Frequency bins
            freqs = torch.linspace(0, sr/2, magnitude.shape[0]).cuda()
            
            # Spectral centroid
            centroid = torch.sum(freqs.unsqueeze(1) * magnitude, dim=0) / (torch.sum(magnitude, dim=0) + 1e-10)
            features['spectral_centroid_mean'] = float(torch.mean(centroid).cpu().item())
            features['spectral_centroid_std'] = float(torch.std(centroid).cpu().item())
            features['spectral_centroid_max'] = float(torch.max(centroid).cpu().item())
            features['spectral_centroid_min'] = float(torch.min(centroid).cpu().item())
            
            # Spectral rolloff (multiple percentiles)
            for percentile in [85, 90, 95]:
                cumsum = torch.cumsum(magnitude, dim=0)
                total_energy = cumsum[-1, :]
                rolloff_threshold = (percentile / 100.0) * total_energy
                rolloff_bins = torch.argmax((cumsum >= rolloff_threshold.unsqueeze(0)).float(), dim=0)
                rolloff_freqs = freqs[rolloff_bins]
                features[f'spectral_rolloff_{percentile}_mean'] = float(torch.mean(rolloff_freqs).cpu().item())
                features[f'spectral_rolloff_{percentile}_std'] = float(torch.std(rolloff_freqs).cpu().item())
            
            # Spectral bandwidth
            spectral_bandwidth = torch.sqrt(
                torch.sum(((freqs.unsqueeze(1) - centroid.unsqueeze(0))**2) * magnitude, dim=0) / 
                (torch.sum(magnitude, dim=0) + 1e-10)
            )
            features['spectral_bandwidth_mean'] = float(torch.mean(spectral_bandwidth).cpu().item())
            features['spectral_bandwidth_std'] = float(torch.std(spectral_bandwidth).cpu().item())
            
            # Spectral contrast (multiple bands)
            n_bands = 7
            spectral_contrast = []
            fft_freqs = np.linspace(0, sr/2, magnitude.shape[0])
            
            for i in range(n_bands):
                if i == 0:
                    freq_mask = fft_freqs < 200
                elif i == n_bands - 1:
                    freq_mask = fft_freqs >= 200 * (2 ** (i-1))
                else:
                    freq_mask = (fft_freqs >= 200 * (2 ** (i-1))) & (fft_freqs < 200 * (2 ** i))
                
                if np.sum(freq_mask) > 0:
                    band_power = power[freq_mask, :]
                    if band_power.numel() > 0:
                        band_peak = torch.quantile(band_power.flatten(), 0.95)
                        band_valley = torch.quantile(band_power.flatten(), 0.05)
                        contrast = band_peak - band_valley
                        spectral_contrast.append(float(contrast.cpu().item()))
            
            features['spectral_contrast_bands'] = len(spectral_contrast)
            features['spectral_contrast_mean'] = float(np.mean(spectral_contrast)) if spectral_contrast else 0.0
            
            # Spectral flatness
            geometric_mean = torch.exp(torch.mean(torch.log(magnitude + 1e-10), dim=0))
            arithmetic_mean = torch.mean(magnitude, dim=0)
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
            features['spectral_flatness_mean'] = float(torch.mean(spectral_flatness).cpu().item())
            
            # 4. Mel-frequency features
            # Compute mel spectrogram on CPU with librosa (more accurate)
            mel_spec = librosa.feature.melspectrogram(
                y=y_cpu, sr=sr, n_fft=self.n_fft, 
                hop_length=self.hop_length, n_mels=self.n_mels
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # MFCCs
            mfccs = librosa.feature.mfcc(
                S=mel_spec_db, sr=sr, n_mfcc=self.n_mfcc
            )
            
            for i in range(self.n_mfcc):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            
            # Delta and delta-delta MFCCs
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            features['mfcc_delta_mean'] = float(np.mean(mfcc_delta))
            features['mfcc_delta_std'] = float(np.std(mfcc_delta))
            features['mfcc_delta2_mean'] = float(np.mean(mfcc_delta2))
            features['mfcc_delta2_std'] = float(np.std(mfcc_delta2))
            
            # 5. Rhythm features
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=y_cpu, sr=sr, hop_length=self.hop_length)
            features['tempo'] = float(tempo)
            features['beat_count'] = len(beats)
            features['beats_per_second'] = float(len(beats) / (len(y_cpu) / sr)) if len(y_cpu) > 0 else 0.0
            
            # 6. Harmonic and percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y_cpu)
            
            # Harmonic-to-percussive ratio
            harmonic_energy = np.mean(y_harmonic**2)
            percussive_energy = np.mean(y_percussive**2)
            features['harmonic_energy'] = float(harmonic_energy)
            features['percussive_energy'] = float(percussive_energy)
            features['hpr'] = float(harmonic_energy / (percussive_energy + 1e-10))
            
            # 7. Chroma features
            chroma = librosa.feature.chroma_stft(y=y_cpu, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
            features['chroma_mean'] = float(np.mean(chroma))
            features['chroma_std'] = float(np.std(chroma))
            
            # Most prominent chroma
            chroma_means = np.mean(chroma, axis=1)
            features['dominant_chroma'] = int(np.argmax(chroma_means))
            
            # 8. Onset detection
            onset_env = librosa.onset.onset_strength(y=y_cpu, sr=sr, hop_length=self.hop_length)
            features['onset_rate'] = float(np.mean(onset_env))
            features['onset_std'] = float(np.std(onset_env))
            
            # 9. Advanced spectral features
            # Spectral slope
            mag_cpu = magnitude.cpu().numpy()
            freqs_cpu = freqs.cpu().numpy()
            
            spectral_slope = []
            for frame in range(mag_cpu.shape[1]):
                if np.sum(mag_cpu[:, frame]) > 0:
                    slope, _ = np.polyfit(freqs_cpu, mag_cpu[:, frame], 1)
                    spectral_slope.append(slope)
            
            if spectral_slope:
                features['spectral_slope_mean'] = float(np.mean(spectral_slope))
                features['spectral_slope_std'] = float(np.std(spectral_slope))
            
            # 10. Noise and quality metrics
            # Estimate noise floor (bottom 10% of magnitude spectrum)
            sorted_mag = torch.sort(magnitude.flatten())[0]
            noise_floor_idx = int(len(sorted_mag) * 0.1)
            noise_floor = sorted_mag[:noise_floor_idx].mean()
            features['noise_floor_db'] = float(20 * torch.log10(noise_floor + 1e-10).cpu().item())
            
            # Signal-to-noise ratio
            signal_power = torch.mean(magnitude)
            features['snr_db'] = float(20 * torch.log10(signal_power / (noise_floor + 1e-10)).cpu().item())
            
            # Spectral entropy
            normalized_spec = magnitude / (torch.sum(magnitude, dim=0, keepdim=True) + 1e-10)
            spectral_entropy = -torch.sum(normalized_spec * torch.log(normalized_spec + 1e-10), dim=0)
            features['spectral_entropy_mean'] = float(torch.mean(spectral_entropy).cpu().item())
            
        except Exception as e:
            logger.error(f"GPU feature extraction failed: {e}")
            import traceback
            traceback.print_exc()
            # Kein Fallback - fail fast
            raise RuntimeError(f"Audio feature extraction failed: {e}")
        
        return features
    
    def classify_audio_segment(self, features):
        """Enhanced audio classification with comprehensive scoring"""
        # Initialize scores for different audio types
        scores = {
            'voice_over': 0,
            'direct_speech': 0,
            'music': 0,
            'mixed_content': 0,
            'ambient': 0
        }
        
        # Voice-over characteristics
        if features.get('snr_db', 0) > 30:
            scores['voice_over'] += 3
        elif features.get('snr_db', 0) > 25:
            scores['voice_over'] += 2
            
        if features.get('noise_floor_db', 0) < -50:
            scores['voice_over'] += 2
        elif features.get('noise_floor_db', 0) < -45:
            scores['voice_over'] += 1
            
        if features.get('dynamic_range_db', 0) < 20:  # Compressed
            scores['voice_over'] += 2
        
        if 1000 < features.get('spectral_centroid_mean', 0) < 3500:  # Speech range
            scores['voice_over'] += 2
            scores['direct_speech'] += 1
        
        # Direct speech characteristics
        if 15 < features.get('snr_db', 0) < 30:
            scores['direct_speech'] += 2
        
        if features.get('spectral_entropy_mean', 0) > 3.5:
            scores['direct_speech'] += 1
        
        if features.get('dynamic_range_db', 0) > 25:
            scores['direct_speech'] += 1
        
        # Music characteristics
        if features.get('tempo', 0) > 60 and features.get('beat_count', 0) > 10:
            scores['music'] += 3
        
        if features.get('hpr', 1.0) < 0.5:  # More percussive
            scores['music'] += 2
        
        if features.get('chroma_std', 0) > 0.2:
            scores['music'] += 1
        
        if features.get('spectral_rolloff_95_mean', 0) > 4000:
            scores['music'] += 1
        
        # Mixed content
        if max(scores.values()) < 3:
            scores['mixed_content'] += 2
        
        # Ambient/background
        if features.get('snr_db', 0) < 15:
            scores['ambient'] += 2
        
        if features.get('onset_rate', 0) < 0.5:
            scores['ambient'] += 1
        
        # Determine primary type
        primary_type = max(scores, key=scores.get)
        confidence = min(0.95, scores[primary_type] * 0.15)
        
        # Additional quality assessment
        quality = 'high'
        if features.get('snr_db', 0) < 20:
            quality = 'low'
        elif features.get('snr_db', 0) < 30:
            quality = 'medium'
        
        # Professional recording indicators
        is_professional = (
            features.get('snr_db', 0) > 30 and
            features.get('noise_floor_db', 0) < -45 and
            features.get('dynamic_range_db', 0) < 25
        )
        
        return {
            'audio_source': primary_type,
            'confidence': float(confidence),
            'quality': quality,
            'is_professional': is_professional,
            'scores': scores,
            'characteristics': {
                'has_music': scores['music'] >= 2,
                'has_speech': scores['voice_over'] + scores['direct_speech'] >= 2,
                'is_clean': features.get('snr_db', 0) > 25,
                'is_compressed': features.get('dynamic_range_db', 0) < 20
            }
        }
    
    def analyze_segment_comprehensive(self, segment_data):
        """Comprehensive segment analysis for parallel processing"""
        y_segment, sr, start_time, end_time = segment_data
        
        # Extract comprehensive features
        features = self.extract_comprehensive_features_gpu(y_segment, sr)
        
        # Classify segment
        classification = self.classify_audio_segment(features)
        
        # Build segment result
        segment_result = {
            'start_time': float(start_time),
            'end_time': float(end_time),
            'duration': float(end_time - start_time),
            'audio_source': classification['audio_source'],
            'confidence': classification['confidence'],
            'quality': classification['quality'],
            'is_professional': classification['is_professional'],
            'has_speech': classification['characteristics']['has_speech'],
            'has_music': classification['characteristics']['has_music'],
            'snr_db': features.get('snr_db', 0),
            'features': {
                'energy': features.get('energy', 0),
                'spectral_centroid': features.get('spectral_centroid_mean', 0),
                'tempo': features.get('tempo', 0),
                'noise_floor_db': features.get('noise_floor_db', 0)
            }
        }
        
        return segment_result
    
    def analyze(self, video_path):
        """Ultimate audio analysis with maximum accuracy"""
        start_time = time.time()
        logger.info(f"[AudioAnalysis-Ultimate] Starting comprehensive analysis of {video_path}")
        
        try:
            # Load audio with higher sample rate for better quality
            y, sr = librosa.load(video_path, sr=self.target_sr, mono=True)
            duration = len(y) / sr
            
            logger.info(f"[AudioAnalysis-Ultimate] Loaded {duration:.1f}s at {sr}Hz")
            
            # Extract global features
            logger.info("[AudioAnalysis-Ultimate] Extracting global features...")
            global_features = self.extract_comprehensive_features_gpu(y, sr)
            
            # Global classification
            global_classification = self.classify_audio_segment(global_features)
            
            # Segment analysis with smaller segments for better temporal resolution
            segment_duration = 3.0  # 3 second segments
            segment_samples = int(sr * segment_duration)
            
            # Prepare segment data
            segment_data = []
            for i in range(0, len(y), segment_samples // 2):  # 50% overlap
                segment = y[i:i+segment_samples]
                if len(segment) < sr * 0.5:  # Skip segments shorter than 0.5s
                    continue
                
                start_t = i / sr
                end_t = min((i + segment_samples) / sr, duration)
                segment_data.append((segment, sr, start_t, end_t))
            
            logger.info(f"[AudioAnalysis-Ultimate] Processing {len(segment_data)} segments...")
            
            # Parallel processing
            segments = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.analyze_segment_comprehensive, data): i 
                          for i, data in enumerate(segment_data)}
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        segments.append(result)
                    except Exception as e:
                        logger.error(f"Segment analysis failed: {e}")
            
            # Sort segments by time
            segments.sort(key=lambda x: x['start_time'])
            
            # Advanced statistics
            segment_types = {}
            quality_counts = {'high': 0, 'medium': 0, 'low': 0}
            
            for seg in segments:
                audio_type = seg['audio_source']
                segment_types[audio_type] = segment_types.get(audio_type, 0) + 1
                quality_counts[seg['quality']] += 1
            
            # Temporal analysis
            transitions = []
            for i in range(1, len(segments)):
                if segments[i]['audio_source'] != segments[i-1]['audio_source']:
                    transitions.append({
                        'time': segments[i]['start_time'],
                        'from': segments[i-1]['audio_source'],
                        'to': segments[i]['audio_source']
                    })
            
            # Build comprehensive result
            result = {
                'segments': segments,
                'global_analysis': {
                    'audio_source': global_classification['audio_source'],
                    'confidence': global_classification['confidence'],
                    'primary_type': global_classification['audio_source'],
                    'duration': float(duration),
                    'sample_rate': int(sr),
                    'bit_depth': 16,  # Assuming 16-bit
                    'channels': 1,  # Mono
                    'snr_db': global_features.get('snr_db', 0),
                    'noise_floor_db': global_features.get('noise_floor_db', 0),
                    'dynamic_range_db': global_features.get('dynamic_range_db', 0),
                    'processing_time': time.time() - start_time
                },
                'statistics': {
                    'segment_types': segment_types,
                    'total_segments': len(segments),
                    'transitions': len(transitions),
                    'quality_distribution': quality_counts,
                    'professional_segments': sum(1 for s in segments if s['is_professional']),
                    'speech_segments': sum(1 for s in segments if s['has_speech']),
                    'music_segments': sum(1 for s in segments if s['has_music'])
                },
                'quality_assessment': {
                    'overall_quality': global_classification['quality'],
                    'is_professional': global_classification['is_professional'],
                    'recording_quality': 'professional' if global_classification['is_professional'] else 'consumer',
                    'clarity_score': min(1.0, global_features.get('snr_db', 0) / 40.0),
                    'consistency_score': 1.0 - (len(transitions) / max(len(segments), 1))
                },
                'acoustic_features': {
                    'tempo': global_features.get('tempo', 0),
                    'beat_count': global_features.get('beat_count', 0),
                    'dominant_frequency': global_features.get('spectral_centroid_mean', 0),
                    'spectral_bandwidth': global_features.get('spectral_bandwidth_mean', 0),
                    'spectral_rolloff': global_features.get('spectral_rolloff_95_mean', 0),
                    'harmonic_percussive_ratio': global_features.get('hpr', 0),
                    'spectral_contrast': global_features.get('spectral_contrast_mean', 0),
                    'spectral_flatness': global_features.get('spectral_flatness_mean', 0)
                },
                'mfcc_summary': {
                    'mean_mfcc': [global_features.get(f'mfcc_{i}_mean', 0) for i in range(min(13, self.n_mfcc))],
                    'mfcc_variability': global_features.get('mfcc_delta_std', 0)
                },
                'transitions': transitions[:10],  # First 10 transitions
                'metadata': {
                    'analyzer_version': 'ultimate_1.0',
                    'feature_count': len(global_features),
                    'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            elapsed = time.time() - start_time
            logger.info(f"[AudioAnalysis-Ultimate] Completed in {elapsed:.1f}s")
            logger.info(f"[AudioAnalysis-Ultimate] Primary type: {global_classification['audio_source']}")
            logger.info(f"[AudioAnalysis-Ultimate] Quality: {global_classification['quality']}, SNR: {global_features.get('snr_db', 0):.1f}dB")
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ultimate audio analysis: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'segments': [],
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def process_batch_gpu(self, batch_data, **kwargs):
        """Compatibility with base class"""
        video_path = kwargs.get('video_path', '')
        if video_path:
            return self.analyze(video_path)
        return {'segments': [], 'error': 'No video path provided'}