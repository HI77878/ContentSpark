#!/usr/bin/env python3
"""
Audio Processing Fix mit forkserver multiprocessing
Löst ProcessPoolExecutor Probleme für Audio-Analyzer
"""

import multiprocessing
multiprocessing.set_start_method('forkserver', force=True)

import os
import sys
import time
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Fix FFmpeg environment
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "protocol_whitelist=file,http,https,tcp,tls"
os.environ['OPENCV_VIDEOIO_PRIORITY_BACKEND'] = "4"
os.environ['OPENCV_FFMPEG_MULTITHREADED'] = "0"

sys.path.insert(0, '/home/user/tiktok_production')

logger = logging.getLogger(__name__)

def process_audio_segment(args):
    """Top-level function für multiprocessing - alle imports innerhalb!"""
    audio_path, start_time, end_time, analysis_type = args
    
    try:
        # Alle imports innerhalb der Funktion für forkserver
        import librosa
        import numpy as np
        import warnings
        warnings.filterwarnings('ignore')
        
        # Audio laden mit librosa
        y, sr = librosa.load(audio_path, offset=start_time, duration=end_time-start_time, sr=16000)
        
        if len(y) == 0:
            return None
            
        result = {
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time
        }
        
        if analysis_type == 'spectral':
            # Spectral features
            if len(y) > 512:  # Minimum für FFT
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
                
                result.update({
                    'spectral_centroid': float(np.mean(spectral_centroid)),
                    'spectral_rolloff': float(np.mean(spectral_rolloff)),
                    'zero_crossing_rate': float(np.mean(zero_crossing_rate)),
                    'rms_energy': float(np.sqrt(np.mean(y**2)))
                })
        
        elif analysis_type == 'mfcc':
            # MFCC features
            if len(y) > 512:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                result.update({
                    'mfcc_mean': mfcc.mean(axis=1).tolist(),
                    'mfcc_std': mfcc.std(axis=1).tolist()
                })
        
        elif analysis_type == 'emotion':
            # Speech emotion features
            if len(y) > 512:
                # Prosodic features
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
                
                # Tempo
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                
                result.update({
                    'pitch_mean': float(pitch_mean),
                    'tempo': float(tempo),
                    'intensity': float(np.mean(np.abs(y))),
                    'speech_rate': float(len(y) / sr)  # Duration-based speech rate
                })
        
        return result
        
    except Exception as e:
        logger.error(f"Audio processing error {start_time}-{end_time}s: {e}")
        return None

class AudioAnalyzerFixed:
    """Fixed audio analyzer mit forkserver multiprocessing"""
    
    def __init__(self):
        self.segment_duration = 1.0  # 1 second segments
        self.max_workers = 4
        
    def analyze(self, video_path: str, analysis_type: str = 'spectral'):
        """
        Analyze audio with fixed multiprocessing
        
        Args:
            video_path: Path to video file
            analysis_type: 'spectral', 'mfcc', or 'emotion'
        """
        start_time = time.time()
        
        try:
            # Extract audio to temp file
            audio_path = self._extract_audio(video_path)
            
            if not audio_path or not Path(audio_path).exists():
                logger.error(f"Failed to extract audio from {video_path}")
                return {'segments': [], 'error': 'Audio extraction failed'}
            
            # Get audio duration
            duration = self._get_audio_duration(audio_path)
            
            if duration <= 0:
                logger.error(f"Invalid audio duration: {duration}")
                return {'segments': [], 'error': 'Invalid audio duration'}
            
            # Create segments
            segments = []
            for i in range(0, int(duration), int(self.segment_duration)):
                segment_start = i
                segment_end = min(i + self.segment_duration, duration)
                segments.append((audio_path, segment_start, segment_end, analysis_type))
            
            logger.info(f"Processing {len(segments)} audio segments with forkserver")
            
            # Process with forkserver multiprocessing
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(process_audio_segment, segments))
            
            # Clean up temp file
            self._cleanup_temp_file(audio_path)
            
            # Filter None results
            valid_results = [r for r in results if r is not None]
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Audio analysis complete: {len(valid_results)} segments in {elapsed:.1f}s")
            
            return {
                'segments': valid_results,
                'metadata': {
                    'duration': duration,
                    'analysis_type': analysis_type,
                    'processing_time': elapsed,
                    'segment_count': len(valid_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return {'segments': [], 'error': str(e)}
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video"""
        try:
            # Create temp audio file
            temp_audio = video_path.replace('.mp4', '_temp_audio.wav')
            
            # Use ffmpeg to extract audio
            cmd = f'ffmpeg -i "{video_path}" -ac 1 -ar 16000 -y "{temp_audio}" 2>/dev/null'
            result = os.system(cmd)
            
            if result == 0 and Path(temp_audio).exists():
                return temp_audio
            else:
                logger.error(f"FFmpeg extraction failed with code {result}")
                return None
                
        except Exception as e:
            logger.error(f"Audio extraction error: {e}")
            return None
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration using librosa"""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) / sr
        except Exception as e:
            logger.error(f"Duration calculation error: {e}")
            return 0.0
    
    def _cleanup_temp_file(self, audio_path: str):
        """Clean up temporary audio file"""
        try:
            if Path(audio_path).exists():
                os.remove(audio_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {audio_path}: {e}")

# Specific analyzer implementations
class AudioAnalysisUltimateFixed(AudioAnalyzerFixed):
    """Fixed version of UltimateAudioAnalysis"""
    
    def analyze(self, video_path: str):
        return super().analyze(video_path, 'spectral')

class AudioEnvironmentEnhancedFixed(AudioAnalyzerFixed):
    """Fixed version of AudioEnvironmentEnhanced"""
    
    def analyze(self, video_path: str):
        return super().analyze(video_path, 'spectral')

class GPUBatchSpeechEmotionFixed(AudioAnalyzerFixed):
    """Fixed version of GPUBatchSpeechEmotion"""
    
    def analyze(self, video_path: str):
        return super().analyze(video_path, 'emotion')

class GPUBatchSpeechRateEnhancedFixed(AudioAnalyzerFixed):
    """Fixed version of GPUBatchSpeechRateEnhanced"""
    
    def analyze(self, video_path: str):
        return super().analyze(video_path, 'emotion')

class GPUBatchSpeechFlowFixed(AudioAnalyzerFixed):
    """Fixed version of GPUBatchSpeechFlow"""
    
    def analyze(self, video_path: str):
        return super().analyze(video_path, 'mfcc')

# Test function
def test_audio_fix():
    """Test the fixed audio analyzer"""
    
    video_path = "/home/user/tiktok_production/test_videos/test1.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Test video not found: {video_path}")
        return
    
    print("🎵 Testing fixed audio analyzer...")
    
    analyzer = AudioAnalysisUltimateFixed()
    result = analyzer.analyze(video_path)
    
    segments = len(result.get('segments', []))
    error = result.get('error', '')
    
    if segments > 0:
        print(f"✅ Audio analyzer works: {segments} segments")
        print(f"📊 Sample segment: {result['segments'][0]}")
    else:
        print(f"❌ Audio analyzer failed: {error}")
    
    return result

if __name__ == "__main__":
    test_audio_fix()