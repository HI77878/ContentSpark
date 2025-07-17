#!/usr/bin/env python3
"""
Optimized Speech Transcription using faster-whisper (4x speed improvement)
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class SpeechTranscriptionFaster:
    """Optimized Whisper using faster-whisper (4x speed)"""
    
    def __init__(self):
        self.model = None
        self.model_size = "large-v3"
        self.device = "cpu"  # Use CPU to avoid cuDNN issues
        self.compute_type = "int8"
        
    def _load_model(self):
        """Lazy load faster-whisper model"""
        if self.model is None:
            try:
                logger.info(f"Loading faster-whisper model: {self.model_size}")
                from faster_whisper import WhisperModel
                
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    num_workers=4,
                    cpu_threads=4
                )
                logger.info("✅ Faster-whisper loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load faster-whisper: {e}")
                # Fallback to CPU if CUDA fails
                try:
                    self.model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="int8",
                        num_workers=2
                    )
                    logger.info("✅ Faster-whisper loaded on CPU")
                except Exception as e2:
                    logger.error(f"Failed to load faster-whisper on CPU: {e2}")
                    raise e2
                    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Transcribe speech using faster-whisper"""
        start_time = time.time()
        
        try:
            # Load model if not already loaded
            self._load_model()
            
            if self.model is None:
                return {
                    'segments': [],
                    'error': 'Model not loaded'
                }
            
            logger.info(f"🎤 Transcribing: {Path(video_path).name}")
            
            # Transcribe with VAD and optimal settings
            segments, info = self.model.transcribe(
                video_path,
                beam_size=5,
                best_of=5,
                patience=1,
                length_penalty=1,
                repetition_penalty=1.2,
                no_repeat_ngram_size=1,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=400
                )
            )
            
            # Convert generator to list
            results = []
            for segment in segments:
                if segment.text.strip():  # Only include non-empty segments
                    results.append({
                        'start_time': float(segment.start),
                        'end_time': float(segment.end),
                        'text': segment.text.strip(),
                        'confidence': float(segment.avg_logprob),
                        'no_speech_prob': float(segment.no_speech_prob),
                        'words': [
                            {
                                'start': float(word.start),
                                'end': float(word.end),
                                'word': word.word,
                                'probability': float(word.probability)
                            }
                            for word in segment.words
                        ] if hasattr(segment, 'words') and segment.words else []
                    })
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Transcription complete: {len(results)} segments in {elapsed:.1f}s")
            
            return {
                'segments': results,
                'metadata': {
                    'language': info.language,
                    'language_probability': float(info.language_probability),
                    'duration': float(info.duration),
                    'processing_time': elapsed,
                    'model_size': self.model_size,
                    'device': self.device,
                    'vad_enabled': True
                }
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Transcription error: {e}")
            return {
                'segments': [],
                'error': str(e),
                'metadata': {
                    'processing_time': elapsed,
                    'model_size': self.model_size,
                    'device': self.device
                }
            }
    
    def cleanup(self):
        """Clean up model resources"""
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("🧹 Faster-whisper model cleaned up")

# Alias for compatibility
UltimateSpeechTranscription = SpeechTranscriptionFaster

def test_faster_whisper():
    """Test faster-whisper implementation"""
    
    video_path = "/home/user/tiktok_production/test_videos/test1.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Test video not found: {video_path}")
        return
    
    print("🎤 Testing faster-whisper transcription...")
    
    analyzer = SpeechTranscriptionFaster()
    result = analyzer.analyze(video_path)
    
    segments = len(result.get('segments', []))
    error = result.get('error', '')
    processing_time = result.get('metadata', {}).get('processing_time', 0)
    
    if segments > 0:
        print(f"✅ Faster-whisper works: {segments} segments in {processing_time:.1f}s")
        print(f"📊 Sample segment: {result['segments'][0]['text'][:100]}...")
        print(f"🌍 Language: {result['metadata']['language']}")
    else:
        print(f"❌ Faster-whisper failed: {error}")
    
    # Cleanup
    analyzer.cleanup()
    
    return result

if __name__ == "__main__":
    test_faster_whisper()