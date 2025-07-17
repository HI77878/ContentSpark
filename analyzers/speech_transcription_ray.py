#!/usr/bin/env python3
"""
Speech Transcription with Ray Model Sharing
"""

import ray
import whisper
import numpy as np
import librosa
import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Global Ray actor handle
_whisper_actor = None

@ray.remote(num_gpus=0.2)
class WhisperActor:
    """Whisper model as Ray actor for shared access"""
    
    def __init__(self):
        logger.info("Loading Whisper model in Ray actor...")
        self.model = whisper.load_model("base", device="cuda")
        logger.info("✅ Whisper model loaded in Ray actor")
    
    def transcribe(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Transcribe audio array"""
        result = self.model.transcribe(
            audio_array,
            language=None,
            task='transcribe',
            verbose=False,
            temperature=0.0,
            word_timestamps=True,
            beam_size=5,
            best_of=5,
            fp16=True
        )
        return result

class SpeechTranscriptionRay:
    """Speech transcription analyzer using Ray for model sharing"""
    
    def __init__(self):
        self.analyzer_name = "speech_transcription_ray"
        self._ensure_actor()
    
    def _ensure_actor(self):
        """Ensure Ray actor is initialized"""
        global _whisper_actor
        if _whisper_actor is None:
            ray.init(ignore_reinit_error=True)
            _whisper_actor = WhisperActor.remote()
            logger.info("Created Whisper Ray actor")
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Analyze speech in video"""
        try:
            # Load audio
            audio, sr = librosa.load(video_path, sr=16000)
            
            # Normalize
            audio_normalized = audio / (np.max(np.abs(audio)) + 1e-10)
            audio_fp32 = audio_normalized.astype(np.float32)
            
            # Transcribe with Ray actor
            result = ray.get(_whisper_actor.transcribe.remote(audio_fp32))
            
            # Format segments
            segments = []
            for seg in result.get('segments', []):
                segments.append({
                    'start_time': float(seg['start']),
                    'end_time': float(seg['end']),
                    'text': seg['text'].strip(),
                    'language': result.get('language', 'unknown'),
                    'confidence': seg.get('confidence', 0.95)
                })
            
            # Check for Marc Gebauer CTAs
            cta_segments = []
            marc_patterns = ['noch mal bestellen', 'was nun', 'verstehe die frage nicht']
            
            for seg in segments:
                text_lower = seg['text'].lower()
                if any(pattern in text_lower for pattern in marc_patterns):
                    cta_segments.append({
                        'start_time': seg['start_time'],
                        'end_time': seg['end_time'],
                        'text': seg['text'],
                        'is_cta': True,
                        'cta_type': 'marc_gebauer_pattern'
                    })
            
            return {
                'segments': segments,
                'language': result.get('language'),
                'text': result.get('text', ''),
                'total_segments': len(segments),
                'cta_segments': cta_segments,
                'has_marc_gebauer_cta': len(cta_segments) > 0,
                'ray_actor_used': True
            }
            
        except Exception as e:
            logger.error(f"Speech transcription failed: {e}")
            return {"error": str(e)}

# For testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyzer = SpeechTranscriptionRay()
        result = analyzer.analyze(sys.argv[1])
        
        print(f"Language: {result.get('language')}")
        print(f"Segments: {result.get('total_segments')}")
        print(f"Marc Gebauer CTA: {result.get('has_marc_gebauer_cta')}")
        
        if result.get('cta_segments'):
            print("\nCTA Segments:")
            for seg in result['cta_segments']:
                print(f"  [{seg['start_time']:.1f}s]: {seg['text']}")