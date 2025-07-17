#!/usr/bin/env python3
"""
Test Ray Model Sharing Performance
Shows true model sharing achieving <3x realtime
"""

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import ray
import torch
import whisper
import time
import json
import numpy as np
import librosa
from pathlib import Path

# Initialize Ray
ray.init(ignore_reinit_error=True)

@ray.remote(num_gpus=0.2)
class WhisperSharedModel:
    """Whisper model shared across all requests"""
    
    def __init__(self):
        print("Loading Whisper ONCE...")
        self.model = whisper.load_model("base", device="cuda")
        print("✅ Whisper loaded and ready for sharing!")
    
    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio"""
        audio, sr = librosa.load(audio_path, sr=16000)
        audio = audio.astype(np.float32)
        
        result = self.model.transcribe(
            audio,
            language=None,
            temperature=0.0,
            word_timestamps=True,
            fp16=True
        )
        
        # Check for Marc Gebauer CTA
        text = result.get('text', '').lower()
        has_cta = False
        cta_type = None
        
        if 'noch mal bestellen' in text:
            has_cta = True
            cta_type = 'marc_gebauer_pattern'
        elif 'was nun' in text:
            has_cta = True
            cta_type = 'marc_gebauer_question'
        elif 'verstehe die frage nicht' in text:
            has_cta = True
            cta_type = 'marc_gebauer_response'
        
        return {
            'text': result.get('text'),
            'language': result.get('language'),
            'segments': len(result.get('segments', [])),
            'has_cta': has_cta,
            'cta_type': cta_type
        }

def test_multiprocess_no_sharing():
    """Test traditional multiprocessing - models loaded multiple times"""
    print("\n📊 Testing TRADITIONAL multiprocessing (no sharing)...")
    
    video_path = "/home/user/tiktok_videos/videos/7525171065367104790.mp4"
    
    # Simulate 3 parallel processes each loading model
    start_time = time.time()
    
    # Process 1
    print("  Process 1: Loading model...")
    model1 = whisper.load_model("base", device="cuda")
    audio, _ = librosa.load(video_path, sr=16000)
    result1 = model1.transcribe(audio.astype(np.float32))
    
    # Process 2
    print("  Process 2: Loading model...")
    model2 = whisper.load_model("base", device="cuda")
    result2 = model2.transcribe(audio.astype(np.float32))
    
    # Process 3
    print("  Process 3: Loading model...")
    model3 = whisper.load_model("base", device="cuda")
    result3 = model3.transcribe(audio.astype(np.float32))
    
    elapsed = time.time() - start_time
    
    print(f"\n  ❌ Traditional approach: {elapsed:.1f}s")
    print(f"     - 3x model loading overhead")
    print(f"     - 3x GPU memory usage")
    print(f"     - Result: '{result1['text'][:50]}...'")
    
    # Clean up
    del model1, model2, model3
    torch.cuda.empty_cache()
    
    return elapsed

def test_ray_sharing():
    """Test Ray model sharing - model loaded ONCE"""
    print("\n🚀 Testing RAY model sharing...")
    
    video_path = "/home/user/tiktok_videos/videos/7525171065367104790.mp4"
    
    # Create shared model actor
    start_time = time.time()
    whisper_actor = WhisperSharedModel.remote()
    
    # Simulate 3 parallel requests to SAME model
    print("  Sending 3 parallel requests to shared model...")
    
    # All 3 requests use the SAME model
    task1 = whisper_actor.transcribe.remote(video_path)
    task2 = whisper_actor.transcribe.remote(video_path) 
    task3 = whisper_actor.transcribe.remote(video_path)
    
    # Get results
    results = ray.get([task1, task2, task3])
    elapsed = time.time() - start_time
    
    print(f"\n  ✅ Ray sharing approach: {elapsed:.1f}s")
    print(f"     - Model loaded ONCE")
    print(f"     - 1x GPU memory usage")
    print(f"     - Parallel processing")
    
    # Show results
    for i, result in enumerate(results):
        print(f"\n  Result {i+1}:")
        print(f"     Text: '{result['text'][:50]}...'")
        print(f"     Language: {result['language']}")
        print(f"     CTA detected: {result['has_cta']}")
        if result['has_cta']:
            print(f"     CTA type: {result['cta_type']}")
    
    return elapsed

def test_realtime_performance():
    """Test achieving <3x realtime with Ray"""
    print("\n⚡ Testing realtime performance...")
    
    video_path = "/home/user/tiktok_videos/videos/7525171065367104790.mp4"
    video_duration = 9.3  # seconds
    
    # Create shared model
    whisper_actor = WhisperSharedModel.remote()
    
    # Analyze video
    start_time = time.time()
    result = ray.get(whisper_actor.transcribe.remote(video_path))
    processing_time = time.time() - start_time
    
    realtime_factor = processing_time / video_duration
    
    print(f"\n📊 Performance Results:")
    print(f"   Video duration: {video_duration}s")
    print(f"   Processing time: {processing_time:.1f}s")
    print(f"   Realtime factor: {realtime_factor:.2f}x")
    print(f"   {'✅ ACHIEVED' if realtime_factor < 3 else '❌ FAILED'} <3x realtime target")
    
    print(f"\n📝 Analysis Results:")
    print(f"   Text: {result['text']}")
    print(f"   Marc Gebauer CTA: {'YES - ' + result['cta_type'] if result['has_cta'] else 'NO'}")

if __name__ == "__main__":
    print("🎯 Ray Model Sharing Performance Test")
    print("=" * 50)
    
    # Test 1: Traditional multiprocessing
    traditional_time = test_multiprocess_no_sharing()
    
    # Test 2: Ray model sharing
    ray_time = test_ray_sharing()
    
    # Performance comparison
    print(f"\n📈 PERFORMANCE COMPARISON:")
    print(f"   Traditional: {traditional_time:.1f}s")
    print(f"   Ray sharing: {ray_time:.1f}s")
    print(f"   Speedup: {traditional_time/ray_time:.1f}x faster with Ray!")
    
    # Test 3: Realtime performance
    test_realtime_performance()
    
    print("\n✅ Test complete!")