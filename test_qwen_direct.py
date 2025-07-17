#!/usr/bin/env python3
"""Direct Qwen2-VL test"""
import time
import torch
print("🔥 DIRECT QWEN2-VL OPTIMIZATION TEST")
print("="*60)

# GPU Status before
if torch.cuda.is_available():
    print(f"GPU Memory before: {torch.cuda.memory_allocated()/1024**3:.1f}GB allocated")

# Import and test
print("\n🚀 Loading Qwen2-VL...")
from analyzers.qwen2_vl_temporal_analyzer import Qwen2VLTemporalAnalyzer

analyzer = Qwen2VLTemporalAnalyzer()

# GPU Status after loading
if torch.cuda.is_available():
    print(f"GPU Memory after loading: {torch.cuda.memory_allocated()/1024**3:.1f}GB allocated")

video = "/home/user/tiktok_production/test_videos/test1.mp4"
print(f"\n📹 Analyzing: {video}")

start = time.time()
result = analyzer.analyze(video)
elapsed = time.time() - start

segments = len(result.get('segments', []))
metadata = result.get('metadata', {})

print(f"\n✅ Results:")
print(f"   Segments: {segments}")
print(f"   Total time: {elapsed:.1f}s")
print(f"   Batch time: {metadata.get('batch_time', 'N/A')}s")
print(f"   Per segment: {metadata.get('per_segment_time', 'N/A')}s")

if segments > 0:
    print(f"\n📝 First segment description:")
    print(f"   {result['segments'][0]['description'][:200]}...")
    
if elapsed < 15:
    print(f"\n🎉 SUCCESS! Qwen2-VL runs in {elapsed:.1f}s (<15s target)")
elif elapsed < 30:
    print(f"\n👍 GOOD! Qwen2-VL runs in {elapsed:.1f}s (<30s acceptable)")
else:
    print(f"\n❌ TOO SLOW: {elapsed:.1f}s")

# Final GPU status
if torch.cuda.is_available():
    print(f"\nGPU Memory final: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

print("="*60)