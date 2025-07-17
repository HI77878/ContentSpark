#!/usr/bin/env python3
"""
BRUTAL OPTIMIZATION TEST - Qwen2-VL muss <15s schaffen!
"""
import time
import torch
import psutil
import GPUtil

print("🔥 BRUTAL OPTIMIZATION TEST")
print("="*60)

# GPU Status
gpus = GPUtil.getGPUs()
if gpus:
    gpu = gpus[0]
    print(f"GPU: {gpu.name}")
    print(f"Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
    print(f"Utilization: {gpu.load*100:.1f}%")

# Test Qwen2-VL direkt
print("\n🚀 Testing Qwen2-VL OPTIMIZED...")
from analyzers.qwen2_vl_temporal_analyzer import Qwen2VLTemporalAnalyzer

analyzer = Qwen2VLTemporalAnalyzer()
video = "/home/user/tiktok_production/test_videos/test1.mp4"

start = time.time()
result = analyzer.analyze(video)
elapsed = time.time() - start

segments = len(result.get('segments', []))
print(f"\n✅ Qwen2-VL Results:")
print(f"   Segments: {segments}")
print(f"   Time: {elapsed:.1f}s")
print(f"   Per segment: {elapsed/segments:.1f}s" if segments > 0 else "")
print(f"   Target: <3s per segment")

if elapsed < 15:
    print(f"\n🎉 SUCCESS! Qwen2-VL optimized to {elapsed:.1f}s!")
else:
    print(f"\n❌ STILL TOO SLOW: {elapsed:.1f}s")

# Test full API
print("\n📊 Testing Full API...")
import requests

start = time.time()
response = requests.post("http://localhost:8003/analyze", 
                       json={"video_path": video},
                       timeout=300)
api_time = time.time() - start

if response.status_code == 200:
    print(f"✅ API Response in {api_time:.1f}s")
    print(f"   Realtime factor: {api_time/10:.1f}x")
    
    if api_time < 30:
        print(f"🎉 GOAL ACHIEVED: <3x realtime!")
else:
    print(f"❌ API Error: {response.status_code}")

print("="*60)