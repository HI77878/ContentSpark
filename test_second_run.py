#!/usr/bin/env python3
"""Quick re-test - Model should be loaded"""
import time
import requests

print("🔄 SECOND RUN TEST (Model already loaded)")
print("="*50)

video_path = "/home/user/tiktok_production/test_videos/test1.mp4"

start = time.time()
response = requests.post(
    "http://localhost:8003/analyze",
    json={"video_path": video_path},
    timeout=300
)
elapsed = time.time() - start

if response.status_code == 200:
    print(f"✅ Success in {elapsed:.1f}s")
    print(f"🎯 Realtime factor: {elapsed/10:.1f}x")
    
    if elapsed < 30:
        print(f"\n🎉 GOAL ACHIEVED: <3x realtime!")
    else:
        print(f"\n⏱️ Still optimizing needed")
else:
    print(f"❌ Error: {response.status_code}")