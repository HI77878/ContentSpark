#!/usr/bin/env python3
"""Test video analysis with the new API"""
import requests
import json
import time
from pathlib import Path
from mass_processing.tiktok_downloader import TikTokDownloader

# Configuration
API_URL = "http://localhost:8003"
VIDEO_URL = "https://www.tiktok.com/@leon_schliebach/video/7446489995663117590"

print("🎬 Starting video analysis test...")

# Check API
response = requests.get(f"{API_URL}/analyzers")
if response.status_code == 200:
    data = response.json()
    print(f"✅ API running with {data['total']} analyzers")
    print(f"   Key analyzers active: face_emotion={data['key_analyzers']['face_emotion']}, "
          f"body_pose={data['key_analyzers']['body_pose']}, "
          f"cross_analyzer_intelligence={data['key_analyzers']['cross_analyzer_intelligence']}")
else:
    print("❌ API not responding")
    exit(1)

# Download video
print(f"\n📥 Downloading video from {VIDEO_URL}")
downloader = TikTokDownloader()
result = downloader.download_video(VIDEO_URL)

if result['status'] == 'success':
    video_path = result['video_path']
    print(f"✅ Video downloaded: {video_path}")
else:
    print(f"❌ Download failed: {result['error']}")
    exit(1)

# Analyze video
print(f"\n🔬 Analyzing video...")
start_time = time.time()

response = requests.post(
    f"{API_URL}/analyze",
    json={"video_path": video_path}
)

if response.status_code == 200:
    result = response.json()
    print(f"✅ Analysis complete!")
    print(f"   Time: {result['processing_time']:.1f}s")
    print(f"   Successful: {result['successful_analyzers']}/{result['total_analyzers']} analyzers")
    print(f"   Results: {result['results_file']}")
else:
    print(f"❌ Analysis failed: {response.text}")