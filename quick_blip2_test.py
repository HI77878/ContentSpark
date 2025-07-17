#!/usr/bin/env python3
"""Quick BLIP-2 test without full API"""
import sys
sys.path.append('/home/user/tiktok_production')

from analyzers.blip2_video_captioning_optimized import BLIP2VideoCaptioningOptimized
import time
import json

video_path = '/home/user/tiktok_videos/videos/7522589683939921165.mp4'

print("Testing BLIP-2 directly...")
print(f"Video: {video_path}")

# Initialize analyzer
analyzer = BLIP2VideoCaptioningOptimized()

# Run analysis
start = time.time()
result = analyzer.analyze(video_path)
elapsed = time.time() - start

# Results
print(f"\nAnalysis completed in {elapsed:.1f}s")
print(f"Description length: {len(result.get('overall_description', ''))} chars")
print(f"Segments: {len(result.get('segments', []))}")

print("\nDescription:")
print("-" * 60)
desc = result.get('overall_description', '')
print(desc[:800] + "..." if len(desc) > 800 else desc)
print("-" * 60)

# Save
with open('blip2_direct_test.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nResults saved to blip2_direct_test.json")