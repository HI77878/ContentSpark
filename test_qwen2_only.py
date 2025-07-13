#!/usr/bin/env python3
"""Test Qwen2-VL Temporal Analyzer directly"""

import sys
sys.path.append('/home/user/tiktok_production')

from analyzers.qwen2_vl_temporal_fixed import Qwen2VLTemporalFixed
import time
import json

# Test video
video_path = "/home/user/tiktok_videos/videos/7446489995663117590.mp4"

print("🚀 Testing Qwen2-VL Temporal Analyzer ONLY...")
print(f"📹 Video: {video_path}")

# Create analyzer
analyzer = Qwen2VLTemporalFixed()

# Run analysis
print("\n⏳ Running analysis...")
start_time = time.time()

try:
    result = analyzer.analyze(video_path)
    elapsed = time.time() - start_time
    
    print(f"\n✅ Analysis complete in {elapsed:.1f}s!")
    print(f"📊 Found {len(result.get('segments', []))} temporal segments")
    
    # Show first 3 segments
    print("\n📝 First 3 temporal descriptions:")
    for i, seg in enumerate(result.get('segments', [])[:3]):
        print(f"\n{i+1}. {seg['start_time']:.1f}s - {seg['end_time']:.1f}s:")
        print(f"   \"{seg['description']}\"")
        
    # Save full result
    output_file = "/home/user/tiktok_production/qwen2_vl_test_result.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Full results saved to: {output_file}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()