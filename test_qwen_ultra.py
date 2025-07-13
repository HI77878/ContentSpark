#!/usr/bin/env python3
"""Test Qwen2-VL Ultra Detailed Analyzer"""

import sys
sys.path.append('/home/user/tiktok_production')

from analyzers.qwen2_vl_ultra_detailed import Qwen2VLUltraDetailedAnalyzer
import time
import json

# Test video - nur erste 10 Sekunden für schnellen Test
video_path = "/home/user/tiktok_videos/videos/7446489995663117590.mp4"

print("🚀 Testing Qwen2-VL ULTRA DETAILED Analyzer...")
print("📊 Settings: 8 frames/segment, 2s segments, 300 tokens, detailed prompts")
print(f"📹 Video: {video_path}")

# Create analyzer
analyzer = Qwen2VLUltraDetailedAnalyzer()

# Temporarily reduce for quick test
analyzer.segment_duration = 3.0  # Test with 3 second segments
print(f"\n⚡ Using {analyzer.segment_duration}s segments for test")

# Run analysis on first 15 seconds
print("\n⏳ Running ultra-detailed analysis...")
start_time = time.time()

try:
    # Create a short test video (first 15 seconds)
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create temp video with first 15 seconds
    test_video = "/tmp/test_15s.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video, fourcc, fps, 
                         (int(cap.get(3)), int(cap.get(4))))
    
    frames_to_write = int(15 * fps)  # 15 seconds
    for i in range(frames_to_write):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    
    print(f"Created 15s test video: {test_video}")
    
    # Analyze
    result = analyzer.analyze(test_video)
    elapsed = time.time() - start_time
    
    print(f"\n✅ Analysis complete in {elapsed:.1f}s!")
    
    # Show results
    summary = result.get('summary', {})
    print(f"\n📊 RESULTS:")
    print(f"  - Segments analyzed: {summary.get('total_segments', 0)}")
    print(f"  - Total words generated: {summary.get('total_words_generated', 0)}")
    print(f"  - Avg words/segment: {summary.get('average_words_per_segment', 0):.1f}")
    print(f"  - Processing time: {elapsed:.1f}s for 15s video")
    print(f"  - Realtime factor: {elapsed/15:.2f}x")
    
    # Show first 2 segments
    segments = result.get('segments', [])
    print(f"\n📝 ULTRA DETAILED DESCRIPTIONS:")
    for i, seg in enumerate(segments[:2]):
        if 'error' not in seg:
            print(f"\n{'='*70}")
            print(f"Segment {i+1}: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s")
            print(f"Words: {seg.get('word_count', 0)}")
            print(f"Description:\n{seg['description']}")
        
    # Save full result
    output_file = "/home/user/tiktok_production/qwen2_ultra_test_result.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Full results saved to: {output_file}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()