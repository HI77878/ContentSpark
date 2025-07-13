#!/usr/bin/env python3
"""Validate if video can be reconstructed from analysis data"""
import json
import os
from collections import defaultdict

# Use the specific result file
result_path = "/home/user/tiktok_production/results/leon_schliebach_7446489995663117590_multiprocess_20250707_120558.json"

print(f"Analyzing: {result_path}")

with open(result_path) as f:
    data = json.load(f)

# Get metadata - handle different formats
metadata = data.get('metadata', {})
video_metadata = data.get('video_metadata', {})

# Try different duration fields
video_duration = metadata.get('video_duration', 0)
if video_duration == 0:
    video_duration = video_metadata.get('duration', 0)
if video_duration == 0:
    video_duration = metadata.get('duration', 0)
if video_duration == 0:
    # Fallback: estimate from timestamp data
    max_timestamp = 0
    for analyzer_data in data.get('analyzer_results', {}).values():
        for seg in analyzer_data.get('segments', []):
            ts = seg.get('timestamp', seg.get('end_time', seg.get('start_time', 0)))
            if isinstance(ts, (int, float)):
                max_timestamp = max(max_timestamp, ts)
    video_duration = max_timestamp if max_timestamp > 0 else 49  # Default to known duration
print(f"\nVideo Duration: {video_duration:.1f}s")

# Get analyzer results
analyzer_results = data.get('analyzer_results', {})

# Check if streaming_dense_captioning is present
print("\n=== ANALYZER STATUS ===")
print(f"Total analyzers: {len(analyzer_results)}")
print(f"Analyzers present: {', '.join(sorted(analyzer_results.keys()))}")

# Check for streaming_dense_captioning
if 'streaming_dense_captioning' in analyzer_results:
    sdc = analyzer_results['streaming_dense_captioning']
    sdc_segments = sdc.get('segments', [])
    sdc_metadata = sdc.get('metadata', {})
    sdc_coverage = sdc_metadata.get('temporal_coverage', '0%')
    print(f"\n=== STREAMING DENSE CAPTIONING ===")
    print(f"Status: FOUND")
    print(f"Coverage: {sdc_coverage}")
    print(f"Segments: {len(sdc_segments)}")
else:
    print("\n=== STREAMING DENSE CAPTIONING ===")
    print("Status: NOT FOUND - Using alternative analyzers")
    sdc_segments = []

# Build reconstruction timeline
reconstruction_data = defaultdict(dict)

# 1. Scene descriptions from various sources
if sdc_segments:
    for seg in sdc_segments:
        time = int(seg['start_time'])
        reconstruction_data[time]['scene'] = seg.get('description', seg.get('caption', ''))

# 2. Objects from object_detection
if 'object_detection' in analyzer_results:
    od = analyzer_results['object_detection']
    for seg in od.get('segments', []):
        time = int(seg.get('timestamp', 0))
        objects = []
        for det in seg.get('detections', []):
            obj_class = det.get('object_class', det.get('class', det.get('label', 'unknown')))
            objects.append(obj_class)
        if objects:
            reconstruction_data[time]['objects'] = objects

# 3. Text overlays
if 'text_overlay' in analyzer_results:
    to = analyzer_results['text_overlay']
    for seg in to.get('segments', []):
        time = int(seg.get('timestamp', seg.get('start_time', 0)))
        texts = []
        # Handle different formats
        if 'text' in seg:
            texts.append(seg['text'])
        elif 'text_detections' in seg:
            texts.extend([t.get('text', '') for t in seg['text_detections']])
        elif 'ocr_text' in seg:
            texts.append(seg['ocr_text'])
        if texts:
            reconstruction_data[time]['text'] = texts

# 4. Speech transcription
if 'speech_transcription' in analyzer_results:
    st = analyzer_results['speech_transcription']
    for seg in st.get('segments', []):
        start = int(seg.get('start', seg.get('start_time', 0)))
        text = seg.get('text', seg.get('transcript', ''))
        if text:
            reconstruction_data[start]['speech'] = text

# 5. Camera movements
if 'camera_analysis' in analyzer_results:
    ca = analyzer_results['camera_analysis']
    for seg in ca.get('segments', []):
        time = int(seg.get('timestamp', 0))
        movement = seg.get('movement', {})
        if movement and isinstance(movement, dict):
            reconstruction_data[time]['camera'] = movement.get('type', 'unknown')
        elif isinstance(movement, str):
            reconstruction_data[time]['camera'] = movement

# 6. Visual effects
if 'visual_effects' in analyzer_results:
    ve = analyzer_results['visual_effects']
    for seg in ve.get('segments', []):
        time = int(seg.get('timestamp', 0))
        effects = seg.get('effects', [])
        if effects:
            reconstruction_data[time]['effects'] = effects

print(f"\n=== RECONSTRUCTION TIMELINE (0-15s) ===")
for second in range(min(15, int(video_duration))):
    if second in reconstruction_data:
        data_at_second = reconstruction_data[second]
        print(f"\n[{second:02d}s]")
        if 'scene' in data_at_second:
            scene = data_at_second['scene']
            if len(scene) > 100:
                scene = scene[:97] + "..."
            print(f"  Scene: {scene}")
        if 'objects' in data_at_second:
            print(f"  Objects: {', '.join(data_at_second['objects'][:5])}")
        if 'text' in data_at_second:
            print(f"  Text: {', '.join(data_at_second['text'])}")
        if 'speech' in data_at_second:
            print(f"  Speech: \"{data_at_second['speech']}\"")
        if 'camera' in data_at_second:
            print(f"  Camera: {data_at_second['camera']}")
        if 'effects' in data_at_second:
            print(f"  Effects: {', '.join(data_at_second['effects'])}")

# Calculate reconstruction metrics
seconds_with_data = len(reconstruction_data)
total_seconds = int(video_duration)
data_types_per_second = []

for sec_data in reconstruction_data.values():
    types = len([k for k in ['scene', 'objects', 'text', 'speech', 'camera', 'effects'] if k in sec_data])
    data_types_per_second.append(types)

avg_data_types = sum(data_types_per_second) / len(data_types_per_second) if data_types_per_second else 0

print(f"\n=== RECONSTRUCTION METRICS ===")
print(f"Seconds with data: {seconds_with_data}/{total_seconds} ({seconds_with_data/total_seconds*100:.1f}%)")
print(f"Average data types per second: {avg_data_types:.1f}")
print(f"Total data points: {sum(data_types_per_second)}")

# Check specific coverage
has_scene = sum(1 for d in reconstruction_data.values() if 'scene' in d)
has_objects = sum(1 for d in reconstruction_data.values() if 'objects' in d)
has_text = sum(1 for d in reconstruction_data.values() if 'text' in d)
has_speech = sum(1 for d in reconstruction_data.values() if 'speech' in d)

print(f"\nData type coverage:")
print(f"  Scene descriptions: {has_scene} seconds")
print(f"  Object detection: {has_objects} seconds")
print(f"  Text overlay: {has_text} seconds")
print(f"  Speech: {has_speech} seconds")

# Calculate final score
temporal_coverage_score = (seconds_with_data / total_seconds) * 40  # 40% weight
data_richness_score = min(40, avg_data_types * 10)  # 40% weight
scene_coverage_score = (has_scene / total_seconds) * 20  # 20% weight for scene descriptions

reconstruction_score = temporal_coverage_score + data_richness_score + scene_coverage_score

print(f"\n=== RECONSTRUCTION SCORE: {reconstruction_score:.0f}% ===")
print(f"  Temporal coverage: {temporal_coverage_score:.0f}/40")
print(f"  Data richness: {data_richness_score:.0f}/40")
print(f"  Scene coverage: {scene_coverage_score:.0f}/20")

if reconstruction_score >= 90:
    print("\n✅ Video kann mit hoher Genauigkeit rekonstruiert werden!")
elif reconstruction_score >= 75:
    print("\n⚠️  Video kann gut rekonstruiert werden, einige Details fehlen")
else:
    print("\n❌ Nicht genug Daten für akkurate Rekonstruktion")

# Save reconstruction data
with open('/home/user/tiktok_production/reconstruction_data.json', 'w') as f:
    json.dump({
        'video_duration': video_duration,
        'reconstruction_score': reconstruction_score,
        'metrics': {
            'seconds_with_data': seconds_with_data,
            'total_seconds': total_seconds,
            'avg_data_types': avg_data_types,
            'coverage_percentage': (seconds_with_data/total_seconds*100)
        },
        'timeline': dict(reconstruction_data)
    }, f, indent=2)