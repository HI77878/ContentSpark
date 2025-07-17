#!/usr/bin/env python3
import json
import os

# Lade Response
with open('/tmp/chase_analysis.json') as f:
    response = json.load(f)

results_file = response['results_file']
print(f"Analyzing results from: {results_file}")

# Lade vollständige Ergebnisse
with open(results_file) as f:
    data = json.load(f)

# Get metadata
metadata = data.get('metadata', {})
analyzer_results = data.get('analyzer_results', {})

# Video Info
print(f"\n=== VIDEO INFO ===")
print(f"Video: {metadata.get('video_filename', 'Unknown')}")
print(f"Duration: {metadata.get('duration', 0):.1f}s")
print(f"Resolution: {metadata.get('width', 0)}x{metadata.get('height', 0)}")
print(f"FPS: {metadata.get('fps', 0)}")
print(f"Processing Time: {metadata.get('processing_time_seconds', 0):.1f}s")
print(f"Realtime Factor: {metadata.get('realtime_factor', 0):.2f}x")
print(f"Reconstruction Score: {metadata.get('reconstruction_score', 0):.1f}%")

# StreamingDenseCaptioning Analyse
sdc = analyzer_results.get('streaming_dense_captioning', {})
sdc_segments = sdc.get('segments', [])

print(f"\n=== STREAMING DENSE CAPTIONING ===")
print(f"Segments: {len(sdc_segments)}")
if metadata.get('duration', 0) > 0:
    print(f"Segments/Second: {len(sdc_segments) / metadata.get('duration', 1):.2f}")

# Check metadata for coverage
if 'metadata' in sdc:
    sdc_meta = sdc['metadata']
    print(f"Coverage: {sdc_meta.get('temporal_coverage', 'N/A')}")
    print(f"Processing Mode: {sdc_meta.get('processing_mode', 'N/A')}")

# Zeige erste 5 Szenen-Beschreibungen
if sdc_segments:
    print(f"\n=== ERSTE SZENEN ===")
    for i, seg in enumerate(sdc_segments[:5]):
        start = seg.get('start_time', 0)
        end = seg.get('end_time', 0)
        caption = seg.get('caption', seg.get('description', ''))
        print(f"[{start:.1f}-{end:.1f}s] {caption[:100]}...")
else:
    print(f"\n=== STREAMING ISSUE DETECTED ===")
    print(f"No segments found! Checking for errors...")
    if 'error' in sdc:
        print(f"Error: {sdc['error']}")
    print(f"Raw SDC data: {sdc}")

# Analyzer Übersicht
print(f"\n=== ANALYZER SUMMARY ===")
analyzer_summary = []
for analyzer_name in sorted(analyzer_results.keys()):
    analyzer_data = analyzer_results[analyzer_name]
    if isinstance(analyzer_data, dict):
        segments = len(analyzer_data.get('segments', []))
        analyzer_summary.append((analyzer_name, segments))

for name, count in sorted(analyzer_summary, key=lambda x: x[1], reverse=True):
    print(f"{name}: {count} segments")

# Text Overlay Details
text_data = analyzer_results.get('text_overlay', {})
text_segments = text_data.get('segments', [])
if text_segments:
    print(f"\n=== TEXT OVERLAYS ===")
    unique_texts = set()
    for seg in text_segments:
        for detection in seg.get('text_detections', []):
            text = detection.get('text', '')
            if text:
                unique_texts.add(text)
    
    print(f"Unique texts found: {len(unique_texts)}")
    for text in list(unique_texts)[:10]:
        print(f"  - {text}")

# Speech Transcription
speech_data = analyzer_results.get('speech_transcription', {})
speech_segments = speech_data.get('segments', [])
if speech_segments:
    print(f"\n=== SPEECH TRANSCRIPTION ===")
    total_words = sum(len(seg.get('text', '').split()) for seg in speech_segments)
    print(f"Total segments: {len(speech_segments)}")
    print(f"Total words: {total_words}")
    if speech_segments:
        print(f"\nFirst transcript:")
        print(f"{speech_segments[0].get('text', '')[:200]}...")