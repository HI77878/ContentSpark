#!/usr/bin/env python3
import json

# Load results
with open('/home/user/tiktok_production/results/chaseridgewayy_7522589683939921165_multiprocess_20250707_160400.json') as f:
    data = json.load(f)

metadata = data.get('metadata', {})
analyzer_results = data.get('analyzer_results', {})

print("=== CHASE VIDEO COMPLETE ANALYSIS ===")
print(f"Processing Time: {metadata.get('processing_time_seconds', 0):.1f}s")
print(f"Reconstruction Score: {metadata.get('reconstruction_score', 0):.1f}%")
print(f"Successful Analyzers: {metadata.get('successful_analyzers')}/{metadata.get('total_analyzers')}")

# CHECK STREAMINGDENSECAPTIONING
sdc = analyzer_results.get('streaming_dense_captioning', {})
sdc_segments = sdc.get('segments', [])

print(f"\n🔥 STREAMING DENSE CAPTIONING STATUS:")
print(f"Segments: {len(sdc_segments)}")

if sdc_segments:
    print(f"✅ IT FUCKING WORKS!")
    print(f"\nFirst 5 captions:")
    for i, seg in enumerate(sdc_segments[:5]):
        print(f"[{seg['start_time']:.0f}-{seg['end_time']:.0f}s] {seg['caption']}")
    
    # Coverage calculation
    video_duration = 68.5  # from ffprobe
    coverage = len(sdc_segments) * 2 / video_duration * 100
    print(f"\nCoverage: {coverage:.1f}%")
else:
    print(f"❌ STILL BROKEN!")

# Show all analyzer segment counts
print(f"\n=== ALL ANALYZER RESULTS ===")
for name in sorted(analyzer_results.keys()):
    segments = len(analyzer_results[name].get('segments', []))
    print(f"{name}: {segments} segments")