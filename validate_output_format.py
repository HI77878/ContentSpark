#!/usr/bin/env python3
"""Validate output format and data quality"""
import json
from pathlib import Path
import sys
from datetime import datetime

# Find latest analysis result
results = list(Path("results").glob("*.json"))
if not results:
    print("❌ No analysis results found in results/")
    sys.exit(1)

latest_file = sorted(results)[-1]
print(f"Validating: {latest_file.name}")
print(f"File size: {latest_file.stat().st_size / 1024 / 1024:.2f} MB\n")

with open(latest_file) as f:
    data = json.load(f)

print("=== OUTPUT VALIDATION ===\n")

# 1. Metadata Check
print("1. METADATA CHECK:")
required_metadata = [
    'video_path', 'tiktok_url', 'creator_username', 'duration', 
    'processing_time_seconds', 'successful_analyzers', 'total_analyzers',
    'reconstruction_score', 'realtime_factor'
]

meta = data.get('metadata', {})
for field in required_metadata:
    exists = field in meta
    value = meta.get(field, 'N/A')
    print(f"  {field}: {'✅' if exists else '❌'} ({value})")

# 2. Analyzer Results Check
print("\n2. ANALYZER RESULTS:")
results = data.get('analyzer_results', {})
print(f"  Total analyzers: {len(results)}")
print(f"  Expected: 20+")

# Check each analyzer
analyzer_quality = {}
for analyzer, result in results.items():
    if 'error' in result:
        analyzer_quality[analyzer] = {'status': 'ERROR', 'segments': 0}
    elif 'segments' in result and result['segments']:
        segments = result['segments']
        analyzer_quality[analyzer] = {
            'status': 'OK',
            'segments': len(segments),
            'has_timestamps': any('timestamp' in s or 'start_time' in s for s in segments),
            'avg_data_per_segment': len(str(segments[0])) if segments else 0
        }
    else:
        analyzer_quality[analyzer] = {'status': 'NO_DATA', 'segments': 0}

# 3. Data Quality Check
print("\n3. DATA QUALITY:")
good_analyzers = [a for a, q in analyzer_quality.items() if q['status'] == 'OK' and q['segments'] > 5]
print(f"  High-quality analyzers (>5 segments): {len(good_analyzers)}/{len(analyzer_quality)}")

for analyzer, quality in sorted(analyzer_quality.items()):
    status_icon = "✅" if quality['status'] == 'OK' else "❌"
    segments = quality.get('segments', 0)
    print(f"  {status_icon} {analyzer}: {segments} segments ({quality['status']})")

# 4. Temporal Coverage Check
print("\n4. TEMPORAL COVERAGE:")
duration = meta.get('duration', 10)
for analyzer in ['qwen2_vl_temporal', 'object_detection', 'speech_transcription', 'visual_effects']:
    if analyzer in results and 'segments' in results[analyzer]:
        segments = results[analyzer]['segments']
        if segments:
            coverage = len(segments) / duration if duration > 0 else 0
            print(f"  {analyzer}: {coverage:.1f} segments/second")

# 5. Data Completeness
print("\n5. DATA COMPLETENESS:")
total_segments = sum(
    len(r.get('segments', [])) 
    for r in results.values() 
    if isinstance(r, dict)
)
print(f"  Total segments across all analyzers: {total_segments}")
print(f"  Average segments per analyzer: {total_segments/len(results):.1f}")

# 6. TikTok Integration Check
print("\n6. TIKTOK INTEGRATION:")
has_url = 'tiktok_url' in meta and meta['tiktok_url']
has_username = 'creator_username' in meta and meta['creator_username']
print(f"  TikTok URL: {'✅' if has_url else '❌ MISSING'}")
print(f"  Creator username: {'✅' if has_username else '❌ MISSING'}")

# 7. Performance Metrics
print("\n7. PERFORMANCE:")
print(f"  Processing time: {meta.get('processing_time_seconds', 'N/A')}s")
print(f"  Realtime factor: {meta.get('realtime_factor', 'N/A')}x")
print(f"  Success rate: {meta.get('successful_analyzers', 0)}/{meta.get('total_analyzers', 0)}")

# Summary
print("\n" + "="*50)
print("SUMMARY:")
success_rate = len(good_analyzers) / len(analyzer_quality) * 100 if analyzer_quality else 0
print(f"✅ Quality score: {success_rate:.0f}%")
print(f"{'✅' if has_url else '❌'} TikTok URL integration")
print(f"{'✅' if total_segments > 100 else '❌'} Data density ({total_segments} total segments)")

# Save validation report
report = {
    "validation_timestamp": datetime.now().isoformat(),
    "file_analyzed": str(latest_file),
    "quality_score": success_rate,
    "has_tiktok_url": has_url,
    "total_segments": total_segments,
    "analyzer_quality": analyzer_quality
}

with open("output_validation_report.json", "w") as f:
    json.dump(report, f, indent=2)
    
print(f"\n📄 Validation report saved: output_validation_report.json")