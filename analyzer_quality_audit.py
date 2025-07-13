#!/usr/bin/env python3
"""Analyzer Quality Audit - Check data quality of all analyzers"""
import sys
sys.path.append('/home/user/tiktok_production')
import json
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

# Find latest analysis result
results = list(Path("results").glob("*.json"))
if not results:
    print("❌ No analysis results found in results/")
    sys.exit(1)

latest_file = sorted(results)[-1]
print(f"Analyzing: {latest_file.name}\n")

with open(latest_file) as f:
    data = json.load(f)
    
video_duration = data['metadata'].get('duration', 10.0)

print("=== ANALYZER QUALITÄTS-AUDIT ===\n")
print(f"Video-Dauer: {video_duration}s")
print(f"{'Analyzer':<30} {'Status':<10} {'Segmente':<10} {'Sek/Seg':<10} {'Problem':<30}")
print("-"*90)

problems = {}
good_analyzers = []

for analyzer, result in data['analyzer_results'].items():
    if 'error' in result:
        problems[analyzer] = f"ERROR: {result['error'][:50]}"
        print(f"{analyzer:<30} {'❌ ERROR':<10} {'-':<10} {'-':<10} {problems[analyzer]:<30}")
        continue
        
    if 'segments' not in result or not result['segments']:
        problems[analyzer] = "Keine Segmente"
        print(f"{analyzer:<30} {'❌ LEER':<10} {'0':<10} {'-':<10} {problems[analyzer]:<30}")
        continue
    
    segments = result['segments']
    
    # Check temporal coverage
    timestamps = []
    for seg in segments:
        # Try different timestamp fields
        ts = seg.get('timestamp', seg.get('start_time', seg.get('time')))
        if ts is not None:
            timestamps.append(float(ts))
    
    if not timestamps:
        problems[analyzer] = "Keine Zeitstempel"
        print(f"{analyzer:<30} {'⚠️ ZEIT':<10} {len(segments):<10} {'-':<10} {problems[analyzer]:<30}")
        continue
    
    # Calculate coverage
    coverage = (max(timestamps) - min(timestamps)) / video_duration if video_duration > 0 else 0
    seconds_per_segment = video_duration / len(segments) if segments else 999
    
    # Check data quality - look for generic/placeholder data
    sample = str(segments[0]).lower()
    generic_terms = ['balanced', 'natural', 'moderate', 'normal', 'placeholder', 'neutral']
    has_generic = any(term in sample for term in generic_terms)
    
    # Check for actual data variety
    unique_values = set()
    for seg in segments[:5]:  # Check first 5 segments
        if isinstance(seg, dict):
            for key, value in seg.items():
                if key not in ['timestamp', 'start_time', 'end_time']:
                    unique_values.add(str(value))
    
    data_variety = len(unique_values) > 3
    
    # Determine problems
    if coverage < 0.8:
        problems[analyzer] = f"Nur {coverage*100:.0f}% Abdeckung"
    elif seconds_per_segment > 2:
        problems[analyzer] = f"Zu wenig Segmente ({seconds_per_segment:.1f}s/Seg)"
    elif has_generic and not data_variety:
        problems[analyzer] = "Generische Daten"
    elif len(segments) < 5:
        problems[analyzer] = f"Nur {len(segments)} Segmente"
    else:
        problems[analyzer] = None
        good_analyzers.append(analyzer)
    
    status = "✅ OK" if not problems[analyzer] else "⚠️ PROBLEM"
    print(f"{analyzer:<30} {status:<10} {len(segments):<10} {seconds_per_segment:.1f}s{' ':<6} {problems[analyzer] or 'Gut':<30}")

# Summary
print(f"\n📊 ZUSAMMENFASSUNG:")
print(f"✅ Gute Analyzer: {len(good_analyzers)}/{len(data['analyzer_results'])}")
print(f"❌ Problematische Analyzer: {len([p for p in problems.values() if p])}")

if problems:
    print("\n🔧 PROBLEME IM DETAIL:")
    for analyzer, problem in sorted(problems.items()):
        if problem:
            print(f"  - {analyzer}: {problem}")

# Recommendations
print("\n💡 EMPFEHLUNGEN:")
print("1. Analyzer mit 'Generische Daten' brauchen spezifischere Prompts")
print("2. Analyzer mit wenig Segmenten brauchen höhere Sampling-Rate")
print("3. Analyzer mit Errors müssen debuggt werden")
print("4. Ziel: Mindestens 1 Segment pro Sekunde für alle Analyzer!")

# Save audit report
audit_report = {
    "audit_timestamp": str(datetime.now()),
    "analyzed_file": str(latest_file),
    "video_duration": video_duration,
    "total_analyzers": len(data['analyzer_results']),
    "good_analyzers": len(good_analyzers),
    "problems": problems
}

with open("analyzer_quality_audit_report.json", "w") as f:
    json.dump(audit_report, f, indent=2)
    
print(f"\n📄 Audit-Report gespeichert: analyzer_quality_audit_report.json")