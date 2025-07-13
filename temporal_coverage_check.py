#!/usr/bin/env python3
"""Check temporal coverage for each analyzer"""
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

# Load latest analysis result
results_dir = Path("/home/user/tiktok_production/results")
analysis_files = list(results_dir.glob("test_video_*.json"))

if not analysis_files:
    print("❌ Keine Analyse-Ergebnisse gefunden!")
    sys.exit(1)

latest_result = sorted(analysis_files)[-1]
print(f"Lade Analyse: {latest_result.name}")

with open(latest_result) as f:
    data = json.load(f)

# Get video duration
video_duration = data.get('metadata', {}).get('duration', 10)
if video_duration == 0:
    video_duration = 10  # Default for test video

print(f"\n=== ZEITLICHE ABDECKUNG PRO ANALYZER ===")
print(f"Video-Dauer: {video_duration}s")
print("="*80)

# Analyze temporal coverage
coverage_report = {}
analyzer_results = data.get('analyzer_results', {})

print(f"\nANALYZER | SEGMENTE | ABDECKUNG | TIMELINE")
print("-"*80)

for analyzer_name, result in sorted(analyzer_results.items()):
    if 'error' in result:
        coverage_report[analyzer_name] = {
            'status': 'error',
            'segments': 0,
            'coverage': 0,
            'timeline': [],
            'gaps': []
        }
        print(f"{analyzer_name:25} | {'ERROR':^8} | {'0%':^9} | ❌ {result['error'][:30]}")
        continue
    
    if 'segments' not in result or not result['segments']:
        coverage_report[analyzer_name] = {
            'status': 'no_data',
            'segments': 0,
            'coverage': 0,
            'timeline': [],
            'gaps': []
        }
        print(f"{analyzer_name:25} | {0:^8} | {'0%':^9} | ⚪ Keine Daten")
        continue
    
    segments = result['segments']
    
    # Extract timestamps
    timestamps = []
    for seg in segments:
        # Try different timestamp fields
        ts = seg.get('timestamp', seg.get('start_time', seg.get('time', None)))
        if ts is not None:
            timestamps.append(float(ts))
    
    if not timestamps:
        coverage_report[analyzer_name] = {
            'status': 'no_timestamps',
            'segments': len(segments),
            'coverage': 0,
            'timeline': [],
            'gaps': []
        }
        print(f"{analyzer_name:25} | {len(segments):^8} | {'?%':^9} | ⚠️ Keine Zeitstempel")
        continue
    
    timestamps.sort()
    
    # Create second-by-second timeline
    timeline = [0] * int(video_duration + 1)
    for ts in timestamps:
        idx = int(ts)
        if 0 <= idx < len(timeline):
            timeline[idx] = 1
    
    # Calculate coverage
    covered_seconds = sum(timeline)
    total_seconds = len(timeline) - 1  # Exclude last second
    coverage_percent = (covered_seconds / total_seconds * 100) if total_seconds > 0 else 0
    
    # Find gaps
    gaps = []
    in_gap = False
    gap_start = 0
    
    for i, covered in enumerate(timeline[:-1]):  # Exclude last element
        if not covered and not in_gap:
            in_gap = True
            gap_start = i
        elif covered and in_gap:
            in_gap = False
            gaps.append((gap_start, i-1))
    
    if in_gap:
        gaps.append((gap_start, len(timeline)-2))
    
    # Store report
    coverage_report[analyzer_name] = {
        'status': 'ok',
        'segments': len(segments),
        'coverage': coverage_percent,
        'timeline': timeline,
        'gaps': gaps,
        'timestamps': timestamps
    }
    
    # Print timeline visualization
    timeline_str = ""
    for i, covered in enumerate(timeline[:-1]):
        if i % 5 == 0:
            timeline_str += "|"
        timeline_str += "█" if covered else "░"
    
    print(f"{analyzer_name:25} | {len(segments):^8} | {coverage_percent:>3.0f}%{' ':6} | {timeline_str}")

# Summary by coverage level
print("\n" + "="*80)
print("ZUSAMMENFASSUNG NACH ABDECKUNG:")
print("-"*80)

coverage_levels = {
    'Vollständig (>90%)': [],
    'Gut (70-90%)': [],
    'Mittel (50-70%)': [],
    'Schlecht (20-50%)': [],
    'Minimal (<20%)': [],
    'Keine Daten': []
}

for name, report in coverage_report.items():
    if report['status'] in ['error', 'no_data', 'no_timestamps']:
        coverage_levels['Keine Daten'].append(name)
    elif report['coverage'] > 90:
        coverage_levels['Vollständig (>90%)'].append((name, report['coverage']))
    elif report['coverage'] > 70:
        coverage_levels['Gut (70-90%)'].append((name, report['coverage']))
    elif report['coverage'] > 50:
        coverage_levels['Mittel (50-70%)'].append((name, report['coverage']))
    elif report['coverage'] > 20:
        coverage_levels['Schlecht (20-50%)'].append((name, report['coverage']))
    else:
        coverage_levels['Minimal (<20%)'].append((name, report['coverage']))

for level, analyzers in coverage_levels.items():
    if analyzers and level != 'Keine Daten':
        print(f"\n{level}: {len(analyzers)}")
        for item in sorted(analyzers, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {item[0]}: {item[1]:.0f}%")
        if len(analyzers) > 5:
            print(f"  ... und {len(analyzers)-5} weitere")
    elif analyzers:
        print(f"\n{level}: {len(analyzers)}")

# Identify problematic gaps
print("\n" + "="*80)
print("PROBLEMATISCHE ZEITLÜCKEN (>3 Sekunden):")
print("-"*80)

problem_count = 0
for name, report in coverage_report.items():
    if report['status'] == 'ok' and report['gaps']:
        large_gaps = [(start, end) for start, end in report['gaps'] if end - start >= 3]
        if large_gaps:
            problem_count += 1
            print(f"\n{name}:")
            for start, end in large_gaps[:3]:
                gap_size = end - start + 1
                print(f"  - Lücke von {start}s bis {end}s ({gap_size}s)")

if problem_count == 0:
    print("\n✅ Keine problematischen Zeitlücken gefunden!")

# Critical analyzer coverage check
print("\n" + "="*80)
print("KRITISCHE ANALYZER ZEITABDECKUNG:")
print("-"*80)

critical = ['qwen2_vl_temporal', 'speech_transcription', 'object_detection', 'face_emotion', 'body_pose']

for analyzer in critical:
    if analyzer in coverage_report:
        report = coverage_report[analyzer]
        if report['status'] == 'ok':
            status = "✅" if report['coverage'] > 80 else "⚠️" if report['coverage'] > 50 else "❌"
            print(f"{status} {analyzer}: {report['coverage']:.0f}% ({report['segments']} Segmente)")
            
            # Show gap summary
            if report['gaps']:
                total_gap_time = sum(end - start + 1 for start, end in report['gaps'])
                print(f"   Lücken: {len(report['gaps'])} Stück, gesamt {total_gap_time}s")
        else:
            print(f"❌ {analyzer}: {report['status']}")
    else:
        print(f"❌ {analyzer}: NICHT GEFUNDEN")

# Save temporal coverage report
coverage_file = f"results/temporal_coverage_{latest_result.stem}_{int(Path(latest_result).stat().st_mtime)}.json"
with open(coverage_file, 'w') as f:
    json.dump({
        'source_file': str(latest_result),
        'video_duration': video_duration,
        'analyzer_count': len(coverage_report),
        'coverage_report': coverage_report,
        'summary': {
            'full_coverage': len(coverage_levels['Vollständig (>90%)']),
            'good_coverage': len(coverage_levels['Gut (70-90%)']),
            'medium_coverage': len(coverage_levels['Mittel (50-70%)']),
            'poor_coverage': len(coverage_levels['Schlecht (20-50%)']),
            'minimal_coverage': len(coverage_levels['Minimal (<20%)']),
            'no_data': len(coverage_levels['Keine Daten'])
        }
    }, f, indent=2)

print(f"\n📊 Zeitabdeckungs-Bericht gespeichert: {coverage_file}")

# Final assessment
full_coverage_count = len(coverage_levels['Vollständig (>90%)']) + len(coverage_levels['Gut (70-90%)'])
total_with_data = len(coverage_report) - len(coverage_levels['Keine Daten'])

if total_with_data > 0:
    good_coverage_percent = full_coverage_count / total_with_data * 100
    print(f"\n🎯 {good_coverage_percent:.0f}% der Analyzer mit Daten haben gute Zeitabdeckung (>70%)")