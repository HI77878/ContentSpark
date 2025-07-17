#!/usr/bin/env python3
<<<<<<< HEAD
"""Detaillierte Performance-Analyse"""

import json
from pathlib import Path

latest_file = max(Path('/home/user/tiktok_production/results').glob('*.json'), key=lambda x: x.stat().st_mtime)
with open(latest_file, 'r') as f:
    data = json.load(f)

print('⏱️ PERFORMANCE-ANALYSE FÜR VIDEO-VERARBEITUNG:')
print('='*60)

# Video-Info
video_meta = data.get('metadata', {})
video_duration = video_meta.get('duration', 10)  # Annahme: 10s Video
print(f'Video-Dauer: {video_duration}s')

# Sammle Zeiten
analyzer_times = []
for name, result in data.get('analyzer_results', {}).items():
    if isinstance(result, dict):
        proc_time = result.get('metadata', {}).get('processing_time')
        if proc_time:
            analyzer_times.append((name, proc_time))

# Sortiere nach Zeit
analyzer_times.sort(key=lambda x: x[1], reverse=True)

print(f'\n🐌 LANGSAMSTE ANALYZER:')
for name, time in analyzer_times[:10]:
    realtime_factor = time / video_duration
    print(f'  {name}: {time:.1f}s ({realtime_factor:.1f}x realtime)')
    if name == 'qwen2_vl_temporal':
        # Hochrechnung für 30s Video
        estimated_30s = (time / video_duration) * 30
        print(f'    → Geschätzt für 30s Video: {estimated_30s:.1f}s')

# Gesamtzeit
total_time = sum(t for _, t in analyzer_times)
print(f'\n📊 GESAMT-STATISTIK:')
print(f'  Summe aller Analyzer-Zeiten: {total_time:.1f}s')
print(f'  Durchschnitt pro Analyzer: {total_time/len(analyzer_times):.1f}s')
print(f'  Realtime-Faktor (sequentiell): {total_time/video_duration:.1f}x')

# Parallelisierung
print(f'\n🚀 MIT PARALLELISIERUNG:')
# Annahme: Staged execution mit längster Stage als Bottleneck
stage1_time = next((t for n, t in analyzer_times if n == 'qwen2_vl_temporal'), 0)
other_max = max((t for n, t in analyzer_times if n != 'qwen2_vl_temporal'), default=0)
parallel_time = stage1_time + other_max  # Vereinfachte Schätzung
print(f'  Geschätzte Parallelzeit: {parallel_time:.1f}s')
print(f'  Parallelisierter Realtime-Faktor: {parallel_time/video_duration:.1f}x')

# Speicherung
print(f'\n💾 DATENSPEICHERUNG:')
print(f'  Dateigröße: {latest_file.stat().st_size / 1024:.1f} KB')
print(f'  Dateiname: {latest_file.name}')
print(f'  Alle Daten in EINER JSON-Datei: ✅')
=======
import json
import glob
import os
from datetime import datetime

# Finde alle Results
results = glob.glob('/home/user/tiktok_production/results/*.json')
performance_data = []

for r in sorted(results)[-20:]:  # Last 20 results
    try:
        with open(r, 'r') as f:
            data = json.load(f)
        
        meta = data.get('metadata', {})
        analyzer_times = {}
        
        # Sammle Processing Times pro Analyzer
        for name, result in data.get('analyzer_results', {}).items():
            if isinstance(result, dict):
                if 'processing_time' in result:
                    analyzer_times[name] = result['processing_time']
                elif 'summary' in result and 'processing_time_seconds' in result['summary']:
                    analyzer_times[name] = result['summary']['processing_time_seconds']
        
        performance_data.append({
            'file': os.path.basename(r),
            'duration': meta.get('duration', 0),
            'total_time': meta.get('processing_time_seconds', 0),
            'realtime': meta.get('realtime_factor', 0),
            'analyzer_times': analyzer_times,
            'timestamp': r.split('_')[-2] + '_' + r.split('_')[-1].replace('.json', '')
        })
    except Exception as e:
        print(f"Error reading {r}: {e}")

# Zeige Trends
print("PERFORMANCE TREND (last 20 runs):")
print("-" * 80)
for p in sorted(performance_data, key=lambda x: x['timestamp']):
    print(f"{p['timestamp']} - {p['file'][:30]}... - {p['realtime']:.2f}x realtime ({p['total_time']:.1f}s for {p['duration']:.1f}s video)")

# Finde die langsamsten Analyzer
all_times = {}
for p in performance_data:
    for analyzer, time in p['analyzer_times'].items():
        if analyzer not in all_times:
            all_times[analyzer] = []
        all_times[analyzer].append(time)

print("\nLANGSAMSTE ANALYZER (Durchschnitt der letzten Runs):")
print("-" * 80)
avg_times = [(sum(times)/len(times), name, len(times)) for name, times in all_times.items() if times]
for time, name, count in sorted(avg_times, reverse=True)[:15]:
    print(f"{name:30} : {time:6.1f}s (from {count} runs)")

# Find best performing run
best_run = min(performance_data, key=lambda x: x['realtime'] if x['realtime'] > 0 else float('inf'))
print(f"\nBEST PERFORMING RUN: {best_run['file']} - {best_run['realtime']:.2f}x realtime")
print("Analyzer times for best run:")
for name, time in sorted(best_run['analyzer_times'].items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {name:30} : {time:6.1f}s")
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
