#!/usr/bin/env python3
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