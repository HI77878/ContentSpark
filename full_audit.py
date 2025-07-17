#!/usr/bin/env python3
import json
import glob
import os
from datetime import datetime
import statistics

print("="*80)
print("VOLLSTÄNDIGER PERFORMANCE-AUDIT")
print("="*80)

# Finde alle Results der letzten 2 Stunden
results = []
for file in glob.glob('/home/user/tiktok_production/results/*.json'):
    if os.path.getmtime(file) > datetime.now().timestamp() - 7200:
        results.append(file)

results = sorted(results)[-10:]  # Letzte 10

performance_data = []
analyzer_stats = {}
errors_found = []

for result_file in results:
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        meta = data.get('metadata', {})
        
        # Sammle Performance-Daten
        perf = {
            'file': os.path.basename(result_file),
            'video_id': meta.get('tiktok_video_id', 'unknown'),
            'creator': meta.get('creator_username', 'unknown'),
            'duration': meta.get('duration', 0),
            'processing_time': meta.get('processing_time_seconds', 0),
            'realtime_factor': meta.get('realtime_factor', 0),
            'successful_analyzers': meta.get('successful_analyzers', 0),
            'total_analyzers': meta.get('total_analyzers', 0),
            'timestamp': meta.get('analysis_timestamp', '')
        }
        performance_data.append(perf)
        
        # Sammle Analyzer-Statistiken
        for analyzer, result in data.get('analyzer_results', {}).items():
            if analyzer not in analyzer_stats:
                analyzer_stats[analyzer] = {
                    'runs': 0,
                    'successes': 0,
                    'segments': [],
                    'errors': []
                }
            
            analyzer_stats[analyzer]['runs'] += 1
            
            if isinstance(result, dict):
                if 'error' in result:
                    analyzer_stats[analyzer]['errors'].append(result['error'])
                    errors_found.append(f"{perf['video_id']}: {analyzer} - {result['error'][:100]}")
                elif 'segments' in result:
                    seg_count = len(result['segments'])
                    analyzer_stats[analyzer]['segments'].append(seg_count)
                    if seg_count > 0:
                        analyzer_stats[analyzer]['successes'] += 1
                    
                    # Check for segment errors
                    for seg in result['segments']:
                        if 'error' in seg:
                            errors_found.append(f"{perf['video_id']}: {analyzer} segment - {seg['error'][:100]}")
                else:
                    analyzer_stats[analyzer]['successes'] += 1
    
    except Exception as e:
        errors_found.append(f"Failed to parse {result_file}: {str(e)}")

# Performance Zusammenfassung
print("\n1. PERFORMANCE-ÜBERSICHT")
print("-" * 80)
print(f"Analysierte Videos: {len(performance_data)}")

if performance_data:
    # Realtime Faktoren
    rt_factors = [p['realtime_factor'] for p in performance_data if p['realtime_factor'] > 0]
    if rt_factors:
        print(f"\nREALTIME-FAKTOREN:")
        print(f"  Durchschnitt: {statistics.mean(rt_factors):.2f}x")
        print(f"  Median: {statistics.median(rt_factors):.2f}x")
        print(f"  Min: {min(rt_factors):.2f}x")
        print(f"  Max: {max(rt_factors):.2f}x")
        print(f"  Unter 3x: {sum(1 for x in rt_factors if x < 3)}/{len(rt_factors)}")
        print(f"  Unter 5x: {sum(1 for x in rt_factors if x < 5)}/{len(rt_factors)}")
    
    # Details pro Video
    print(f"\nDETAILS PRO VIDEO:")
    for p in performance_data[-5:]:  # Letzte 5
        print(f"\n  {p['creator']} - {p['video_id']}")
        print(f"    Dauer: {p['duration']:.1f}s")
        print(f"    Verarbeitung: {p['processing_time']:.1f}s")
        print(f"    Faktor: {p['realtime_factor']:.2f}x {'✅' if p['realtime_factor'] < 5 else '❌'}")
        print(f"    Analyzer: {p['successful_analyzers']}/{p['total_analyzers']}")

# Analyzer Zuverlässigkeit
print("\n2. ANALYZER-ZUVERLÄSSIGKEIT")
print("-" * 80)

analyzer_reliability = []
for name, stats in analyzer_stats.items():
    if stats['runs'] > 0:
        success_rate = (stats['successes'] / stats['runs']) * 100
        avg_segments = statistics.mean(stats['segments']) if stats['segments'] else 0
        analyzer_reliability.append((success_rate, name, stats))

print(f"\nANALYZER ERFOLGSRATEN:")
for success_rate, name, stats in sorted(analyzer_reliability, reverse=True):
    status = "✅" if success_rate == 100 else "⚠️" if success_rate >= 80 else "❌"
    print(f"  {status} {name}: {success_rate:.1f}% ({stats['successes']}/{stats['runs']})")
    if stats['segments']:
        print(f"     Durchschnitt: {statistics.mean(stats['segments']):.1f} Segmente")
    if stats['errors']:
        print(f"     Fehler: {len(stats['errors'])}")

# Fehler
print("\n3. GEFUNDENE FEHLER")
print("-" * 80)
if errors_found:
    print(f"Insgesamt {len(errors_found)} Fehler gefunden:")
    for error in errors_found[:10]:
        print(f"  - {error}")
    if len(errors_found) > 10:
        print(f"  ... und {len(errors_found) - 10} weitere")
else:
    print("Keine Fehler gefunden")

# Datenqualität Check
print("\n4. DATENQUALITÄT")
print("-" * 80)

# Prüfe die Qualität der Analyzer-Ausgaben
latest = sorted(glob.glob('/home/user/tiktok_production/results/*.json'))[-1]
with open(latest, 'r') as f:
    data = json.load(f)

print(f"Analysiere: {os.path.basename(latest)}")

# Qwen2-VL Qualität
qwen = None
for key in ['qwen2_vl_temporal', 'qwen2_vl_ultra', 'qwen2_vl_optimized']:
    if key in data['analyzer_results']:
        qwen = data['analyzer_results'][key]
        break

if qwen and 'segments' in qwen:
    print(f"\nQWEN2-VL ANALYSE:")
    success_segs = [s for s in qwen['segments'] if 'description' in s and s['description']]
    print(f"  Segmente: {len(success_segs)}/{len(qwen['segments'])}")
    
    if success_segs:
        # Prüfe Beschreibungsqualität
        lengths = [len(s['description']) for s in success_segs]
        print(f"  Beschreibungslänge: {min(lengths)}-{max(lengths)} Zeichen (Ø {statistics.mean(lengths):.0f})")
        
        # Zeige Beispiel
        print(f"\n  BEISPIEL-BESCHREIBUNG:")
        print(f"  {success_segs[0]['description'][:300]}...")

# Speech Transcription
speech = data['analyzer_results'].get('speech_transcription', {})
if speech and 'segments' in speech:
    transcribed = [s for s in speech['segments'] if s.get('text')]
    print(f"\nSPEECH TRANSCRIPTION:")
    print(f"  Transkribierte Segmente: {len(transcribed)}")
    if transcribed:
        total_text = ' '.join(s['text'] for s in transcribed)
        print(f"  Gesamttext: {len(total_text)} Zeichen")
        print(f"  Sprache: {speech.get('language', 'unknown')}")

# Object Detection
obj_det = data['analyzer_results'].get('object_detection', {})
if obj_det and 'segments' in obj_det:
    print(f"\nOBJECT DETECTION:")
    print(f"  Segmente: {len(obj_det['segments'])}")
    total_objects = sum(len(s.get('objects', [])) for s in obj_det['segments'])
    print(f"  Erkannte Objekte gesamt: {total_objects}")

print("\n" + "="*80)

# Store results for checklist
globals()['rt_factors'] = rt_factors if 'rt_factors' in locals() else []
globals()['analyzer_stats'] = analyzer_stats
globals()['errors_found'] = errors_found