#!/usr/bin/env python3
"""Test mit kurzem Test-Video für schnelle Ergebnisse"""
import requests
import json
import time
from datetime import datetime

# Verwende Test-Video (10s)
video_path = "/home/user/tiktok_production/test_video.mp4"
video_duration = 10.0
TIKTOK_URL = "https://www.tiktok.com/@testuser/video/123456789"

print("=== SCHNELLTEST DER OPTIMIERTEN ANALYZER ===")
print(f"Video: {video_path}")
print(f"Dauer: {video_duration}s")
print(f"Start: {datetime.now()}\n")

# Analyse mit TikTok URL
print("ANALYSIERE TEST-VIDEO...")
analysis_start = time.time()

response = requests.post(
    "http://localhost:8003/analyze",
    json={
        "video_path": video_path,
        "tiktok_url": TIKTOK_URL,
        "creator_username": "testuser"
    },
    timeout=300
)

analysis_time = time.time() - analysis_start

if response.status_code == 200:
    result = response.json()
    print(f"✅ Analyse erfolgreich!")
    print(f"   Zeit: {analysis_time:.1f}s ({analysis_time/video_duration:.1f}x realtime)")
    print(f"   Erfolgreiche Analyzer: {result['successful_analyzers']}/{result['total_analyzers']}")
    print(f"   Ergebnis: {result['results_file']}\n")
    
    # Lade Ergebnisse
    with open(result['results_file']) as f:
        data = json.load(f)
    
    # Check TikTok URL
    print("TIKTOK INTEGRATION:")
    print(f"  URL gespeichert: {'✅' if data['metadata'].get('tiktok_url') else '❌'}")
    print(f"  Creator: {data['metadata'].get('creator_username', 'N/A')}")
    print()
    
    # Analyzer Detail-Prüfung
    print("ANALYZER DETAIL-CHECK:")
    print("-" * 70)
    print(f"{'Analyzer':<25} {'Status':<10} {'Segmente':<10} {'Pro Sekunde':<12} {'Qualität'}")
    print("-" * 70)
    
    analyzer_stats = {}
    for analyzer, analyzer_data in sorted(data['analyzer_results'].items()):
        if 'error' in analyzer_data:
            status = "ERROR"
            segments = 0
            quality = "❌ Fehler"
        elif 'segments' in analyzer_data:
            segments = len(analyzer_data['segments'])
            if segments > 0:
                status = "OK"
                # Prüfe Datenqualität
                sample = str(analyzer_data['segments'][0])
                if any(term in sample.lower() for term in ['placeholder', 'balanced', 'moderate', 'normal']):
                    quality = "⚠️ Generisch"
                else:
                    quality = "✅ Spezifisch"
            else:
                status = "NO_DATA"
                quality = "❌ Leer"
        else:
            status = "NO_SEGMENTS"
            segments = 0
            quality = "❌ Keine Daten"
        
        segments_per_sec = segments / video_duration if segments > 0 else 0
        
        icon = "✅" if segments_per_sec >= 1.0 else "⚠️" if segments_per_sec > 0.5 else "❌"
        print(f"{analyzer:<25} {status:<10} {segments:<10} {segments_per_sec:>6.2f}/s     {quality}")
    
    # Kritische Analyzer im Detail
    print("\n" + "="*70)
    print("KRITISCHE ANALYZER IM DETAIL:")
    
    critical = ['qwen2_vl_temporal', 'object_detection', 'speech_transcription', 'visual_effects']
    for analyzer in critical:
        if analyzer in data['analyzer_results']:
            result_data = data['analyzer_results'][analyzer]
            print(f"\n{analyzer.upper()}:")
            
            if 'segments' in result_data and result_data['segments']:
                segments = result_data['segments']
                print(f"  Segmente: {len(segments)} ({len(segments)/video_duration:.2f}/s)")
                
                # Zeige erste 3 Segmente
                for i, seg in enumerate(segments[:3]):
                    timestamp = seg.get('timestamp', seg.get('start_time', 'N/A'))
                    
                    # Extrahiere relevante Daten je nach Analyzer
                    if analyzer == 'qwen2_vl_temporal':
                        content = seg.get('description', 'N/A')[:80]
                    elif analyzer == 'object_detection':
                        objects = seg.get('objects', [])
                        content = f"{len(objects)} objects" if objects else "No objects"
                    elif analyzer == 'speech_transcription':
                        content = seg.get('text', 'N/A')[:80]
                    elif analyzer == 'visual_effects':
                        effects = [k for k in seg.keys() if k not in ['timestamp', 'confidence']]
                        content = f"Effects: {', '.join(effects[:3])}"
                    else:
                        content = str(seg)[:80]
                    
                    print(f"    [{timestamp}s] {content}...")
            else:
                print(f"  ❌ Keine Segmente")
    
    # Zusammenfassung
    print("\n" + "="*70)
    print("OPTIMIERUNGS-ERFOLG:")
    
    # Berechne Verbesserungen
    old_scores = {
        'qwen2_vl_temporal': 4,
        'object_detection': 0,
        'speech_transcription': 0,
        'visual_effects': 30
    }
    
    for analyzer in critical:
        if analyzer in data['analyzer_results'] and 'segments' in data['analyzer_results'][analyzer]:
            new_count = len(data['analyzer_results'][analyzer]['segments'])
            old_count = old_scores.get(analyzer, 0)
            improvement = (new_count / old_count * 100 - 100) if old_count > 0 else float('inf')
            
            if improvement == float('inf'):
                print(f"  {analyzer}: {old_count} → {new_count} segmente (NEU!)")
            else:
                print(f"  {analyzer}: {old_count} → {new_count} segmente (+{improvement:.0f}%)")
    
else:
    print(f"❌ Analyse fehlgeschlagen: {response.status_code}")
    print(response.text)