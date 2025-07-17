#!/usr/bin/env python3
"""Test der optimierten Analyzer mit existierendem Video"""
import requests
import json
import time
from datetime import datetime

# Video bereits heruntergeladen
TIKTOK_URL = "https://www.tiktok.com/@leon_schliebach/video/7446489995663117590"
video_path = "/home/user/tiktok_videos/videos/7446489995663117590.mp4"
video_duration = 48.97

print("=== TEST DER OPTIMIERTEN ANALYZER ===")
print(f"TikTok URL: {TIKTOK_URL}")
print(f"Video: {video_path}")
print(f"Dauer: {video_duration:.1f}s")
print(f"Start: {datetime.now()}\n")

# Analyse mit TikTok URL
print("ANALYSIERE VIDEO...")
analysis_start = time.time()

# WICHTIG: TikTok URL mitschicken!
response = requests.post(
    "http://localhost:8003/analyze",
    json={
        "video_path": video_path,
        "tiktok_url": TIKTOK_URL,
        "creator_username": "leon_schliebach"
    },
    timeout=600
)

analysis_time = time.time() - analysis_start

if response.status_code == 200:
    result = response.json()
    print(f"✅ Analyse erfolgreich!")
    print(f"   Zeit: {analysis_time:.1f}s ({analysis_time/video_duration:.1f}x realtime)")
    print(f"   Erfolgreiche Analyzer: {result['successful_analyzers']}/{result['total_analyzers']}")
    print(f"   Ergebnis: {result['results_file']}\n")
    
    # Lade und prüfe Ergebnisse
    with open(result['results_file']) as f:
        data = json.load(f)
    
    print("ANALYZER-ERGEBNISSE:")
    print("-" * 60)
    
    analyzer_stats = {}
    for analyzer, analyzer_data in sorted(data['analyzer_results'].items()):
        if 'error' in analyzer_data:
            status = "ERROR"
            segments = 0
        elif 'segments' in analyzer_data:
            segments = len(analyzer_data['segments'])
            if segments > 0:
                status = "OK"
            else:
                status = "NO_DATA"
        else:
            status = "NO_SEGMENTS"
            segments = 0
        
        segments_per_sec = segments / video_duration if segments > 0 else 0
        analyzer_stats[analyzer] = {
            'status': status,
            'segments': segments,
            'per_second': segments_per_sec
        }
        
        icon = "✅" if status == "OK" else "❌"
        print(f"{icon} {analyzer:<25} {segments:>4} segments ({segments_per_sec:.2f}/s)")
    
    # Zusammenfassung
    ok_analyzers = [a for a, s in analyzer_stats.items() if s['status'] == 'OK']
    print("\n" + "="*60)
    print(f"ZUSAMMENFASSUNG:")
    print(f"  Erfolgreiche Analyzer: {len(ok_analyzers)}/{len(analyzer_stats)} ({len(ok_analyzers)/len(analyzer_stats)*100:.0f}%)")
    print(f"  Durchschnitt Segmente/Sekunde: {sum(s['per_second'] for s in analyzer_stats.values())/len(analyzer_stats):.2f}")
    print(f"  Processing: {analysis_time/video_duration:.1f}x realtime")
    
    # Kritische Analyzer Detail-Check
    print("\nKRITISCHE ANALYZER:")
    critical = ['qwen2_vl_temporal', 'object_detection', 'speech_transcription', 'visual_effects']
    for analyzer in critical:
        if analyzer in data['analyzer_results']:
            stats = analyzer_stats[analyzer]
            print(f"\n{analyzer}:")
            print(f"  Status: {stats['status']}")
            print(f"  Segmente: {stats['segments']} ({stats['per_second']:.2f}/s)")
            
            if stats['segments'] > 0:
                sample = data['analyzer_results'][analyzer]['segments'][0]
                print(f"  Beispiel: {str(sample)[:150]}...")
    
else:
    print(f"❌ Analyse fehlgeschlagen: {response.status_code}")
    print(response.text)