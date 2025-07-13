#!/usr/bin/env python3
"""Test der optimierten Analyzer"""
import requests
import json
import time
from datetime import datetime
import sys
sys.path.append('/home/user/tiktok_production')

# TikTok Video mit klarer Sprache und Objekten
TIKTOK_URL = "https://www.tiktok.com/@leon_schliebach/video/7446489995663117590"

print("=== TEST DER OPTIMIERTEN ANALYZER ===")
print(f"TikTok URL: {TIKTOK_URL}")
print(f"Start: {datetime.now()}\n")

# 1. Download Video
from mass_processing.tiktok_downloader import TikTokDownloader
downloader = TikTokDownloader()

print("1. DOWNLOAD VIDEO...")
download_start = time.time()
result = downloader.download_video(TIKTOK_URL)

if not result or 'local_path' not in result:
    print("❌ Download fehlgeschlagen!")
    exit(1)

video_path = result['local_path']
video_duration = result.get('duration', 0)
download_time = time.time() - download_start

print(f"✅ Download erfolgreich!")
print(f"   Pfad: {video_path}")
print(f"   Dauer: {video_duration}s")
print(f"   Download-Zeit: {download_time:.1f}s\n")

# 2. Analyse mit TikTok URL
print("2. ANALYSIERE VIDEO...")
analysis_start = time.time()

# WICHTIG: TikTok URL mitschicken!
response = requests.post(
    "http://localhost:8003/analyze",
    json={
        "video_path": video_path,
        "tiktok_url": TIKTOK_URL,
        "creator_username": TIKTOK_URL.split('@')[1].split('/')[0]
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
    
    # 3. Lade und prüfe Ergebnisse
    with open(result['results_file']) as f:
        data = json.load(f)
    
    print("3. ANALYZER-ERGEBNISSE:")
    print("-" * 50)
    
    for analyzer, analyzer_data in sorted(data['analyzer_results'].items()):
        if 'error' in analyzer_data:
            print(f"❌ {analyzer}: ERROR")
        elif 'segments' in analyzer_data:
            segments = analyzer_data['segments']
            if segments:
                print(f"✅ {analyzer}: {len(segments)} Segmente")
            else:
                print(f"⚠️ {analyzer}: 0 Segmente")
        else:
            print(f"⚠️ {analyzer}: Keine Segmente")
    
    # 4. Speichere Report
    report = {
        'test_time': datetime.now().isoformat(),
        'tiktok_url': TIKTOK_URL,
        'video_duration': video_duration,
        'analysis_time': analysis_time,
        'realtime_factor': analysis_time/video_duration,
        'result_file': result['results_file']
    }
    
    with open('optimized_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Test-Report gespeichert: optimized_test_report.json")
    
else:
    print(f"❌ Analyse fehlgeschlagen: {response.status_code}")
    print(response.text)
    exit(1)