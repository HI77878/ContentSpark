#!/usr/bin/env python3
"""Finaler Test mit echtem TikTok Video"""
import requests
import json
import time
from datetime import datetime
import sys
sys.path.append('/home/user/tiktok_production')

# TikTok Video mit Sprache, Objekten und Effekten
TIKTOK_URL = "https://www.tiktok.com/@marcgebauer/video/7387444736228306193"

print("=== FINALER TEST MIT ECHTEM TIKTOK VIDEO ===")
print(f"URL: {TIKTOK_URL}")
print(f"Start: {datetime.now()}\n")

# 1. Download Video
from mass_processing.tiktok_downloader import TikTokDownloader
downloader = TikTokDownloader()

print("1. DOWNLOAD VIDEO...")
download_start = time.time()
result = downloader.download_video(TIKTOK_URL)

if not result or 'local_path' not in result:
    print("❌ Download fehlgeschlagen!")
    # Prüfe ob bereits heruntergeladen
    video_id = TIKTOK_URL.split('/')[-1]
    video_path = f"/home/user/tiktok_videos/videos/{video_id}.mp4"
    import os
    if os.path.exists(video_path):
        print(f"✅ Video bereits vorhanden: {video_path}")
        result = {'local_path': video_path}
    else:
        exit(1)
else:
    video_path = result['local_path']

# Hole Video-Dauer
import subprocess
duration_cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
video_duration = float(subprocess.check_output(duration_cmd, shell=True).decode().strip())
download_time = time.time() - download_start

print(f"✅ Video bereit!")
print(f"   Pfad: {video_path}")
print(f"   Dauer: {video_duration:.1f}s\n")

# 2. Analyse mit TikTok URL
print("2. ANALYSIERE VIDEO MIT OPTIMIERTEN ANALYZERN...")
analysis_start = time.time()

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
    
    # Lade Ergebnisse
    with open(result['results_file']) as f:
        data = json.load(f)
    
    # 3. Detaillierte Auswertung
    print("3. DETAILLIERTE AUSWERTUNG")
    print("="*80)
    
    # TikTok Integration Check
    print("TIKTOK INTEGRATION:")
    print(f"  URL: {'✅' if data['metadata'].get('tiktok_url') == TIKTOK_URL else '❌'}")
    print(f"  Creator: {data['metadata'].get('creator_username', 'N/A')}")
    print()
    
    # Analyzer Performance
    print("ANALYZER PERFORMANCE:")
    print(f"{'Analyzer':<25} {'Segmente':<12} {'Pro Sekunde':<12} {'Status'}")
    print("-"*70)
    
    analyzer_stats = {}
    for analyzer, result_data in sorted(data['analyzer_results'].items()):
        if 'error' in result_data:
            segments = 0
            status = f"❌ ERROR: {result_data['error'][:30]}..."
        elif 'segments' in result_data:
            segments = len(result_data['segments'])
            if segments > 0:
                status = "✅ OK"
            else:
                status = "⚠️ Keine Daten"
        else:
            segments = 0
            status = "❌ Keine Segmente"
        
        rate = segments / video_duration
        analyzer_stats[analyzer] = {'segments': segments, 'rate': rate}
        
        icon = "🟢" if rate >= 1.0 else "🟡" if rate >= 0.5 else "🔴"
        print(f"{icon} {analyzer:<23} {segments:<12} {rate:<12.2f} {status}")
    
    # Kritische Analyzer im Detail
    print("\n" + "="*80)
    print("KRITISCHE ANALYZER IM DETAIL:\n")
    
    critical = ['qwen2_vl_temporal', 'object_detection', 'speech_transcription', 'visual_effects']
    for analyzer in critical:
        if analyzer in data['analyzer_results']:
            result_data = data['analyzer_results'][analyzer]
            stats = analyzer_stats[analyzer]
            
            print(f"{analyzer.upper()}:")
            print(f"  Segmente: {stats['segments']} ({stats['rate']:.2f}/s)")
            
            if 'segments' in result_data and result_data['segments']:
                segments = result_data['segments']
                
                # Zeige Beispiele
                if analyzer == 'qwen2_vl_temporal':
                    for i, seg in enumerate(segments[:3]):
                        desc = seg.get('description', 'N/A')
                        print(f"    [{seg.get('timestamp', 0):.1f}s] {desc[:100]}...")
                        
                elif analyzer == 'object_detection':
                    # Zähle Objekte
                    object_types = {}
                    for seg in segments:
                        obj_type = seg.get('object', 'unknown')
                        object_types[obj_type] = object_types.get(obj_type, 0) + 1
                    print(f"    Erkannte Objekte: {dict(sorted(object_types.items(), key=lambda x: x[1], reverse=True)[:5])}")
                    
                elif analyzer == 'speech_transcription':
                    # Zeige erste Worte
                    total_words = sum(seg.get('word_count', 0) for seg in segments)
                    print(f"    Wörter gesamt: {total_words}")
                    for i, seg in enumerate(segments[:3]):
                        text = seg.get('text', '')
                        if text:
                            print(f"    [{seg.get('start_time', 0):.1f}s] \"{text}\"")
                            
                elif analyzer == 'visual_effects':
                    # Zähle Effekte
                    effect_counts = {}
                    for seg in segments:
                        if 'effects' in seg:
                            for effect in seg['effects']:
                                if effect not in ['timestamp']:
                                    effect_counts[effect] = effect_counts.get(effect, 0) + 1
                    print(f"    Top Effekte: {dict(sorted(effect_counts.items(), key=lambda x: x[1], reverse=True)[:5])}")
            else:
                print(f"    ❌ Keine Segmente")
            print()
    
    # Zusammenfassung
    print("="*80)
    print("ZUSAMMENFASSUNG:\n")
    
    # Erfolgsmetriken
    successful = len([a for a, s in analyzer_stats.items() if s['segments'] > 0])
    high_quality = len([a for a, s in analyzer_stats.items() if s['rate'] >= 1.0])
    
    print(f"✅ Erfolgreiche Analyzer: {successful}/{len(analyzer_stats)} ({successful/len(analyzer_stats)*100:.0f}%)")
    print(f"🌟 High-Quality (≥1/s): {high_quality} Analyzer")
    print(f"⚡ Performance: {analysis_time/video_duration:.1f}x realtime")
    print(f"💾 Datengröße: {len(json.dumps(data))/1024/1024:.1f} MB")
    
    # Top Performer
    print("\nTOP PERFORMER (Segmente/Sekunde):")
    top_performers = sorted(analyzer_stats.items(), key=lambda x: x[1]['rate'], reverse=True)[:5]
    for analyzer, stats in top_performers:
        print(f"  🏆 {analyzer}: {stats['rate']:.2f}/s ({stats['segments']} Segmente)")
    
    # Verbesserungen gegenüber Baseline
    print("\nVERBESSERUNGEN:")
    baseline = {
        'visual_effects': 30/10,  # 30 segments in 10s video
        'qwen2_vl_temporal': 4/10,
        'object_detection': 0,
        'speech_transcription': 0
    }
    
    for analyzer in critical:
        if analyzer in analyzer_stats:
            current = analyzer_stats[analyzer]['segments']
            old = baseline.get(analyzer, 0) * video_duration / 10
            if old > 0:
                improvement = (current / old - 1) * 100
                print(f"  {analyzer}: {old:.0f} → {current} Segmente (+{improvement:.0f}%)")
            elif current > 0:
                print(f"  {analyzer}: 0 → {current} Segmente (NEU!)")
    
    print(f"\n✅ TEST ERFOLGREICH ABGESCHLOSSEN!")
    
else:
    print(f"❌ Analyse fehlgeschlagen: {response.status_code}")
    print(response.text)