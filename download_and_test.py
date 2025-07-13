#!/usr/bin/env python3
"""Download and analyze a TikTok video"""
import sys
sys.path.append('/home/user/tiktok_production')
from mass_processing.tiktok_downloader import TikTokDownloader
import requests
import json
import time
from pathlib import Path

# Test video
tiktok_url = "https://www.tiktok.com/@marcgebauer/video/7525171065367104790"

print("=== TESTE MIT ECHTEM TIKTOK VIDEO ===")
print(f"URL: {tiktok_url}")

# 1. Download video using TikTokDownloader
print("\n1. Lade Video herunter...")
downloader = TikTokDownloader()

try:
    result = downloader.download_video(tiktok_url)
    video_path = result.get('file_path')
    
    if video_path and Path(video_path).exists():
        print(f"✅ Video heruntergeladen: {video_path}")
        print(f"   Titel: {result.get('title', 'N/A')}")
        print(f"   Dauer: {result.get('duration', 0)}s")
        print(f"   Views: {result.get('view_count', 0):,}")
    else:
        print("❌ Download fehlgeschlagen - keine Datei gefunden")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Download fehlgeschlagen: {e}")
    # Use test video as fallback
    video_path = "/home/user/tiktok_production/test_video.mp4"
    if Path(video_path).exists():
        print(f"⚠️ Verwende Test-Video: {video_path}")
    else:
        print("❌ Kein Test-Video gefunden")
        sys.exit(1)

# 2. Analyze video
print("\n2. Analysiere Video...")
start_time = time.time()

analyze_response = requests.post("http://localhost:8003/analyze",
    json={"video_path": str(video_path)})

if analyze_response.status_code == 200:
    result = analyze_response.json()
    duration = time.time() - start_time
    
    print(f"✅ Analyse abgeschlossen in {duration:.1f}s")
    print(f"   Erfolgreiche Analyzer: {result.get('successful_analyzers', 0)}")
    print(f"   Gesamt Analyzer: {result.get('total_analyzers', 0)}")
    
    if result.get('results_path'):
        print(f"   Ergebnisse: {result['results_path']}")
        
        # Load and check results
        results_path = Path(result['results_path'])
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
            
            print("\n3. Prüfe Datenqualität...")
            
            # Check critical analyzers
            critical = {
                'qwen2_vl_temporal': 'Temporal Video Understanding',
                'speech_transcription': 'Speech Transcription', 
                'object_detection': 'Object Detection',
                'face_emotion': 'Face Emotion',
                'audio_analysis': 'Audio Analysis'
            }
            
            good_quality = 0
            bad_quality = 0
            
            for analyzer, desc in critical.items():
                print(f"\n{desc}:")
                if analyzer in data.get('analyzer_results', {}):
                    result = data['analyzer_results'][analyzer]
                    if 'error' in result:
                        print(f"  ❌ FEHLER: {result['error'][:80]}...")
                        bad_quality += 1
                    elif 'segments' in result and result['segments']:
                        seg_count = len(result['segments'])
                        first_seg = result['segments'][0]
                        
                        # Check data quality
                        if analyzer == 'qwen2_vl_temporal':
                            desc_text = first_seg.get('description', '')
                            if len(desc_text) > 100 and 'person' not in desc_text.lower():
                                print(f"  ✅ {seg_count} Segmente - HOHE QUALITÄT")
                                print(f"     {desc_text[:150]}...")
                                good_quality += 1
                            else:
                                print(f"  ⚠️ {seg_count} Segmente - NIEDRIGE QUALITÄT")
                                bad_quality += 1
                        else:
                            print(f"  ✅ {seg_count} Segmente")
                            sample = str(first_seg)[:100]
                            print(f"     {sample}...")
                            good_quality += 1
                    else:
                        print(f"  ⚠️ Keine Daten")
                        bad_quality += 1
                else:
                    print(f"  ❌ FEHLT in Ergebnissen")
                    bad_quality += 1
            
            # Overall summary
            total_analyzers = len(data.get('analyzer_results', {}))
            successful = sum(1 for r in data.get('analyzer_results', {}).values() 
                           if 'error' not in r and ('segments' in r or 'data' in r))
            
            print(f"\n=== ZUSAMMENFASSUNG ===")
            print(f"Analyzer gesamt: {total_analyzers}")
            print(f"Erfolgreich: {successful}")
            print(f"Kritische Analyzer: {good_quality}/{len(critical)} gut")
            print(f"Erfolgsrate: {successful/total_analyzers*100:.0f}%")
            
            # Save quality report
            quality_report = {
                'total_analyzers': total_analyzers,
                'successful': successful,
                'critical_good': good_quality,
                'critical_bad': bad_quality,
                'success_rate': successful/total_analyzers*100
            }
            
            with open('/home/user/tiktok_production/quality_report.json', 'w') as f:
                json.dump(quality_report, f, indent=2)
            
            print(f"\nQualitätsbericht gespeichert: quality_report.json")
else:
    print(f"❌ Analyse fehlgeschlagen: {analyze_response.text}")