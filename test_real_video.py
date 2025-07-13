#!/usr/bin/env python3
"""Test the system with a real TikTok video"""
import requests
import json
import time
from pathlib import Path
import sys

# Test video
tiktok_url = "https://www.tiktok.com/@marcgebauer/video/7525171065367104790"

print("=== TESTE MIT ECHTEM TIKTOK VIDEO ===")
print(f"URL: {tiktok_url}")

# 1. Download video
print("\n1. Lade Video herunter...")
download_response = requests.post("http://localhost:8003/download", 
    json={"url": tiktok_url})

if download_response.status_code == 200:
    download_data = download_response.json()
    video_path = download_data.get('video_path')
    print(f"✅ Video heruntergeladen: {video_path}")
else:
    print(f"❌ Download fehlgeschlagen: {download_response.text}")
    sys.exit(1)

# 2. Analyze video
print("\n2. Analysiere Video...")
start_time = time.time()

analyze_response = requests.post("http://localhost:8003/analyze",
    json={"video_path": video_path})

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
            critical = ['qwen2_vl_temporal', 'qwen2_vl_optimized', 'speech_transcription', 
                       'object_detection', 'face_emotion']
            
            for analyzer in critical:
                if analyzer in data.get('analyzer_results', {}):
                    result = data['analyzer_results'][analyzer]
                    if 'error' in result:
                        print(f"❌ {analyzer}: FEHLER - {result['error'][:50]}...")
                    elif 'segments' in result and result['segments']:
                        seg_count = len(result['segments'])
                        sample = str(result['segments'][0])[:100]
                        print(f"✅ {analyzer}: {seg_count} Segmente")
                        print(f"   Beispiel: {sample}...")
                    else:
                        print(f"⚠️ {analyzer}: Keine Daten")
                else:
                    print(f"❌ {analyzer}: FEHLT in Ergebnissen")
            
            # Overall quality check
            total_analyzers = len(data.get('analyzer_results', {}))
            successful = sum(1 for r in data.get('analyzer_results', {}).values() 
                           if 'error' not in r and 'segments' in r and r['segments'])
            
            print(f"\n=== ZUSAMMENFASSUNG ===")
            print(f"Erfolgreiche Analyzer: {successful}/{total_analyzers}")
            print(f"Erfolgsrate: {successful/total_analyzers*100:.0f}%")
            
            # Check Qwen2-VL quality
            if 'qwen2_vl_temporal' in data.get('analyzer_results', {}):
                qwen_result = data['analyzer_results']['qwen2_vl_temporal']
                if 'segments' in qwen_result and qwen_result['segments']:
                    desc = qwen_result['segments'][0].get('description', '')
                    if len(desc) > 100:
                        print(f"\n✅ Qwen2-VL liefert gute Beschreibungen ({len(desc)} Zeichen)")
                    else:
                        print(f"\n⚠️ Qwen2-VL Beschreibungen zu kurz ({len(desc)} Zeichen)")
else:
    print(f"❌ Analyse fehlgeschlagen: {analyze_response.text}")