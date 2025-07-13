#!/usr/bin/env python3
"""Full quality test with real TikTok video"""
import sys
sys.path.append('/home/user/tiktok_production')
import time
import json
from pathlib import Path
import requests
from datetime import datetime
import psutil
import subprocess
from system_monitor import SystemMonitor

# Video URL
VIDEO_URL = "https://www.tiktok.com/@leon_schliebach/video/7446489995663117590?q=vlog&t=1750908402121"

print("=== VOLLSTÄNDIGER QUALITÄTSTEST MIT ECHTEM TIKTOK VIDEO ===")
print(f"Video: {VIDEO_URL}")
print(f"Start: {datetime.now()}\n")

# 1. System Baseline
print("1. SYSTEM BASELINE:")
print(f"CPU Cores: {psutil.cpu_count()}")
print(f"CPU: {psutil.cpu_percent()}%")
print(f"RAM: {psutil.virtual_memory().used / 1024**3:.1f}GB / {psutil.virtual_memory().total / 1024**3:.1f}GB")

# GPU Check
try:
    gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits']).decode()
    gpu_used, gpu_total, gpu_util = map(int, gpu_info.strip().split(', '))
    print(f"GPU: {gpu_used}MB / {gpu_total}MB ({gpu_util}%)\n")
except:
    print("GPU: Not available\n")

# 2. Start System Monitor
print("2. STARTE SYSTEM MONITORING...")
monitor = SystemMonitor()
monitor.start()

# 3. Video Download
print("\n3. DOWNLOAD VIDEO:")
start_download = time.time()

from mass_processing.tiktok_downloader import TikTokDownloader
downloader = TikTokDownloader()

try:
    result = downloader.download_video(VIDEO_URL)
    
    if result and result.get('file_path'):
        video_path = result['file_path']
        download_time = time.time() - start_download
        print(f"✅ Download erfolgreich: {video_path}")
        print(f"   Dauer: {download_time:.1f}s")
        print(f"   Video-Länge: {result.get('duration', 'unknown')}s")
        print(f"   Titel: {result.get('title', 'N/A')}")
        print(f"   Views: {result.get('view_count', 0):,}\n")
        
        video_duration = result.get('duration', 11.27)
    else:
        print("❌ Download fehlgeschlagen - kein Pfad")
        # Use test video as fallback
        video_path = "/home/user/tiktok_production/test_video.mp4"
        video_duration = 10
        if Path(video_path).exists():
            print(f"⚠️ Verwende Test-Video: {video_path}\n")
        else:
            print("❌ Kein Test-Video gefunden!")
            sys.exit(1)
except Exception as e:
    print(f"❌ Download Fehler: {e}")
    video_path = "/home/user/tiktok_production/test_video.mp4"
    video_duration = 10
    if Path(video_path).exists():
        print(f"⚠️ Verwende Test-Video: {video_path}\n")
    else:
        print("❌ Kein Test-Video gefunden!")
        sys.exit(1)

# 4. API Status
print("4. API STATUS:")
try:
    api_resp = requests.get("http://localhost:8003/", timeout=5)
    if api_resp.status_code == 200:
        api_data = api_resp.json()
        print(f"✅ API läuft: {api_data.get('total_analyzers', 'unknown')} Analyzer")
    else:
        print(f"⚠️ API Status Code: {api_resp.status_code}")
except Exception as e:
    print(f"❌ API nicht erreichbar: {e}")
    sys.exit(1)

# 5. Analyse mit Monitoring
print("\n5. STARTE ANALYSE MIT MONITORING:")
print("=" * 60)
analysis_start = time.time()

# Clear results directory
results_dir = Path("/home/user/tiktok_production/results")
results_dir.mkdir(exist_ok=True)

# Send analysis request
try:
    analyze_resp = requests.post(
        "http://localhost:8003/analyze",
        json={"video_path": str(video_path)},
        timeout=600  # 10 minutes timeout
    )
    
    analysis_time = time.time() - analysis_start
    
    if analyze_resp.status_code == 200:
        result_data = analyze_resp.json()
        print(f"\n✅ Analyse abgeschlossen in {analysis_time:.1f}s")
        print(f"   Erfolgreiche Analyzer: {result_data.get('successful_analyzers', 0)}")
        print(f"   Gesamt Analyzer: {result_data.get('total_analyzers', 0)}")
        
        # Get results file path
        results_path = result_data.get('results_path')
        if results_path and Path(results_path).exists():
            print(f"   Ergebnis-Datei: {results_path}")
            
            # Load results
            with open(results_path) as f:
                analysis_data = json.load(f)
        else:
            print("❌ Keine Ergebnis-Datei gefunden")
            analysis_data = None
    else:
        print(f"❌ Analyse fehlgeschlagen: {analyze_resp.status_code}")
        print(analyze_resp.text)
        analysis_data = None
        
except Exception as e:
    print(f"❌ Analyse Fehler: {e}")
    analysis_data = None
    analysis_time = time.time() - analysis_start

# 6. Stop monitoring and save stats
print("\n6. SYSTEM NACH ANALYSE:")
monitor_stats = monitor.stop()
monitor_summary = monitor.get_summary()

# Save monitoring data
monitor_file = f"results/monitor_stats_{int(time.time())}.json"
monitor.save_stats(monitor_file)
print(f"Monitor-Daten gespeichert: {monitor_file}")

# Print summary
print(f"\nMONITORING ZUSAMMENFASSUNG:")
print(f"Dauer: {monitor_summary['duration_seconds']:.1f}s")
print(f"CPU Durchschnitt: {monitor_summary['cpu']['avg']:.1f}% (Max: {monitor_summary['cpu']['max']:.1f}%)")
print(f"RAM Durchschnitt: {monitor_summary['memory_gb']['avg']:.1f}GB (Max: {monitor_summary['memory_gb']['max']:.1f}GB)")
print(f"GPU Auslastung: {monitor_summary['gpu_util']['avg']:.1f}% (Max: {monitor_summary['gpu_util']['max']:.1f}%)")
print(f"GPU Memory: {monitor_summary['gpu_memory_mb']['avg']:.0f}MB (Max: {monitor_summary['gpu_memory_mb']['max']:.0f}MB)")

# Performance metrics
if video_duration > 0:
    speed_factor = analysis_time / video_duration
    print(f"\nPERFORMANCE: {speed_factor:.1f}x realtime")
    if speed_factor < 3:
        print("✅ Ziel erreicht: <3x realtime")
    else:
        print("⚠️ Ziel verfehlt: >3x realtime")

# Save complete test results
test_results = {
    'test_timestamp': datetime.now().isoformat(),
    'video_url': VIDEO_URL,
    'video_duration': video_duration,
    'download_time': download_time if 'download_time' in locals() else 0,
    'analysis_time': analysis_time,
    'speed_factor': analysis_time / video_duration if video_duration > 0 else 0,
    'monitoring_summary': monitor_summary,
    'analysis_results': analysis_data
}

results_file = f"results/full_quality_test_{int(time.time())}.json"
with open(results_file, 'w') as f:
    json.dump(test_results, f, indent=2)

print(f"\nTest-Ergebnisse gespeichert: {results_file}")

# Quick quality check
if analysis_data and 'analyzer_results' in analysis_data:
    print("\n7. QUICK QUALITY CHECK:")
    critical_found = 0
    critical_working = 0
    
    critical_analyzers = ['qwen2_vl', 'speech_transcription', 'object_detection', 'face_emotion']
    
    for critical in critical_analyzers:
        found = False
        for analyzer_name in analysis_data['analyzer_results']:
            if critical in analyzer_name:
                found = True
                result = analysis_data['analyzer_results'][analyzer_name]
                if 'error' not in result and 'segments' in result and result['segments']:
                    critical_working += 1
                    print(f"✅ {critical}: {len(result['segments'])} Segmente")
                else:
                    print(f"❌ {critical}: Fehler oder keine Daten")
                break
        
        if found:
            critical_found += 1
        else:
            print(f"❌ {critical}: NICHT GEFUNDEN")
    
    print(f"\nKritische Analyzer: {critical_working}/{len(critical_analyzers)} funktionieren")
    
print("\n=== TEST ABGESCHLOSSEN ===")
print(f"Führe quality_validator.py aus für detaillierte Analyse...")