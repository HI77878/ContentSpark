#!/usr/bin/env python3
"""
Frische Analyse mit Chase Ridgeway Video und vollständigem Monitoring
"""
import os
import sys
import json
import glob
import time
import subprocess
from datetime import datetime
from system_monitor import SystemMonitor

print("\n🎬 FRISCHE ANALYSE - CHASE RIDGEWAY VIDEO")
print("="*80)

# Überprüfe ob API läuft
try:
    import requests
    health = requests.get("http://localhost:8003/health", timeout=5)
    if health.status_code != 200:
        print("❌ API ist nicht healthy!")
        sys.exit(1)
    print("✅ API läuft und ist bereit")
except:
    print("❌ API läuft nicht! Bitte starten mit:")
    print("   cd /home/user/tiktok_production")
    print("   source fix_ffmpeg_env.sh")
    print("   python3 api/stable_production_api_multiprocess.py &")
    sys.exit(1)

# Start Monitoring
monitor = SystemMonitor()
monitor.start()

# Chase Ridgeway Video URL
url = "https://www.tiktok.com/@chaseridgewayy/video/7522589683939921165"
print(f"\n📥 Starte Download und Analyse: {url}")
print("="*80)

try:
    # Führe Download und Analyse aus
    cmd = f"python3 /home/user/tiktok_production/download_and_analyze.py {url}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Zeige Output live
    for line in process.stdout:
        if line.strip():  # Ignoriere leere Zeilen
            print(line.strip())
    
    process.wait()
    
    if process.returncode == 0:
        print("\n✅ ANALYSE ERFOLGREICH ABGESCHLOSSEN")
    else:
        print("\n❌ ANALYSE FEHLGESCHLAGEN")
        stderr = process.stderr.read()
        if stderr:
            print(f"Fehler: {stderr}")
        
except Exception as e:
    print(f"❌ Fehler: {e}")
finally:
    # Stoppe Monitoring
    monitor.stop()

# Finde die neue Analyse-Datei
print("\n🔍 Suche Analyse-Ergebnisse...")
result_files = sorted(glob.glob('/home/user/tiktok_production/results/*7522589683939921165*.json'))

if result_files:
    latest_result = result_files[-1]
    print(f"\n📊 ANALYSE-ERGEBNISSE: {os.path.basename(latest_result)}")
    print("="*80)
    
    with open(latest_result) as f:
        data = json.load(f)
    
    # Metadata
    meta = data['metadata']
    print("\n📋 METADATA:")
    print(f"   TikTok URL: {meta.get('tiktok_url', 'N/A')}")
    print(f"   Creator: @{meta.get('creator_username', 'N/A')}")
    print(f"   Video ID: {meta.get('tiktok_video_id', 'N/A')}")
    print(f"   Duration: {meta.get('duration', 0):.1f}s")
    print(f"   Processing Time: {meta.get('processing_time_seconds', 0):.1f}s")
    print(f"   Realtime Factor: {meta.get('realtime_factor', 0):.2f}x")
    print(f"   Analyzer Success: {meta.get('successful_analyzers', 0)}/{meta.get('total_analyzers', 0)}")
    print(f"   Reconstruction Score: {meta.get('reconstruction_score', 0):.1f}%")
    
    # Top Analyzer nach Zeit (aus Logs wenn möglich)
    print("\n⏱️ ANALYZER PERFORMANCE:")
    
    # Datenqualität pro Analyzer
    print("\n📊 DATENQUALITÄT:")
    empty_analyzers = []
    full_analyzers = []
    
    for name, result in sorted(data['analyzer_results'].items()):
        segments = result.get('segments', [])
        if segments:
            full_analyzers.append((name, len(segments)))
        else:
            # Prüfe ob andere Daten vorhanden sind
            if result and any(k != 'metadata' for k in result.keys()):
                full_analyzers.append((name, 'andere Daten'))
            else:
                empty_analyzers.append(name)
    
    # Zeige erfolgreiche Analyzer
    print(f"\n✅ Erfolgreiche Analyzer ({len(full_analyzers)}):")
    for name, count in full_analyzers:
        if isinstance(count, int):
            print(f"   {name}: {count} Segmente")
        else:
            print(f"   {name}: {count}")
    
    # Zeige leere Analyzer
    if empty_analyzers:
        print(f"\n❌ Analyzer ohne Daten ({len(empty_analyzers)}):")
        for name in empty_analyzers:
            print(f"   {name}")
    
    # Key Insights
    print("\n🔍 KEY INSIGHTS:")
    
    # Speech
    speech = data['analyzer_results'].get('speech_transcription', {}).get('segments', [])
    if speech:
        lang = speech[0].get('language', 'N/A')
        print(f"\n📢 Sprache: {lang}")
        total_text = " ".join([s.get('text', '') for s in speech if s.get('text')])
        print(f"   Transkription ({len(speech)} Segmente): {total_text[:100]}...")
    
    # Qwen2-VL Temporal
    qwen = data['analyzer_results'].get('qwen2_vl_temporal', {}).get('segments', [])
    if qwen:
        print(f"\n🎥 Video Understanding (Qwen2-VL):")
        for i, seg in enumerate(qwen[:3]):  # Erste 3 Segmente
            print(f"   [{seg.get('timestamp', 0):.1f}s]: {seg.get('description', 'N/A')[:80]}...")
    
    # Objects
    objects = data['analyzer_results'].get('object_detection', {}).get('segments', [])
    if objects:
        total_objects = sum(len(s.get('objects', [])) for s in objects)
        print(f"\n📦 Objekte erkannt: {total_objects} in {len(objects)} Frames")
        
        # Häufigste Objekte
        from collections import Counter
        obj_counter = Counter()
        for seg in objects:
            for obj in seg.get('objects', []):
                obj_name = obj.get('object', obj.get('class', obj.get('label', 'unknown')))
                obj_counter[obj_name] += 1
        
        print("   Top Objekte:")
        for obj, count in obj_counter.most_common(5):
            print(f"   - {obj}: {count}x")
    
    # Visual Effects
    effects = data['analyzer_results'].get('visual_effects', {}).get('segments', [])
    if effects:
        effect_types = Counter()
        for seg in effects:
            effect_types[seg.get('type', 'unknown')] += 1
        
        print(f"\n✨ Visuelle Effekte: {len(effects)} erkannt")
        for effect, count in effect_types.most_common(3):
            print(f"   - {effect}: {count}x")
    
    # Save comprehensive report
    report = {
        'analysis_id': os.path.basename(latest_result),
        'video_url': url,
        'timestamp': datetime.now().isoformat(),
        'metadata': meta,
        'monitoring_summary': {
            'gpu_max_util': max([m['gpu_util'] for m in monitor.metrics]) if monitor.metrics else 0,
            'gpu_max_mem_gb': max([m['gpu_mem_gb'] for m in monitor.metrics]) if monitor.metrics else 0,
            'gpu_max_temp': max([m['gpu_temp'] for m in monitor.metrics]) if monitor.metrics else 0,
            'cpu_max': max([m['cpu_percent'] for m in monitor.metrics]) if monitor.metrics else 0,
            'ram_max': max([m['ram_percent'] for m in monitor.metrics]) if monitor.metrics else 0,
        },
        'data_quality': {
            'successful_analyzers': len(full_analyzers),
            'empty_analyzers': len(empty_analyzers),
            'analyzers_with_data': [name for name, _ in full_analyzers],
            'analyzers_without_data': empty_analyzers
        }
    }
    
    report_file = '/home/user/tiktok_production/chase_fresh_analysis_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Vollständiger Report gespeichert: {report_file}")
    
else:
    print("\n❌ Keine Analyse-Ergebnisse gefunden!")

print("\n" + "="*80)
print("✅ FRISCHE ANALYSE KOMPLETT")
print("="*80)