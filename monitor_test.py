#!/usr/bin/env python3
"""Überwachter Testlauf mit Live-Monitoring"""

import json
import requests
import time
import subprocess
import threading
from datetime import datetime

# Test video
test_url = "https://www.tiktok.com/@leon_schliebach/video/7446489995663117590"

print("🔍 ÜBERWACHTER TESTLAUF MIT LIVE-MONITORING")
print("="*70)
print(f"TikTok URL: {test_url}")
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# GPU Monitoring Thread
gpu_stats = []
def monitor_gpu():
    while monitoring:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_util, mem_used, mem_total = map(float, result.stdout.strip().split(', '))
                gpu_stats.append({
                    'time': time.time(),
                    'gpu_util': gpu_util,
                    'memory_percent': (mem_used / mem_total) * 100
                })
        except:
            pass
        time.sleep(1)

# Start GPU monitoring
monitoring = True
gpu_thread = threading.Thread(target=monitor_gpu)
gpu_thread.start()

# Start analysis
print("📡 Starte Analyse...")
start_time = time.time()

response = requests.post(
    "http://localhost:8003/analyze",
    json={"tiktok_url": test_url}
)

end_time = time.time()
monitoring = False
gpu_thread.join()

print(f"⏱️  Analyse dauerte: {end_time - start_time:.1f}s")

if response.status_code != 200:
    print(f"❌ API Error: {response.status_code}")
    print(response.text)
    exit(1)

# Get result file
result_data = response.json()
print(f"✅ API Response erhalten")

# Find result file from logs
import re
log_check = subprocess.run(['tail', '-100', '/home/user/tiktok_production/logs/automatic_api.log'], 
                          capture_output=True, text=True)
match = re.search(r'Results saved to (/home/user/tiktok_production/results/.+\.json)', log_check.stdout)
if match:
    result_file = match.group(1)
    print(f"📁 Result file: {result_file}")
else:
    print("❌ Could not find result file")
    exit(1)

# Load results
with open(result_file, 'r') as f:
    data = json.load(f)

print("\n" + "="*70)
print("📊 ANALYZER-BY-ANALYZER ÜBERWACHUNG:")
print("="*70)

# Detailed analyzer monitoring
analyzers_to_check = ['age_estimation', 'object_detection', 'speech_emotion']
overall_status = []

for analyzer_name in analyzers_to_check:
    print(f"\n{'🎭' if analyzer_name == 'age_estimation' else '🎯' if analyzer_name == 'object_detection' else '😊'} {analyzer_name.upper()}:")
    print("-"*50)
    
    if analyzer_name not in data['analyzer_results']:
        print(f"❌ FEHLER: Analyzer nicht in Ergebnissen!")
        overall_status.append((analyzer_name, False, "Nicht gefunden"))
        continue
    
    analyzer_data = data['analyzer_results'][analyzer_name]
    
    if analyzer_name == 'age_estimation':
        segments = analyzer_data.get('segments', [])
        faces_detected = sum(s.get('faces_detected', 0) for s in segments)
        frames_with_faces = sum(1 for s in segments if s.get('faces_detected', 0) > 0)
        detection_rate = frames_with_faces / len(segments) * 100 if segments else 0
        
        print(f"  Segments analysiert: {len(segments)}")
        print(f"  Frames mit Gesichtern: {frames_with_faces} ({detection_rate:.1f}%)")
        print(f"  Gesichter insgesamt: {faces_detected}")
        
        # Sample output
        if segments and frames_with_faces > 0:
            for seg in segments:
                if seg.get('faces_detected', 0) > 0:
                    print(f"\n  Beispiel bei {seg['timestamp']}s:")
                    for face in seg.get('faces', [])[:1]:
                        print(f"    - Alter: {face.get('age', 'N/A')} Jahre")
                        print(f"    - Geschlecht: {face.get('gender', 'N/A')}")
                        print(f"    - Konfidenz: {face.get('confidence', {}).get('detection', 0):.3f}")
                    break
        
        success = detection_rate > 20
        overall_status.append((analyzer_name, success, f"{detection_rate:.1f}% Erkennungsrate"))
        
    elif analyzer_name == 'object_detection':
        segments = analyzer_data.get('segments', [])
        if not segments:
            print(f"❌ FEHLER: Keine Segments!")
            overall_status.append((analyzer_name, False, "Keine Segments"))
            continue
            
        frames_with_person = sum(1 for s in segments if s.get('has_person', False))
        person_rate = frames_with_person / len(segments) * 100 if segments else 0
        total_objects = sum(s.get('objects_detected', 0) for s in segments)
        
        print(f"  Segments analysiert: {len(segments)}")
        print(f"  Frames mit Person: {frames_with_person} ({person_rate:.1f}%)")
        print(f"  Objekte insgesamt: {total_objects}")
        
        # Object type distribution
        object_types = {}
        for seg in segments:
            for obj in seg.get('objects', []):
                obj_type = obj.get('object_class', 'unknown')
                object_types[obj_type] = object_types.get(obj_type, 0) + 1
        
        print(f"  Objekttypen: {', '.join(f'{k}({v})' for k, v in sorted(object_types.items(), key=lambda x: x[1], reverse=True)[:5])}")
        
        success = person_rate > 30
        overall_status.append((analyzer_name, success, f"{person_rate:.1f}% Person-Erkennung"))
        
    elif analyzer_name == 'speech_emotion':
        segments = analyzer_data.get('segments', [])
        emotions = set()
        for seg in segments:
            if seg.get('dominant_emotion'):
                emotions.add(seg['dominant_emotion'])
        
        print(f"  Segments analysiert: {len(segments)}")
        print(f"  Emotionen gefunden: {', '.join(emotions) if emotions else 'KEINE'}")
        
        if analyzer_data.get('summary'):
            summary = analyzer_data['summary']
            print(f"  Gesamtton: {summary.get('overall_tone', 'N/A')}")
            print(f"  Emotionale Stabilität: {summary.get('emotional_stability', 'N/A')}")
        
        # Sample emotions
        if segments:
            print("\n  Beispiel-Emotionen:")
            for seg in segments[:3]:
                if seg.get('dominant_emotion'):
                    print(f"    - {seg['timestamp']}s: {seg['dominant_emotion']} (Konfidenz: {seg['confidence']:.2f})")
        
        success = len(emotions) > 0 and 'none' not in str(emotions).lower()
        overall_status.append((analyzer_name, success, f"{len(emotions)} Emotionen erkannt"))

# GPU Stats Summary
print("\n" + "="*70)
print("🖥️  GPU ÜBERWACHUNG:")
print("="*70)
if gpu_stats:
    max_gpu = max(s['gpu_util'] for s in gpu_stats)
    avg_gpu = sum(s['gpu_util'] for s in gpu_stats) / len(gpu_stats)
    max_mem = max(s['memory_percent'] for s in gpu_stats)
    avg_mem = sum(s['memory_percent'] for s in gpu_stats) / len(gpu_stats)
    
    print(f"  GPU Auslastung: Ø {avg_gpu:.1f}% (Max: {max_gpu:.1f}%)")
    print(f"  GPU Speicher: Ø {avg_mem:.1f}% (Max: {max_mem:.1f}%)")

# Performance Summary
print("\n" + "="*70)
print("⚡ PERFORMANCE:")
print("="*70)
print(f"  Verarbeitungszeit: {data['metadata']['processing_time_seconds']:.1f}s")
print(f"  Realtime-Faktor: {data['metadata']['realtime_factor']:.1f}x")
print(f"  Erfolgreiche Analyzer: {data['metadata']['successful_analyzers']}/{data['metadata']['total_analyzers']}")
print(f"  Reconstruction Score: {data['metadata']['reconstruction_score']:.1f}%")

# Final Status
print("\n" + "="*70)
print("🎯 GESAMTSTATUS:")
print("="*70)
for name, status, detail in overall_status:
    print(f"  {'✅' if status else '❌'} {name}: {detail}")

all_success = all(status for _, status, _ in overall_status)
print(f"\n{'🎉 ALLE ANALYZER ERFOLGREICH!' if all_success else '⚠️  Einige Analyzer brauchen noch Arbeit'}")

# Check logs for errors
print("\n📋 LOG-ANALYSE:")
error_check = subprocess.run(['grep', '-c', 'ERROR', '/home/user/tiktok_production/logs/automatic_api.log'], 
                           capture_output=True, text=True)
print(f"  Fehler in Logs: {error_check.stdout.strip()}")

print(f"\nEnde: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")