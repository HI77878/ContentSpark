#!/usr/bin/env python3
"""Test the fixed system with real video"""
import requests
import json
import time
from pathlib import Path
from datetime import datetime

print("=== TEST DES REPARIERTEN SYSTEMS ===")
print(f"Start: {datetime.now()}")

# Use test video
video_path = "/home/user/tiktok_production/test_video.mp4"

# 1. Check API health
print("\n1. API Health Check:")
health = requests.get("http://localhost:8003/health").json()
print(f"✅ Status: {health['status']}")
print(f"✅ Active Analyzers: {health['active_analyzers']}")
print(f"✅ GPU: {health['gpu']['gpu_name']}")
print(f"✅ Parallelization: {health['parallelization']}")

# 2. Analyze video
print(f"\n2. Analysiere Video: {video_path}")
start_time = time.time()

response = requests.post(
    "http://localhost:8003/analyze",
    json={"video_path": video_path}
)

analysis_time = time.time() - start_time

if response.status_code == 200:
    result = response.json()
    print(f"✅ Analyse abgeschlossen in {analysis_time:.1f}s")
    print(f"   Erfolgreiche Analyzer: {result['successful_analyzers']}")
    print(f"   Gesamt Analyzer: {result['total_analyzers']}")
    print(f"   Ergebnis-Datei: {result['results_file']}")
    
    # Load and check results
    results_file = Path(result['results_file'])
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        
        print("\n3. ANALYZER STATUS:")
        
        # Check critical analyzers
        critical = {
            'qwen2_vl_temporal': 'Qwen2-VL Temporal',
            'object_detection': 'Object Detection',
            'face_emotion': 'Face Emotion',
            'speech_transcription': 'Speech Transcription',
            'body_pose': 'Body Pose'
        }
        
        for analyzer_key, name in critical.items():
            if analyzer_key in data['analyzer_results']:
                result = data['analyzer_results'][analyzer_key]
                if 'error' in result:
                    print(f"❌ {name}: ERROR - {result['error'][:50]}")
                elif 'segments' in result and result['segments']:
                    print(f"✅ {name}: {len(result['segments'])} Segmente")
                else:
                    print(f"⚠️ {name}: Keine Daten")
            else:
                print(f"❌ {name}: NICHT GEFUNDEN")
        
        # Overall stats
        total = len(data['analyzer_results'])
        successful = sum(1 for r in data['analyzer_results'].values() 
                        if 'error' not in r and 'segments' in r and r['segments'])
        
        print(f"\n4. ZUSAMMENFASSUNG:")
        print(f"Erfolgreiche Analyzer: {successful}/{total} ({successful/total*100:.0f}%)")
        print(f"Realtime Factor: {data['metadata'].get('realtime_factor', 'N/A')}x")
        print(f"Reconstruction Score: {data['metadata'].get('reconstruction_score', 'N/A')}%")
        
        # Save test results
        test_report = {
            'test_time': datetime.now().isoformat(),
            'analysis_time': analysis_time,
            'successful_analyzers': successful,
            'total_analyzers': total,
            'success_rate': successful/total*100,
            'critical_analyzers_working': {
                name: analyzer_key in data['analyzer_results'] and 
                      'error' not in data['analyzer_results'][analyzer_key]
                for analyzer_key, name in critical.items()
            }
        }
        
        with open('fixed_system_test_report.json', 'w') as f:
            json.dump(test_report, f, indent=2)
        
        print(f"\n✅ Test-Report gespeichert: fixed_system_test_report.json")
        
else:
    print(f"❌ Analyse fehlgeschlagen: {response.status_code}")
    print(response.text)