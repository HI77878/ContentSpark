#!/usr/bin/env python3
"""Test mit anderem Video um zu prüfen ob es video-spezifisch ist"""

import json
import requests
import time

# Test mit einem anderen Video
test_urls = [
    # Original
    ("Leon Schliebach", "https://www.tiktok.com/@leon_schliebach/video/7446489995663117590"),
    # Ein anderes Video zum Vergleich - nehmen wir ein populäres
    ("Charli D'Amelio", "https://www.tiktok.com/@charlidamelio/video/7308296028948090155")
]

print("🔍 TEST MIT MEHREREN VIDEOS")
print("="*70)

for creator, url in test_urls:
    print(f"\n📹 Teste: {creator}")
    print(f"   URL: {url}")
    print()
    
    # Start analysis
    start_time = time.time()
    response = requests.post(
        "http://localhost:8003/analyze",
        json={"tiktok_url": url}
    )
    end_time = time.time()
    
    print(f"   ⏱️  Analyse dauerte: {end_time - start_time:.1f}s")
    
    if response.status_code != 200:
        print(f"   ❌ API Error: {response.status_code}")
        continue
    
    # Get result file
    import subprocess
    import re
    time.sleep(1)  # Wait for log write
    log_check = subprocess.run(['tail', '-200', '/home/user/tiktok_production/logs/automatic_api.log'], 
                              capture_output=True, text=True)
    match = re.search(rf'Results saved to (/home/user/tiktok_production/results/.*{url.split("/")[-1]}.*\.json)', log_check.stdout)
    if not match:
        print("   ❌ Kein Result File gefunden")
        continue
    
    result_file = match.group(1)
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    # Quick check of age estimation
    age_data = data['analyzer_results'].get('age_estimation', {})
    age_segments = age_data.get('segments', [])
    frames_with_faces = sum(1 for s in age_segments if s.get('faces_detected', 0) > 0)
    face_rate = frames_with_faces / len(age_segments) * 100 if age_segments else 0
    
    print(f"\n   📊 Age Estimation Ergebnis:")
    print(f"      - Segments: {len(age_segments)}")
    print(f"      - Gesichtserkennung: {face_rate:.1f}% ({frames_with_faces}/{len(age_segments)})")
    print(f"      - Status: {'✅ GUT' if face_rate > 30 else '❌ NIEDRIG'}")
    
    # Check if it's running on GPU
    log_extract = subprocess.run(['grep', '-A5', 'AgeGenderInsightFace.*Available ONNX', 
                                 '/home/user/tiktok_production/logs/automatic_api.log'], 
                                capture_output=True, text=True)
    if 'CUDA' in log_extract.stdout:
        print(f"      - Backend: GPU (CUDA)")
    else:
        print(f"      - Backend: CPU")

print("\n" + "="*70)
print("FAZIT:")
print("Wenn beide Videos niedrige Erkennungsraten haben -> Analyzer-Problem")
print("Wenn nur Leon's Video niedrig ist -> Video-spezifisches Problem")