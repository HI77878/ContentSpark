#!/usr/bin/env python3
"""Finaler Test mit GPU-Unterstützung für Age Estimation"""

import json
import requests
import time
import subprocess

print("🚀 FINALER TEST MIT GPU-UNTERSTÜTZUNG")
print("="*70)
print("Teste Age Estimation mit CUDA-aktiviertem InsightFace...")
print()

# Test analysis
start_time = time.time()
response = requests.post(
    "http://localhost:8003/analyze",
    json={"tiktok_url": "https://www.tiktok.com/@leon_schliebach/video/7446489995663117590"}
)
end_time = time.time()

print(f"⏱️  Analyse dauerte: {end_time - start_time:.1f}s")

if response.status_code != 200:
    print(f"❌ API Error: {response.status_code}")
    exit(1)

# Get result file
time.sleep(2)
log_check = subprocess.run(['tail', '-100', '/home/user/tiktok_production/logs/gpu_fixed_api.log'], 
                          capture_output=True, text=True)
import re
match = re.search(r'Results saved to (/home/user/tiktok_production/results/.+\.json)', log_check.stdout)
if not match:
    print("❌ Kein Result File gefunden")
    exit(1)

result_file = match.group(1)
with open(result_file, 'r') as f:
    data = json.load(f)

print(f"📊 Ergebnisse: {result_file}")
print()

# Check if InsightFace is using GPU
gpu_check = subprocess.run(['grep', '-B2', '-A2', 'AgeGenderInsightFace.*ONNX providers', 
                           '/home/user/tiktok_production/logs/gpu_fixed_api.log'], 
                          capture_output=True, text=True)
print("🖥️  GPU STATUS:")
print("-"*70)
if 'CUDAExecutionProvider' in gpu_check.stdout:
    print("✅ InsightFace läuft auf GPU!")
else:
    print("❌ InsightFace läuft noch auf CPU")
print()

# Check all three fixed analyzers
print("📊 ANALYZER ERGEBNISSE:")
print("-"*70)

# 1. Age Estimation
age_data = data['analyzer_results'].get('age_estimation', {})
age_segments = age_data.get('segments', [])
frames_with_faces = sum(1 for s in age_segments if s.get('faces_detected', 0) > 0)
face_rate = frames_with_faces / len(age_segments) * 100 if age_segments else 0
total_faces = sum(s.get('faces_detected', 0) for s in age_segments)

print(f"\n🎭 AGE ESTIMATION:")
print(f"   Segments analysiert: {len(age_segments)}")
print(f"   Gesichtserkennung: {face_rate:.1f}% ({frames_with_faces}/{len(age_segments)} frames)")
print(f"   Gesichter total: {total_faces}")
print(f"   ➡️  {'✅ ERFOLGREICH' if face_rate > 30 else '❌ NOCH PROBLEMATISCH'}")

# Show example detections
if age_segments:
    print("\n   Beispiel-Erkennungen:")
    count = 0
    for seg in age_segments:
        if seg.get('faces_detected', 0) > 0 and count < 3:
            print(f"      {seg['timestamp']:.1f}s: {seg['faces_detected']} Gesicht(er)")
            count += 1

# 2. Object Detection
obj_data = data['analyzer_results'].get('object_detection', {})
obj_segments = obj_data.get('segments', [])
frames_with_person = sum(1 for s in obj_segments if s.get('has_person', False))
person_rate = frames_with_person / len(obj_segments) * 100 if obj_segments else 0

print(f"\n🎯 OBJECT DETECTION:")
print(f"   Person-Erkennung: {person_rate:.1f}% ({frames_with_person}/{len(obj_segments)} frames)")
print(f"   ➡️  {'✅ ERFOLGREICH' if person_rate > 50 else '❌ PROBLEMATISCH'}")

# 3. Speech Emotion
emotion_data = data['analyzer_results'].get('speech_emotion', {})
emotion_segments = emotion_data.get('segments', [])
emotions = set()
for seg in emotion_segments:
    if seg.get('dominant_emotion'):
        emotions.add(seg['dominant_emotion'])

print(f"\n😊 SPEECH EMOTION:")
print(f"   Emotionen: {', '.join(emotions) if emotions else 'KEINE'}")
print(f"   ➡️  {'✅ ERFOLGREICH' if emotions else '❌ PROBLEMATISCH'}")

# Performance
print(f"\n⚡ PERFORMANCE:")
print(f"   Verarbeitungszeit: {data['metadata']['processing_time_seconds']:.1f}s")
print(f"   Realtime-Faktor: {data['metadata']['realtime_factor']:.1f}x")

# Final result
print("\n" + "="*70)
success_count = sum([
    face_rate > 30,
    person_rate > 50,
    bool(emotions)
])

if success_count == 3:
    print("🎉 ALLE 3 ANALYZER ERFOLGREICH MIT GPU-UNTERSTÜTZUNG!")
else:
    print(f"📊 {success_count}/3 Analyzer erfolgreich")
    if face_rate <= 30:
        print("\n⚠️  Age Estimation braucht weitere Optimierung")