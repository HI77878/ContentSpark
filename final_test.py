#!/usr/bin/env python3
"""Finaler Test aller Fixes"""

import json
import requests
import time

print("🎯 FINALER TEST ALLER ANALYZER-FIXES")
print("="*70)
print("Teste mit Leon Schliebach Video...")
print()

# Start analysis
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
import subprocess
import re
log_check = subprocess.run(['tail', '-100', '/home/user/tiktok_production/logs/automatic_api.log'], 
                          capture_output=True, text=True)
match = re.search(r'Results saved to (/home/user/tiktok_production/results/.+\.json)', log_check.stdout)
if not match:
    print("❌ Kein Result File gefunden")
    exit(1)

result_file = match.group(1)
with open(result_file, 'r') as f:
    data = json.load(f)

print(f"📊 Ergebnisse: {result_file}")
print()

# Check all three fixed analyzers
print("🔍 ANALYZER-STATUS:")
print("-"*70)

# 1. Age Estimation
age_data = data['analyzer_results'].get('age_estimation', {})
age_segments = age_data.get('segments', [])
frames_with_faces = sum(1 for s in age_segments if s.get('faces_detected', 0) > 0)
face_rate = frames_with_faces / len(age_segments) * 100 if age_segments else 0
total_faces = sum(s.get('faces_detected', 0) for s in age_segments)

print(f"\n🎭 AGE ESTIMATION:")
print(f"   Segments: {len(age_segments)}")
print(f"   Erkennungsrate: {face_rate:.1f}% ({frames_with_faces}/{len(age_segments)} frames)")
print(f"   Gesichter total: {total_faces}")
print(f"   ➡️  {'✅ ERFOLGREICH' if face_rate > 30 else '❌ PROBLEM'} (Ziel: >30%)")

# 2. Object Detection
obj_data = data['analyzer_results'].get('object_detection', {})
obj_segments = obj_data.get('segments', [])
frames_with_person = sum(1 for s in obj_segments if s.get('has_person', False))
person_rate = frames_with_person / len(obj_segments) * 100 if obj_segments else 0
total_objects = sum(s.get('objects_detected', 0) for s in obj_segments)

print(f"\n🎯 OBJECT DETECTION:")
print(f"   Segments: {len(obj_segments)}")
print(f"   Person-Erkennung: {person_rate:.1f}% ({frames_with_person}/{len(obj_segments)} frames)")
print(f"   Objekte total: {total_objects}")
print(f"   ➡️  {'✅ ERFOLGREICH' if person_rate > 50 else '❌ PROBLEM'} (Ziel: >50%)")

# 3. Speech Emotion
emotion_data = data['analyzer_results'].get('speech_emotion', {})
emotion_segments = emotion_data.get('segments', [])
emotions = set()
for seg in emotion_segments:
    if seg.get('dominant_emotion'):
        emotions.add(seg['dominant_emotion'])

print(f"\n😊 SPEECH EMOTION:")
print(f"   Segments: {len(emotion_segments)}")
print(f"   Emotionen: {', '.join(emotions) if emotions else 'KEINE'}")
print(f"   ➡️  {'✅ ERFOLGREICH' if emotions and 'none' not in str(emotions).lower() else '❌ PROBLEM'}")

# Overall Result
print("\n" + "="*70)
print("📈 GESAMTERGEBNIS:")
print("="*70)

results = [
    ("Age Estimation", face_rate > 30, f"{face_rate:.1f}%"),
    ("Object Detection", person_rate > 50, f"{person_rate:.1f}%"),
    ("Speech Emotion", bool(emotions) and 'none' not in str(emotions).lower(), f"{len(emotions)} emotions")
]

success_count = sum(1 for _, success, _ in results if success)

for name, success, detail in results:
    print(f"  {'✅' if success else '❌'} {name}: {detail}")

print(f"\nPerformance:")
print(f"  - Verarbeitungszeit: {data['metadata']['processing_time_seconds']:.1f}s")
print(f"  - Realtime-Faktor: {data['metadata']['realtime_factor']:.1f}x")
print(f"  - Erfolgreiche Analyzer: {data['metadata']['successful_analyzers']}/{data['metadata']['total_analyzers']}")

if success_count == 3:
    print("\n🎉 ALLE 3 ANALYZER ERFOLGREICH REPARIERT!")
else:
    print(f"\n⚠️  {success_count}/3 Analyzer erfolgreich repariert")