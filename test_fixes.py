#!/usr/bin/env python3
"""Test script to verify all analyzer fixes work correctly"""

import json
import requests
import time

# Test with a new TikTok video
test_url = "https://www.tiktok.com/@leon_schliebach/video/7446489995663117590"

print("🧪 TESTING ALL FIXED ANALYZERS")
print("="*50)
print(f"TikTok URL: {test_url}")
print()

# Call the API
print("📡 Calling API...")
response = requests.post(
    "http://localhost:8003/analyze",
    json={"tiktok_url": test_url}
)

if response.status_code != 200:
    print(f"❌ API Error: {response.status_code}")
    print(response.text)
    exit(1)

result = response.json()

# Handle different response formats
if 'result_file' in result:
    result_file = result['result_file']
    print(f"✅ Analysis completed: {result_file}")
    # Load from file
    with open(result_file, 'r') as f:
        data = json.load(f)
elif 'results' in result:
    # Direct results in response
    data = result['results']
    print(f"✅ Analysis completed (direct response)")
else:
    # Try to use the response as the data directly
    data = result
    print(f"✅ Analysis completed")
    
print()

# Debug: print the response structure
print("Response keys:", list(data.keys())[:5], "...")
if 'message' in data:
    print("Message:", data['message'])
if 'status' in data:
    print("Status:", data['status'])
    
# Try to get actual results path
if 'message' in data and 'results' in data['message']:
    import re
    match = re.search(r'results/(.+\.json)', data['message'])
    if match:
        result_file = match.group(0)
        print(f"Found result file: {result_file}")
        with open(result_file, 'r') as f:
            data = json.load(f)

print("📊 ANALYZER RESULTS CHECK:")
print("-"*50)

# Check Age Estimation
age_data = data['analyzer_results'].get('age_estimation', {})
age_segments = age_data.get('segments', [])
faces_detected = sum(s.get('faces_detected', 0) for s in age_segments)
frames_with_faces = sum(1 for s in age_segments if s.get('faces_detected', 0) > 0)
face_detection_rate = frames_with_faces / len(age_segments) * 100 if age_segments else 0

print(f"\n🎭 Age Estimation:")
print(f"   - Total segments: {len(age_segments)}")
print(f"   - Frames with faces: {frames_with_faces}")
print(f"   - Face detection rate: {face_detection_rate:.1f}%")
print(f"   - Total faces: {faces_detected}")
if age_segments and age_segments[0].get('faces'):
    face = age_segments[0]['faces'][0]
    print(f"   - Example: {face.get('age')} years, {face.get('gender')}, confidence: {face.get('confidence', {}).get('detection', 0):.2f}")

# Check Object Detection
obj_data = data['analyzer_results'].get('object_detection', {})
obj_segments = obj_data.get('segments', [])
frames_with_person = sum(1 for s in obj_segments if s.get('has_person', False))
person_detection_rate = frames_with_person / len(obj_segments) * 100 if obj_segments else 0
total_objects = sum(s.get('objects_detected', 0) for s in obj_segments)

print(f"\n🎯 Object Detection:")
print(f"   - Total segments: {len(obj_segments)}")
print(f"   - Frames with person: {frames_with_person}")
print(f"   - Person detection rate: {person_detection_rate:.1f}%")
print(f"   - Total objects detected: {total_objects}")
print(f"   - Unique object types: {obj_data.get('unique_objects', 0)}")
if obj_segments and obj_segments[0].get('objects'):
    obj = obj_segments[0]['objects'][0]
    print(f"   - Example: {obj.get('object_class')}, confidence: {obj.get('confidence_score', 0):.2f}")

# Check Speech Emotion
emotion_data = data['analyzer_results'].get('speech_emotion', {})
emotion_segments = emotion_data.get('segments', [])
emotions_detected = set()
for seg in emotion_segments:
    if seg.get('dominant_emotion'):
        emotions_detected.add(seg['dominant_emotion'])

print(f"\n😊 Speech Emotion:")
print(f"   - Total segments: {len(emotion_segments)}")
print(f"   - Unique emotions: {len(emotions_detected)}")
print(f"   - Emotions found: {', '.join(emotions_detected)}")
if emotion_segments and emotion_segments[0].get('dominant_emotion'):
    seg = emotion_segments[0]
    print(f"   - Example: {seg.get('dominant_emotion')} ({seg.get('dominant_emotion_de')}), confidence: {seg.get('confidence', 0):.2f}")
    print(f"   - Valence: {seg.get('emotion_valence')}, Intensity: {seg.get('emotion_intensity')}")

# Overall success check
print("\n" + "="*50)
print("🎯 FIX VERIFICATION:")
print("-"*50)

age_fixed = face_detection_rate > 30  # Was 7%, should be much higher now
object_fixed = person_detection_rate > 50  # Was 21%, should be much higher now
emotion_fixed = len(emotions_detected) > 0 and 'none' not in str(emotions_detected).lower()

print(f"✅ Age Estimation Fixed: {age_fixed} (Detection rate: {face_detection_rate:.1f}%)")
print(f"✅ Object Detection Fixed: {object_fixed} (Person rate: {person_detection_rate:.1f}%)")
print(f"✅ Speech Emotion Fixed: {emotion_fixed} (Found {len(emotions_detected)} emotions)")

if age_fixed and object_fixed and emotion_fixed:
    print("\n🎉 ALL ANALYZERS SUCCESSFULLY FIXED!")
else:
    print("\n⚠️ Some analyzers still need work")

# Performance metrics
print(f"\n⏱️ Performance:")
print(f"   - Processing time: {data['metadata']['processing_time_seconds']:.1f}s")
print(f"   - Realtime factor: {data['metadata']['realtime_factor']:.1f}x")
print(f"   - Successful analyzers: {data['metadata']['successful_analyzers']}/{data['metadata']['total_analyzers']}")