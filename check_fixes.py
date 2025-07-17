#!/usr/bin/env python3
"""Check the latest analysis results to verify fixes"""

import json

result_file = "/home/user/tiktok_production/results/7446489995663117590_multiprocess_20250712_145448.json"

print("🔍 CHECKING ANALYZER FIXES")
print("="*60)

with open(result_file, 'r') as f:
    data = json.load(f)

print(f"✅ Loaded results from: {result_file}")
print(f"   Video: {data['metadata']['tiktok_metadata']['description']}")
print(f"   Processing time: {data['metadata']['processing_time_seconds']:.1f}s")
print(f"   Realtime factor: {data['metadata']['realtime_factor']:.1f}x")
print()

# Check Age Estimation
print("🎭 AGE ESTIMATION CHECK:")
print("-"*60)
age_data = data['analyzer_results'].get('age_estimation', {})
age_segments = age_data.get('segments', [])
faces_detected = sum(s.get('faces_detected', 0) for s in age_segments)
frames_with_faces = sum(1 for s in age_segments if s.get('faces_detected', 0) > 0)
face_detection_rate = frames_with_faces / len(age_segments) * 100 if age_segments else 0

print(f"Total segments analyzed: {len(age_segments)}")
print(f"Frames with faces detected: {frames_with_faces} ({face_detection_rate:.1f}%)")
print(f"Total faces detected: {faces_detected}")
print(f"Unique individuals: {age_data.get('metadata', {}).get('unique_identities', 0)}")

if age_segments and age_segments[0].get('faces'):
    face = age_segments[0]['faces'][0]
    print(f"\nFirst detection example:")
    print(f"  - Age: {face.get('age')} years")
    print(f"  - Gender: {face.get('gender')}")
    print(f"  - Age group: {face.get('age_group')}")
    print(f"  - Detection confidence: {face.get('confidence', {}).get('detection', 0):.3f}")

print(f"\n✅ FIX STATUS: {'SUCCESSFUL' if face_detection_rate > 30 else 'FAILED'} (was 7%, now {face_detection_rate:.1f}%)")

# Check Object Detection  
print("\n\n🎯 OBJECT DETECTION CHECK:")
print("-"*60)
obj_data = data['analyzer_results'].get('object_detection', {})
obj_segments = obj_data.get('segments', [])

if not obj_segments:
    print("❌ ERROR: No segments found in object detection results")
else:
    # Check new format with objects array
    frames_with_objects = sum(1 for s in obj_segments if s.get('objects_detected', 0) > 0)
    frames_with_person = sum(1 for s in obj_segments if s.get('has_person', False))
    person_detection_rate = frames_with_person / len(obj_segments) * 100 if obj_segments else 0
    total_objects = sum(s.get('objects_detected', 0) for s in obj_segments)
    
    print(f"Total segments analyzed: {len(obj_segments)}")
    print(f"Frames with objects: {frames_with_objects}")
    print(f"Frames with person detected: {frames_with_person} ({person_detection_rate:.1f}%)")
    print(f"Total objects detected: {total_objects}")
    print(f"Unique object types: {obj_data.get('unique_objects', 0)}")
    
    if obj_segments and obj_segments[0].get('objects'):
        obj = obj_segments[0]['objects'][0]
        print(f"\nFirst detection example:")
        print(f"  - Object: {obj.get('object_class')}")
        print(f"  - Confidence: {obj.get('confidence_score', 0):.3f}")
        print(f"  - Category: {obj.get('object_category')}")
        print(f"  - Position: {obj.get('position')}")
    
    print(f"\n✅ FIX STATUS: {'SUCCESSFUL' if person_detection_rate > 50 else 'FAILED'} (was 21%, now {person_detection_rate:.1f}%)")

# Check Speech Emotion
print("\n\n😊 SPEECH EMOTION CHECK:")
print("-"*60)
emotion_data = data['analyzer_results'].get('speech_emotion', {})
emotion_segments = emotion_data.get('segments', [])
emotions_detected = set()
emotion_examples = []

for seg in emotion_segments:
    if seg.get('dominant_emotion'):
        emotions_detected.add(seg['dominant_emotion'])
        if len(emotion_examples) < 3:
            emotion_examples.append(seg)

print(f"Total segments analyzed: {len(emotion_segments)}")
print(f"Unique emotions detected: {len(emotions_detected)}")
print(f"Emotions found: {', '.join(sorted(emotions_detected))}")

if emotion_data.get('summary'):
    summary = emotion_data['summary']
    print(f"\nOverall analysis:")
    print(f"  - Overall tone: {summary.get('overall_tone', 'unknown')}")
    print(f"  - Emotional stability: {summary.get('emotional_stability', 'unknown')}")
    print(f"  - Primary emotion: {summary.get('overall_dominant_emotion', 'unknown')}")

if emotion_examples:
    print(f"\nExample emotions detected:")
    for seg in emotion_examples[:3]:
        print(f"  - {seg['timestamp']:.1f}s: {seg['dominant_emotion']} ({seg.get('dominant_emotion_de', '')}) - confidence: {seg['confidence']:.2f}")

emotion_fixed = len(emotions_detected) > 0 and 'none' not in str(emotions_detected).lower()
print(f"\n✅ FIX STATUS: {'SUCCESSFUL' if emotion_fixed else 'FAILED'} (was returning 'none', now found {len(emotions_detected)} emotions)")

# Overall summary
print("\n\n" + "="*60)
print("🎉 OVERALL FIX VERIFICATION:")
print("="*60)

age_fixed = face_detection_rate > 30
object_fixed = person_detection_rate > 50 if obj_segments else False
emotion_fixed = len(emotions_detected) > 0 and 'none' not in str(emotions_detected).lower()

fixes = [
    ("Age Estimation", age_fixed, f"{face_detection_rate:.1f}% face detection (was 7%)"),
    ("Object Detection", object_fixed, f"{person_detection_rate:.1f}% person detection (was 21%)" if obj_segments else "ERROR: No segments"),
    ("Speech Emotion", emotion_fixed, f"{len(emotions_detected)} emotions found (was 'none')")
]

for name, status, detail in fixes:
    print(f"{'✅' if status else '❌'} {name}: {'FIXED' if status else 'NEEDS WORK'} - {detail}")

all_fixed = all(status for _, status, _ in fixes)
print(f"\n{'🎉 ALL ANALYZERS SUCCESSFULLY FIXED!' if all_fixed else '⚠️  Some analyzers still need work'}")

print(f"\n📊 Performance metrics:")
print(f"   - Processing time: {data['metadata']['processing_time_seconds']:.1f}s")
print(f"   - Realtime factor: {data['metadata']['realtime_factor']:.1f}x (target: <3x)")
print(f"   - Successful analyzers: {data['metadata']['successful_analyzers']}/{data['metadata']['total_analyzers']}")
print(f"   - Reconstruction score: {data['metadata']['reconstruction_score']:.1f}%")