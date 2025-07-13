#!/usr/bin/env python3
"""Debug why data is not showing in report"""

import json

# Load latest results
with open("/home/user/tiktok_production/results/7446489995663117590_multiprocess_20250712_153148.json", 'r') as f:
    data = json.load(f)

results = data['analyzer_results']

# Check specific analyzers that showed as missing
print("🔍 DEBUGGING DATA EXTRACTION:")
print("="*60)

# 1. Body Pose
print("\n1. BODY POSE:")
body_pose = results.get('body_pose', {})
if body_pose and 'segments' in body_pose:
    print(f"   Segments: {len(body_pose['segments'])}")
    sample = body_pose['segments'][0]
    print(f"   Sample segment keys: {list(sample.keys())}")
    print(f"   Sample data: {sample}")

# 2. Face Emotion
print("\n2. FACE EMOTION:")
face_emotion = results.get('face_emotion', {})
if face_emotion and 'segments' in face_emotion:
    print(f"   Segments: {len(face_emotion['segments'])}")
    sample = face_emotion['segments'][0]
    print(f"   Sample segment keys: {list(sample.keys())}")
    if 'faces' in sample:
        print(f"   Faces in first segment: {len(sample['faces'])}")

# 3. Background Segmentation
print("\n3. BACKGROUND SEGMENTATION:")
bg = results.get('background_segmentation', {})
if bg and 'segments' in bg:
    print(f"   Segments: {len(bg['segments'])}")
    sample = bg['segments'][0]
    print(f"   Sample segment keys: {list(sample.keys())}")
    
# 4. Sound Effects
print("\n4. SOUND EFFECTS:")
sound = results.get('sound_effects', {})
print(f"   Data type: {type(sound)}")
print(f"   Keys: {list(sound.keys()) if isinstance(sound, dict) else 'Not a dict'}")

# 5. Color Analysis
print("\n5. COLOR ANALYSIS:")
color = results.get('color_analysis', {})
if color:
    print(f"   Keys: {list(color.keys())}")
    if 'dominant_colors' in color:
        print(f"   Dominant colors: {color['dominant_colors'][:3]}")

# 6. Speech Flow
print("\n6. SPEECH FLOW:")
flow = results.get('speech_flow', {})
if flow and 'segments' in flow:
    sample = flow['segments'][0]
    print(f"   First segment: {sample}")
    
# 7. Age Estimation
print("\n7. AGE ESTIMATION:")
age = results.get('age_estimation', {})
if age and 'segments' in age:
    faces_detected = sum(s.get('faces_detected', 0) for s in age['segments'])
    frames_with_faces = sum(1 for s in age['segments'] if s.get('faces_detected', 0) > 0)
    print(f"   Total faces: {faces_detected}")
    print(f"   Frames with faces: {frames_with_faces}/{len(age['segments'])}")
    # Find a segment with faces
    for seg in age['segments']:
        if seg.get('faces_detected', 0) > 0:
            print(f"   Example segment with face: {seg}")
            break