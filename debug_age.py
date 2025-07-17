#!/usr/bin/env python3
"""Debug Age Estimation on single frame"""

import cv2
from analyzers.age_gender_insightface import AgeGenderInsightFace

# Test video
video_path = "/home/user/tiktok_videos/videos/7446489995663117590.mp4"

print("🔍 DEBUGGING AGE ESTIMATION")
print("="*50)

# Initialize analyzer
analyzer = AgeGenderInsightFace()
analyzer._load_model_impl()

# Extract a few frames manually
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {fps:.1f} FPS, {total_frames} frames total")
print()

# Test on specific frames
test_frames = [0, 30, 60, 90, 120, 150, 300, 500, 700, 900]

for frame_idx in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if not ret:
        continue
        
    timestamp = frame_idx / fps
    print(f"\nFrame {frame_idx} ({timestamp:.1f}s):")
    
    # Run face detection
    faces = analyzer.app.get(frame)
    
    if faces:
        print(f"  ✅ {len(faces)} Gesicht(er) gefunden!")
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            age = int(face.age) if hasattr(face, 'age') else 'N/A'
            gender = analyzer.gender_map.get(face.gender, 'unknown') if hasattr(face, 'gender') else 'unknown'
            conf = float(face.det_score) if hasattr(face, 'det_score') else 0.0
            
            print(f"    Face {i+1}: Alter={age}, Geschlecht={gender}, Konfidenz={conf:.3f}")
            print(f"             Position: x={bbox[0]}, y={bbox[1]}, w={bbox[2]-bbox[0]}, h={bbox[3]-bbox[1]}")
    else:
        print(f"  ❌ Keine Gesichter gefunden")
        # Save frame for inspection
        debug_path = f"/tmp/debug_frame_{frame_idx}.jpg"
        cv2.imwrite(debug_path, frame)
        print(f"     Frame gespeichert: {debug_path}")

cap.release()

# Test with different thresholds
print("\n" + "="*50)
print("TESTING DIFFERENT THRESHOLDS:")
print("="*50)

# Get frame at 10s
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(10 * fps))
ret, test_frame = cap.read()
cap.release()

if ret:
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print(f"\nThreshold {threshold}:")
        analyzer.app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=threshold)
        faces = analyzer.app.get(test_frame)
        print(f"  Gefunden: {len(faces)} Gesicht(er)")
        
# Check if it's a video content issue
print("\n" + "="*50)
print("MÖGLICHE GRÜNDE FÜR NIEDRIGE ERKENNUNG:")
print("="*50)
print("1. Video zeigt keine frontalen Gesichter")
print("2. Gesichter sind zu klein im Frame")
print("3. Bewegungsunschärfe")
print("4. Schlechte Beleuchtung")
print("5. Gesichter sind verdeckt/im Profil")

# Try MTCNN as alternative
print("\n" + "="*50)
print("ALTERNATIVE: MTCNN TEST")
print("="*50)

try:
    from mtcnn import MTCNN
    import tensorflow as tf
    
    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')
    
    detector = MTCNN()
    
    # Test on same frames
    cap = cv2.VideoCapture(video_path)
    for frame_idx in [0, 150, 300, 450]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb_frame)
            print(f"Frame {frame_idx}: MTCNN fand {len(faces)} Gesicht(er)")
    cap.release()
except Exception as e:
    print(f"MTCNN Test fehlgeschlagen: {e}")