#!/usr/bin/env python3
import json
from collections import defaultdict

# Load results
with open('/home/user/tiktok_production/results/chaseridgewayy_7522589683939921165_multiprocess_20250707_160400.json') as f:
    data = json.load(f)

metadata = data.get('metadata', {})
analyzer_results = data.get('analyzer_results', {})

print("=" * 80)
print("CHASE RIDGEWAY TIKTOK - COMPLETE 1:1 VIDEO RECONSTRUCTION")
print("=" * 80)
print(f"Video: chaseridgewayy_7522589683939921165.mp4")
print(f"Duration: 68.5 seconds")
print(f"Resolution: 1078x1920 (Portrait)")
print(f"Reconstruction Score: {metadata.get('reconstruction_score', 0):.1f}%")
print(f"Analysis Time: {metadata.get('processing_time_seconds', 0):.1f}s")
print("=" * 80)

# Build timeline
timeline = defaultdict(dict)

# 1. StreamingDenseCaptioning (BASE)
sdc = analyzer_results.get('streaming_dense_captioning', {})
for seg in sdc.get('segments', []):
    t = int(seg['start_time'])
    timeline[t]['scene'] = seg['caption']

# 2. Speech Transcription
speech = analyzer_results.get('speech_transcription', {})
for seg in speech.get('segments', []):
    t = int(seg.get('start', seg.get('start_time', 0)))
    timeline[t]['speech'] = seg['text']

# 3. Text Overlays
text_overlay = analyzer_results.get('text_overlay', {})
for seg in text_overlay.get('segments', []):
    t = int(seg['timestamp'])
    texts = []
    for det in seg.get('text_detections', []):
        if det.get('text'):
            texts.append(det['text'])
    if texts:
        timeline[t]['text'] = ', '.join(texts[:3])  # Top 3 texts

# 4. Objects
objects = analyzer_results.get('object_detection', {})
for seg in objects.get('segments', []):
    t = int(seg['timestamp'])
    obj_list = []
    for det in seg.get('detections', [])[:5]:  # Top 5 objects
        obj_class = det.get('object_class', det.get('class', det.get('label', 'unknown')))
        if obj_class and obj_class != 'unknown':
            obj_list.append(obj_class)
    if obj_list:
        timeline[t]['objects'] = list(set(obj_list))

# 5. Camera Movement
camera = analyzer_results.get('camera_analysis', {})
for seg in camera.get('segments', []):
    t = int(seg['timestamp'])
    movement = seg.get('movement', {})
    if isinstance(movement, dict):
        move_type = movement.get('type', '')
        if move_type and move_type != 'static':
            timeline[t]['camera'] = move_type
    elif isinstance(movement, str) and movement != 'static':
        timeline[t]['camera'] = movement

# 6. Visual Effects
effects = analyzer_results.get('visual_effects', {})
for seg in effects.get('segments', []):
    t = int(seg['timestamp'])
    fx = []
    for effect in seg.get('effects', []):
        if effect.get('confidence', 0) > 0.7:
            fx.append(effect.get('type', ''))
    if fx:
        timeline[t]['effects'] = fx

# Generate reconstruction
print("\nSCENE-BY-SCENE RECONSTRUCTION:")
print("-" * 80)

for second in sorted(timeline.keys()):
    if second > 68:  # Don't go past video duration
        break
        
    data = timeline[second]
    output = f"\n[{second:02d}s] "
    
    # Scene description
    if 'scene' in data:
        output += f"{data['scene']} "
    
    # Speech
    if 'speech' in data:
        output += f'| Speech: "{data["speech"]}" '
    
    # Text overlays
    if 'text' in data:
        output += f'| Text: "{data["text"]}" '
    
    # Objects
    if 'objects' in data:
        output += f"| Visible: {', '.join(data['objects'][:3])} "
    
    # Camera
    if 'camera' in data:
        output += f"| Camera: {data['camera']} "
    
    # Effects
    if 'effects' in data:
        output += f"| Effects: {', '.join(data['effects'])} "
    
    print(output)

# Summary stats
print("\n" + "=" * 80)
print("RECONSTRUCTION SUMMARY:")
print(f"- Total timeline entries: {len(timeline)}")
print(f"- Temporal coverage: {len(timeline) / 68.5 * 100:.1f}%")
print(f"- Speech segments: {len(speech.get('segments', []))}")
print(f"- Text overlays detected: {len(text_overlay.get('segments', []))}")
print(f"- Objects tracked: {len(objects.get('segments', []))}")
print(f"- Camera movements: {len([s for s in camera.get('segments', []) if s.get('movement', {}).get('type') != 'static'])}")
print("=" * 80)