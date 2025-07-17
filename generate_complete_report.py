#!/usr/bin/env python3
"""Generate COMPLETE analysis report with ALL available data"""

import json
from datetime import datetime

# Load latest results directly
with open("/home/user/tiktok_production/results/7446489995663117590_multiprocess_20250712_153148.json", 'r') as f:
    data = json.load(f)

results = data['analyzer_results']
metadata = data['metadata']

def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

print("📊 TikTok Video Analysis:")
print("="*80)

# Analysis Summary
print("\n📋 **Analysis Summary**")
qwen = results.get('qwen2_vl_temporal', {})
if qwen and 'segments' in qwen:
    descriptions = [s['description'] for s in qwen['segments'] if 'description' in s]
    if descriptions:
        summary = " ".join(descriptions[:5])
        print(summary)
    else:
        print("[DATEN FEHLT - Keine Videobeschreibungen]")
else:
    print("[DATEN FEHLT - Keine Videobeschreibungen]")

# Hook Analysis
print("\n🪝 **Hook Analysis (0-3 sec)**")
print("Initial Elements:")

# Objects in first 3 seconds
objects = results.get('object_detection', {})
if objects and 'segments' in objects:
    for seg in objects['segments'][:3]:
        if seg['timestamp'] <= 3:
            for obj in seg.get('objects', [])[:3]:
                print(f"  {obj['object_class']}: Position {obj.get('position', 'unbekannt')}, Konfidenz {obj['confidence_score']:.0%}")

# Text in first 3 seconds
texts = results.get('text_overlay', {})
if texts and 'segments' in texts:
    for seg in texts['segments']:
        if seg.get('start_time', 0) <= 3 and seg.get('text'):
            print(f"  Text: \"{seg['text']}\" (Position: {seg.get('position', 'unbekannt')})")
            if len([s for s in texts['segments'] if s.get('start_time', 0) <= 3]) >= 3:
                break

# Opening Statement
speech = results.get('speech_transcription', {})
if speech and 'segments' in speech:
    first_speech = next((s['text'] for s in speech['segments'] if s.get('text')), None)
    print(f"Opening Statement: \"{first_speech}\"" if first_speech else "Opening Statement: [Keine Sprache in ersten 3 Sekunden]")
else:
    print("Opening Statement: [Keine Sprache erkannt]")

# First Frame from scene segmentation
scene = results.get('scene_segmentation', {})
if scene and 'segments' in scene and scene['segments']:
    first_scene = scene['segments'][0]
    print(f"First Frame Impact: {first_scene.get('description', '[Keine Beschreibung]')}")
else:
    print("First Frame Impact: [DATEN FEHLT]")

# Audio Analysis
print("\n🎵 **Audio Analysis**")
print("\n   **Speech Content**")

# Language - check actual transcription
language = "Unbekannt"
if speech and 'segments' in speech:
    # Simple German detection
    sample_text = " ".join([s.get('text', '') for s in speech['segments'][:5] if s.get('text')])
    if any(word in sample_text.lower() for word in ['ich', 'und', 'der', 'die', 'das', 'ist', 'ein', 'was', 'mit']):
        language = "Deutsch"
    elif sample_text:
        language = "Englisch"  # Default if text exists
print(f"   Language: {language}")

# Type of speech
print(f"   Type: {'Direkte Ansprache (Vlog-Stil)' if speech and speech.get('segments') else '[Keine Sprache]'}")

# Voice analysis
audio = results.get('audio_analysis', {})
if audio and 'voice_analysis' in audio:
    voice = audio['voice_analysis']
    pitch_stats = voice.get('pitch_statistics', {})
    if pitch_stats.get('mean'):
        print(f"   Voice Pitch (F0): {pitch_stats['mean']:.0f} Hz ± {pitch_stats.get('std', 0):.0f} Hz ({pitch_stats.get('min', 0):.0f}-{pitch_stats.get('max', 0):.0f} Hz)")
    else:
        print("   Voice Pitch (F0): [Keine Stimmanalyse]")
    
    # Gender estimation from pitch
    if pitch_stats.get('mean', 0) > 180:
        print("   Speakers: 1 (wahrscheinlich weiblich basierend auf Tonhöhe)")
    elif pitch_stats.get('mean', 0) > 0:
        print("   Speakers: 1 (wahrscheinlich männlich basierend auf Tonhöhe)")
    else:
        print("   Speakers: [Nicht bestimmbar]")
else:
    print("   Voice Pitch (F0): [Keine Audioanalyse]")
    print("   Speakers: [Nicht bestimmbar]")

# Speaking speed from speech flow
flow = results.get('speech_flow', {})
if flow and 'segments' in flow:
    wpms = [s.get('wpm', 0) for s in flow['segments'] if s.get('wpm', 0) > 0]
    if wpms:
        print(f"   Speaking Speed: {sum(wpms)/len(wpms):.0f} WPM (Range: {min(wpms):.0f}-{max(wpms):.0f} WPM)")
    else:
        print("   Speaking Speed: [Nicht messbar]")
else:
    print("   Speaking Speed: [Keine Daten]")

# Emotional tone
emotion = results.get('speech_emotion', {})
if emotion and 'summary' in emotion:
    print(f"   Emotional Tone (aus Sprache): {emotion['summary'].get('overall_tone', 'Unbekannt')}")
elif emotion and 'segments' in emotion:
    emotions = [s.get('dominant_emotion', 'none') for s in emotion['segments'] if s.get('dominant_emotion')]
    if emotions:
        from collections import Counter
        most_common = Counter(emotions).most_common(1)[0][0]
        print(f"   Emotional Tone (aus Sprache): Überwiegend {most_common}")
    else:
        print("   Emotional Tone (aus Sprache): [Keine Emotionen erkannt]")
else:
    print("   Emotional Tone (aus Sprache): [Keine Analyse]")

# Complete Transcript
print("\n   **Complete Transcript with Timestamps**")
if speech and 'segments' in speech:
    for seg in speech['segments']:
        if seg.get('text'):
            start = format_timestamp(seg.get('start', 0))
            end = format_timestamp(seg.get('end', seg.get('start', 0) + 3))
            print(f"   [{start}-{end}]: \"{seg['text']}\"")
else:
    print("   [Keine Transkription verfügbar]")

# Sound Effects - check different structure
print("\n   **Sound Effects**")
sound_data = results.get('sound_effects', {})
if isinstance(sound_data, dict) and 'segments' in sound_data:
    effects_found = False
    for seg in sound_data['segments']:
        if seg.get('sound_effects'):
            for effect in seg['sound_effects']:
                print(f"   [{format_timestamp(seg['timestamp'])}]: {effect['type']} – Konfidenz: {effect.get('confidence', 0):.0%}")
                effects_found = True
    if not effects_found:
        print("   [Keine Soundeffekte erkannt]")
else:
    print("   [Keine Soundeffekte erkannt]")

# Speech Flow Analysis
print("\n🗣️ **Speech Flow Analysis**")
if flow and 'segments' in flow:
    print("   Emphasized Words:")
    emphasized_count = 0
    for seg in flow['segments']:
        if seg.get('emphasized_words'):
            for word in seg['emphasized_words']:
                print(f"   [{format_timestamp(seg['timestamp'])}]: \"{word['word']}\" ({word['type']})")
                emphasized_count += 1
                if emphasized_count >= 10:
                    break
        if emphasized_count >= 10:
            break
    
    print("\n   Significant Pauses:")
    pause_count = 0
    for seg in flow['segments']:
        if seg.get('pauses'):
            for pause in seg['pauses']:
                print(f"   [{format_timestamp(pause['timestamp'])}]: {pause['duration']:.1f}s")
                pause_count += 1
                if pause_count >= 5:
                    break
    if pause_count == 0:
        print("   [Keine signifikanten Pausen erkannt]")
    
    # Rhythm pattern
    if flow['segments']:
        rhythms = [s.get('rhythm', {}).get('pattern', '') for s in flow['segments'] if s.get('rhythm')]
        if rhythms:
            from collections import Counter
            common_rhythm = Counter(rhythms).most_common(1)[0][0]
            print(f"\n   Speech Rhythm Pattern: Überwiegend {common_rhythm}")
        else:
            print("\n   Speech Rhythm Pattern: [Nicht analysiert]")

# Cut Analysis
print("\n✂️ **Cut Analysis & Dynamics**")
cuts = results.get('cut_analysis', {})
if cuts:
    print(f"   Total Cuts: {cuts.get('total_cuts', 'Unbekannt')}")
    print(f"   Cuts per Minute: {cuts.get('cuts_per_minute', 'Unbekannt'):.1f}")
    print(f"   Average Shot Length: {cuts.get('average_shot_duration', 'Unbekannt'):.1f}s")
    
    # Shot changes
    if 'segments' in cuts:
        print("\n   Shot Changes:")
        for i, seg in enumerate(cuts['segments'][:10]):
            if seg.get('change_detected'):
                print(f"   [{format_timestamp(seg['timestamp'])}]: Schnitt erkannt (Änderung: {seg.get('change_magnitude', 0):.0%})")

# Camera movements
print("\n   Camera Movements:")
camera = results.get('camera_analysis', {})
if camera and 'segments' in camera:
    for seg in camera['segments'][:10]:
        if seg.get('camera_movement') and seg['camera_movement'] != 'static':
            print(f"   [{format_timestamp(seg['timestamp'])}]: {seg['camera_movement']} (Geschwindigkeit: {seg.get('movement_speed', 0):.1f})")

# Gesture & Body Language
print("\n👐 **Gesture & Body Language Analysis**")
body = results.get('body_pose', {})
if body and 'segments' in body:
    # Dominant posture analysis
    postures = []
    for seg in body['segments']:
        if seg.get('poses'):
            for pose in seg['poses']:
                if 'body_language' in pose and pose['body_language'].get('dominant'):
                    postures.append(pose['body_language']['dominant'])
    
    if postures:
        from collections import Counter
        dominant = Counter(postures).most_common(1)[0][0]
        print(f"   Dominant Posture: {dominant.capitalize()} (in {Counter(postures)[dominant]/len(postures)*100:.0f}% der Frames)")
    
    print("\n   Key Gestures:")
    gesture_count = 0
    for seg in body['segments']:
        if seg.get('poses'):
            for pose in seg['poses']:
                if pose.get('gestures'):
                    for gesture in pose['gestures']:
                        print(f"   [{format_timestamp(seg['timestamp'])}]: {gesture['description']} (Konfidenz: {gesture['confidence']:.0%})")
                        gesture_count += 1
                        if gesture_count >= 10:
                            break
    
    # Body language summary
    if body['segments']:
        body_langs = []
        for seg in body['segments']:
            if seg.get('poses'):
                for pose in seg['poses']:
                    if 'body_language' in pose:
                        body_langs.append(pose['body_language'].get('description', ''))
        if body_langs:
            print(f"\n   Non-verbal Communication Summary: {body_langs[0]}")

# Facial Analysis
print("\n😊 **Facial Analysis Over Time**")
age = results.get('age_estimation', {})
face_detected = False
if age and 'segments' in age:
    faces_total = sum(s.get('faces_detected', 0) for s in age['segments'])
    if faces_total > 0:
        face_detected = True
        print(f"   Gesichter erkannt in {sum(1 for s in age['segments'] if s.get('faces_detected', 0) > 0)} von {len(age['segments'])} Frames")
        
        # Age information
        print("\n   Dominant Age Group:")
        for seg in age['segments']:
            if seg.get('faces_detected', 0) > 0:
                print(f"   [{format_timestamp(seg['timestamp'])}]: {seg.get('age_group', 'Unbekannt')} ({seg.get('estimated_age', 'N/A')} Jahre)")
                break

# Face emotion only if faces detected
if face_detected:
    face_emotion = results.get('face_emotion', {})
    if face_emotion and 'segments' in face_emotion:
        print("\n   Emotional Changes (visuell):")
        for seg in face_emotion['segments'][:5]:
            if seg.get('dominant_emotion'):
                print(f"   [{format_timestamp(seg['timestamp'])}]: {seg['dominant_emotion']}")
else:
    print("   [Keine Gesichter für Emotionsanalyse erkannt]")

# Eye tracking
eye = results.get('eye_tracking', {})
if eye and 'segments' in eye:
    print("\n   Eye Tracking (Blickrichtung):")
    for seg in eye['segments'][:10]:
        if seg.get('gaze_data'):
            for gaze in seg['gaze_data']:
                direction = gaze.get('gaze_direction', 'Unbekannt')
                print(f"   [{format_timestamp(seg['timestamp'])}]: {direction}")

# Background Analysis
print("\n🏠 **Background Analysis & Context**")
bg = results.get('background_segmentation', {})
if bg and 'segments' in bg:
    # Collect all detected objects
    all_objects = set()
    for seg in bg['segments']:
        if seg.get('detected_objects'):
            all_objects.update(seg['detected_objects'])
    
    if all_objects:
        print("   Background Objects:")
        for obj in list(all_objects)[:10]:
            print(f"   - {obj}")
    
    # Environment analysis
    environments = [seg.get('environment', '') for seg in bg['segments'] if seg.get('environment')]
    if environments:
        from collections import Counter
        common_env = Counter(environments).most_common(1)[0][0]
        print(f"\n   Environmental Context: Überwiegend {common_env}")

# Visual Analysis
print("\n👁️ **Visual Analysis (Overall)**")
print("\n   **Environment**")

# Setting from scene descriptions
if scene and 'segments' in scene:
    settings = [s.get('setting', '') for s in scene['segments'] if s.get('setting')]
    if settings:
        print(f"   Setting: {settings[0]}")
    else:
        print("   Setting: [Nicht identifiziert]")

# Color analysis
color = results.get('color_analysis', {})
if color:
    if 'summary' in color and 'dominant_palette' in color['summary']:
        colors = color['summary']['dominant_palette']
        print(f"   Color Palette: {', '.join(colors[:5])}")
    elif 'segments' in color:
        # Aggregate colors from segments
        all_colors = []
        for seg in color['segments']:
            if seg.get('dominant_colors'):
                all_colors.extend([c['name'] for c in seg['dominant_colors'][:2]])
        if all_colors:
            from collections import Counter
            common_colors = Counter(all_colors).most_common(5)
            print(f"   Color Palette: {', '.join([c[0] for c in common_colors])}")

# Content quality for visual style
quality = results.get('content_quality', {})
if quality and 'summary' in quality:
    print(f"   Visual Style: {quality['summary'].get('overall_quality', 'Unbekannt')} Qualität")

# Person Detection
print("\n   **Person Detection**")
if objects and 'segments' in objects:
    person_segments = [s for s in objects['segments'] if s.get('has_person', False)]
    if person_segments:
        print(f"   Main Subject(s): Person in {len(person_segments)} von {len(objects['segments'])} Frames erkannt")
        
        # Age from age estimation
        if age and 'segments' in age:
            ages = [s.get('estimated_age') for s in age['segments'] if s.get('estimated_age')]
            if ages:
                print(f"   Age Range (geschätzt, visuell): {min(ages)}-{max(ages)} Jahre")
        
        # Appearance from age data
        for seg in age.get('segments', []):
            if seg.get('appearance'):
                print(f"   Physical Appearance: {seg['appearance']}")
                break

# Text Overlays
print("\n   **On-Screen Text Overlays**")
if texts and 'segments' in texts:
    print(f"   Gesamt: {len(texts['segments'])} Text-Overlays erkannt")
    for i, seg in enumerate(texts['segments'][:20]):
        if seg.get('text'):
            start = format_timestamp(seg.get('start_time', 0))
            end = format_timestamp(seg.get('end_time', seg.get('start_time', 0) + 2))
            text = seg['text']
            position = seg.get('position', seg.get('location', 'Mitte'))
            print(f"   [{start}-{end}]: \"{text}\" – Position: {position}")

# Objects summary
print("\n   **Objects Detected**")
if objects and 'object_counts' in objects:
    sorted_objects = sorted(objects['object_counts'].items(), key=lambda x: x[1], reverse=True)
    for obj_name, count in sorted_objects[:10]:
        # Find confidence from segments
        conf = 0
        for seg in objects['segments']:
            for obj in seg.get('objects', []):
                if obj['object_class'] == obj_name:
                    conf = obj['confidence_score']
                    break
            if conf > 0:
                break
        print(f"   {obj_name}: {count}x erkannt (Konfidenz: {conf:.0%})")

# Scene-by-Scene
print("\n🎬 **Scene-by-Scene Analysis**")
duration = metadata.get('duration', 48)
for start in range(0, int(duration), 5):
    end = min(start + 5, int(duration))
    print(f"\n   Segment {format_timestamp(start)}–{format_timestamp(end)}")
    
    # Actions from Qwen
    if qwen and 'segments' in qwen:
        for seg in qwen['segments']:
            if start <= seg.get('start_time', 0) < end:
                print(f"   Action: {seg.get('description', 'Keine Beschreibung')[:100]}...")
                break
    
    # Key visuals
    visuals = []
    if objects and 'segments' in objects:
        for seg in objects['segments']:
            if start <= seg.get('timestamp', 0) < end:
                for obj in seg.get('objects', [])[:3]:
                    visuals.append(obj['object_class'])
                if visuals:
                    break
    if visuals:
        print(f"   Key Visuals: {', '.join(visuals)}")
    
    # Emotion
    if emotion and 'segments' in emotion:
        for seg in emotion['segments']:
            if start <= seg.get('timestamp', 0) < end:
                print(f"   Emotion: {seg.get('dominant_emotion', 'Neutral')}")
                break

# Interaction Signals
print("\n🔄 **Interaction Signals**")
cta = results.get('comment_cta_detection', {})
if cta and 'cta_elements' in cta and cta['cta_elements']:
    print("   Call-to-Action Elements:")
    for element in cta['cta_elements']:
        print(f"   [{format_timestamp(element.get('timestamp', 0))}]: {element.get('type', 'CTA')} - \"{element.get('text', '')}\"")
else:
    print("   [Keine Call-to-Action Elemente erkannt]")

# Trend participation
trend_cues = []
if texts and 'segments' in texts:
    for seg in texts['segments']:
        text = seg.get('text', '').lower()
        if any(word in text for word in ['challenge', 'trend', 'viral', '#']):
            trend_cues.append(text)

if trend_cues:
    print("\n   Trend Participation Signals:")
    for cue in trend_cues[:3]:
        print(f"   - Text-Overlay: \"{cue}\"")
else:
    print("\n   Trend Participation Signals: [Keine erkannt]")

# Final Summary
print("\n📌 **Abschlussanalyse**")
summary_parts = []

# Basic info
summary_parts.append(f"Das {format_timestamp(duration)} lange Video von @{metadata['tiktok_metadata']['username']}")

# Content
if qwen and 'segments' in qwen:
    summary_parts.append(f"zeigt in {len(qwen['segments'])} analysierten Segmenten einen Tagesablauf")

# Technical
summary_parts.append(f"mit {cuts.get('total_cuts', 'vielen')} Schnitten")
summary_parts.append(f"{len(texts.get('segments', []))} Text-Overlays")

# Speech
if speech and 'segments' in speech:
    summary_parts.append(f"{len(speech['segments'])} Sprachsegmenten")

# Objects
if objects and 'unique_objects' in objects:
    summary_parts.append(f"{objects['unique_objects']} verschiedenen Objekttypen")

print(". ".join(summary_parts) + ".")

print("\n" + "="*80)
print(f"✅ VOLLSTÄNDIGE ANALYSE mit ALLEN verfügbaren Daten abgeschlossen!")