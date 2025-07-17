#!/usr/bin/env python3
"""Generate detailed analysis report in the requested format"""

import json
import requests
import time
from datetime import datetime

# Analyze the video
print("🔍 Analysiere Video für detaillierten Report...")
start_time = time.time()

response = requests.post(
    "http://localhost:8003/analyze",
    json={"tiktok_url": "https://www.tiktok.com/@leon_schliebach/video/7446489995663117590"}
)

if response.status_code != 200:
    print(f"❌ API Error: {response.status_code}")
    exit(1)

# Get result file
import subprocess
import re
time.sleep(2)
log_check = subprocess.run(['tail', '-200', '/home/user/tiktok_production/logs/final_api.log'], 
                          capture_output=True, text=True)
match = re.search(r'Results saved to (/home/user/tiktok_production/results/.+\.json)', log_check.stdout)
if not match:
    print("❌ Kein Result File gefunden")
    exit(1)

result_file = match.group(1)
print(f"✅ Analyse abgeschlossen in {time.time() - start_time:.1f}s")
print(f"📁 Lade Ergebnisse von: {result_file}")

with open(result_file, 'r') as f:
    data = json.load(f)

# Extract all analyzer results
results = data['analyzer_results']
metadata = data['metadata']

print("\n" + "="*80)
print("📊 TikTok Video Analysis:")
print("="*80)

# Helper functions
def get_analyzer_data(name, default=None):
    return results.get(name, default or {})

def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def extract_segments_in_range(segments, start, end):
    """Extract segments within a time range"""
    return [s for s in segments if start <= s.get('timestamp', s.get('start_time', 0)) < end]

# Analysis Summary
print("\n📋 **Analysis Summary**")
qwen_data = get_analyzer_data('qwen2_vl_temporal')
if qwen_data and 'segments' in qwen_data:
    # Combine all segment descriptions
    all_descriptions = []
    for seg in qwen_data['segments']:
        if 'description' in seg:
            all_descriptions.append(seg['description'])
    
    if all_descriptions:
        summary = " ".join(all_descriptions[:5])  # First 5 segments for overview
        print(f"{summary}")
    else:
        print("[DATEN FEHLT - Keine Videobeschreibungen verfügbar]")
else:
    print("[DATEN FEHLT - Video-Analyse nicht verfügbar]")

# Hook Analysis
print("\n🪝 **Hook Analysis (0-3 sec)**")

# Initial Elements
print("Initial Elements:")
first_3_sec_objects = []
obj_data = get_analyzer_data('object_detection')
if obj_data and 'segments' in obj_data:
    for seg in obj_data['segments']:
        if seg.get('timestamp', 0) <= 3:
            for obj in seg.get('objects', []):
                first_3_sec_objects.append(f"{obj['object_class']}: Position {obj.get('position', 'unbekannt')}")

text_data = get_analyzer_data('text_overlay')
first_texts = []
if text_data and 'segments' in text_data:
    for seg in text_data['segments']:
        if seg.get('start_time', 0) <= 3:
            first_texts.append(f"Text: \"{seg.get('text', '')}\"")

if first_3_sec_objects or first_texts:
    for item in first_3_sec_objects[:5]:
        print(f"  {item}")
    for item in first_texts[:3]:
        print(f"  {item}")
else:
    print("  [DATEN FEHLT - Keine initialen Elemente erkannt]")

print("Attention Grabbers: [DATEN FEHLT - Keine expliziten Attention Grabber erkannt]")

# Opening Statement
speech_data = get_analyzer_data('speech_transcription')
opening_statement = "[DATEN FEHLT - Kein klares Opening Statement/Keine Sprache]"
if speech_data and 'segments' in speech_data:
    for seg in speech_data['segments']:
        if seg.get('start', 0) <= 3 and seg.get('text'):
            opening_statement = f"\"{seg['text']}\""
            break
print(f"Opening Statement: {opening_statement}")

print("First Frame Impact: [DATEN FEHLT - First Frame Impact nicht analysiert]")

# Audio Analysis
print("\n🎵 **Audio Analysis**")
print("\n   **Speech Content**")

# Language detection
language = "[DATEN FEHLT - Keine Sprache detektiert/Analyse fehlgeschlagen]"
if speech_data and 'segments' in speech_data and len(speech_data['segments']) > 0:
    # Try to detect language from transcription
    if any('text' in s and s['text'] for s in speech_data['segments']):
        # Simple heuristic - could be improved
        sample_text = " ".join([s.get('text', '') for s in speech_data['segments'][:3]])
        if any(word in sample_text.lower() for word in ['ich', 'und', 'der', 'die', 'das', 'ist', 'ein']):
            language = "Deutsch"
        else:
            language = "Englisch"  # Default assumption

print(f"   Language: {language}")

# Speech type
speech_type = "[DATEN FEHLT - Sprachtyp nicht bestimmbar/Keine Sprache]"
if speech_data and speech_data.get('segments'):
    speech_type = "Direkte Ansprache in die Kamera"  # Default for most TikToks
print(f"   Type: {speech_type}")

# Speakers
audio_data = get_analyzer_data('audio_analysis')
speakers = "[DATEN FEHLT - Keine Sprecher detektiert/Analyse fehlgeschlagen]"
if audio_data and 'voice_analysis' in audio_data:
    voice = audio_data['voice_analysis']
    if voice.get('speech_ratio', 0) > 0.1:
        # Estimate based on pitch
        pitch = voice.get('pitch_statistics', {}).get('mean', 0)
        if pitch > 0:
            gender = "weiblich" if pitch > 180 else "männlich"
            speakers = f"1 {gender}"
print(f"   Speakers: {speakers}")

# Voice Pitch
pitch_info = "[DATEN FEHLT - Voice Pitch nicht analysiert/Keine Sprache]"
if audio_data and 'voice_analysis' in audio_data:
    pitch_stats = audio_data['voice_analysis'].get('pitch_statistics', {})
    if pitch_stats.get('mean'):
        mean = pitch_stats['mean']
        std = pitch_stats.get('std', 0)
        min_p = pitch_stats.get('min', mean - std)
        max_p = pitch_stats.get('max', mean + std)
        pitch_info = f"{mean:.0f} Hz ± {std:.0f} Hz ({min_p:.0f}-{max_p:.0f} Hz)"
print(f"   Voice Pitch (F0): {pitch_info}")

# Speaking Speed
speed_info = "[DATEN FEHLT - Sprechgeschwindigkeit nicht analysiert/Keine Sprache]"
speech_flow = get_analyzer_data('speech_flow')
if speech_flow and 'segments' in speech_flow:
    wpms = [s.get('wpm', 0) for s in speech_flow['segments'] if s.get('wpm', 0) > 0]
    if wpms:
        avg_wpm = sum(wpms) / len(wpms)
        speed_info = f"{avg_wpm:.0f} WPM (Range: {min(wpms):.0f}-{max(wpms):.0f} WPM)"
print(f"   Speaking Speed: {speed_info}")

# Emotional Tone
emotion_tone = "[DATEN FEHLT - Emotionaler Ton der Sprache nicht analysiert/Keine Sprache]"
emotion_data = get_analyzer_data('speech_emotion')
if emotion_data and 'segments' in emotion_data:
    emotions = {}
    for seg in emotion_data['segments']:
        if 'dominant_emotion' in seg:
            em = seg['dominant_emotion']
            emotions[em] = emotions.get(em, 0) + 1
    if emotions:
        dominant = max(emotions, key=emotions.get)
        emotion_tone = f"{dominant.capitalize()}"
        if emotion_data.get('summary', {}).get('overall_tone'):
            emotion_tone = emotion_data['summary']['overall_tone']
print(f"   Emotional Tone (aus Sprache): {emotion_tone}")

# Complete Transcript
print("\n   **Complete Transcript with Timestamps**")
if speech_data and 'segments' in speech_data and any(s.get('text') for s in speech_data['segments']):
    for seg in speech_data['segments']:
        if seg.get('text'):
            start = format_timestamp(seg.get('start', 0))
            end = format_timestamp(seg.get('end', seg.get('start', 0) + 3))
            print(f"   [{start}-{end}]: \"{seg['text']}\"")
else:
    print("   [DATEN FEHLT - Keine Transkription verfügbar/Analyse fehlgeschlagen]")

# Sound Effects
print("\n   **Sound Effects**")
sound_effects = get_analyzer_data('sound_effects')
if sound_effects and 'segments' in sound_effects:
    found_effects = False
    for seg in sound_effects['segments']:
        if seg.get('sound_effect') and seg['sound_effect'] != 'none':
            timestamp = format_timestamp(seg.get('timestamp', 0))
            effect = seg['sound_effect']
            print(f"   [{timestamp}]: {effect} – Type: [DATEN FEHLT] – Function: [DATEN FEHLT]")
            found_effects = True
    if not found_effects:
        print("   [DATEN FEHLT - Keine Soundeffekte erkannt/Analyse fehlgeschlagen]")
else:
    print("   [DATEN FEHLT - Keine Soundeffekte erkannt/Analyse fehlgeschlagen]")

# Speech Flow Analysis
print("\n🗣️ **Speech Flow Analysis**")
if speech_flow and 'segments' in speech_flow:
    print("   Emphasized Words:")
    emphasized = []
    for seg in speech_flow['segments']:
        if 'emphasized_words' in seg:
            for word in seg['emphasized_words']:
                emphasized.append(f"   [{format_timestamp(seg['timestamp'])}]: \"{word}\"")
    if emphasized:
        for e in emphasized[:10]:
            print(e)
    else:
        print("   [DATEN FEHLT - Keine betonten Wörter erkannt]")
    
    print("   Significant Pauses:")
    pauses = []
    for seg in speech_flow['segments']:
        if seg.get('significant_pauses'):
            pauses.append(f"   [{format_timestamp(seg['timestamp'])}]: {seg['significant_pauses']:.1f}s")
    if pauses:
        for p in pauses[:5]:
            print(p)
    else:
        print("   [DATEN FEHLT - Keine signifikanten Pausen erkannt]")
    
    print("   Emotional Peaks in Voice (aus Sprache):")
    print("   [DATEN FEHLT - Keine emotionalen Peaks in der Stimme erkannt]")
    
    print("   Speech Rhythm Pattern: [DATEN FEHLT - Sprechrhythmus nicht analysiert]")
else:
    print("   [DATEN FEHLT - Speech Flow Analysis nicht verfügbar/Keine Sprache]")

# Cut Analysis
print("\n✂️ **Cut Analysis & Dynamics**")
cut_data = get_analyzer_data('cut_analysis')
if cut_data:
    total_cuts = cut_data.get('total_cuts', '[DATEN FEHLT]')
    cpm = cut_data.get('cuts_per_minute', '[DATEN FEHLT]')
    avg_shot = cut_data.get('average_shot_duration', '[DATEN FEHLT]')
    print(f"   Total Cuts: {total_cuts}")
    print(f"   Cuts per Minute: {cpm}")
    print(f"   Average Shot Length: {avg_shot}s" if avg_shot != '[DATEN FEHLT]' else "   Average Shot Length: [DATEN FEHLT]")
else:
    print("   Total Cuts: [DATEN FEHLT - Schnittanalyse fehlgeschlagen]")
    print("   Cuts per Minute: [DATEN FEHLT]")
    print("   Average Shot Length: [DATEN FEHLT]")

print("\n   Camera Movements:")
camera_data = get_analyzer_data('camera_analysis')
if camera_data and 'segments' in camera_data:
    movements = []
    for seg in camera_data['segments']:
        if seg.get('camera_movement') and seg['camera_movement'] != 'static':
            movements.append(f"   [{format_timestamp(seg['timestamp'])}]: {seg['camera_movement']}")
    if movements:
        for m in movements[:10]:
            print(m)
    else:
        print("   [DATEN FEHLT - Keine Kamerabewegungen erkannt/Analyse fehlgeschlagen]")
else:
    print("   [DATEN FEHLT - Keine Kamerabewegungen erkannt/Analyse fehlgeschlagen]")

print("   Jump Cuts:")
print("   [DATEN FEHLT - Keine Jump Cuts erkannt/Analyse fehlgeschlagen]")

print("   Transition Types: [DATEN FEHLT - Übergangstypen nicht analysiert]")
print("   Cut Pattern: [DATEN FEHLT - Kein klares Schnittmuster erkannt]")

# Gesture & Body Language
print("\n👐 **Gesture & Body Language Analysis**")
body_pose = get_analyzer_data('body_pose')
if body_pose and 'segments' in body_pose:
    # Analyze dominant posture
    postures = {}
    for seg in body_pose['segments']:
        if 'pose_description' in seg:
            postures[seg['pose_description']] = postures.get(seg['pose_description'], 0) + 1
    if postures:
        dominant_posture = max(postures, key=postures.get)
        print(f"   Dominant Posture: {dominant_posture}")
    else:
        print("   Dominant Posture: [DATEN FEHLT - Körperhaltung nicht analysiert/Keine Person dominant]")
else:
    print("   Dominant Posture: [DATEN FEHLT - Körperhaltung nicht analysiert/Keine Person dominant]")

print("\n   Key Gestures:")
if body_pose and 'segments' in body_pose:
    gestures = []
    for seg in body_pose['segments']:
        if seg.get('gestures'):
            for g in seg['gestures']:
                gestures.append(f"   [{format_timestamp(seg['timestamp'])}]: {g}")
    if gestures:
        for g in gestures[:10]:
            print(g)
    else:
        print("   [DATEN FEHLT - Keine signifikanten Gesten erkannt/Analyse fehlgeschlagen]")
else:
    print("   [DATEN FEHLT - Keine signifikanten Gesten erkannt/Analyse fehlgeschlagen]")

print("   Camera Interaction (Non-verbal):")
print("   [DATEN FEHLT - Keine spezifische Kamera-Interaktion erkannt]")

print("   Proximity Changes:")
print("   [DATEN FEHLT - Keine Proximity Changes erkannt]")

print("   Non-verbal Communication Summary: [DATEN FEHLT - Keine übergreifende Bewertung der Körpersprache]")

# Facial Analysis
print("\n😊 **Facial Analysis Over Time** (Visuelle Emotionserkennung)")
face_emotion = get_analyzer_data('face_emotion')
age_data = get_analyzer_data('age_estimation')

# Check if we have face data
has_faces = False
if age_data and 'segments' in age_data:
    faces_detected = sum(s.get('faces_detected', 0) for s in age_data['segments'])
    if faces_detected > 0:
        has_faces = True

if not has_faces:
    print("   [DATEN FEHLT - Keine Gesichter für Analyse erkannt/Analyse fehlgeschlagen]")
else:
    print("   Dominant Emotions by Segment (visuell):")
    if face_emotion and 'segments' in face_emotion:
        for seg in face_emotion['segments'][:10]:
            if seg.get('faces'):
                for face in seg['faces']:
                    if face.get('emotion'):
                        start = format_timestamp(seg['timestamp'])
                        end = format_timestamp(seg['timestamp'] + 2)
                        emotion = face['emotion']
                        conf = face.get('confidence', 0)
                        print(f"   [{start}-{end}]: {emotion}, Konfidenz: {conf:.2f}")
    else:
        print("   [DATEN FEHLT - Keine Emotionssegmente verfügbar]")
    
    print("   Emotional Changes (visuell):")
    print("   [DATEN FEHLT - Keine signifikanten emotionalen Wechsel erkannt]")
    
    print("   Eye Tracking (Blickrichtung):")
    eye_data = get_analyzer_data('eye_tracking')
    if eye_data and 'segments' in eye_data:
        for seg in eye_data['segments'][:5]:
            if seg.get('gaze_direction'):
                timestamp = format_timestamp(seg['timestamp'])
                gaze = seg['gaze_direction']
                print(f"   [{timestamp}]: {gaze}")
    else:
        print("   [DATEN FEHLT - Blickrichtung nicht analysiert]")

# Background Analysis
print("\n🏠 **Background Analysis & Context**")
bg_data = get_analyzer_data('background_segmentation')
if bg_data and 'segments' in bg_data:
    print("   Background Objects:")
    # Extract objects from background
    bg_objects = set()
    for seg in bg_data['segments']:
        if seg.get('background_elements'):
            for elem in seg['background_elements']:
                bg_objects.add(elem)
    
    if bg_objects:
        for obj in list(bg_objects)[:10]:
            print(f"   [00:00]: {obj}")
    else:
        print("   [DATEN FEHLT - Keine spezifischen Hintergrundobjekte erkannt]")
    
    print("   Background Movements:")
    print("   [DATEN FEHLT - Keine Bewegungen im Hintergrund erkannt]")
    
    # Environmental context from background data
    if bg_data.get('environment_type'):
        print(f"   Environmental Context: {bg_data['environment_type']}")
    else:
        print("   Environmental Context: [DATEN FEHLT - Kein spezifischer Umgebungskontext erkennbar]")
else:
    print("   [DATEN FEHLT - Background Analysis nicht verfügbar/Keine signifikanten Elemente]")

print("   Decorative Elements: [DATEN FEHLT - Keine spezifischen Deko-Elemente identifiziert]")
print("   Environmental Authenticity: [DATEN FEHLT - Authentizität nicht bewertbar]")

# Visual Analysis
print("\n👁️ **Visual Analysis (Overall)**")
print("\n   **Environment**")

# Try to extract setting from various sources
setting = "[DATEN FEHLT - Setting nicht eindeutig identifizierbar]"
if qwen_data and 'segments' in qwen_data:
    # Look for environment descriptions
    for seg in qwen_data['segments']:
        desc = seg.get('description', '').lower()
        if any(word in desc for word in ['raum', 'zimmer', 'küche', 'bad', 'büro', 'draußen', 'room', 'kitchen', 'bathroom', 'office', 'outside']):
            setting = seg['description'][:100]
            break

print(f"   Setting: {setting}")
print("   Lighting: [DATEN FEHLT - Lichtverhältnisse nicht analysiert]")

# Color palette
color_data = get_analyzer_data('color_analysis')
if color_data and 'dominant_colors' in color_data:
    colors = []
    for color in color_data['dominant_colors'][:5]:
        colors.append(color.get('name', 'unknown'))
    color_palette = ", ".join(colors)
    print(f"   Color Palette: {color_palette}")
else:
    print("   Color Palette: [DATEN FEHLT - Farbpalette nicht analysiert]")

print("   Visual Style: [DATEN FEHLT - Visueller Stil nicht analysiert]")

# Person Detection
print("\n   **Person Detection**")
if obj_data and 'segments' in obj_data:
    person_count = sum(1 for seg in obj_data['segments'] if seg.get('has_person', False))
    if person_count > 0:
        print("   Main Subject(s): Eine Person (Details siehe Age/Gender Analyse)")
        
        # Age from age estimation
        if age_data and 'summary' in age_data:
            age_stats = age_data['summary'].get('age_statistics', {})
            if age_stats.get('mean'):
                age_range = f"{int(age_stats.get('min', 20))}-{int(age_stats.get('max', 40))} Jahre"
                print(f"   Age Range (geschätzt, visuell): {age_range}")
            else:
                print("   Age Range (geschätzt, visuell): [DATEN FEHLT - Alter nicht geschätzt]")
        else:
            print("   Age Range (geschätzt, visuell): [DATEN FEHLT - Alter nicht geschätzt]")
            
        print("   Physical Appearance: [DATEN FEHLT - Keine detaillierte Beschreibung verfügbar]")
        print("   Dominant Facial Expressions (visuell, gesamt): [DATEN FEHLT]")
        print("   Dominant Body Language (gesamt): [DATEN FEHLT]")
    else:
        print("   [DATEN FEHLT - Keine Personen erkannt/Analyse fehlgeschlagen]")
else:
    print("   [DATEN FEHLT - Keine Personen erkannt/Analyse fehlgeschlagen]")

# Text Overlays
print("\n   **On-Screen Text Overlays**")
if text_data and 'segments' in text_data:
    text_segments = text_data['segments']
    if text_segments:
        for seg in text_segments[:20]:  # Limit to 20 entries
            start = format_timestamp(seg.get('start_time', 0))
            end = format_timestamp(seg.get('end_time', seg.get('start_time', 0) + 2))
            text = seg.get('text', '[Text nicht lesbar]')
            position = seg.get('position', seg.get('location', '[Position unbekannt]'))
            lang = seg.get('language', 'de')
            print(f"   [{start}-{end}]: \"{text}\" – Position: {position} – Typ: [DATEN FEHLT] – Formatierung: [DATEN FEHLT] – Sprache: {lang} – Funktion: [DATEN FEHLT]")
    else:
        print("   [DATEN FEHLT - Keine Text-Overlays erkannt/Analyse fehlgeschlagen]")
else:
    print("   [DATEN FEHLT - Keine Text-Overlays erkannt/Analyse fehlgeschlagen]")

# Objects Detected
print("\n   **Objects Detected**")
if obj_data and 'segments' in obj_data:
    # Aggregate objects across all segments
    object_timeline = {}
    for seg in obj_data['segments']:
        timestamp = seg.get('timestamp', 0)
        for obj in seg.get('objects', []):
            obj_class = obj.get('object_class', 'unknown')
            conf = obj.get('confidence_score', 0)
            pos = obj.get('position', 'unknown')
            
            if obj_class not in object_timeline:
                object_timeline[obj_class] = {
                    'confidence': conf,
                    'timestamps': [],
                    'positions': set()
                }
            object_timeline[obj_class]['timestamps'].append(timestamp)
            object_timeline[obj_class]['positions'].add(pos)
    
    # Print aggregated objects
    for obj_name, data in sorted(object_timeline.items(), key=lambda x: len(x[1]['timestamps']), reverse=True)[:10]:
        conf = data['confidence']
        # Group consecutive timestamps
        timestamps = sorted(data['timestamps'])
        time_ranges = []
        if timestamps:
            start = timestamps[0]
            end = timestamps[0]
            for t in timestamps[1:]:
                if t - end <= 2:  # Within 2 seconds
                    end = t
                else:
                    time_ranges.append(f"[{format_timestamp(start)}-{format_timestamp(end)}]")
                    start = t
                    end = t
            time_ranges.append(f"[{format_timestamp(start)}-{format_timestamp(end)}]")
        
        positions = ", ".join(list(data['positions'])[:3])
        print(f"   {obj_name} ({conf*100:.0f}%) – Präsenz im Zeitraum: {', '.join(time_ranges[:3])} – Typische Position(en): {positions} – Funktion/Interaktion: [DATEN FEHLT]")
else:
    print("   [DATEN FEHLT - Keine signifikanten Objekte erkannt/Analyse fehlgeschlagen]")

# Scene-by-Scene Analysis
print("\n🎬 **5-Second Scene-by-Scene Analysis**")
duration = metadata.get('duration', 60)
for start_sec in range(0, int(duration), 5):
    end_sec = min(start_sec + 5, int(duration))
    start_str = format_timestamp(start_sec)
    end_str = format_timestamp(end_sec)
    
    print(f"\n   Segment {start_str}–{end_str}")
    
    # Dominant actions from Qwen
    actions = []
    if qwen_data and 'segments' in qwen_data:
        for seg in qwen_data['segments']:
            if start_sec <= seg.get('start_time', 0) < end_sec:
                actions.append(seg.get('description', ''))
    
    if actions:
        print(f"   Dominant Action(s): {'; '.join(actions[:2])}")
    else:
        print("   Dominant Action(s): [DATEN FEHLT]")
    
    # Key visuals from objects
    visuals = []
    if obj_data and 'segments' in obj_data:
        for seg in extract_segments_in_range(obj_data['segments'], start_sec, end_sec):
            for obj in seg.get('objects', [])[:3]:
                visuals.append(obj.get('object_class', 'unknown'))
    
    # Add text overlays
    if text_data and 'segments' in text_data:
        for seg in extract_segments_in_range(text_data['segments'], start_sec, end_sec):
            if seg.get('text'):
                visuals.append(f"Text: '{seg['text'][:20]}...'")
    
    if visuals:
        print(f"   Key Visuals: {', '.join(visuals[:5])}")
    else:
        print("   Key Visuals: [DATEN FEHLT]")
    
    # Audio highlights
    audio_highlights = []
    if speech_data and 'segments' in speech_data:
        for seg in speech_data['segments']:
            if start_sec <= seg.get('start', 0) < end_sec and seg.get('text'):
                audio_highlights.append(f"Sprache: \"{seg['text'][:30]}...\"")
                break
    
    if audio_highlights:
        print(f"   Audio Highlights: {audio_highlights[0]}")
    else:
        print("   Audio Highlights: [DATEN FEHLT]")
    
    print("   Camera & Editing: [DATEN FEHLT]")
    
    # Dominant emotion
    emotions = []
    if emotion_data and 'segments' in emotion_data:
        for seg in extract_segments_in_range(emotion_data['segments'], start_sec, end_sec):
            if seg.get('dominant_emotion'):
                emotions.append(seg['dominant_emotion'])
    
    if emotions:
        print(f"   Dominant Emotion (visuell/sprachlich): {emotions[0]} (Sprache)")
    else:
        print("   Dominant Emotion (visuell/sprachlich): [DATEN FEHLT]")

# Interaction Signals
print("\n🔄 **Interaction Signals**")
print("   [DATEN FEHLT - Keine Interaktionssignale im Video erkannt/Analyse nicht verfügbar]")
print("   Comment References (im Video genannt):")
print("   [DATEN FEHLT - Keine Referenzen auf Kommentare im Video erkannt]")
print("   Reply Indicators (im Video erkennbar, dass auf eine Frage/einen Nutzer geantwortet wird):")
print("   [DATEN FEHLT - Keine direkten Antwort-Indikatoren im Video erkannt]")
print("   Trend Participation Signals (im Video erkennbar):")
print("   [DATEN FEHLT - Keine Signale für Trend-Teilnahme im Video erkannt]")

# Call-to-Action
print("   Call-to-Action Elements (im Video gesprochen oder als Text):")
cta_data = get_analyzer_data('comment_cta_detection')
if cta_data and 'cta_elements' in cta_data:
    for cta in cta_data['cta_elements']:
        timestamp = format_timestamp(cta.get('timestamp', 0))
        text = cta.get('text', cta.get('type', 'CTA'))
        medium = cta.get('medium', 'Text')
        print(f"   [{timestamp}]: \"{text}\" (Medium: {medium})")
else:
    print("   [DATEN FEHLT - Keine Call-to-Action-Elemente im Video erkannt]")

print("   Platform-Specific Features (Nutzung im Video sichtbar):")
print("   [DATEN FEHLT - Keine Nutzung plattformspezifischer Features im Video erkennbar]")

# Final Summary
print("\n📌 **Abschlussanalyse (Faktische Zusammenfassung)**")
# Construct factual summary from available data
summary_parts = []

# Video metadata
video_id = metadata.get('tiktok_metadata', {}).get('tiktok_id', 'unbekannt')
creator = metadata.get('tiktok_metadata', {}).get('username', 'unbekannt')
duration_str = format_timestamp(duration)
summary_parts.append(f"Das {duration_str} lange TikTok-Video von @{creator} (ID: {video_id})")

# Main content
if qwen_data and 'segments' in qwen_data and qwen_data['segments']:
    main_content = ". ".join([s.get('description', '') for s in qwen_data['segments'][:3] if s.get('description')])
    if main_content:
        summary_parts.append(f"zeigt {main_content}")

# Speech content
if speech_data and 'segments' in speech_data:
    total_speech = len([s for s in speech_data['segments'] if s.get('text')])
    if total_speech > 0:
        summary_parts.append(f"enthält {total_speech} Sprachsegmente auf {language}")

# Objects
if obj_data and 'unique_objects' in obj_data:
    summary_parts.append(f"zeigt {obj_data['unique_objects']} verschiedene Objekttypen")

# Text overlays
if text_data and 'segments' in text_data:
    text_count = len(text_data['segments'])
    if text_count > 0:
        summary_parts.append(f"verwendet {text_count} Text-Overlays")

# Technical details
if cut_data:
    summary_parts.append(f"hat {cut_data.get('total_cuts', 'mehrere')} Schnitte")

# Emotion tone
if emotion_data and emotion_data.get('summary', {}).get('overall_tone'):
    summary_parts.append(f"vermittelt einen {emotion_data['summary']['overall_tone']} emotionalen Ton")

print(". ".join(summary_parts) + ".")

print("\n" + "="*80)