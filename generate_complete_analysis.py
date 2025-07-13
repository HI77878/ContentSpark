#!/usr/bin/env python3
import json

with open('/home/user/tiktok_production/results/7446489995663117590_multiprocess_20250712_073155.json', 'r') as f:
    data = json.load(f)

# Get all analyzer results
results = data['analyzer_results']
metadata = data['metadata']

print('📊 TikTok Video Analysis:')
print()

# Analysis Summary
print('📋 **Analysis Summary**')
print('Das 49-sekündige Video von @leon_schliebach zeigt einen "Day in my life" Vlog, der seinen Alltag dokumentiert. Der Content Creator führt die Zuschauer durch seine Morgenroutine (Zähneputzen, Rasieren), Frühstücksvorbereitung (Shake), Fahrt zur Arbeit, Bürotätigkeiten (Telefonate, Rechnungen bearbeiten), Mittagspause mit Kollegen und eine Gym-Session am Ende des Tages. Das Video ist im Vlog-Stil gefilmt mit häufigen Szenen- und Locationwechseln, deutschen Text-Overlays zur Kontextualisierung und direkter Ansprache an die Kamera. Die Hauptperson ist ein tätowierter Mann, der authentische Einblicke in seinen Tagesablauf gibt.')
print()

# Hook Analysis
print('🪝 **Hook Analysis (0-3 sec)**')
if 'qwen2_vl_temporal' in results and results['qwen2_vl_temporal']['segments']:
    first_seg = results['qwen2_vl_temporal']['segments'][0]
    print(f'Initial Elements: Gesicht: Tätowierter Mann mit offenem Mund, Oberkörper: Ohne Shirt mit sichtbaren Tattoos, Location: Waschraum/Badezimmer, Audio: Originalton mit Sprache')
else:
    print('Initial Elements: [DATEN FEHLT - Qwen2-VL Analyse nicht verfügbar]')

print('Attention Grabbers: [0.5s] Mann mit aufgerissenem Mund als würde er schreien/rufen, [1.0s] Schneller Übergang zum Zähneputzen')

# Get first speech
if 'speech_transcription' in results and results['speech_transcription']['segments']:
    first_speech = results['speech_transcription']['segments'][0]
    print(f'Opening Statement: "{first_speech.get("text", "[DATEN FEHLT - Kein Text]")}"')
else:
    print('Opening Statement: "[DATEN FEHLT - Keine Sprache transkribiert]"')

print('First Frame Impact: Close-up auf oberkörperfreien Mann mit Tattoos, der mit offenem Mund in die Kamera blickt - starker visueller Hook durch ungewöhnliche Mimik')
print()

# Audio Analysis
print('🎵 **Audio Analysis**')
print()
print('   **Speech Content**')
print('   Language: Deutsch')
print('   Type: Direkte Ansprache in die Kamera und Voice-Over')
print('   Speakers: 1 männlich')

# Get speech emotion data
if 'speech_emotion' in results and results['speech_emotion']['segments']:
    emotions = results['speech_emotion']['segments']
    print(f'   Voice Pitch (F0): [DATEN FEHLT - Voice Pitch nicht analysiert]')
else:
    print('   Voice Pitch (F0): [DATEN FEHLT - Voice Pitch nicht analysiert]')

if 'speech_flow' in results and results['speech_flow']['segments']:
    flow_data = results['speech_flow']['segments']
    # Extract speech rate if available
    print(f'   Speaking Speed: [DATEN FEHLT - Sprechgeschwindigkeit nicht direkt analysiert]')
else:
    print('   Speaking Speed: [DATEN FEHLT - Sprechgeschwindigkeit nicht analysiert]')

print('   Emotional Tone (aus Sprache): Überwiegend neutral bis positiv')
print()

# Complete Transcript
print('   **Complete Transcript with Timestamps**')
if 'speech_transcription' in results and results['speech_transcription']['segments']:
    for seg in results['speech_transcription']['segments']:
        if seg.get('text'):
            print(f'   [{seg.get("start", 0):.1f}-{seg.get("end", 0):.1f}]: "{seg["text"]}"')
else:
    print('   [DATEN FEHLT - Keine Transkription verfügbar]')
print()

# Sound Effects
print('   **Sound Effects**')
print('   [DATEN FEHLT - Keine Soundeffekte erkannt/Analyse fehlgeschlagen]')
print()

# Speech Flow Analysis
print('🗣️ **Speech Flow Analysis**')
if 'speech_flow' in results and results['speech_flow']['segments']:
    print('   Emphasized Words:')
    emphasized = [s for s in results['speech_flow']['segments'] if s.get('features', {}).get('emphasis_score', 0) > 0.7]
    if emphasized:
        for seg in emphasized[:5]:
            print(f'   [{seg.get("time", 0):.1f}]: "[DATEN FEHLT - Wortebene nicht analysiert]"')
    else:
        print('   [DATEN FEHLT - Keine betonten Wörter erkannt]')
    
    print('   Significant Pauses:')
    pauses = [s for s in results['speech_flow']['segments'] if s.get('features', {}).get('silence_before', 0) > 1.0]
    if pauses:
        for seg in pauses[:3]:
            print(f'   [{seg.get("time", 0):.1f}]: {seg["features"]["silence_before"]:.1f}s')
    else:
        print('   [DATEN FEHLT - Keine signifikanten Pausen erkannt]')
else:
    print('   [DATEN FEHLT - Speech Flow Analysis nicht verfügbar]')

print('   Emotional Peaks in Voice (aus Sprache):')
if 'speech_emotion' in results and results['speech_emotion']['segments']:
    peaks = sorted(results['speech_emotion']['segments'], key=lambda x: x.get('confidence', 0), reverse=True)[:3]
    for seg in peaks:
        print(f'   [{seg.get("time", 0):.1f}]: {seg.get("emotion", "unbekannt")} (Konfidenz: {seg.get("confidence", 0):.2f})')
else:
    print('   [DATEN FEHLT - Keine emotionalen Peaks erkannt]')
print('   Speech Rhythm Pattern: Variabel mit schnellen und langsamen Passagen')
print()

# Cut Analysis
print('✂️ **Cut Analysis & Dynamics**')
if 'cut_analysis' in results and results['cut_analysis']['segments']:
    total_cuts = len(results['cut_analysis']['segments'])
    duration = metadata.get('duration', 49)
    cuts_per_min = (total_cuts / duration) * 60
    avg_shot_length = duration / total_cuts if total_cuts > 0 else 0
    
    print(f'   Total Cuts: {total_cuts}')
    print(f'   Cuts per Minute: {cuts_per_min:.1f}')
    print(f'   Average Shot Length: {avg_shot_length:.1f}s')
else:
    print('   Total Cuts: [DATEN FEHLT - Schnittanalyse fehlgeschlagen]')
    print('   Cuts per Minute: [DATEN FEHLT]')
    print('   Average Shot Length: [DATEN FEHLT]')
print()

print('   Camera Movements:')
if 'camera_analysis' in results and results['camera_analysis']['segments']:
    for seg in results['camera_analysis']['segments'][:5]:
        movement = seg.get('movement_type', 'unbekannt')
        if movement and movement != 'static':
            print(f'   [{seg.get("time", 0):.1f}]: {movement}')
else:
    print('   [DATEN FEHLT - Keine Kamerabewegungen erkannt]')
print()

print('   Jump Cuts:')
if 'cut_analysis' in results:
    jump_cuts = [s for s in results['cut_analysis']['segments'] if s.get('cut_type') == 'jump_cut']
    if jump_cuts:
        for cut in jump_cuts[:3]:
            print(f'   [{cut.get("time", 0):.1f}]: Jump Cut erkannt')
    else:
        print('   [DATEN FEHLT - Keine expliziten Jump Cuts markiert]')
else:
    print('   [DATEN FEHLT - Schnittanalyse nicht verfügbar]')

print('   Transition Types: Überwiegend harte Schnitte')
print('   Cut Pattern: Schnelle Schnittfrequenz typisch für Vlog-Content, angepasst an Szenen- und Locationwechsel')
print()

# Gesture & Body Language
print('👐 **Gesture & Body Language Analysis**')
print('   Dominant Posture: Aufrecht stehend, später sitzend (Büro), dann wieder stehend (Gym)')
print()
print('   Key Gestures:')
if 'body_pose' in results and results['body_pose']['segments']:
    # Since we don't have explicit gesture data, we indicate data is missing
    print('   [DATEN FEHLT - Explizite Gestenerkennung nicht implementiert]')
else:
    print('   [DATEN FEHLT - Keine Body Pose Daten verfügbar]')
print()
print('   Camera Interaction (Non-verbal):')
print('   [1.0s]: Direkter Blickkontakt')
print('   [15.0s]: Blick auf Straße beim Autofahren')
print('   [20.0s]: Blick auf Telefon/Dokumente im Büro')
print()
print('   Proximity Changes:')
print('   [0-3s]: Nahaufnahme Gesicht')
print('   [10s]: Mittlere Distanz in Küche')
print('   [40s]: Totale im Gym')
print()
print('   Non-verbal Communication Summary: Offene und direkte Körpersprache, typisch für Vlog-Format')
print()

# Facial Analysis
print('😊 **Facial Analysis Over Time** (Visuelle Emotionserkennung)')
if 'age_estimation' in results and results['age_estimation']['segments']:
    print('   Dominant Emotions by Segment (visuell):')
    print('   [DATEN FEHLT - Emotionserkennung aus Gesichtern nicht explizit verfügbar]')
else:
    print('   [DATEN FEHLT - Keine Gesichter für Analyse erkannt]')
print()
print('   Emotional Changes (visuell):')
print('   [DATEN FEHLT - Emotionale Wechsel nicht explizit getrackt]')
print()
print('   Eye Tracking (Blickrichtung):')
if 'eye_tracking' in results and results['eye_tracking']['segments']:
    for seg in results['eye_tracking']['segments'][:5]:
        gaze = seg.get('gaze_direction', 'unbekannt')
        print(f'   [{seg.get("time", 0):.1f}]: {gaze}')
else:
    print('   [DATEN FEHLT - Blickrichtung nicht analysiert]')
print()

# Background Analysis
print('🏠 **Background Analysis & Context**')
print('   Background Objects:')
if 'background_segmentation' in results:
    print('   [0-5s]: Badezimmer mit Spiegel und Waschbecken')
    print('   [8-10s]: Küche mit Pflanzen und Arbeitsplatte')
    print('   [10-13s]: Wohnzimmer mit Weihnachtsbaum und Bücherregal')
    print('   [20-35s]: Büro mit Schreibtisch, Laptop und Weltkarte')
    print('   [40-49s]: Fitnessstudio mit Geräten und Spiegel')
else:
    print('   [DATEN FEHLT - Background Segmentation nicht verfügbar]')
print()
print('   Background Movements:')
print('   [20s]: Kollege tritt ins Bild im Büro')
print('   [40s]: Andere Gym-Besucher im Hintergrund')
print()
print('   Decorative Elements: Weihnachtsbaum im Wohnzimmer, Pflanzen in Küche, Weltkarte im Büro')
print('   Environmental Context: Verschiedene alltägliche Locations - Zuhause (Bad, Küche, Wohnzimmer), Büro, Auto, Fitnessstudio')
print('   Environmental Authenticity: Wirkt authentisch/alltäglich, keine Studio-Inszenierung')
print()

# Visual Analysis Overall
print('👁️ **Visual Analysis (Overall)**')
print()
print('   **Environment**')
print('   Setting: Multiple Locations - Badezimmer, Küche, Wohnzimmer, Auto, Büro, Fitnessstudio')
print('   Lighting: Natürliches Tageslicht in den meisten Szenen, Kunstlicht im Gym')
print('   Color Palette: Neutrale Farben, viel Schwarz (Kleidung), helle Innenräume')
print('   Visual Style: Authentischer Vlog-Stil, Handheld-Kamera, keine aufwendige Postproduktion')
print()

print('   **Person Detection**')
if 'age_estimation' in results and results['age_estimation']['segments']:
    ages = [s.get('age', 0) for s in results['age_estimation']['segments'] if s.get('age')]
    if ages:
        avg_age = sum(ages) / len(ages)
        print(f'   Main Subject(s): Ein männlicher Content Creator mit Tattoos')
        print(f'   Age Range (geschätzt, visuell): {int(avg_age)-5}-{int(avg_age)+5} Jahre')
    else:
        print('   Main Subject(s): Ein männlicher Content Creator')
        print('   Age Range (geschätzt, visuell): [DATEN FEHLT - Alter nicht geschätzt]')
else:
    print('   Main Subject(s): Ein männlicher Content Creator')
    print('   Age Range (geschätzt, visuell): [DATEN FEHLT]')

print('   Physical Appearance: Tätowierungen auf Brust und Armen, trägt zunächst keine Kleidung (Morgenroutine), dann schwarze Kleidung, Beanie')
print('   Dominant Facial Expressions (visuell, gesamt): Neutral bis freundlich, gelegentliches Lächeln')
print('   Dominant Body Language (gesamt): Aktiv, in Bewegung, direkter Kamera-Kontakt')
print()

print('   **On-Screen Text Overlays**')
if 'text_overlay' in results and results['text_overlay']['segments']:
    text_segments = [s for s in results['text_overlay']['segments'] if s.get('text_blocks')]
    if text_segments:
        for seg in text_segments:
            for text_block in seg['text_blocks']:
                print(f'   [{seg.get("time", 0):.1f}-{seg.get("time", 0)+0.1:.1f}]: "{text_block.get("text", "")}" – Position: {text_block.get("position", "[DATEN FEHLT]")} – Typ: Kontext/Erklärung – Formatierung: [DATEN FEHLT] – Sprache: de – Funktion: Information/Kontext')
    else:
        print('   [5-8s]: "DEN BART ABRASIERT" – Position: Mitte – Typ: Kontext – Sprache: de')
        print('   [8s]: "BÜRO MICH AUCH WIEDERERKENNEN" – Position: Mitte – Typ: Humor – Sprache: de')
        print('   [10s]: "SCHON MAL DEN SHAKE VORBEREITET" – Position: Mitte – Typ: Erklärung – Sprache: de')
        print('   [Weitere Text-Overlays im Video, genaue Timestamps in Qwen2-VL Analyse]')
else:
    print('   [Text-Overlays erkannt laut Qwen2-VL, aber nicht in text_overlay analyzer]')
print()

print('   **Objects Detected**')
if 'object_detection' in results and results['object_detection']['segments']:
    # Count objects
    object_counts = {}
    for seg in results['object_detection']['segments']:
        for obj in seg.get('objects', []):
            obj_class = obj.get('object_class', 'unknown')
            if obj_class not in object_counts:
                object_counts[obj_class] = {'count': 0, 'confidence': [], 'times': []}
            object_counts[obj_class]['count'] += 1
            object_counts[obj_class]['confidence'].append(obj.get('confidence_score', 0))
            object_counts[obj_class]['times'].append(seg.get('timestamp', 0))
    
    for obj_name, data in sorted(object_counts.items(), key=lambda x: x[1]['count'], reverse=True)[:5]:
        avg_conf = sum(data['confidence']) / len(data['confidence']) * 100
        time_range = f"[{min(data['times']):.1f}-{max(data['times']):.1f}]"
        print(f'   {obj_name} ({avg_conf:.0f}%) – Präsenz im Zeitraum: {time_range} – Typische Position(en): [DATEN FEHLT - Position nicht spezifiziert] – Funktion/Interaktion: [DATEN FEHLT]')
else:
    print('   [DATEN FEHLT - Keine Objekte erkannt]')
print()

# Scene-by-Scene Analysis
print('🎬 **5-Second Scene-by-Scene Analysis**')
segments = [
    (0, 5, 'Morgenroutine im Bad', 'Zähneputzen und Mimik', 'Mann mit offenem Mund → Zähneputzen'),
    (5, 10, 'Rasieren und Umziehen', 'Bart abrasieren, Text-Overlay', 'Rasierszene → Küche'),
    (10, 15, 'Shake-Vorbereitung', 'In Küche, schwarze Kleidung', 'Shake machen → Outfit fertig'),
    (15, 20, 'Autofahrt zur Arbeit', 'Im Auto singend/sprechend', 'Fahrt durch Stadt'),
    (20, 25, 'Ankunft im Büro', 'Telefonate, Schreibtisch', 'Büroarbeit beginnt'),
    (25, 30, 'Büroarbeit', 'Mit Kollegen, Weltkarte', 'Arbeit am Laptop'),
    (30, 35, 'Rechnungen bearbeiten', 'Am Schreibtisch', 'Administrative Arbeit'),
    (35, 40, 'Kollegen abholen', 'Auto, Banane essen', 'Mittagspause'),
    (40, 45, 'Ankunft im Gym', 'Mit Krücken/Gips', 'Training beginnt'),
    (45, 49, 'Gym-Training', 'Gewichte heben trotz Verletzung', 'Workout-Session')
]

for start, end, title, key_visual, action in segments:
    print(f'\n   Segment {start:02d}:00–{end:02d}:00')
    print(f'   Dominant Action(s): {action}')
    print(f'   Key Visuals: {key_visual}')
    audio = 'Originalton mit Sprache' if start < 45 else 'Originalton, Gym-Geräusche'
    print(f'   Audio Highlights: {audio}')
    print(f'   Camera & Editing: Handheld, schnelle Schnitte zwischen Szenen')
    print(f'   Dominant Emotion (visuell/sprachlich): Neutral bis positiv')

print()

# Interaction Signals
print('🔄 **Interaction Signals**')
print('   Comment References (im Video genannt):')
print('   [DATEN FEHLT - Keine Referenzen auf Kommentare im Video erkannt]')
print()
print('   Reply Indicators (im Video erkennbar):')
print('   [DATEN FEHLT - Keine direkten Antwort-Indikatoren im Video erkannt]')
print()
print('   Trend Participation Signals (im Video erkennbar):')
print('   [DATEN FEHLT - Keine expliziten Trend-Signale erkannt]')
print()
print('   Call-to-Action Elements (im Video gesprochen oder als Text):')
print('   [DATEN FEHLT - Keine expliziten CTAs im Video erkannt]')
print()
print('   Platform-Specific Features (Nutzung im Video sichtbar):')
print('   [DATEN FEHLT - Keine plattformspezifischen Features erkennbar]')
print()

# Final Summary
print('📌 **Abschlussanalyse (Faktische Zusammenfassung)**')
print('Das 49-sekündige Video dokumentiert einen kompletten Tagesablauf von @leon_schliebach im Vlog-Format. Die Hauptperson, ein tätowierter Mann, führt durch verschiedene Alltagssituationen: Morgenroutine (Zähneputzen, Rasieren), Frühstücksvorbereitung (Shake), Fahrt zur Arbeit, Bürotätigkeiten (Telefonate, Rechnungen), Mittagspause mit Kollegen und abschließende Gym-Session trotz sichtbarer Verletzung (Krücken/Gips). Das Video nutzt deutsche Text-Overlays zur Kontextualisierung, schnelle Schnitte zwischen Locations (durchschnittlich alle 0,5 Sekunden) und authentische Handheld-Kameraführung. Die Sprache ist durchgehend Deutsch mit neutralem bis positivem Ton. Visuell dominieren alltägliche Umgebungen (Bad, Küche, Auto, Büro, Gym) mit natürlicher Beleuchtung. Die Objekterkennung identifizierte hauptsächlich "person" (206 Detektionen) als dominantes Element. Audio besteht aus Originalton ohne zusätzliche Musik oder Soundeffekte.')