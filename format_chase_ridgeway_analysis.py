#!/usr/bin/env python3
"""
Format Chase Ridgeway video analysis according to the template
"""

import json
from datetime import datetime

def format_analysis(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    results = data['analyzer_results']
    
    # Start formatting the output
    output = []
    output.append("# CHASE RIDGEWAY VIDEO ANALYSE")
    output.append(f"\n**Video ID:** {metadata['video_filename'].replace('.mp4', '')}")
    output.append(f"**Analysedatum:** {metadata['analysis_timestamp']}")
    output.append(f"**Verarbeitungszeit:** {metadata['processing_time_seconds']:.2f} Sekunden")
    output.append(f"**Anzahl Analyzer:** {metadata['total_analyzers']}")
    output.append("\n---\n")
    
    # 1. SPEECH TRANSCRIPTION
    output.append("## 1. SPEECH TRANSCRIPTION (Whisper)")
    speech = results.get('speech_transcription', {})
    if speech.get('segments'):
        output.append("\n### Transkribierte Segmente:")
        for seg in speech['segments']:
            output.append(f"\n**[{seg['start_time']:.2f}s - {seg['end_time']:.2f}s]**")
            output.append(f"Text: \"{seg['text']}\"")
            output.append(f"Sprache: {seg['language']}")
            output.append(f"Konfidenz: {seg.get('confidence', 0):.2%}")
            if seg.get('words'):
                output.append("Wörter:")
                for word in seg['words']:
                    output.append(f"  - \"{word['word']}\" [{word['start']:.2f}s - {word['end']:.2f}s] (Prob: {word['probability']:.2%})")
    else:
        output.append("\n[DATEN FEHLT - Keine Sprachsegmente gefunden]")
    
    # 2. QWEN2-VL TEMPORAL DESCRIPTIONS
    output.append("\n## 2. QWEN2-VL TEMPORAL DESCRIPTIONS")
    qwen = results.get('qwen2_vl_temporal', {})
    if qwen.get('segments'):
        output.append("\n### Sekündliche Beschreibungen:")
        for seg in qwen['segments'][:30]:  # First 30 seconds
            output.append(f"\n**[{seg['start_time']:.0f}s - {seg['end_time']:.0f}s]**")
            output.append(f"{seg['description']}")
    else:
        output.append("\n[DATEN FEHLT - Keine temporalen Beschreibungen]")
    
    # 3. OBJECT DETECTION
    output.append("\n## 3. OBJECT DETECTION (YOLOv8)")
    objects = results.get('object_detection', {})
    if objects.get('segments'):
        # Group by object type
        obj_dict = {}
        for obj in objects['segments']:
            key = obj['object']
            if key not in obj_dict:
                obj_dict[key] = []
            obj_dict[key].append(obj)
        
        output.append("\n### Erkannte Objekte:")
        output.append(f"Gesamt: {objects.get('total_objects_detected', 0)} Objekte in {objects.get('total_frames_processed', 0)} Frames")
        output.append(f"Einzigartige Objekte: {objects.get('unique_objects', 0)}")
        
        for obj_type, instances in obj_dict.items():
            output.append(f"\n**{obj_type.upper()}:**")
            output.append(f"  - Kategorie: {instances[0].get('category', 'unknown')}")
            output.append(f"  - Vorkommen: {len(instances)} mal")
            # Show first 5 instances
            for inst in instances[:5]:
                output.append(f"    - Zeit: {inst['timestamp']:.1f}s | Konfidenz: {inst['confidence']:.2%} | Position: {inst['bbox']}")
            if len(instances) > 5:
                output.append(f"    - ... und {len(instances) - 5} weitere Vorkommen")
    else:
        output.append("\n[DATEN FEHLT - Keine Objekte erkannt]")
    
    # 4. VISUAL EFFECTS
    output.append("\n## 4. VISUAL EFFECTS ANALYSIS")
    vfx = results.get('visual_effects', {})
    if vfx.get('segments'):
        # Group effects by type
        effect_dict = {}
        for seg in vfx['segments']:
            if seg.get('effects'):
                for effect in seg['effects']:
                    key = effect
                    if key not in effect_dict:
                        effect_dict[key] = []
                    effect_dict[key].append(seg['timestamp'])
        
        if effect_dict:
            output.append("\n### Erkannte Effekte:")
            for effect_type, timestamps in effect_dict.items():
                output.append(f"\n**{effect_type.upper()}:**")
                output.append(f"  - Vorkommen: {len(timestamps)} mal")
                output.append(f"  - Zeitpunkte: {', '.join([f'{t:.1f}s' for t in timestamps[:5]])}")
                if len(timestamps) > 5:
                    output.append(f"    ... und {len(timestamps) - 5} weitere")
        else:
            output.append("\n[DATEN FEHLT - Keine visuellen Effekte erkannt]")
    else:
        output.append("\n[DATEN FEHLT - Keine visuellen Effekte erkannt]")
    
    # 5. TEXT OVERLAY
    output.append("\n## 5. TEXT OVERLAY DETECTION (EasyOCR)")
    text_overlay = results.get('text_overlay', {})
    if text_overlay.get('segments'):
        text_segments = [seg for seg in text_overlay['segments'] if seg.get('texts')]
        if text_segments:
            output.append("\n### Erkannte Texte:")
            output.append(f"Gesamt: {len(text_segments)} Frames mit Text")
            
            # Show first 10 text detections
            for seg in text_segments[:10]:
                output.append(f"\n**Zeit: {seg['timestamp']:.1f}s**")
                for text_data in seg['texts']:
                    output.append(f"  - Text: \"{text_data['text']}\"")
                    output.append(f"    - Konfidenz: {text_data['confidence']:.2%}")
                    if 'bbox' in text_data:
                        output.append(f"    - Position: {text_data['bbox']}")
            
            if len(text_segments) > 10:
                output.append(f"\n... und {len(text_segments) - 10} weitere Frames mit Text")
        else:
            output.append("\n[DATEN FEHLT - Keine Textoverlays erkannt]")
    else:
        output.append("\n[DATEN FEHLT - Keine Textoverlays erkannt]")
    
    # 6. CAMERA MOVEMENTS
    output.append("\n## 6. CAMERA MOVEMENTS")
    camera = results.get('camera_analysis', {})
    if camera.get('segments'):
        output.append("\n### Kamerabewegungen:")
        output.append(f"Gesamt: {len(camera['segments'])} analysierte Segmente")
        
        # Group by movement type
        move_dict = {}
        for seg in camera['segments']:
            key = seg.get('movement', 'unknown')
            if key not in move_dict:
                move_dict[key] = []
            move_dict[key].append(seg)
        
        for move_type, segments in move_dict.items():
            output.append(f"\n**{move_type.upper()}:**")
            output.append(f"  - Vorkommen: {len(segments)} Segmente")
            # Show first 3 examples
            for seg in segments[:3]:
                output.append(f"    - [{seg['start_time']:.1f}s - {seg['end_time']:.1f}s] {seg.get('description', 'no description')}")
                if seg.get('shot_type'):
                    output.append(f"      Shot: {seg['shot_type'].get('type', 'unknown')}")
    else:
        output.append("\n[DATEN FEHLT - Keine Kamerabewegungen erkannt]")
    
    # 7. AUDIO ANALYSIS
    output.append("\n## 7. AUDIO ANALYSIS")
    audio = results.get('audio_analysis', {})
    if audio:
        output.append("\n### Audio-Metriken:")
        if 'energy' in audio:
            output.append(f"  - Durchschnittliche Energie: {audio['energy'].get('mean', 0):.2f}")
        if 'tempo' in audio:
            output.append(f"  - Tempo: {audio['tempo']:.1f} BPM")
        if 'key' in audio:
            output.append(f"  - Tonart: {audio['key']}")
    
    # Speech emotion
    speech_emotion = results.get('speech_emotion', {})
    if speech_emotion.get('segments'):
        output.append("\n### Sprach-Emotionen:")
        for seg in speech_emotion['segments']:
            output.append(f"\n**[{seg['timestamp']:.1f}s - Dauer: {seg.get('duration', 0):.1f}s]**")
            output.append(f"  - Dominante Emotion: {seg.get('dominant_emotion', 'unknown')}")
            output.append(f"  - Dominante Emotion (DE): {seg.get('dominant_emotion_de', 'unbekannt')}")
            output.append(f"  - Konfidenz: {seg.get('confidence', 0):.2%}")
            if seg.get('emotions'):
                output.append("  - Emotionswerte:")
                for emotion, score in seg['emotions'].items():
                    output.append(f"    - {emotion}: {score:.2%}")
    else:
        output.append("\n[DATEN FEHLT - Keine Sprach-Emotionen erkannt]")
    
    # 8. EYE TRACKING
    output.append("\n## 8. EYE TRACKING DATA")
    eye_tracking = results.get('eye_tracking', {})
    if eye_tracking.get('segments'):
        eye_segs = [seg for seg in eye_tracking['segments'] if seg.get('eye_data')]
        if eye_segs:
            output.append("\n### Eye Tracking Daten:")
            output.append(f"Gesamt: {len(eye_segs)} Frames mit Eye-Tracking")
            
            # Show first 10 segments
            for seg in eye_segs[:10]:
                output.append(f"\n**Zeit: {seg['timestamp']:.1f}s**")
                if seg.get('eye_data', {}).get('left_eye'):
                    left = seg['eye_data']['left_eye']
                    output.append(f"  - Linkes Auge: offen={left.get('is_open', False)}, Konfidenz={left.get('confidence', 0):.2%}")
                if seg.get('eye_data', {}).get('right_eye'):
                    right = seg['eye_data']['right_eye']
                    output.append(f"  - Rechtes Auge: offen={right.get('is_open', False)}, Konfidenz={right.get('confidence', 0):.2%}")
                if seg.get('gaze_direction'):
                    output.append(f"  - Blickrichtung: {seg['gaze_direction']}")
            
            if len(eye_segs) > 10:
                output.append(f"\n... und {len(eye_segs) - 10} weitere Frames mit Eye-Tracking")
        else:
            output.append("\n[DATEN FEHLT - Keine Eye-Tracking-Daten mit Augendaten]")
    else:
        output.append("\n[DATEN FEHLT - Keine Eye-Tracking-Daten]")
    
    # 9. PRODUCT DETECTION
    output.append("\n## 9. PRODUCT DETECTION")
    products = results.get('product_detection', {})
    if products.get('segments'):
        # Extract all product detections
        all_products = []
        for seg in products['segments']:
            if seg.get('products'):
                for prod in seg['products']:
                    prod['timestamp'] = seg['timestamp']
                    all_products.append(prod)
        
        if all_products:
            # Group by product type
            prod_dict = {}
            for prod in all_products:
                key = prod['product']
                if key not in prod_dict:
                    prod_dict[key] = []
                prod_dict[key].append(prod)
            
            output.append("\n### Erkannte Produkte:")
            output.append(f"Gesamt: {len(all_products)} Produkterkennungen")
            
            for prod_type, instances in prod_dict.items():
                output.append(f"\n**{prod_type.upper()}:**")
                output.append(f"  - Kategorie: {instances[0].get('category', 'unknown')}")
                output.append(f"  - Vorkommen: {len(instances)} mal")
                # Show first 3 instances
                for inst in instances[:3]:
                    output.append(f"    - Zeit: {inst['timestamp']:.1f}s | Konfidenz: {inst['confidence']:.2%}")
                if len(instances) > 3:
                    output.append(f"    - ... und {len(instances) - 3} weitere")
        else:
            output.append("\n[DATEN FEHLT - Keine Produkte in Segmenten gefunden]")
    else:
        output.append("\n[DATEN FEHLT - Keine Produkte erkannt]")
    
    # 10. SCENE SEGMENTATION
    output.append("\n## 10. SCENE SEGMENTATION")
    scenes = results.get('scene_segmentation', {})
    if scenes.get('scenes'):
        output.append("\n### Szenen:")
        for i, scene in enumerate(scenes['scenes']):
            output.append(f"\n**Szene {i+1}: [{scene['start_time']:.1f}s - {scene['end_time']:.1f}s]**")
            output.append(f"  - Dauer: {scene['duration']:.1f}s")
            output.append(f"  - Typ: {scene.get('scene_type', 'unknown')}")
            if scene.get('confidence'):
                output.append(f"  - Konfidenz: {scene['confidence']:.2%}")
    else:
        output.append("\n[DATEN FEHLT - Keine Szenensegmentierung]")
    
    # 11. COLOR ANALYSIS
    output.append("\n## 11. COLOR ANALYSIS")
    colors = results.get('color_analysis', {})
    if colors.get('segments'):
        # Extract color data from segments
        all_colors = {}
        for seg in colors['segments']:
            if seg.get('dominant_colors'):
                for color in seg['dominant_colors']:
                    color_key = tuple(color['color'])
                    if color_key not in all_colors:
                        all_colors[color_key] = {'count': 0, 'percentage': 0}
                    all_colors[color_key]['count'] += 1
                    all_colors[color_key]['percentage'] += color['percentage']
        
        if all_colors:
            output.append("\n### Dominante Farben (über alle Frames):")
            # Sort by frequency
            sorted_colors = sorted(all_colors.items(), key=lambda x: x[1]['count'], reverse=True)
            for color_rgb, data in sorted_colors[:5]:
                avg_percentage = data['percentage'] / data['count']
                output.append(f"  - RGB({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}) - {data['count']} Frames, Ø {avg_percentage:.1%} pro Frame")
        
        # Check for color harmony in segments
        harmonies = {}
        for seg in colors['segments']:
            if seg.get('color_harmony'):
                harmony = seg['color_harmony']
                harmonies[harmony] = harmonies.get(harmony, 0) + 1
        
        if harmonies:
            output.append("\n### Farbharmonien:")
            for harmony, count in sorted(harmonies.items(), key=lambda x: x[1], reverse=True):
                output.append(f"  - {harmony}: {count} Frames")
    else:
        output.append("\n[DATEN FEHLT - Keine Farbanalyse]")
    
    # 12. SUMMARY STATISTICS
    output.append("\n## 12. ZUSAMMENFASSENDE STATISTIKEN")
    
    # Get video duration from temporal_flow
    temporal = results.get('temporal_flow', {})
    duration = temporal.get('video_info', {}).get('duration', 0)
    if duration == 0 and temporal.get('segments'):
        # Calculate from segments
        last_seg = max(temporal['segments'], key=lambda x: x.get('end_time', 0))
        duration = last_seg.get('end_time', 0)
    
    output.append(f"\n- **Videodauer:** {duration:.1f} Sekunden")
    output.append(f"- **Erfolgreiche Analyzer:** {metadata['successful_analyzers']}/{metadata['total_analyzers']}")
    output.append(f"- **Rekonstruktions-Score:** {metadata['reconstruction_score']:.1f}%")
    output.append(f"- **Echtzeit-Faktor:** {metadata['realtime_factor']:.2f}x")
    
    # Additional analyzers summary
    output.append("\n### Weitere verfügbare Analyzer:")
    for analyzer in ['background_segmentation', 'content_quality', 'cut_analysis', 
                     'age_estimation', 'sound_effects', 'speech_rate', 'comment_cta_detection',
                     'audio_environment', 'speech_flow', 'temporal_flow']:
        if analyzer in results and results[analyzer]:
            output.append(f"  - **{analyzer}:** ✓ Daten vorhanden")
        else:
            output.append(f"  - **{analyzer}:** ✗ Keine Daten")
    
    return '\n'.join(output)

if __name__ == "__main__":
    json_path = "/home/user/tiktok_production/results/7522589683939921165_multiprocess_20250708_142055.json"
    formatted = format_analysis(json_path)
    
    # Save to file
    output_path = "/home/user/tiktok_production/chase_ridgeway_formatted_analysis.md"
    with open(output_path, 'w') as f:
        f.write(formatted)
    
    print(f"Formatted analysis saved to: {output_path}")
    print(f"\nTotal length: {len(formatted)} characters")