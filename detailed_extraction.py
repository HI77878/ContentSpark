#!/usr/bin/env python3
import json
import sys
from collections import defaultdict

def extract_detailed_analysis(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = data['analyzer_results']
    
    # Create formatted report
    report = []
    report.append("=== DETAILED VIDEO ANALYSIS EXTRACTION ===")
    report.append(f"\nVideo: {data['metadata']['video_filename']}")
    report.append(f"Duration: ~{data['metadata']['processing_time_seconds'] / data['metadata']['realtime_factor']:.1f} seconds")
    report.append(f"Analysis Date: {data['metadata']['analysis_timestamp']}")
    report.append(f"Processing Time: {data['metadata']['processing_time_seconds']:.2f}s")
    report.append(f"Analyzers: {data['metadata']['successful_analyzers']}/{data['metadata']['total_analyzers']} successful\n")
    
    # 1. Speech Transcription with timestamps
    report.append("1. SPEECH TRANSCRIPTION SEGMENTS")
    speech = results.get('speech_transcription', {})
    segments = speech.get('segments', [])
    if segments:
        for seg in segments:
            report.append(f"  [{seg['start']:.2f}s - {seg['end']:.2f}s]: \"{seg['text']}\"")
    else:
        report.append("  [No speech detected in video]")
    report.append(f"  • Language: {speech.get('language', 'Not detected')}")
    report.append(f"  • Speaking Rate: {speech.get('speaking_rate_wpm', 0)} WPM")
    report.append(f"  • Voice Type: {speech.get('pitch_category', 'Unknown')}")
    report.append(f"  • Pitch Mean: {speech.get('pitch_analysis', {}).get('pitch_mean_hz', 0):.1f} Hz")
    report.append("")
    
    # 2. Video-LLaVA descriptions
    report.append("2. BLIP-2 VIDEO DESCRIPTIONS (Video-LLaVA)")
    video_llava = results.get('video_llava', {})
    if video_llava.get('segments'):
        for seg in video_llava['segments']:
            report.append(f"  [{seg.get('timestamp', 0):.2f}s]: {seg.get('description', 'No description')}")
    else:
        report.append("  [Video-LLaVA analysis not available or failed]")
    if video_llava.get('summary'):
        report.append(f"  • Summary: {video_llava['summary']}")
    report.append("")
    
    # 3. Text Overlays
    report.append("3. TEXT OVERLAYS DETECTED")
    text_overlay = results.get('text_overlay', {})
    text_segs = text_overlay.get('segments', [])
    if text_segs:
        for seg in text_segs:
            text = seg.get('text', '').strip()
            if text:
                report.append(f"  [{seg['timestamp']:.2f}s]: \"{text}\"")
                report.append(f"    Position: {seg.get('position', 'Unknown')}")
                report.append(f"    BBox: {seg.get('bbox', [])}")
            else:
                report.append(f"  [{seg['timestamp']:.2f}s]: [Empty or unreadable text detected]")
    else:
        report.append("  [No text overlays detected]")
    report.append("")
    
    # 4. Object Detection Results
    report.append("4. OBJECT DETECTION RESULTS")
    obj_det = results.get('object_detection', {})
    obj_segs = obj_det.get('segments', [])
    if obj_segs:
        # Group by timestamp for cleaner display
        objects_by_time = defaultdict(list)
        for seg in obj_segs:
            ts = seg['timestamp']
            objects_by_time[ts].append({
                'object': seg['object'],
                'confidence': seg['confidence'],
                'bbox': seg.get('bbox', [])
            })
        
        # Show first 10 timestamps
        sorted_times = sorted(objects_by_time.keys())
        for i, ts in enumerate(sorted_times[:10]):
            objs = objects_by_time[ts]
            obj_list = [f"{o['object']} ({o['confidence']:.2f})" for o in objs]
            report.append(f"  [{ts:.2f}s]: {', '.join(obj_list)}")
        
        if len(sorted_times) > 10:
            report.append(f"  ... and {len(sorted_times) - 10} more timestamps with objects")
        
        # Object summary
        if obj_det.get('summary'):
            summary = obj_det['summary']
            report.append(f"\n  Object Summary:")
            if summary.get('unique_objects'):
                report.append(f"    • Unique objects: {', '.join(summary['unique_objects'])}")
            if summary.get('total_detections'):
                report.append(f"    • Total detections: {summary['total_detections']}")
    else:
        report.append("  [No objects detected]")
    report.append("")
    
    # 5. Visual Effects and Camera Movements
    report.append("5. VISUAL EFFECTS AND CAMERA MOVEMENTS")
    
    # Visual Effects
    report.append("  A. Visual Effects:")
    visual_fx = results.get('visual_effects', {})
    vfx_segs = visual_fx.get('segments', [])
    if vfx_segs:
        for seg in vfx_segs[:10]:
            effect = seg.get('effect', 'Unknown effect')
            if effect:
                report.append(f"    [{seg['timestamp']:.2f}s]: {effect} (confidence: {seg.get('confidence', 0):.2f})")
        if len(vfx_segs) > 10:
            report.append(f"    ... and {len(vfx_segs) - 10} more effects")
    else:
        report.append("    [No visual effects detected]")
    
    # Camera Movements
    report.append("\n  B. Camera Movements:")
    camera = results.get('camera_analysis', {})
    cam_segs = camera.get('segments', [])
    if cam_segs:
        for seg in cam_segs[:10]:
            movement = seg.get('movement', 'Unknown movement')
            if movement:
                report.append(f"    [{seg['timestamp']:.2f}s]: {movement} (confidence: {seg.get('confidence', 0):.2f})")
        if len(cam_segs) > 10:
            report.append(f"    ... and {len(cam_segs) - 10} more movements")
    else:
        report.append("    [No camera movements detected]")
    report.append("")
    
    # 6. Cut Analysis
    report.append("6. CUT ANALYSIS DATA")
    cuts = results.get('cut_analysis', {})
    cut_segs = cuts.get('segments', [])
    if cut_segs:
        report.append(f"  Total cuts: {len(cut_segs)}")
        report.append("  Cut timestamps:")
        for i, seg in enumerate(cut_segs[:15]):
            report.append(f"    Cut {i+1}: {seg['timestamp']:.2f}s")
        if len(cut_segs) > 15:
            report.append(f"    ... and {len(cut_segs) - 15} more cuts")
        
        # Statistics
        stats = cuts.get('statistics', {})
        if stats:
            report.append(f"\n  Statistics:")
            report.append(f"    • Average shot duration: {stats.get('avg_shot_duration', 0):.2f}s")
            report.append(f"    • Shortest shot: {stats.get('min_shot_duration', 0):.2f}s")
            report.append(f"    • Longest shot: {stats.get('max_shot_duration', 0):.2f}s")
            report.append(f"    • Cut rate: {stats.get('cut_rate_per_minute', 0):.1f} cuts/min")
    else:
        report.append("  [No cuts detected]")
    report.append("")
    
    # 7. Face/Emotion Detection
    report.append("7. FACE/EMOTION DETECTION")
    
    # Eye tracking
    report.append("  A. Eye Tracking:")
    eye_track = results.get('eye_tracking', {})
    eye_segs = eye_track.get('segments', [])
    if eye_segs:
        for seg in eye_segs[:5]:
            report.append(f"    [{seg['timestamp']:.2f}s]: Face {seg.get('face_id', 0)}, Gaze: {seg.get('gaze_direction', 'Unknown')}")
    else:
        report.append("    [No eye tracking data]")
    
    # Age estimation
    report.append("\n  B. Age Estimation:")
    age_est = results.get('age_estimation', {})
    age_segs = age_est.get('segments', [])
    if age_segs:
        for seg in age_segs[:5]:
            if seg.get('age', 0) > 0:
                report.append(f"    [{seg['timestamp']:.2f}s]: Face {seg.get('face_id', 0)}, Age: {seg.get('age', 0)}")
    else:
        report.append("    [No age estimation data]")
    
    # Speech emotion
    report.append("\n  C. Speech Emotion:")
    speech_emo = results.get('speech_emotion', {})
    emo_segs = speech_emo.get('segments', [])
    if emo_segs:
        for seg in emo_segs[:5]:
            emotion = seg.get('emotion', 'Unknown')
            if emotion:
                report.append(f"    [{seg['timestamp']:.2f}s]: {emotion} (confidence: {seg.get('confidence', 0):.2f})")
    else:
        report.append("    [No speech emotion data]")
    report.append("")
    
    # 8. Audio Analysis
    report.append("8. AUDIO ANALYSIS")
    
    # General audio analysis
    report.append("  A. General Audio Features:")
    audio_analysis = results.get('audio_analysis', {})
    audio_segs = audio_analysis.get('segments', [])
    if audio_segs:
        for seg in audio_segs[:3]:
            ts = seg.get('timestamp', seg.get('time', 0))
            report.append(f"    [{ts:.2f}s]: Energy: {seg.get('energy', 0):.2f}, Tempo: {seg.get('tempo', 0):.1f}")
    
    # Sound effects
    report.append("\n  B. Sound Effects:")
    sound_fx = results.get('sound_effects', {})
    sfx_segs = sound_fx.get('segments', [])
    if sfx_segs:
        for seg in sfx_segs[:10]:
            effect = seg.get('effect', 'Unknown')
            if effect:
                report.append(f"    [{seg['timestamp']:.2f}s]: {effect} (confidence: {seg.get('confidence', 0):.2f})")
    else:
        report.append("    [No sound effects detected]")
    
    # Audio environment
    report.append("\n  C. Audio Environment:")
    audio_env = results.get('audio_environment', {})
    env_segs = audio_env.get('segments', [])
    if env_segs:
        for seg in env_segs[:5]:
            env = seg.get('environment', 'Unknown')
            if env:
                report.append(f"    [{seg['timestamp']:.2f}s]: {env}")
    else:
        report.append("    [No audio environment data]")
    report.append("")
    
    # 9. Scene Segmentation
    report.append("9. SCENE SEGMENTATION")
    scene_seg = results.get('scene_segmentation', {})
    scene_segs = scene_seg.get('segments', [])
    if scene_segs:
        report.append(f"  Total scenes: {len(scene_segs)}")
        for i, seg in enumerate(scene_segs[:10]):
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            duration = seg.get('duration', end - start)
            report.append(f"  Scene {i+1}: {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
            if seg.get('description'):
                report.append(f"    Description: {seg['description']}")
            if seg.get('dominant_color'):
                report.append(f"    Dominant color: {seg['dominant_color']}")
        if len(scene_segs) > 10:
            report.append(f"  ... and {len(scene_segs) - 10} more scenes")
    else:
        report.append("  [No scene segmentation data]")
    report.append("")
    
    # 10. Other Relevant Analyzers
    report.append("10. OTHER RELEVANT ANALYZER RESULTS")
    
    # Composition Analysis
    report.append("  A. Composition Analysis:")
    comp = results.get('composition_analysis', {})
    if comp.get('segments'):
        seg = comp['segments'][0] if comp['segments'] else {}
        report.append(f"    Overall score: {seg.get('composition_score', 0):.2f}")
        report.append(f"    Rule of thirds: {seg.get('rule_of_thirds', 0):.2f}")
        report.append(f"    Balance: {seg.get('balance', 0):.2f}")
    
    # Color Analysis
    report.append("\n  B. Color Analysis:")
    color = results.get('color_analysis', {})
    if color.get('segments'):
        for seg in color['segments'][:3]:
            report.append(f"    [{seg['timestamp']:.2f}s]: Dominant: {seg.get('dominant_color', 'Unknown')}, Palette: {seg.get('palette', [])}")
    
    # Content Quality
    report.append("\n  C. Content Quality:")
    quality = results.get('content_quality', {})
    if quality.get('segments'):
        seg = quality['segments'][0] if quality['segments'] else {}
        report.append(f"    Overall quality: {seg.get('overall_quality', 0):.2f}")
        report.append(f"    Technical score: {seg.get('technical_score', 0):.2f}")
        report.append(f"    Aesthetic score: {seg.get('aesthetic_score', 0):.2f}")
    
    # Product Detection
    report.append("\n  D. Product Detection:")
    products = results.get('product_detection', {})
    prod_segs = products.get('segments', [])
    if prod_segs:
        unique_products = set(seg['product'] for seg in prod_segs)
        report.append(f"    Unique products detected: {', '.join(unique_products)}")
        for seg in prod_segs[:5]:
            report.append(f"    [{seg['timestamp']:.2f}s]: {seg['product']} (confidence: {seg['confidence']:.2f})")
    
    report.append("\n" + "="*50)
    
    # Summary of data availability
    report.append("\nDATA AVAILABILITY SUMMARY:")
    report.append("✓ Analyzers with data:")
    for analyzer, data in results.items():
        if data and isinstance(data, dict) and (data.get('segments') or data.get('summary')):
            report.append(f"  • {analyzer}")
    
    report.append("\n✗ Analyzers with no/empty data:")
    for analyzer, data in results.items():
        if not data or (isinstance(data, dict) and not data.get('segments') and not data.get('summary')):
            report.append(f"  • {analyzer}")
    
    return "\n".join(report)

if __name__ == "__main__":
    json_path = "/home/user/tiktok_production/results/7425998222721633569_multiprocess_20250705_071729.json"
    report = extract_detailed_analysis(json_path)
    print(report)