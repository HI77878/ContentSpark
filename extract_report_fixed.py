#!/usr/bin/env python3
"""Extract and format analysis results from JSON file"""

import json
import sys
from datetime import datetime

def format_time(seconds):
    """Convert seconds to mm:ss format"""
    if seconds is None:
        return "N/A"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def extract_analyzer_results(json_path):
    """Extract and format results from each analyzer"""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("="*80)
    print(f"VIDEO ANALYSIS REPORT")
    print("="*80)
    
    # Metadata
    metadata = data.get('metadata', {})
    print(f"\nVideo Path: {metadata.get('video_path', 'N/A')}")
    print(f"Video Filename: {metadata.get('video_filename', 'N/A')}")
    print(f"Analysis Timestamp: {metadata.get('analysis_timestamp', 'N/A')}")
    print(f"Processing Time: {metadata.get('processing_time_seconds', 0):.2f} seconds")
    print(f"Total Analyzers: {metadata.get('total_analyzers', 0)}")
    print(f"Successful Analyzers: {metadata.get('successful_analyzers', 0)}")
    print(f"Reconstruction Score: {metadata.get('reconstruction_score', 0):.1f}%")
    print(f"Realtime Factor: {metadata.get('realtime_factor', 0):.2f}x")
    print(f"API Version: {metadata.get('api_version', 'N/A')}")
    
    # Performance metrics
    perf = data.get('performance_metrics', {})
    if perf:
        print(f"\nPerformance:")
        print(f"  Total Time: {perf.get('total_time', 0):.2f}s")
        print(f"  Speed Ratio: {perf.get('speed_ratio', 0):.2f}x realtime")
        print(f"  Peak GPU Memory: {perf.get('peak_gpu_memory_gb', 0):.2f} GB")
        print(f"  Avg GPU Utilization: {perf.get('avg_gpu_utilization', 0):.1f}%")
    
    print("\n" + "="*80)
    print("ANALYZER RESULTS")
    print("="*80)
    
    # Analyzer results
    analyzers = data.get('analyzer_results', {})
    
    # Group analyzers by category
    categories = {
        'Visual Analysis': ['object_detection', 'composition_analysis', 'visual_effects', 
                           'camera_analysis', 'color_analysis', 'background_segmentation'],
        'Content Understanding': ['video_llava', 'text_overlay', 'product_detection',
                                'content_quality', 'scene_segmentation'],
        'Human Analysis': ['age_estimation', 'eye_tracking'],
        'Production Analysis': ['cut_analysis', 'temporal_flow'],
        'Audio Analysis': ['speech_transcription', 'speech_emotion', 'speech_rate',
                          'audio_analysis', 'audio_environment', 'sound_effects']
    }
    
    for category, analyzer_names in categories.items():
        print(f"\n{category.upper()}")
        print("-" * len(category))
        
        for analyzer_name in analyzer_names:
            if analyzer_name in analyzers:
                result = analyzers[analyzer_name]
                print(f"\n{analyzer_name}:")
                
                # Extract key insights based on analyzer type
                if analyzer_name == 'object_detection':
                    segments = result.get('segments', [])
                    objects = {}
                    for seg in segments:
                        for obj in seg.get('objects', []):
                            label = obj['label']
                            objects[label] = objects.get(label, 0) + 1
                    if objects:
                        print(f"  Objects detected: {', '.join([f'{k} ({v}x)' for k, v in sorted(objects.items(), key=lambda x: x[1], reverse=True)[:5]])}")
                
                elif analyzer_name == 'face_detection':
                    segments = result.get('segments', [])
                    total_faces = sum(seg.get('face_count', 0) for seg in segments)
                    print(f"  Total face detections: {total_faces}")
                    if segments:
                        max_faces = max(seg.get('face_count', 0) for seg in segments)
                        print(f"  Max faces in frame: {max_faces}")
                
                elif analyzer_name == 'video_llava':
                    summary = result.get('summary', {})
                    segments = result.get('segments', [])
                    if summary:
                        print(f"  Main Activity: {summary.get('main_activity', 'N/A')}")
                        print(f"  Setting: {summary.get('setting', 'N/A')}")
                        print(f"  Key Objects: {', '.join(summary.get('key_objects', [])[:5])}")
                        narr = summary.get('narrative_description', '')
                        if narr:
                            print(f"  Description: {narr[:150]}...")
                    if segments:
                        print(f"  Video segments analyzed: {len(segments)}")
                
                elif analyzer_name == 'speech_transcription':
                    segments = result.get('segments', [])
                    # Get full transcript from segments
                    transcript = ' '.join(seg.get('text', '') for seg in segments if seg.get('text'))
                    if transcript:
                        print(f"  Transcript: \"{transcript[:200]}...\"" if len(transcript) > 200 else f"  Transcript: \"{transcript}\"")
                    
                    # Additional speech info
                    if 'speaking_rate_wpm' in result:
                        print(f"  Speaking rate: {result['speaking_rate_wpm']:.1f} words/minute")
                    if 'pitch_category' in result:
                        print(f"  Pitch: {result['pitch_category']}")
                    if 'emphasized_words' in result and result['emphasized_words']:
                        emphasized = result['emphasized_words']
                        if isinstance(emphasized[0], dict):
                            words = [w.get('word', '') for w in emphasized[:5]]
                        else:
                            words = emphasized[:5]
                        print(f"  Emphasized words: {', '.join(words)}")
                
                elif analyzer_name == 'emotion_detection':
                    emotion_summary = result.get('emotion_summary', {})
                    if emotion_summary:
                        dominant = emotion_summary.get('dominant_emotion', 'N/A')
                        print(f"  Dominant emotion: {dominant}")
                        dist = emotion_summary.get('emotion_distribution', {})
                        if dist:
                            top_emotions = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
                            print(f"  Top emotions: {', '.join([f'{e[0]} ({e[1]:.1f}%)' for e in top_emotions])}")
                
                elif analyzer_name == 'text_overlay':
                    segments = result.get('segments', [])
                    all_text = []
                    for seg in segments:
                        for text_item in seg.get('texts', []):
                            text = text_item.get('text', '')
                            if text and text not in all_text:
                                all_text.append(text)
                    if all_text:
                        print(f"  Text found: {', '.join(all_text[:5])}")
                        if len(all_text) > 5:
                            print(f"  ... and {len(all_text) - 5} more")
                    else:
                        print(f"  Text segments analyzed: {len(segments)}")
                
                elif analyzer_name == 'music_analysis':
                    segments = result.get('segments', [])
                    if segments:
                        avg_tempo = sum(s.get('tempo', 0) for s in segments) / len(segments)
                        print(f"  Average tempo: {avg_tempo:.1f} BPM")
                        # Get unique keys
                        keys = list(set(s.get('key', '') for s in segments if s.get('key')))
                        if keys:
                            print(f"  Musical keys: {', '.join(keys[:3])}")
                
                elif analyzer_name == 'scene_classification':
                    segments = result.get('segments', [])
                    scenes = {}
                    for seg in segments:
                        scene = seg.get('scene', 'unknown')
                        scenes[scene] = scenes.get(scene, 0) + 1
                    if scenes:
                        top_scenes = sorted(scenes.items(), key=lambda x: x[1], reverse=True)[:3]
                        print(f"  Top scenes: {', '.join([f'{s[0]} ({s[1]}x)' for s in top_scenes])}")
                
                elif analyzer_name == 'cut_analysis':
                    cuts = result.get('cuts', [])
                    segments = result.get('segments', [])
                    if cuts:
                        print(f"  Total cuts: {len(cuts)}")
                        if cuts:
                            avg_duration = sum(c.get('duration', 0) for c in cuts) / len(cuts)
                            print(f"  Average shot duration: {avg_duration:.1f}s")
                    elif segments:
                        print(f"  Cut segments analyzed: {len(segments)}")
                
                elif analyzer_name == 'camera_analysis':
                    segments = result.get('segments', [])
                    techniques = result.get('camera_techniques', {})
                    if techniques:
                        print(f"  Techniques found: {', '.join(techniques.keys())}")
                    elif segments:
                        print(f"  Camera segments analyzed: {len(segments)}")
                        # Extract camera movements
                        movements = []
                        for seg in segments:
                            if 'camera_movement' in seg:
                                movement = seg['camera_movement']
                                if isinstance(movement, str):
                                    movements.append(movement)
                                elif isinstance(movement, dict):
                                    movements.append(movement.get('type', 'unknown'))
                        if movements:
                            unique_movements = list(set(movements))
                            print(f"  Camera movements: {', '.join(unique_movements[:5])}")
                
                elif analyzer_name == 'body_pose':
                    segments = result.get('segments', [])
                    total_poses = sum(len(seg.get('poses', [])) for seg in segments)
                    print(f"  Total pose detections: {total_poses}")
                
                elif analyzer_name == 'content_quality':
                    segments = result.get('segments', [])
                    if segments:
                        print(f"  Quality segments analyzed: {len(segments)}")
                        if segments[0]:
                            first = segments[0]
                            for key in ['quality_score', 'clarity', 'lighting_quality']:
                                if key in first:
                                    avg_val = sum(s.get(key, 0) for s in segments) / len(segments)
                                    print(f"  Average {key}: {avg_val:.2f}")
                                    break
                
                elif analyzer_name == 'color_analysis':
                    segments = result.get('segments', [])
                    if segments:
                        print(f"  Color segments analyzed: {len(segments)}")
                        # Extract dominant colors
                        all_colors = []
                        for seg in segments:
                            if 'dominant_colors' in seg:
                                colors = seg['dominant_colors']
                                if isinstance(colors, list):
                                    for color in colors:
                                        if isinstance(color, dict) and 'hex' in color:
                                            all_colors.append(color['hex'])
                                        elif isinstance(color, str):
                                            all_colors.append(color)
                        if all_colors:
                            print(f"  Sample dominant colors: {', '.join(all_colors[:5])}")
                
                elif analyzer_name == 'audio_analysis' or analyzer_name == 'audio_environment':
                    segments = result.get('segments', [])
                    if segments:
                        print(f"  Audio segments analyzed: {len(segments)}")
                        # Get audio characteristics
                        if segments[0]:
                            for key in ['energy', 'loudness', 'tempo', 'environment']:
                                if key in segments[0]:
                                    values = [s.get(key, 0) for s in segments if key in s]
                                    if values:
                                        avg_val = sum(values) / len(values) if isinstance(values[0], (int, float)) else values[0]
                                        print(f"  {key.capitalize()}: {avg_val}")
                
                elif analyzer_name == 'speech_emotion':
                    segments = result.get('segments', [])
                    if segments:
                        print(f"  Emotion segments analyzed: {len(segments)}")
                        # Extract emotions
                        emotions = {}
                        for seg in segments:
                            emotion = seg.get('emotion', seg.get('predicted_emotion', 'unknown'))
                            if emotion:
                                emotions[emotion] = emotions.get(emotion, 0) + 1
                        if emotions:
                            top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                            print(f"  Detected emotions: {', '.join([f'{e[0]} ({e[1]})' for e in top_emotions])}")
                
                elif analyzer_name == 'speech_rate':
                    segments = result.get('segments', [])
                    if segments:
                        rates = [s.get('words_per_minute', 0) for s in segments if 'words_per_minute' in s]
                        if rates:
                            avg_rate = sum(rates) / len(rates)
                            print(f"  Average speech rate: {avg_rate:.1f} words/minute")
                        print(f"  Speech segments: {len(segments)}")
                
                elif analyzer_name == 'temporal_flow':
                    segments = result.get('segments', [])
                    if segments:
                        print(f"  Temporal segments analyzed: {len(segments)}")
                        # Look for pacing or rhythm info
                        if segments[0]:
                            for key in ['pacing', 'rhythm', 'flow_score']:
                                if key in segments[0]:
                                    print(f"  {key.capitalize()} tracked")
                
                elif analyzer_name == 'scene_segmentation':
                    segments = result.get('segments', [])
                    if segments:
                        print(f"  Scene segments detected: {len(segments)}")
                        # Extract scene types
                        scene_types = {}
                        for seg in segments:
                            scene_type = seg.get('scene_type', seg.get('category', 'unknown'))
                            if scene_type:
                                scene_types[scene_type] = scene_types.get(scene_type, 0) + 1
                        if scene_types:
                            print(f"  Scene types: {', '.join([f'{k} ({v})' for k, v in list(scene_types.items())[:3]])}")
                
                elif analyzer_name == 'sound_effects':
                    segments = result.get('segments', [])
                    if segments:
                        print(f"  Sound effect segments: {len(segments)}")
                        # Extract detected effects
                        effects = []
                        for seg in segments:
                            if 'sound_type' in seg:
                                effects.append(seg['sound_type'])
                            elif 'effects' in seg:
                                effects.extend(seg['effects'])
                        if effects:
                            unique_effects = list(set(effects))
                            print(f"  Effects detected: {', '.join(unique_effects[:5])}")
                
                elif analyzer_name == 'age_estimation':
                    segments = result.get('segments', [])
                    if segments:
                        print(f"  Age estimation segments: {len(segments)}")
                        # Extract age ranges
                        ages = []
                        for seg in segments:
                            if 'age_range' in seg:
                                ages.append(seg['age_range'])
                            elif 'estimated_age' in seg:
                                ages.append(f"{seg['estimated_age']} years")
                        if ages:
                            print(f"  Age estimates: {', '.join(ages[:3])}")
                
                elif analyzer_name == 'eye_tracking':
                    segments = result.get('segments', [])
                    if segments:
                        print(f"  Eye tracking segments: {len(segments)}")
                        # Extract gaze patterns
                        gaze_areas = []
                        for seg in segments:
                            if 'gaze_direction' in seg:
                                gaze_areas.append(seg['gaze_direction'])
                            elif 'focus_area' in seg:
                                gaze_areas.append(seg['focus_area'])
                        if gaze_areas:
                            unique_gaze = list(set(gaze_areas))
                            print(f"  Gaze patterns: {', '.join(unique_gaze[:3])}")
                
                elif analyzer_name == 'product_detection':
                    segments = result.get('segments', [])
                    products = {}
                    for seg in segments:
                        for prod in seg.get('products', []):
                            label = prod.get('label', prod.get('category', 'unknown'))
                            products[label] = products.get(label, 0) + 1
                    if products:
                        print(f"  Products detected: {', '.join([f'{k} ({v}x)' for k, v in sorted(products.items(), key=lambda x: x[1], reverse=True)[:5]])}")
                    else:
                        print(f"  Product segments analyzed: {len(segments)}")
                
                elif analyzer_name == 'visual_effects':
                    segments = result.get('segments', [])
                    if segments:
                        print(f"  Visual effect segments: {len(segments)}")
                        # Extract effects
                        effects = []
                        for seg in segments:
                            if 'effects' in seg:
                                effects.extend(seg['effects'])
                            elif 'effect_type' in seg:
                                effects.append(seg['effect_type'])
                        if effects:
                            unique_effects = list(set(effects))
                            print(f"  Effects detected: {', '.join(unique_effects[:5])}")
                
                elif analyzer_name == 'composition_analysis':
                    segments = result.get('segments', [])
                    if segments:
                        print(f"  Composition segments analyzed: {len(segments)}")
                        # Extract composition patterns
                        compositions = []
                        for seg in segments:
                            if 'composition_type' in seg:
                                compositions.append(seg['composition_type'])
                            elif 'rule_of_thirds' in seg:
                                if seg['rule_of_thirds']:
                                    compositions.append('rule of thirds')
                        if compositions:
                            unique_comp = list(set(compositions))
                            print(f"  Composition patterns: {', '.join(unique_comp[:3])}")
                
                elif analyzer_name == 'background_segmentation':
                    segments = result.get('segments', [])
                    if segments:
                        print(f"  Background segments analyzed: {len(segments)}")
                        # Extract background info
                        backgrounds = []
                        for seg in segments:
                            if 'background_type' in seg:
                                backgrounds.append(seg['background_type'])
                            elif 'scene_type' in seg:
                                backgrounds.append(seg['scene_type'])
                        if backgrounds:
                            unique_bg = list(set(backgrounds))
                            print(f"  Background types: {', '.join(unique_bg[:3])}")
                
                else:
                    # Generic summary for other analyzers
                    if 'segments' in result:
                        print(f"  Segments analyzed: {len(result['segments'])}")
                    if 'summary' in result:
                        summary_keys = list(result['summary'].keys())[:3]
                        for key in summary_keys:
                            value = result['summary'][key]
                            if isinstance(value, (int, float)):
                                print(f"  {key}: {value:.2f}")
                            elif isinstance(value, str):
                                print(f"  {key}: {value[:50]}...")
    
    # Summary stats
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    total_analyzers = len(analyzers)
    print(f"\nTotal Analyzers Run: {total_analyzers}")
    print(f"Successful: {metadata.get('successful_analyzers', total_analyzers)}")
    print(f"Failed: {total_analyzers - metadata.get('successful_analyzers', total_analyzers)}")
    
    # List all analyzers that ran
    print("\nAnalyzers executed:")
    for name in sorted(analyzers.keys()):
        print(f"  - {name}")

if __name__ == "__main__":
    json_path = "/home/user/tiktok_production/results/7425998222721633569_multiprocess_20250705_075800.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    
    extract_analyzer_results(json_path)