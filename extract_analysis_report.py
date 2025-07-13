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
                        if summary:
                            print(f"  Main Activity: {summary.get('main_activity', 'N/A')}")
                            print(f"  Setting: {summary.get('setting', 'N/A')}")
                            print(f"  Key Objects: {', '.join(summary.get('key_objects', [])[:5])}")
                            narr = summary.get('narrative_description', '')
                            if narr:
                                print(f"  Description: {narr[:150]}...")
                    
                    elif analyzer_name == 'speech_recognition' or analyzer_name == 'speech_transcription':
                        transcript = result.get('full_transcript', '')
                        segments = result.get('segments', [])
                        # Check for text in segments if no full_transcript
                        if not transcript and segments:
                            transcript = ' '.join(seg.get('text', '') for seg in segments)
                        if transcript:
                            print(f"  Transcript: \"{transcript[:100]}...\"" if len(transcript) > 100 else f"  Transcript: \"{transcript}\"")
                        print(f"  Speech segments: {len(segments)}")
                    
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
                            for text in seg.get('texts', []):
                                if text['text'] not in all_text:
                                    all_text.append(text['text'])
                        if all_text:
                            print(f"  Text found: {', '.join(all_text[:5])}")
                            if len(all_text) > 5:
                                print(f"  ... and {len(all_text) - 5} more")
                    
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
                        print(f"  Total cuts: {len(cuts)}")
                        if cuts:
                            avg_duration = sum(c.get('duration', 0) for c in cuts) / len(cuts)
                            print(f"  Average shot duration: {avg_duration:.1f}s")
                    
                    elif analyzer_name == 'camera_analysis':
                        techniques = result.get('camera_techniques', {})
                        if techniques:
                            print(f"  Techniques found: {', '.join(techniques.keys())}")
                    
                    elif analyzer_name == 'body_pose':
                        segments = result.get('segments', [])
                        total_poses = sum(len(seg.get('poses', [])) for seg in segments)
                        print(f"  Total pose detections: {total_poses}")
                    
                    elif analyzer_name == 'quality_assessment' or analyzer_name == 'content_quality':
                        metrics = result.get('quality_metrics', {})
                        if metrics:
                            print(f"  Overall score: {metrics.get('overall_score', 0):.2f}/5.0")
                            print(f"  Sharpness: {metrics.get('sharpness', 0):.2f}")
                            print(f"  Brightness: {metrics.get('brightness', 0):.2f}")
                        # Check for segments
                        segments = result.get('segments', [])
                        if segments and not metrics:
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
                                    all_colors.extend(seg['dominant_colors'])
                            if all_colors:
                                print(f"  Sample dominant colors: {all_colors[:3]}")
                    
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
                                scene_types[scene_type] = scene_types.get(scene_type, 0) + 1
                            if scene_types:
                                print(f"  Scene types: {', '.join([f'{k} ({v})' for k, v in scene_types.items()][:3])}")
                    
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
                
                elif status == 'error':
                    error = result.get('error', 'Unknown error')
                    print(f"\n{analyzer_name}: ERROR - {error}")
    
    # Summary stats
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    total_analyzers = len(analyzers)
    successful = sum(1 for a in analyzers.values() if a.get('status') == 'success')
    failed = sum(1 for a in analyzers.values() if a.get('status') == 'error')
    
    print(f"\nTotal Analyzers Run: {total_analyzers}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed Analyzers:")
        for name, result in analyzers.items():
            if result.get('status') == 'error':
                print(f"  - {name}: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    json_path = "/home/user/tiktok_production/results/7425998222721633569_multiprocess_20250705_075800.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    
    extract_analyzer_results(json_path)