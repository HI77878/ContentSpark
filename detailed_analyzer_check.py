#!/usr/bin/env python3
import json

def detailed_analyzer_check(json_path):
    """Perform detailed check on specific analyzers."""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    analyzer_results = data.get('analyzer_results', {})
    
    # Specific analyzers to check in detail
    analyzers_to_check = [
        'video_llava', 'text_overlay', 'visual_effects', 'camera_analysis',
        'speech_transcription', 'scene_segmentation', 'composition_analysis',
        'content_quality'
    ]
    
    print("=== DETAILED ANALYZER INSPECTION ===\n")
    
    for analyzer in analyzers_to_check:
        print(f"\n{'='*50}")
        print(f"ANALYZER: {analyzer}")
        print('='*50)
        
        if analyzer not in analyzer_results:
            print("STATUS: MISSING FROM RESULTS!")
            continue
            
        result = analyzer_results[analyzer]
        
        if not result:
            print("STATUS: EMPTY RESULT")
            continue
            
        # Print structure summary
        print(f"TYPE: {type(result).__name__}")
        if isinstance(result, dict):
            print(f"KEYS: {list(result.keys())}")
            
            # Check segments
            if 'segments' in result:
                segments = result['segments']
                print(f"SEGMENTS COUNT: {len(segments)}")
                
                if segments:
                    # Show first segment structure
                    print(f"\nFIRST SEGMENT SAMPLE:")
                    first_seg = segments[0]
                    print(json.dumps(first_seg, indent=2)[:500] + "..." if len(json.dumps(first_seg)) > 500 else json.dumps(first_seg, indent=2))
                    
                    # Analyzer-specific checks
                    if analyzer == 'text_overlay':
                        text_count = sum(len(seg.get('texts', [])) for seg in segments)
                        readable_text = []
                        for seg in segments:
                            for text in seg.get('texts', []):
                                if isinstance(text, dict) and text.get('text') and text['text'] != 'unknown':
                                    readable_text.append(text['text'])
                        print(f"\nTEXT STATS:")
                        print(f"  Total text overlays detected: {text_count}")
                        print(f"  Readable texts: {len(readable_text)}")
                        if readable_text:
                            print(f"  Sample texts: {readable_text[:3]}")
                            
                    elif analyzer == 'visual_effects':
                        effect_count = sum(len(seg.get('effects', [])) for seg in segments)
                        effect_types = []
                        for seg in segments:
                            for effect in seg.get('effects', []):
                                if isinstance(effect, dict) and effect.get('type'):
                                    effect_types.append(effect['type'])
                        print(f"\nEFFECT STATS:")
                        print(f"  Total effects detected: {effect_count}")
                        print(f"  Effect types: {set(effect_types) if effect_types else 'None'}")
                        
                    elif analyzer == 'camera_analysis':
                        movement_count = sum(len(seg.get('movements', [])) for seg in segments)
                        movement_types = []
                        for seg in segments:
                            for movement in seg.get('movements', []):
                                if isinstance(movement, dict) and movement.get('type'):
                                    movement_types.append(movement['type'])
                        print(f"\nMOVEMENT STATS:")
                        print(f"  Total movements detected: {movement_count}")
                        print(f"  Movement types: {set(movement_types) if movement_types else 'None'}")
                        
                    elif analyzer == 'scene_segmentation':
                        print(f"\nSCENE STATS:")
                        for i, seg in enumerate(segments[:3]):  # First 3 segments
                            start = seg.get('start_time', 'N/A')
                            end = seg.get('end_time', 'N/A')
                            print(f"  Segment {i}: {start} -> {end} (duration: {end-start if isinstance(start, (int,float)) and isinstance(end, (int,float)) else 'N/A'})")
                            
                    elif analyzer == 'composition_analysis':
                        print(f"\nCOMPOSITION STATS:")
                        all_scores = []
                        for seg in segments:
                            scores = seg.get('scores', {})
                            all_scores.append(scores)
                        if all_scores:
                            first_scores = all_scores[0]
                            print(f"  First segment scores: {first_scores}")
                            # Check if all are zero
                            all_zero = all(
                                all(score == 0 for score in scores.values())
                                for scores in all_scores if isinstance(scores, dict)
                            )
                            print(f"  All scores zero: {all_zero}")
                            
                    elif analyzer == 'content_quality':
                        print(f"\nQUALITY STATS:")
                        all_metrics = []
                        for seg in segments:
                            metrics = seg.get('metrics', {})
                            all_metrics.append(metrics)
                        if all_metrics:
                            first_metrics = all_metrics[0]
                            print(f"  First segment metrics: {first_metrics}")
                            # Check if all are zero
                            all_zero = all(
                                all(metric == 0 for metric in metrics.values())
                                for metrics in all_metrics if isinstance(metrics, dict)
                            )
                            print(f"  All metrics zero: {all_zero}")
                else:
                    print("  SEGMENTS: EMPTY!")
                    
            # Check for other relevant fields
            if analyzer == 'speech_transcription':
                print(f"\nFULL TRANSCRIPT: {'Present' if result.get('full_transcript') else 'MISSING'}")
                if result.get('full_transcript'):
                    print(f"  Length: {len(result['full_transcript'])}")
                    print(f"  Content: '{result['full_transcript'][:100]}...'")
                    
            if analyzer == 'video_llava':
                print(f"\nVIDEO LLAVA SPECIFIC:")
                print(f"  Has segments: {'segments' in result}")
                print(f"  Result keys: {list(result.keys())}")
                # Print full result if small
                if len(str(result)) < 1000:
                    print(f"  Full result: {result}")

if __name__ == "__main__":
    json_path = "/home/user/tiktok_production/results/7425998222721633569_multiprocess_20250705_071729.json"
    detailed_analyzer_check(json_path)