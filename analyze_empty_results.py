#!/usr/bin/env python3
import json
import sys

def analyze_empty_results(json_path):
    """Analyze which analyzers are producing empty or placeholder data."""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    analyzer_results = data.get('analyzer_results', {})
    
    empty_analyzers = []
    problematic_analyzers = []
    
    for analyzer_name, result in analyzer_results.items():
        issues = []
        
        # Check for empty results
        if not result:
            empty_analyzers.append((analyzer_name, "Completely empty result"))
            continue
            
        # Check for empty segments
        if 'segments' in result and not result['segments']:
            issues.append("Empty segments array")
            
        # Check for all zero values
        if isinstance(result, dict):
            numeric_values = []
            for k, v in result.items():
                if isinstance(v, (int, float)) and k not in ['processing_time', 'timestamp', 'start_time', 'end_time']:
                    numeric_values.append(v)
            
            if numeric_values and all(v == 0 for v in numeric_values):
                issues.append("All numeric values are zero")
        
        # Check specific analyzer issues
        if analyzer_name == 'video_llava':
            if 'error' in str(result).lower():
                issues.append("Error in result")
            if not result.get('segments'):
                issues.append("Missing segments")
                
        elif analyzer_name == 'text_overlay':
            if 'segments' in result:
                has_text = False
                for seg in result['segments']:
                    if seg.get('texts') and any(t.get('text') and t['text'] != 'unknown' for t in seg['texts']):
                        has_text = True
                        break
                if not has_text:
                    issues.append("No readable text despite detecting overlays")
                    
        elif analyzer_name == 'visual_effects':
            if 'segments' in result:
                has_descriptions = False
                for seg in result['segments']:
                    effects = seg.get('effects', [])
                    if effects:
                        for e in effects:
                            if isinstance(e, dict) and e.get('description') and e['description'] != 'unknown':
                                has_descriptions = True
                                break
                            elif isinstance(e, str) and e != 'unknown':
                                has_descriptions = True
                                break
                if not has_descriptions and any(seg.get('effects') for seg in result['segments']):
                    issues.append("No effect descriptions despite detecting effects")
                    
        elif analyzer_name == 'camera_analysis':
            if 'segments' in result:
                has_types = False
                for seg in result['segments']:
                    movements = seg.get('movements', [])
                    if movements:
                        for m in movements:
                            if isinstance(m, dict) and m.get('type') and m['type'] != 'unknown':
                                has_types = True
                                break
                if not has_types and any(seg.get('movements') for seg in result['segments']):
                    issues.append("No movement types despite detecting movements")
                    
        elif analyzer_name == 'speech_transcription':
            if 'segments' in result and not result['segments']:
                issues.append("No speech detected")
            elif not result.get('full_transcript'):
                issues.append("No full transcript")
                
        elif analyzer_name == 'scene_segmentation':
            if 'segments' in result:
                for i, seg in enumerate(result['segments']):
                    if seg.get('end_time', 0) <= seg.get('start_time', 0):
                        issues.append(f"Invalid timestamps in segment {i}")
                        break
                        
        elif analyzer_name == 'composition_analysis':
            if 'segments' in result:
                all_zero = True
                for seg in result['segments']:
                    scores = seg.get('scores', {})
                    if any(scores.get(k, 0) != 0 for k in ['rule_of_thirds', 'balance', 'symmetry', 'depth', 'leading_lines']):
                        all_zero = False
                        break
                if all_zero:
                    issues.append("All composition scores are zero")
                    
        elif analyzer_name == 'content_quality':
            if 'segments' in result:
                all_zero = True
                for seg in result['segments']:
                    metrics = seg.get('metrics', {})
                    if any(metrics.get(k, 0) != 0 for k in ['sharpness', 'exposure', 'contrast', 'saturation', 'noise_level']):
                        all_zero = False
                        break
                if all_zero:
                    issues.append("All quality metrics are zero")
        
        if issues:
            problematic_analyzers.append((analyzer_name, issues))
    
    print("=== ANALYZER ISSUES REPORT ===\n")
    
    print(f"Total analyzers: {len(analyzer_results)}")
    print(f"Empty analyzers: {len(empty_analyzers)}")
    print(f"Problematic analyzers: {len(problematic_analyzers)}\n")
    
    if empty_analyzers:
        print("COMPLETELY EMPTY ANALYZERS:")
        for name, issue in empty_analyzers:
            print(f"  - {name}: {issue}")
        print()
    
    if problematic_analyzers:
        print("PROBLEMATIC ANALYZERS:")
        for name, issues in problematic_analyzers:
            print(f"\n  {name}:")
            for issue in issues:
                print(f"    - {issue}")
    
    # Check for missing analyzers
    print("\n=== MISSING ANALYZERS CHECK ===")
    expected_analyzers = [
        'video_llava', 'text_overlay', 'visual_effects', 'camera_analysis',
        'speech_transcription', 'scene_segmentation', 'composition_analysis',
        'content_quality'
    ]
    
    for analyzer in expected_analyzers:
        if analyzer not in analyzer_results:
            print(f"  - {analyzer}: MISSING from results")

if __name__ == "__main__":
    json_path = "/home/user/tiktok_production/results/7425998222721633569_multiprocess_20250705_071729.json"
    analyze_empty_results(json_path)