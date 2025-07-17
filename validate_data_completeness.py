#!/usr/bin/env python3
"""
Validate data completeness of analysis results
"""
import json
import sys
from pathlib import Path

def validate_result_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"\n🔍 VALIDATING: {Path(file_path).name}")
    print("=" * 60)
    
    # Check metadata
    metadata = data.get('metadata', {})
    print("\n📊 Metadata Validation:")
    required_fields = ['duration', 'fps', 'frame_count', 'processing_time_seconds', 'reconstruction_score']
    for field in required_fields:
        value = metadata.get(field, 'MISSING')
        status = '✅' if field in metadata else '❌'
        print(f"  {status} {field}: {value}")
    
    # Check analyzers
    results = data.get('analyzer_results', {})
    print(f"\n📋 Analyzer Results ({len(results)} total):")
    
    critical_analyzers = [
        'qwen2_vl_temporal',
        'object_detection', 
        'speech_transcription',
        'audio_analysis',
        'scene_segmentation'
    ]
    
    for analyzer in critical_analyzers:
        if analyzer in results and 'segments' in results[analyzer]:
            segments = len(results[analyzer]['segments'])
            print(f"  ✅ {analyzer}: {segments} segments")
        else:
            print(f"  ❌ {analyzer}: MISSING or ERROR")
    
    # Object detection details
    if 'object_detection' in results:
        obj_segments = results['object_detection'].get('segments', [])
        total_objects = sum(len(s.get('objects', [])) for s in obj_segments)
        print(f"\n🎯 Object Detection:")
        print(f"  Total segments: {len(obj_segments)}")
        print(f"  Total objects: {total_objects}")
        if obj_segments and obj_segments[0].get('objects'):
            first_obj = obj_segments[0]['objects'][0]
            print(f"  Sample: {first_obj.get('object_class', 'N/A')} ({first_obj.get('confidence_score', 0):.2f})")
    
    # Qwen2-VL quality
    if 'qwen2_vl_temporal' in results:
        qwen_segments = results['qwen2_vl_temporal'].get('segments', [])
        if qwen_segments:
            avg_desc_len = sum(len(s.get('description', '')) for s in qwen_segments) / len(qwen_segments)
            print(f"\n🎬 Qwen2-VL Quality:")
            print(f"  Segments: {len(qwen_segments)}")
            print(f"  Avg description length: {avg_desc_len:.0f} chars")
            print(f"  Coverage: {len(qwen_segments) * 3}s of {metadata.get('duration', 0):.0f}s video")
    
    # Final score
    print(f"\n🏆 FINAL VALIDATION:")
    print(f"  Reconstruction Score: {metadata.get('reconstruction_score', 0):.0f}/100")
    print(f"  All critical analyzers present: {'YES' if all(a in results for a in critical_analyzers) else 'NO'}")
    
    return data

if __name__ == "__main__":
    if len(sys.argv) > 1:
        validate_result_file(sys.argv[1])
    else:
        # Find latest result
        results_dir = Path("results")
        latest = max(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        validate_result_file(latest)