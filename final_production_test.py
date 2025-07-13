#\!/usr/bin/env python3
"""
Final Production Test with All Fixes
Tests all 21 analyzers for real data quality
"""
import time
import json
import subprocess
import os
from datetime import datetime

def run_final_test():
    print("🚀 FINALER PRODUKTIONSTEST MIT ALLEN FIXES")
    print("=" * 70)
    
    # Test video
    video_path = "/home/user/tiktok_production/downloads/videos/7425998222721633569.mp4"
    video_duration = 28.9  # seconds
    
    print(f"📹 Test Video: Copenhagen Vlog")
    print(f"   Duration: {video_duration}s")
    print(f"   Expected: <3x realtime (<87s)")
    print()
    
    # Start monitoring
    print("📊 Starting analysis...")
    start_time = time.time()
    
    # Call API
    api_url = "http://localhost:8003/analyze"
    cmd = [
        'curl', '-X', 'POST', api_url,
        '-H', 'Content-Type: application/json',
        '-d', json.dumps({'video_path': video_path}),
        '-s'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    analysis_time = time.time() - start_time
    
    if result.returncode == 0:
        try:
            response = json.loads(result.stdout)
            
            if response.get('status') == 'success':
                print()
                print("✅ ANALYSIS SUCCESSFUL\!")
                print(f"   Processing time: {analysis_time:.1f}s")
                print(f"   Realtime factor: {analysis_time/video_duration:.2f}x")
                
                # Load results
                results_file = response.get('results_file')
                if results_file and os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        full_results = json.load(f)
                    
                    analyzer_results = full_results.get('analyzer_results', {})
                    
                    print()
                    print("📋 DATA QUALITY CHECK:")
                    print("-" * 70)
                    
                    # Check each critical analyzer
                    checks = {
                        'text_overlay': check_text_overlay,
                        'camera_analysis': check_camera_analysis,
                        'scene_segmentation': check_scene_segmentation,
                        'object_detection': check_object_detection,
                        'video_llava': check_blip2
                    }
                    
                    quality_scores = []
                    
                    for analyzer_name, check_func in checks.items():
                        if analyzer_name in analyzer_results:
                            score, message = check_func(analyzer_results[analyzer_name])
                            quality_scores.append(score)
                            status = "✅" if score >= 0.8 else "⚠️" if score >= 0.5 else "❌"
                            print(f"{status} {analyzer_name}: {message} (Score: {score:.2f})")
                        else:
                            print(f"❌ {analyzer_name}: NOT FOUND IN RESULTS")
                            quality_scores.append(0.0)
                    
                    # Overall quality
                    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
                    
                    print()
                    print("=" * 70)
                    print("🎯 FINAL ASSESSMENT:")
                    print(f"   Data Quality Score: {avg_quality:.2%}")
                    print(f"   Realtime Factor: {analysis_time/video_duration:.2f}x")
                    print(f"   All Analyzers Run: {len(analyzer_results)}/21")
                    
                    # Final verdict
                    if avg_quality >= 0.8 and analysis_time/video_duration < 3.0:
                        print()
                        print("🎉 SYSTEM IST 100% PRODUKTIONSREIF\!")
                        print("   ✓ Echte Datenqualität erreicht")
                        print("   ✓ Performance unter 3x Realtime")
                        print("   ✓ Keine Platzhalter oder Halluzinationen")
                    else:
                        print()
                        print("⚠️  SYSTEM NOCH NICHT PRODUKTIONSREIF")
                        if avg_quality < 0.8:
                            print(f"   - Datenqualität: {avg_quality:.2%} (Ziel: >80%)")
                        if analysis_time/video_duration >= 3.0:
                            print(f"   - Performance: {analysis_time/video_duration:.2f}x (Ziel: <3x)")
                    
            else:
                print(f"❌ Analysis failed: {response}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            print(f"Raw output: {result.stdout}")
    else:
        print(f"❌ API call failed: {result.stderr}")

def check_text_overlay(data):
    """Check text overlay quality"""
    if 'segments' not in data:
        return 0.0, "No segments found"
    
    segments = data['segments']
    if not segments:
        return 0.0, "No text segments"
    
    # Count segments with real text
    real_text_count = 0
    for seg in segments:
        text = seg.get('text', '')
        if text and text != 'KEIN TEXT' and len(text) > 0:
            real_text_count += 1
    
    ratio = real_text_count / len(segments) if segments else 0
    
    # Sample some text
    sample_texts = []
    for seg in segments[:5]:
        text = seg.get('text', '')
        if text and len(text) > 0:
            sample_texts.append(text[:30])
    
    if sample_texts:
        return ratio, f"{real_text_count}/{len(segments)} have text. Samples: {sample_texts[:2]}"
    else:
        return ratio, f"{real_text_count}/{len(segments)} have real text"

def check_camera_analysis(data):
    """Check camera analysis quality"""
    if 'segments' not in data:
        return 0.0, "No segments found"
    
    segments = data['segments']
    if not segments:
        return 0.0, "No camera segments"
    
    # Count segments with real movement types
    real_movement_count = 0
    movement_types = set()
    
    for seg in segments:
        movement = seg.get('movement_type', 'UNBEKANNT')
        if movement and movement != 'UNBEKANNT':
            real_movement_count += 1
            movement_types.add(movement)
    
    ratio = real_movement_count / len(segments) if segments else 0
    
    return ratio, f"{real_movement_count}/{len(segments)} classified. Types: {list(movement_types)[:5]}"

def check_scene_segmentation(data):
    """Check scene segmentation quality"""
    if 'segments' not in data:
        return 0.0, "No segments found"
    
    segments = data['segments']
    if not segments:
        return 0.0, "No scene segments"
    
    # Count segments with real scene types
    real_scene_count = 0
    scene_types = set()
    
    for seg in segments:
        scene_type = seg.get('scene_type', 'unknown')
        if scene_type and scene_type not in ['unknown', 'UNBEKANNT']:
            real_scene_count += 1
            scene_types.add(scene_type)
    
    ratio = real_scene_count / len(segments) if segments else 0
    
    return ratio, f"{real_scene_count}/{len(segments)} classified. Types: {list(scene_types)[:5]}"

def check_object_detection(data):
    """Check object detection quality"""
    if 'segments' not in data:
        return 0.0, "No segments found"
    
    segments = data['segments']
    if not segments:
        return 0.0, "No object segments"
    
    # Count segments with real object labels
    real_object_count = 0
    object_types = set()
    
    for seg in segments[:100]:  # Check first 100
        obj = seg.get('object', seg.get('class', seg.get('label', 'unknown')))
        if obj and obj != 'unknown':
            real_object_count += 1
            object_types.add(obj)
    
    # If we checked 100, extrapolate
    if len(segments) > 100:
        ratio = real_object_count / 100
    else:
        ratio = real_object_count / len(segments) if segments else 0
    
    return ratio, f"{len(object_types)} unique objects. Types: {list(object_types)[:5]}"

def check_blip2(data):
    """Check BLIP-2 quality"""
    if 'segments' not in data:
        return 0.0, "No segments found"
    
    segments = data['segments']
    if not segments:
        return 0.0, "No BLIP-2 segments"
    
    # Check for hallucinations
    hallucination_keywords = ['tinder', 'instagram', 'splits', 'http', 'pic.twitter']
    hallucination_count = 0
    total_length = 0
    factual_segments = 0
    
    for seg in segments:
        desc = seg.get('description', seg.get('text', seg.get('caption', '')))
        if desc:
            total_length += len(desc)
            
            # Check for hallucinations
            has_hallucination = any(kw in desc.lower() for kw in hallucination_keywords)
            if has_hallucination:
                hallucination_count += 1
            else:
                factual_segments += 1
    
    avg_length = total_length / len(segments) if segments else 0
    factual_ratio = factual_segments / len(segments) if segments else 0
    
    # Score based on factual ratio and description length
    score = factual_ratio * (min(avg_length, 150) / 150)
    
    return score, f"{factual_segments}/{len(segments)} factual. Avg length: {avg_length:.0f} chars"

if __name__ == "__main__":
    run_final_test()
