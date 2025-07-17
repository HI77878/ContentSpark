#!/usr/bin/env python3
"""
Comprehensive Analysis Test with Detailed Monitoring
Tests all analyzers and verifies data quality for video reconstruction
"""

import requests
import json
import time
import sys
from pathlib import Path
from datetime import datetime

API_URL = "http://localhost:8003"

def monitor_gpu():
    """Monitor GPU usage"""
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                           '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if result.returncode == 0:
        values = result.stdout.strip().split(', ')
        return {
            'gpu_util': float(values[0]),
            'memory_used': float(values[1]),
            'memory_total': float(values[2])
        }
    return None

def analyze_video_comprehensive(video_path: str):
    """Run comprehensive analysis with detailed monitoring"""
    
    print(f"\n🎬 COMPREHENSIVE VIDEO ANALYSIS")
    print("=" * 60)
    print(f"📹 Video: {Path(video_path).name}")
    print(f"🕐 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initial GPU status
    gpu_start = monitor_gpu()
    if gpu_start:
        print(f"\n📊 Initial GPU Status:")
        print(f"   - GPU Utilization: {gpu_start['gpu_util']:.1f}%")
        print(f"   - Memory: {gpu_start['memory_used']:.0f}/{gpu_start['memory_total']:.0f} MB")
    
    # Submit analysis
    print(f"\n🚀 Starting Analysis...")
    start_time = time.time()
    
    try:
        response = requests.post(f"{API_URL}/analyze", json={"video_path": video_path})
        response.raise_for_status()
        result = response.json()
        
        processing_time = time.time() - start_time
        
        print(f"\n✅ Analysis Complete!")
        print(f"   - Processing Time: {result['processing_time']:.1f}s")
        print(f"   - Successful Analyzers: {result['successful_analyzers']}/{result['total_analyzers']}")
        print(f"   - Results File: {result['results_file']}")
        
        # Load and analyze results
        with open(result['results_file'], 'r') as f:
            analysis_data = json.load(f)
        
        return analysis_data, result['results_file']
        
    except Exception as e:
        print(f"\n❌ Analysis Failed: {e}")
        return None, None

def verify_analyzer_outputs(analysis_data: dict):
    """Verify each analyzer's output quality"""
    
    print(f"\n📋 DETAILED ANALYZER VERIFICATION")
    print("=" * 60)
    
    metadata = analysis_data.get('metadata', {})
    video_duration = metadata.get('duration', 0)
    fps = metadata.get('fps', 0)
    frame_count = metadata.get('frame_count', 0)
    
    print(f"\n📊 Video Metadata:")
    print(f"   - Duration: {video_duration:.2f}s")
    print(f"   - FPS: {fps:.2f}")
    print(f"   - Frame Count: {frame_count}")
    
    analyzer_results = analysis_data.get('analyzer_results', {})
    
    # Detailed verification for each analyzer
    verification_results = {}
    
    for analyzer_name, data in analyzer_results.items():
        print(f"\n🔍 Analyzer: {analyzer_name}")
        print("-" * 40)
        
        verification = {
            'has_data': False,
            'segment_count': 0,
            'data_quality': 'poor',
            'issues': [],
            'sample_data': None
        }
        
        if isinstance(data, dict):
            segments = data.get('segments', [])
            verification['has_data'] = len(segments) > 0
            verification['segment_count'] = len(segments)
            
            if segments:
                # Check temporal coverage
                if segments[0].get('time') is not None:
                    times = [s.get('time', 0) for s in segments]
                    coverage = (max(times) - min(times)) / video_duration if video_duration > 0 else 0
                    print(f"   ✓ Segments: {len(segments)}")
                    print(f"   ✓ Temporal Coverage: {coverage*100:.1f}%")
                else:
                    print(f"   ✓ Segments: {len(segments)} (no timestamps)")
                
                # Sample first segment
                first_segment = segments[0]
                verification['sample_data'] = first_segment
                
                # Analyzer-specific checks
                if analyzer_name == 'qwen2_vl_temporal':
                    # Check description quality
                    desc_lengths = [len(s.get('description', '')) for s in segments]
                    avg_length = sum(desc_lengths) / len(desc_lengths) if desc_lengths else 0
                    print(f"   ✓ Avg Description Length: {avg_length:.0f} chars")
                    if avg_length < 100:
                        verification['issues'].append("Short descriptions")
                    
                    # Sample description
                    if first_segment.get('description'):
                        print(f"   ✓ Sample: \"{first_segment['description'][:150]}...\"")
                    
                elif analyzer_name == 'object_detection':
                    # Check object detection quality - objects are in 'objects' array after normalization
                    total_objects = sum(len(s.get('objects', [])) for s in segments)
                    print(f"   ✓ Total Objects Detected: {total_objects}")
                    if total_objects < 10:
                        verification['issues'].append("Few objects detected")
                    
                    # Sample objects
                    if first_segment.get('objects'):
                        objects = [obj.get('object_class', '') for obj in first_segment['objects'][:5]]
                        print(f"   ✓ Sample Objects: {', '.join(objects)}")
                
                elif analyzer_name == 'speech_transcription':
                    # Check transcription quality
                    total_words = sum(len(s.get('text', '').split()) for s in segments)
                    print(f"   ✓ Total Words Transcribed: {total_words}")
                    
                    # Sample transcription
                    for s in segments[:3]:
                        if s.get('text'):
                            print(f"   ✓ [{s.get('start', 0):.1f}s]: \"{s['text']}\"")
                            break
                
                elif analyzer_name == 'text_overlay':
                    # Check text detection
                    texts_found = sum(1 for s in segments if s.get('text_blocks'))
                    print(f"   ✓ Frames with Text: {texts_found}")
                    
                    # Sample text
                    for s in segments:
                        if s.get('text_blocks'):
                            texts = [t.get('text', '') for t in s['text_blocks'][:3]]
                            print(f"   ✓ Sample Texts: {texts}")
                            break
                
                elif analyzer_name == 'camera_analysis':
                    # Check camera movements
                    movements = [s.get('movement_type', '') for s in segments if s.get('movement_type')]
                    unique_movements = set(movements)
                    print(f"   ✓ Movement Types: {', '.join(unique_movements)}")
                    
                elif analyzer_name == 'audio_analysis':
                    # Check audio features
                    if segments:
                        features = first_segment.get('features', {})
                        if features:
                            print(f"   ✓ Energy: {features.get('energy', 0):.3f}")
                            print(f"   ✓ Tempo: {features.get('tempo', 0):.1f} BPM")
                
                # Determine quality
                if len(segments) > 5 and not verification['issues']:
                    verification['data_quality'] = 'excellent'
                elif len(segments) > 2:
                    verification['data_quality'] = 'good'
                elif len(segments) > 0:
                    verification['data_quality'] = 'fair'
                
            else:
                print(f"   ❌ No segments found!")
                verification['issues'].append("No data")
        else:
            print(f"   ❌ Invalid data format!")
            verification['issues'].append("Invalid format")
        
        # Quality verdict
        quality_emoji = {
            'excellent': '🟢',
            'good': '🟡', 
            'fair': '🟠',
            'poor': '🔴'
        }
        print(f"   Quality: {quality_emoji[verification['data_quality']]} {verification['data_quality'].upper()}")
        
        if verification['issues']:
            print(f"   Issues: {', '.join(verification['issues'])}")
        
        verification_results[analyzer_name] = verification
    
    return verification_results

def assess_reconstruction_capability(analysis_data: dict, verification_results: dict):
    """Assess if video can be reconstructed from the data"""
    
    print(f"\n🎯 VIDEO RECONSTRUCTION CAPABILITY ASSESSMENT")
    print("=" * 60)
    
    reconstruction_score = 0
    max_score = 100
    required_elements = {
        'visual_timeline': False,
        'audio_timeline': False,
        'text_content': False,
        'object_tracking': False,
        'scene_changes': False,
        'camera_movements': False,
        'speech_content': False,
        'temporal_consistency': False
    }
    
    # Check visual timeline (Qwen2-VL)
    qwen_data = verification_results.get('qwen2_vl_temporal', {})
    if qwen_data.get('segment_count', 0) > 5 and qwen_data.get('data_quality') in ['excellent', 'good']:
        required_elements['visual_timeline'] = True
        reconstruction_score += 20
        print("✅ Visual Timeline: Complete temporal descriptions available")
    else:
        print("❌ Visual Timeline: Insufficient temporal descriptions")
    
    # Check audio timeline
    audio_data = verification_results.get('audio_analysis', {})
    speech_data = verification_results.get('speech_transcription', {})
    if audio_data.get('has_data') or speech_data.get('has_data'):
        required_elements['audio_timeline'] = True
        reconstruction_score += 15
        print("✅ Audio Timeline: Audio features and/or speech captured")
    else:
        print("❌ Audio Timeline: No audio data")
    
    # Check text content
    text_data = verification_results.get('text_overlay', {})
    if text_data.get('has_data'):
        required_elements['text_content'] = True
        reconstruction_score += 10
        print("✅ Text Content: On-screen text detected")
    else:
        print("⚠️  Text Content: No on-screen text found (may not exist)")
    
    # Check object tracking
    object_data = verification_results.get('object_detection', {})
    if object_data.get('segment_count', 0) > 5:
        required_elements['object_tracking'] = True
        reconstruction_score += 15
        print("✅ Object Tracking: Objects detected across timeline")
    else:
        print("❌ Object Tracking: Insufficient object data")
    
    # Check scene changes
    scene_data = verification_results.get('scene_segmentation', {})
    cut_data = verification_results.get('cut_analysis', {})
    if scene_data.get('has_data') or cut_data.get('has_data'):
        required_elements['scene_changes'] = True
        reconstruction_score += 10
        print("✅ Scene Changes: Scene transitions detected")
    else:
        print("❌ Scene Changes: No scene segmentation data")
    
    # Check camera movements
    camera_data = verification_results.get('camera_analysis', {})
    if camera_data.get('has_data'):
        required_elements['camera_movements'] = True
        reconstruction_score += 10
        print("✅ Camera Movements: Camera motion tracked")
    else:
        print("❌ Camera Movements: No camera analysis data")
    
    # Check speech content
    if speech_data.get('segment_count', 0) > 0:
        required_elements['speech_content'] = True
        reconstruction_score += 10
        print("✅ Speech Content: Speech transcribed")
    else:
        print("⚠️  Speech Content: No speech detected (may be music only)")
    
    # Check temporal consistency
    analyzers_with_time = sum(1 for v in verification_results.values() 
                             if v.get('segment_count', 0) > 0 and v.get('sample_data', {}).get('time') is not None)
    if analyzers_with_time > 5:
        required_elements['temporal_consistency'] = True
        reconstruction_score += 10
        print("✅ Temporal Consistency: Multiple analyzers with synchronized timestamps")
    else:
        print("❌ Temporal Consistency: Few analyzers with proper timestamps")
    
    # Final assessment
    print(f"\n📊 RECONSTRUCTION SCORE: {reconstruction_score}/100")
    
    if reconstruction_score >= 80:
        print("✅ VERDICT: Video can be reconstructed with HIGH accuracy")
        print("   All major visual and audio elements are captured")
    elif reconstruction_score >= 60:
        print("🟡 VERDICT: Video can be reconstructed with MODERATE accuracy")
        print("   Most key elements captured, some details may be missing")
    elif reconstruction_score >= 40:
        print("🟠 VERDICT: Video can be partially reconstructed")
        print("   Basic structure captured, but missing important details")
    else:
        print("🔴 VERDICT: Insufficient data for meaningful reconstruction")
        print("   Critical analyzers failed or provided poor data")
    
    return reconstruction_score, required_elements

def main():
    # Choose a video to analyze
    video_path = "/home/user/tiktok_videos/videos/7425998222721633569.mp4"
    
    print("🔬 COMPREHENSIVE VIDEO ANALYSIS & RECONSTRUCTION TEST")
    print("=" * 60)
    
    # Check API health
    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        print("✅ API is running and healthy")
    except:
        print("❌ API is not running. Please start it first.")
        sys.exit(1)
    
    # Run comprehensive analysis
    analysis_data, results_file = analyze_video_comprehensive(video_path)
    
    if analysis_data:
        # Verify analyzer outputs
        verification_results = verify_analyzer_outputs(analysis_data)
        
        # Assess reconstruction capability
        reconstruction_score, elements = assess_reconstruction_capability(analysis_data, verification_results)
        
        # GPU status at end
        gpu_end = monitor_gpu()
        if gpu_end:
            print(f"\n📊 Final GPU Status:")
            print(f"   - GPU Utilization: {gpu_end['gpu_util']:.1f}%")
            print(f"   - Memory: {gpu_end['memory_used']:.0f}/{gpu_end['memory_total']:.0f} MB")
        
        # Summary
        print(f"\n📋 ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"✅ Results saved to: {results_file}")
        print(f"📊 Total Analyzers: {len(verification_results)}")
        working_analyzers = sum(1 for v in verification_results.values() if v['has_data'])
        print(f"✅ Working Analyzers: {working_analyzers}/{len(verification_results)}")
        print(f"🎯 Reconstruction Score: {reconstruction_score}/100")
        
        # Detailed issues if any
        analyzers_with_issues = [(name, v) for name, v in verification_results.items() if v['issues']]
        if analyzers_with_issues:
            print(f"\n⚠️  ANALYZERS WITH ISSUES:")
            for name, v in analyzers_with_issues:
                print(f"   - {name}: {', '.join(v['issues'])}")

if __name__ == "__main__":
    main()