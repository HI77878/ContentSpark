#!/usr/bin/env python3
import sys
import torch
import time
from pathlib import Path

# Fix FFmpeg environment
import os
os.environ['LD_LIBRARY_PATH'] = '/home/user/ffmpeg-install/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PATH'] = '/home/user/ffmpeg-install/bin:' + os.environ.get('PATH', '')

sys.path.insert(0, '/home/user/tiktok_production')

test_video = "test_local_video.mp4"
analyzers = [
    'qwen2_vl_temporal', 'object_detection', 'text_overlay', 
    'speech_transcription', 'body_pose', 'background_segmentation',
    'camera_analysis', 'scene_segmentation', 'color_analysis',
    'content_quality', 'cut_analysis', 'age_estimation',
    'eye_tracking', 'audio_analysis', 'audio_environment',
    'speech_emotion', 'speech_flow', 'temporal_flow',
    'cross_analyzer_intelligence'
]

# Results summary
results_summary = {}

for analyzer_name in analyzers:
    print(f"\n{'='*50}")
    print(f"Testing: {analyzer_name}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        # Clear GPU memory before each test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Import analyzer
        if analyzer_name == 'speech_transcription':
            from analyzers.speech_transcription_ultimate import UltimateSpeechTranscription as AnalyzerClass
        elif analyzer_name == 'audio_analysis':
            from analyzers.audio_analysis_ultimate import UltimateAudioAnalysis as AnalyzerClass
        elif analyzer_name == 'audio_environment':
            from analyzers.audio_environment_enhanced import AudioEnvironmentEnhanced as AnalyzerClass
        elif analyzer_name == 'speech_emotion':
            from analyzers.gpu_batch_speech_emotion import GPUBatchSpeechEmotion as AnalyzerClass
        elif analyzer_name == 'speech_flow':
            from analyzers.gpu_batch_speech_flow import GPUBatchSpeechFlow as AnalyzerClass
        elif analyzer_name == 'temporal_flow':
            from analyzers.narrative_analysis_wrapper import NarrativeAnalysisWrapper as AnalyzerClass
        elif analyzer_name == 'cross_analyzer_intelligence':
            from analyzers.cross_analyzer_intelligence import CrossAnalyzerIntelligence as AnalyzerClass
        elif analyzer_name == 'qwen2_vl_temporal':
            from analyzers.qwen2_vl_temporal_analyzer import Qwen2VLTemporalAnalyzer as AnalyzerClass
        elif analyzer_name == 'object_detection':
            from analyzers.gpu_batch_object_detection_yolo import GPUBatchObjectDetectionYOLO as AnalyzerClass
        elif analyzer_name == 'text_overlay':
            from analyzers.text_overlay_tiktok_fixed import TikTokTextOverlayAnalyzer as AnalyzerClass
        elif analyzer_name == 'body_pose':
            from analyzers.body_pose_yolov8 import BodyPoseYOLOv8 as AnalyzerClass
        elif analyzer_name == 'background_segmentation':
            from analyzers.background_segmentation_light import GPUBatchBackgroundSegmentationLight as AnalyzerClass
        elif analyzer_name == 'camera_analysis':
            from analyzers.camera_analysis_fixed import GPUBatchCameraAnalysisFixed as AnalyzerClass
        elif analyzer_name == 'scene_segmentation':
            from analyzers.scene_segmentation_fixed import SceneSegmentationFixedAnalyzer as AnalyzerClass
        elif analyzer_name == 'color_analysis':
            from analyzers.gpu_batch_color_analysis import GPUBatchColorAnalysis as AnalyzerClass
        elif analyzer_name == 'content_quality':
            from analyzers.gpu_batch_content_quality_fixed import GPUBatchContentQualityFixed as AnalyzerClass
        elif analyzer_name == 'cut_analysis':
            from analyzers.cut_analysis_fixed import CutAnalysisFixedAnalyzer as AnalyzerClass
        elif analyzer_name == 'age_estimation':
            from analyzers.age_gender_insightface import AgeGenderInsightFace as AnalyzerClass
        elif analyzer_name == 'eye_tracking':
            from analyzers.gpu_batch_eye_tracking import GPUBatchEyeTracking as AnalyzerClass
        else:
            raise ValueError(f"Unknown analyzer: {analyzer_name}")
        
        # Initialize analyzer
        analyzer = AnalyzerClass()
        
        # Special case for cross_analyzer_intelligence - needs previous results
        if analyzer_name == 'cross_analyzer_intelligence':
            # Create dummy previous results
            previous_results = {
                'object_detection': {'segments': [{'timestamp': 0, 'objects': []}]},
                'text_overlay': {'segments': [{'timestamp': 0, 'text': 'test'}]}
            }
            result = analyzer.analyze(previous_results)
        else:
            # Normal analysis
            result = analyzer.analyze(test_video)
        
        # Check result
        duration = time.time() - start_time
        
        if result and isinstance(result, dict):
            # Check for segments
            segments = result.get('segments', result.get('data', []))
            if isinstance(segments, list) and len(segments) > 0:
                print(f"✅ SUCCESS: Got {len(segments)} segments in {duration:.1f}s")
                
                # Special check for Qwen2-VL - verify 2 second intervals
                if analyzer_name == 'qwen2_vl_temporal' and len(segments) > 1:
                    interval = segments[1]['start_time'] - segments[0]['start_time']
                    print(f"   Qwen2-VL segment interval: {interval:.1f}s")
                
                results_summary[analyzer_name] = {'status': 'success', 'segments': len(segments), 'time': duration}
            else:
                print(f"⚠️ WARNING: No segments found (got {type(segments)}) in {duration:.1f}s")
                results_summary[analyzer_name] = {'status': 'warning', 'segments': 0, 'time': duration}
        else:
            print(f"⚠️ WARNING: Invalid result type: {type(result)} in {duration:.1f}s")
            results_summary[analyzer_name] = {'status': 'warning', 'segments': 0, 'time': duration}
            
        # Cleanup
        del analyzer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ FAILED: {e} in {duration:.1f}s")
        results_summary[analyzer_name] = {'status': 'failed', 'error': str(e), 'time': duration}
        import traceback
        traceback.print_exc()

# Final summary
print("\n\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

successful = sum(1 for r in results_summary.values() if r['status'] == 'success')
warnings = sum(1 for r in results_summary.values() if r['status'] == 'warning')
failed = sum(1 for r in results_summary.values() if r['status'] == 'failed')

print(f"\n✅ Successful: {successful}/{len(analyzers)}")
print(f"⚠️ Warnings: {warnings}/{len(analyzers)}")
print(f"❌ Failed: {failed}/{len(analyzers)}")

print("\nDetailed Results:")
for analyzer, result in results_summary.items():
    status_icon = "✅" if result['status'] == 'success' else "⚠️" if result['status'] == 'warning' else "❌"
    print(f"{status_icon} {analyzer}: {result['status']} ({result['time']:.1f}s)")
    if 'segments' in result:
        print(f"   Segments: {result['segments']}")
    if 'error' in result:
        print(f"   Error: {result['error'][:100]}...")

print("\n\nDONE! Check results above.")