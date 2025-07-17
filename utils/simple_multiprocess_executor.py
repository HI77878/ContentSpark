#!/usr/bin/env python3
"""Simplified multiprocess executor without dynamic imports"""

import torch
import torch.multiprocessing as mp
import time
import logging
import os
import gc
from typing import Dict, List, Any
from collections import defaultdict
import cv2
from pathlib import Path

# Set up multiprocessing
mp.set_start_method('spawn', force=True)

logger = logging.getLogger(__name__)

def analyze_single(analyzer_name: str, video_path: str) -> Dict[str, Any]:
    """Run single analyzer in subprocess"""
    import sys
    sys.path.append('/home/user/tiktok_production')
    
    # ALL 29 ANALYZERS - Direct imports
    if analyzer_name == 'video_llava':
        from analyzers.video_comprehensive_description import VideoComprehensiveDescription
        analyzer = VideoComprehensiveDescription()
    elif analyzer_name == 'speech_transcription':
        from analyzers.speech_transcription_ultimate import UltimateSpeechTranscription
        analyzer = UltimateSpeechTranscription()
    elif analyzer_name == 'text_overlay':
        from analyzers.text_overlay_tiktok_fixed import TikTokTextOverlayAnalyzer
        analyzer = TikTokTextOverlayAnalyzer()
    elif analyzer_name == 'face_detection':
        from analyzers.gpu_batch_face_detection_optimized import GPUBatchFaceDetectionOptimized
        analyzer = GPUBatchFaceDetectionOptimized()
    elif analyzer_name == 'object_detection':
        from analyzers.gpu_batch_object_detection_yolo import GPUBatchObjectDetectionYOLO
        analyzer = GPUBatchObjectDetectionYOLO()
    elif analyzer_name == 'emotion_detection':
        from analyzers.emotion_detection_real import GPUBatchEmotionDetectionReal
        analyzer = GPUBatchEmotionDetectionReal()
    elif analyzer_name == 'body_pose':
        from analyzers.gpu_batch_body_pose_yolo import GPUBatchBodyPoseYOLO
        analyzer = GPUBatchBodyPoseYOLO()
    elif analyzer_name == 'hand_gesture':
        from analyzers.gpu_batch_hand_gesture import GPUBatchHandGesture
        analyzer = GPUBatchHandGesture()
    elif analyzer_name == 'eye_tracking':
        from analyzers.gpu_batch_eye_tracking import GPUBatchEyeTracking
        analyzer = GPUBatchEyeTracking()
    elif analyzer_name == 'age_estimation':
        from analyzers.age_estimation_light import GPUBatchAgeEstimationLight
        analyzer = GPUBatchAgeEstimationLight()
    elif analyzer_name == 'gesture_recognition':
        from analyzers.gpu_batch_gesture_recognition_fixed import GPUBatchGestureRecognitionFixed
        analyzer = GPUBatchGestureRecognitionFixed()
    elif analyzer_name == 'content_quality':
        from analyzers.gpu_batch_content_quality_fixed import GPUBatchContentQualityFixed
        analyzer = GPUBatchContentQualityFixed()
    elif analyzer_name == 'composition_analysis':
        from analyzers.composition_analysis_light import GPUBatchCompositionAnalysisLight
        analyzer = GPUBatchCompositionAnalysisLight()
    elif analyzer_name == 'product_detection':
        from analyzers.gpu_batch_product_detection_light import GPUBatchProductDetectionLight
        analyzer = GPUBatchProductDetectionLight()
    elif analyzer_name == 'visual_effects':
        from analyzers.visual_effects_light_fixed import VisualEffectsLight
        analyzer = VisualEffectsLight()
    elif analyzer_name == 'background_segmentation':
        from analyzers.background_segmentation_light import GPUBatchBackgroundSegmentationLight
        analyzer = GPUBatchBackgroundSegmentationLight()
    elif analyzer_name == 'camera_analysis':
        from analyzers.camera_analysis_fixed import GPUBatchCameraAnalysisFixed
        analyzer = GPUBatchCameraAnalysisFixed()
    elif analyzer_name == 'cut_analysis':
        from analyzers.cut_analysis_fixed import CutAnalysisFixedAnalyzer
        analyzer = CutAnalysisFixedAnalyzer()
    elif analyzer_name == 'scene_segmentation':
        from analyzers.scene_segmentation_fixed import SceneSegmentationFixedAnalyzer
        analyzer = SceneSegmentationFixedAnalyzer()
    elif analyzer_name == 'color_analysis':
        from analyzers.gpu_batch_color_analysis import GPUBatchColorAnalysis
        analyzer = GPUBatchColorAnalysis()
    elif analyzer_name == 'audio_analysis':
        from analyzers.audio_analysis_ultimate import UltimateAudioAnalysis
        analyzer = UltimateAudioAnalysis()
    elif analyzer_name == 'audio_environment':
        from analyzers.audio_environment_enhanced import AudioEnvironmentEnhanced
        analyzer = AudioEnvironmentEnhanced()
    elif analyzer_name == 'speech_emotion':
        from analyzers.gpu_batch_speech_emotion import GPUBatchSpeechEmotion
        analyzer = GPUBatchSpeechEmotion()
    elif analyzer_name == 'speech_rate':
        from analyzers.gpu_batch_speech_rate_enhanced import GPUBatchSpeechRateEnhanced
        analyzer = GPUBatchSpeechRateEnhanced()
    elif analyzer_name == 'sound_effects':
        from analyzers.gpu_batch_sound_effects_enhanced import GPUBatchSoundEffectsEnhanced
        analyzer = GPUBatchSoundEffectsEnhanced()
    elif analyzer_name == 'speech_flow':
        from analyzers.gpu_batch_speech_flow import GPUBatchSpeechFlow
        analyzer = GPUBatchSpeechFlow()
    elif analyzer_name == 'comment_cta_detection':
        from analyzers.gpu_batch_comment_cta_detection import GPUBatchCommentCTADetection
        analyzer = GPUBatchCommentCTADetection()
    elif analyzer_name == 'temporal_flow':
        from analyzers.cpu_batch_temporal_flow import CPUBatchTemporalFlow
        analyzer = CPUBatchTemporalFlow()
    else:
        return {"error": f"Unknown analyzer: {analyzer_name}"}
    
    try:
        result = analyzer.analyze(video_path)
        return result
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


class SimpleMultiprocessExecutor:
    """Simplified executor using process pool"""
    
    def __init__(self, num_processes: int = 4):
        self.num_processes = num_processes
        
    def execute_parallel(self, video_path: str, analyzers: List[str]) -> Dict[str, Any]:
        """Execute analyzers in parallel"""
        start_time = time.time()
        
        # Validate video
        if not Path(video_path).exists():
            raise ValueError(f"Video not found: {video_path}")
        
        # Use process pool
        with mp.Pool(processes=self.num_processes) as pool:
            # Submit all tasks
            tasks = [(analyzer, video_path) for analyzer in analyzers]
            results_async = [pool.apply_async(analyze_single, args=task) for task in tasks]
            
            # Collect results
            results = {}
            for i, (analyzer, _) in enumerate(tasks):
                try:
                    result = results_async[i].get(timeout=120)  # 2 minute timeout per analyzer
                    results[analyzer] = result
                    logger.info(f"Completed {analyzer}: {i+1}/{len(analyzers)}")
                except Exception as e:
                    results[analyzer] = {"error": str(e)}
                    logger.error(f"Failed {analyzer}: {e}")
        
        # Get video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Add metadata
        total_time = time.time() - start_time
        results['metadata'] = {
            'video_path': video_path,
            'duration': duration,
            'analysis_time': total_time,
            'realtime_factor': total_time / duration if duration > 0 else 0,
            'analyzers_used': analyzers,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"Analysis complete in {total_time:.1f}s ({total_time/duration:.2f}x realtime)")
        
        return results