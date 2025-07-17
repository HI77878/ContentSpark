#!/usr/bin/env python3
"""
Fixed Multiprocess GPU Executor with dedicated BLIP-2 worker
Solves the BLIP-2 loading timeout issue by isolating it in its own process
"""

import torch
import torch.multiprocessing as mp
import time
import logging
import os
import gc
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import traceback
import cv2
from pathlib import Path
from queue import Empty

# Set spawn method at module level
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

logger = logging.getLogger(__name__)

def blip2_worker_process(result_queue: mp.Queue, video_path: str):
    """Dedicated worker process for BLIP-2 only"""
    try:
        # Set environment
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        torch.cuda.set_device(0)
        
        # Import in worker to avoid serialization
        import sys
        sys.path.append('/home/user/tiktok_production')
        
        logger.info("BLIP-2 Worker: Starting dedicated BLIP-2 process")
        
        try:
            # Load BLIP-2
            from analyzers.blip2_video_captioning_optimized import BLIP2VideoCaptioningOptimized
            start_load = time.time()
            analyzer = BLIP2VideoCaptioningOptimized()
            load_time = time.time() - start_load
            logger.info(f"BLIP-2 Worker: Model loaded in {load_time:.1f}s")
            
            # Run analysis
            start_analysis = time.time()
            result = analyzer.analyze(video_path)
            analysis_time = time.time() - start_analysis
            
            logger.info(f"BLIP-2 Worker: Analysis completed in {analysis_time:.1f}s")
            result_queue.put(('blip2', result))
            
        except Exception as e:
            logger.error(f"BLIP-2 Worker: Failed: {e}")
            logger.error(traceback.format_exc())
            result_queue.put(('blip2', {"error": str(e)}))
            
    except Exception as e:
        logger.error(f"BLIP-2 Worker: Fatal error: {e}")
    finally:
        logger.info("BLIP-2 Worker: Shutting down")

def standard_worker_process(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue, 
                          video_path: str, analyzer_configs: Dict[str, Dict]):
    """Worker process for all non-BLIP-2 analyzers"""
    try:
        # Set environment
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)
        
        # Import analyzers in worker to avoid serialization
        import sys
        sys.path.append('/home/user/tiktok_production')
        
        loaded_analyzers = {}
        
        while True:
            try:
                task = task_queue.get(timeout=1)
                if task is None:  # Shutdown signal
                    break
                    
                analyzer_name = task
                
                # Skip BLIP-2 (handled by dedicated worker)
                if analyzer_name == 'blip2':
                    logger.info(f"Worker {gpu_id}: Skipping BLIP-2 (handled by dedicated worker)")
                    continue
                
                # Load analyzer if needed
                if analyzer_name not in loaded_analyzers:
                    try:
                        config = analyzer_configs[analyzer_name]
                        module = __import__(config['module'], fromlist=[config['class']])
                        analyzer_class = getattr(module, config['class'])
                        loaded_analyzers[analyzer_name] = analyzer_class()
                        logger.info(f"Worker {gpu_id}: Loaded {analyzer_name}")
                    except Exception as e:
                        logger.error(f"Worker {gpu_id}: Failed to load {analyzer_name}: {e}")
                        result_queue.put((analyzer_name, {"error": str(e)}))
                        continue
                
                # Run analysis
                try:
                    start_time = time.time()
                    analyzer = loaded_analyzers[analyzer_name]
                    result = analyzer.analyze(video_path)
                    elapsed = time.time() - start_time
                    
                    logger.info(f"Worker {gpu_id}: {analyzer_name} completed in {elapsed:.1f}s")
                    result_queue.put((analyzer_name, result))
                    
                except Exception as e:
                    logger.error(f"Worker {gpu_id}: {analyzer_name} failed: {e}")
                    logger.error(traceback.format_exc())
                    result_queue.put((analyzer_name, {"error": str(e)}))
                
                # Clear GPU cache
                torch.cuda.empty_cache()
                gc.collect()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {gpu_id} error: {e}")
                
    except Exception as e:
        logger.error(f"Worker {gpu_id} fatal error: {e}")
    finally:
        logger.info(f"Worker {gpu_id} shutting down")


class MultiprocessGPUExecutorBlip2Fix:
    """Fixed executor with dedicated BLIP-2 worker"""
    
    def __init__(self, num_gpu_processes: int = 3):
        self.num_gpu_processes = num_gpu_processes
        self.analyzer_configs = self._get_analyzer_configs()
        
    def _get_analyzer_configs(self) -> Dict[str, Dict]:
        """Get analyzer configurations with FINAL CORRECT mappings"""
        return {
            # Priority analyzers - VERIFIED
            'blip2': {
                'module': 'analyzers.blip2_video_captioning_optimized',
                'class': 'BLIP2VideoCaptioningOptimized',
                'priority': 1
            },
            'speech_transcription': {
                'module': 'analyzers.speech_transcription_ultimate',
                'class': 'UltimateSpeechTranscription',
                'priority': 1
            },
            'text_overlay': {
                'module': 'analyzers.text_overlay_tiktok_fixed',
                'class': 'TikTokTextOverlayAnalyzer',
                'priority': 1
            },
            # GPU analyzers - CORRECTED FROM ML_ANALYZER_REGISTRY
            'object_detection': {
                'module': 'analyzers.gpu_batch_object_detection_yolo',
                'class': 'GPUBatchObjectDetectionYOLO',
                'priority': 2
            },
            'camera_analysis': {
                'module': 'analyzers.camera_analysis_fixed',
                'class': 'GPUBatchCameraAnalysisFixed',
                'priority': 2
            },
            'visual_effects': {
                'module': 'analyzers.visual_effects_light_fixed',
                'class': 'VisualEffectsLight',
                'priority': 2
            },
            'composition_analysis': {
                'module': 'analyzers.composition_analysis_light',
                'class': 'GPUBatchCompositionAnalysisLight',
                'priority': 2
            },
            'product_detection': {
                'module': 'analyzers.gpu_batch_product_detection_light',
                'class': 'GPUBatchProductDetectionLight',
                'priority': 2
            },
            'background_segmentation': {
                'module': 'analyzers.background_segmentation_light',
                'class': 'GPUBatchBackgroundSegmentationLight',
                'priority': 2
            },
            'color_analysis': {
                'module': 'analyzers.gpu_batch_color_analysis',
                'class': 'GPUBatchColorAnalysis',
                'priority': 3
            },
            'scene_segmentation': {
                'module': 'analyzers.scene_segmentation_fixed',
                'class': 'SceneSegmentationFixedAnalyzer',
                'priority': 3
            },
            'content_quality': {
                'module': 'analyzers.gpu_batch_content_quality_fixed',
                'class': 'GPUBatchContentQualityFixed',
                'priority': 3
            },
            'cut_analysis': {
                'module': 'analyzers.cut_analysis_fixed',
                'class': 'CutAnalysisFixedAnalyzer',
                'priority': 3
            },
            'body_pose': {
                'module': 'analyzers.gpu_batch_body_pose_yolo',
                'class': 'GPUBatchBodyPoseYOLO',
                'priority': 3
            },
            'hand_gesture': {
                'module': 'analyzers.gpu_batch_hand_gesture',
                'class': 'GPUBatchHandGesture',
                'priority': 3
            },
            'eye_tracking': {
                'module': 'analyzers.gpu_batch_eye_tracking',
                'class': 'GPUBatchEyeTracking',
                'priority': 3
            },
            'age_estimation': {
                'module': 'analyzers.age_estimation_light',
                'class': 'GPUBatchAgeEstimationLight',
                'priority': 3
            },
            # Audio analyzers - CORRECTED FROM REGISTRY
            'audio_analysis': {
                'module': 'analyzers.audio_analysis_ultimate',
                'class': 'UltimateAudioAnalysis',
                'priority': 4
            },
            'audio_environment': {
                'module': 'analyzers.audio_environment_enhanced',
                'class': 'AudioEnvironmentEnhanced',
                'priority': 4
            },
            'speech_emotion': {
                'module': 'analyzers.gpu_batch_speech_emotion',
                'class': 'GPUBatchSpeechEmotion',
                'priority': 4
            },
            'speech_rate': {
                'module': 'analyzers.gpu_batch_speech_rate',
                'class': 'GPUBatchSpeechRate',
                'priority': 4
            },
            'sound_effects': {
                'module': 'analyzers.sound_effects_detector',
                'class': 'SoundEffectsDetector',
                'priority': 4
            },
            'temporal_flow': {
                'module': 'analyzers.cpu_batch_temporal_flow',
                'class': 'CPUBatchTemporalFlow',
                'priority': 4
            },
            # Additional analyzers from registry
            'face_detection': {
                'module': 'analyzers.gpu_batch_face_detection_optimized',
                'class': 'GPUBatchFaceDetectionOptimized',
                'priority': 3
            },
            'emotion_detection': {
                'module': 'analyzers.emotion_detection_real',
                'class': 'GPUBatchEmotionDetectionReal',
                'priority': 3
            }
        }
    
    def execute_parallel(self, video_path: str, selected_analyzers: List[str]) -> Dict[str, Any]:
        """Execute analyzers in parallel with dedicated BLIP-2 worker"""
        start_time = time.time()
        
        # Validate video
        if not Path(video_path).exists():
            raise ValueError(f"Video not found: {video_path}")
            
        # Filter analyzers that exist in config
        valid_analyzers = [a for a in selected_analyzers if a in self.analyzer_configs]
        if len(valid_analyzers) < len(selected_analyzers):
            missing = set(selected_analyzers) - set(valid_analyzers)
            logger.warning(f"Skipping unknown analyzers: {missing}")
        
        # Check if BLIP-2 is requested
        run_blip2 = 'blip2' in valid_analyzers
        
        # Group analyzers by priority (excluding BLIP-2)
        priority_groups = defaultdict(list)
        for analyzer in valid_analyzers:
            if analyzer != 'blip2':  # BLIP-2 handled separately
                priority = self.analyzer_configs[analyzer]['priority']
                priority_groups[priority].append(analyzer)
        
        # Setup multiprocessing
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        processes = []
        
        # Start dedicated BLIP-2 worker if needed
        if run_blip2:
            blip2_process = mp.Process(
                target=blip2_worker_process,
                args=(result_queue, video_path)
            )
            blip2_process.start()
            processes.append(blip2_process)
            logger.info("Started dedicated BLIP-2 worker process")
        
        # Start standard worker processes
        for i in range(self.num_gpu_processes):
            p = mp.Process(
                target=standard_worker_process,
                args=(0, task_queue, result_queue, video_path, self.analyzer_configs)
            )
            p.start()
            processes.append(p)
            
        logger.info(f"Started {self.num_gpu_processes} standard GPU worker processes")
        
        # Submit tasks by priority (excluding BLIP-2)
        total_tasks = 0
        for priority in sorted(priority_groups.keys()):
            for analyzer in priority_groups[priority]:
                task_queue.put(analyzer)
                total_tasks += 1
        
        # Add BLIP-2 to total if running
        if run_blip2:
            total_tasks += 1
                
        # Send shutdown signals to standard workers
        for _ in range(self.num_gpu_processes):
            task_queue.put(None)
            
        # Collect results
        results = {}
        completed = 0
        
        while completed < total_tasks:
            try:
                # Increased timeout for BLIP-2 loading
                analyzer_name, result = result_queue.get(timeout=600)  # 10 minutes
                results[analyzer_name] = result
                completed += 1
                logger.info(f"Progress: {completed}/{total_tasks} analyzers completed")
            except:
                logger.warning("Timeout waiting for results (600s)")
                break
                
        # Wait for processes to finish
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                
        # Calculate performance
        total_time = time.time() - start_time
        
        # Get video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Add metadata
        results['metadata'] = {
            'video_path': video_path,
            'duration': duration,
            'analysis_time': total_time,
            'realtime_factor': total_time / duration if duration > 0 else 0,
            'analyzers_used': valid_analyzers,
            'analyzers_completed': completed,
            'total_analyzers': total_tasks,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"Analysis complete in {total_time:.1f}s ({total_time/duration:.2f}x realtime)")
        logger.info(f"Completed {completed}/{total_tasks} analyzers")
        
        return results