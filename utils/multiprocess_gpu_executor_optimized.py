#!/usr/bin/env python3
"""
Optimized Multiprocess GPU Executor with Smart Task Distribution
Ensures Qwen2-VL Ultra runs concurrently with other analyzers
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
from queue import Empty
import threading

# Set spawn method at module level
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

logger = logging.getLogger(__name__)

# Analyzer timing estimates for smart scheduling
ANALYZER_TIMINGS = {
    'qwen2_vl_ultra': 400.0,  # Heavy - runs alone
    'object_detection': 15.0,
    'background_segmentation': 18.0,
    'camera_analysis': 18.0,
    'text_overlay': 25.0,
    'visual_effects': 22.5,
    'color_analysis': 16.4,
    'content_quality': 11.7,
    'scene_segmentation': 10.6,
    'speech_rate': 10.0,
    'cut_analysis': 4.1,
    'age_estimation': 1.1,
    # CPU analyzers
    'speech_transcription': 4.5,
    'sound_effects': 5.9,
    'speech_emotion': 2.8,
    'speech_flow': 45.5,
    'comment_cta_detection': 0.1,
    'audio_environment': 2.0,
    'temporal_flow': 1.8,
    'audio_analysis': 7.5,
}

def worker_process(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue, 
                  video_path: str, analyzer_configs: Dict[str, Dict], use_cuda_streams: bool = True):
    """Worker process that runs analyzers with optional CUDA streams"""
    try:
        # Set environment
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id % torch.cuda.device_count())
        torch.cuda.set_device(0)
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Import analyzers in worker
        import sys
        sys.path.append('/home/user/tiktok_production')
        from registry_loader import get_ml_analyzer
        
        loaded_analyzers = {}
        cuda_streams = {}
        
        while True:
            try:
                task = task_queue.get(timeout=1)
                if task is None:  # Shutdown signal
                    break
                    
                analyzer_name = task
                logger.info(f"Worker {gpu_id}: Processing {analyzer_name}")
                
                # Load analyzer if not already loaded
                if analyzer_name not in loaded_analyzers:
                    analyzer = get_ml_analyzer(analyzer_name)
                    if analyzer:
                        # Check if it's already an instance or a class
                        if hasattr(analyzer, 'analyze'):
                            # It's already an instance
                            loaded_analyzers[analyzer_name] = analyzer
                        else:
                            # It's a class, instantiate it
                            loaded_analyzers[analyzer_name] = analyzer()
                        logger.info(f"Worker {gpu_id}: Loaded {analyzer_name} - {type(loaded_analyzers[analyzer_name]).__name__}")
                        
                        # Create dedicated CUDA stream for heavy analyzers
                        if use_cuda_streams and analyzer_name in ['qwen2_vl_ultra', 'object_detection', 'background_segmentation']:
                            cuda_streams[analyzer_name] = torch.cuda.Stream()
                    else:
                        logger.error(f"Worker {gpu_id}: Failed to load {analyzer_name}")
                        result_queue.put((analyzer_name, {'error': 'Failed to load analyzer'}))
                        continue
                
                # Run analyzer
                analyzer = loaded_analyzers.get(analyzer_name)
                if analyzer:
                    try:
                        start_time = time.time()
                        
                        # Use CUDA stream for heavy analyzers
                        if analyzer_name in cuda_streams:
                            with torch.cuda.stream(cuda_streams[analyzer_name]):
                                result = analyzer.analyze(video_path)
                                torch.cuda.current_stream().synchronize()
                        else:
                            result = analyzer.analyze(video_path)
                        
                        end_time = time.time()
                        logger.info(f"Worker {gpu_id}: {analyzer_name} completed in {end_time - start_time:.1f}s")
                        result_queue.put((analyzer_name, result))
                    except Exception as e:
                        logger.error(f"Worker {gpu_id}: {analyzer_name} failed: {e}")
                        logger.error(traceback.format_exc())
                        result_queue.put((analyzer_name, {'error': str(e)}))
                
                # Clean GPU memory after heavy analyzers
                if analyzer_name in ['qwen2_vl_ultra', 'object_detection', 'background_segmentation']:
                    torch.cuda.empty_cache()
                    gc.collect()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {gpu_id} error: {e}")
                
    except Exception as e:
        logger.error(f"Worker {gpu_id} fatal error: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"Worker {gpu_id} shutting down")


class OptimizedMultiprocessGPUExecutor:
    """Optimized executor with smart task distribution"""
    
    def __init__(self, num_gpu_processes: int = 4):
        self.num_gpu_processes = num_gpu_processes
        
        # Enable CUDA MPS for better GPU sharing
        os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = '100'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.7'
        
    def create_optimized_task_groups(self, analyzer_list: List[str]) -> Dict[int, List[str]]:
        """Create optimized task groups for parallel execution"""
        # Separate analyzers by type
        heavy_analyzer = None
        gpu_analyzers = []
        cpu_analyzers = []
        
        for analyzer in analyzer_list:
            if analyzer == 'qwen2_vl_ultra':
                heavy_analyzer = analyzer
            elif analyzer in ['speech_transcription', 'sound_effects', 'speech_emotion', 
                            'speech_flow', 'comment_cta_detection', 'audio_environment', 
                            'temporal_flow', 'audio_analysis']:
                cpu_analyzers.append(analyzer)
            else:
                gpu_analyzers.append(analyzer)
        
        # Sort GPU analyzers by estimated time (longest first)
        gpu_analyzers.sort(key=lambda x: ANALYZER_TIMINGS.get(x, 10.0), reverse=True)
        
        # Create task groups
        task_groups = {i: [] for i in range(self.num_gpu_processes)}
        
        # Assign heavy analyzer to worker 0
        if heavy_analyzer:
            task_groups[0].append(heavy_analyzer)
        
        # Distribute other GPU analyzers across remaining workers
        worker_id = 1
        for analyzer in gpu_analyzers:
            task_groups[worker_id].append(analyzer)
            worker_id = (worker_id + 1) % self.num_gpu_processes
            if worker_id == 0:  # Skip worker 0 if it has heavy analyzer
                worker_id = 1
        
        # Distribute CPU analyzers evenly
        for i, analyzer in enumerate(cpu_analyzers):
            worker_id = (i % (self.num_gpu_processes - 1)) + 1
            task_groups[worker_id].append(analyzer)
        
        return task_groups
    
    def execute_parallel(self, video_path: str, analyzer_list: List[str]) -> Dict[str, Any]:
        """Execute analysis with optimized multiprocess GPU parallelization"""
        logger.info(f"Starting optimized analysis with {len(analyzer_list)} analyzers across {self.num_gpu_processes} GPU processes")
        
        # Extract metadata first
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        metadata = {
            'video_path': str(video_path),
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'total_analyzers': len(analyzer_list)
        }
        
        # Create optimized task groups
        task_groups = self.create_optimized_task_groups(analyzer_list)
        
        # Log task distribution
        for worker_id, tasks in task_groups.items():
            total_time = sum(ANALYZER_TIMINGS.get(t, 10.0) for t in tasks)
            logger.info(f"Worker {worker_id}: {len(tasks)} tasks, estimated {total_time:.1f}s - {tasks}")
        
        # Create queues
        task_queues = {i: mp.Queue() for i in range(self.num_gpu_processes)}
        result_queue = mp.Queue()
        
        # Start worker processes
        processes = []
        for gpu_id in range(self.num_gpu_processes):
            # Enable CUDA streams for better parallelism
            use_streams = gpu_id == 0  # Use streams for worker 0 with heavy analyzer
            p = mp.Process(target=worker_process, 
                          args=(gpu_id, task_queues[gpu_id], result_queue, video_path, {}, use_streams))
            p.start()
            processes.append(p)
            
        logger.info(f"Started {self.num_gpu_processes} optimized GPU worker processes")
        
        # Submit tasks to appropriate workers
        total_tasks = 0
        for worker_id, tasks in task_groups.items():
            for task in tasks:
                task_queues[worker_id].put(task)
                total_tasks += 1
        
        # Signal workers to shut down after tasks
        for worker_id in range(self.num_gpu_processes):
            task_queues[worker_id].put(None)
        
        # Collect results
        results = {}
        completed = 0
        
        while completed < total_tasks:
            try:
                analyzer_name, result = result_queue.get(timeout=600)  # 10 minute timeout
                results[analyzer_name] = result
                completed += 1
                logger.info(f"Progress: {completed}/{total_tasks} analyzers completed")
            except Exception as e:
                logger.error(f"Error collecting results: {e}")
                break
        
        # Wait for processes to finish
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                logger.warning(f"Terminating stuck process {p.pid}")
                p.terminate()
                p.join()
        
        # Add metadata to results
        results['metadata'] = metadata
        
        return results