#!/usr/bin/env python3
"""
Ultimate GPU Executor - Optimized for <3x Realtime Performance
"""
import multiprocessing as mp
import torch
import time
import logging
import gc
import traceback
from queue import Empty
from typing import Dict, List, Any, Optional
import numpy as np

# Import performance config
from configs.ultimate_performance_config import (
    GPU_PROCESS_CONFIG, ULTIMATE_BATCH_SIZES, 
    ULTIMATE_FRAME_SAMPLING, ANALYZER_TIMEOUTS
)

logger = logging.getLogger(__name__)


def gpu_worker_ultimate(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue,
                       analyzer_configs: Dict[str, Dict], video_path: str,
                       process_assignment: Dict[str, List[str]]):
    """Ultimate GPU worker with optimized batch processing"""
    try:
        # Set GPU and optimize settings
        torch.cuda.set_device(gpu_id % torch.cuda.device_count())
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Pre-allocate GPU memory fraction
        torch.cuda.set_per_process_memory_fraction(
            GPU_PROCESS_CONFIG['gpu_memory_fraction']
        )
        
        loaded_analyzers = {}
        my_analyzers = process_assignment.get(f'process_{gpu_id}', [])
        
        # Pre-load assigned analyzers
        for analyzer_name in my_analyzers:
            if analyzer_name in analyzer_configs and analyzer_name != 'others':
                try:
                    config = analyzer_configs[analyzer_name]
                    module = __import__(config['module'], fromlist=[config['class']])
                    analyzer_class = getattr(module, config['class'])
                    analyzer = analyzer_class()
                    
                    # Set batch size if applicable
                    if hasattr(analyzer, 'batch_size'):
                        analyzer.batch_size = ULTIMATE_BATCH_SIZES.get(
                            analyzer_name, 8
                        )
                    
                    loaded_analyzers[analyzer_name] = analyzer
                    logger.info(f"Worker {gpu_id}: Pre-loaded {analyzer_name}")
                except Exception as e:
                    logger.error(f"Worker {gpu_id}: Failed to pre-load {analyzer_name}: {e}")
        
        # Process tasks
        while True:
            try:
                analyzer_name = task_queue.get(timeout=2)
                if analyzer_name is None:
                    break
                
                # Skip if not assigned to this worker (unless in 'others')
                if analyzer_name not in my_analyzers and 'others' not in my_analyzers:
                    task_queue.put(analyzer_name)  # Put back for another worker
                    continue
                
                # Load analyzer if needed
                if analyzer_name not in loaded_analyzers:
                    try:
                        config = analyzer_configs[analyzer_name]
                        module = __import__(config['module'], fromlist=[config['class']])
                        analyzer_class = getattr(module, config['class'])
                        analyzer = analyzer_class()
                        
                        # Set batch size
                        if hasattr(analyzer, 'batch_size'):
                            analyzer.batch_size = ULTIMATE_BATCH_SIZES.get(
                                analyzer_name, 8
                            )
                        
                        loaded_analyzers[analyzer_name] = analyzer
                        logger.info(f"Worker {gpu_id}: Loaded {analyzer_name}")
                    except Exception as e:
                        logger.error(f"Worker {gpu_id}: Failed to load {analyzer_name}: {e}")
                        result_queue.put((analyzer_name, {"error": str(e)}))
                        continue
                
                # Run analysis with timeout
                try:
                    start_time = time.time()
                    analyzer = loaded_analyzers[analyzer_name]
                    
                    # Apply frame sampling if configured
                    if hasattr(analyzer, 'frame_interval') and analyzer_name in ULTIMATE_FRAME_SAMPLING:
                        sampling = ULTIMATE_FRAME_SAMPLING[analyzer_name]
                        analyzer.frame_interval = sampling['interval']
                        if hasattr(analyzer, 'max_frames'):
                            analyzer.max_frames = sampling['max_frames']
                    
                    # Run analysis
                    result = analyzer.analyze(video_path)
                    elapsed = time.time() - start_time
                    
                    logger.info(f"Worker {gpu_id}: {analyzer_name} completed in {elapsed:.1f}s")
                    result_queue.put((analyzer_name, result))
                    
                    # Memory cleanup after heavy models
                    if analyzer_name in ['video_llava', 'object_detection', 'background_segmentation']:
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                except Exception as e:
                    logger.error(f"Worker {gpu_id}: {analyzer_name} failed: {e}")
                    logger.error(traceback.format_exc())
                    result_queue.put((analyzer_name, {"error": str(e)}))
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {gpu_id} error: {e}")
                
    except Exception as e:
        logger.error(f"Worker {gpu_id} fatal error: {e}")
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        logger.info(f"Worker {gpu_id} shutting down")


class UltimateGPUExecutor:
    """Ultimate executor optimized for <3x realtime performance"""
    
    def __init__(self, analyzer_configs: Dict[str, Dict]):
        self.analyzer_configs = analyzer_configs
        self.num_processes = GPU_PROCESS_CONFIG['num_processes']
        self.process_assignment = GPU_PROCESS_CONFIG['batch_priority']
        
    def execute_parallel(self, video_path: str, 
                        selected_analyzers: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute analysis with optimized GPU parallelization"""
        
        if selected_analyzers is None:
            selected_analyzers = list(self.analyzer_configs.keys())
        
        # Filter out unknown analyzers
        valid_analyzers = [a for a in selected_analyzers if a in self.analyzer_configs]
        unknown = set(selected_analyzers) - set(valid_analyzers)
        if unknown:
            logger.warning(f"Skipping unknown analyzers: {unknown}")
        
        start_time = time.time()
        
        # Setup multiprocessing
        mp.set_start_method('spawn', force=True)
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # Sort analyzers by priority
        sorted_analyzers = sorted(
            valid_analyzers,
            key=lambda x: self.analyzer_configs[x].get('priority', 99)
        )
        
        # Add tasks to queue
        for analyzer in sorted_analyzers:
            task_queue.put(analyzer)
        
        # Add sentinels
        for _ in range(self.num_processes):
            task_queue.put(None)
        
        # Start worker processes
        processes = []
        for i in range(self.num_processes):
            p = mp.Process(
                target=gpu_worker_ultimate,
                args=(i, task_queue, result_queue, self.analyzer_configs, 
                      video_path, self.process_assignment)
            )
            p.start()
            processes.append(p)
        
        logger.info(f"Started {self.num_processes} GPU worker processes")
        
        # Collect results with progress tracking
        results = {}
        completed = 0
        total = len(valid_analyzers)
        
        # Timeout based on video duration (estimate 3x realtime max)
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = frame_count / fps if fps > 0 else 60
        cap.release()
        
        max_timeout = max(video_duration * 3, 180)  # 3x realtime or 3 minutes min
        timeout_time = time.time() + max_timeout
        
        while completed < total and time.time() < timeout_time:
            try:
                analyzer_name, result = result_queue.get(timeout=5)
                results[analyzer_name] = result
                completed += 1
                logger.info(f"Progress: {completed}/{total} analyzers completed")
            except Empty:
                # Check if processes are still alive
                alive = sum(1 for p in processes if p.is_alive())
                if alive == 0:
                    logger.warning("All worker processes have terminated")
                    break
        
        if time.time() >= timeout_time:
            logger.warning("Timeout waiting for results")
        
        # Terminate processes
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        
        processing_time = time.time() - start_time
        realtime_factor = processing_time / video_duration if video_duration > 0 else 0
        
        logger.info(f"Analysis complete in {processing_time:.1f}s ({realtime_factor:.2f}x realtime)")
        
        # Calculate reconstruction score
        successful = sum(1 for r in results.values() if 'error' not in r)
        reconstruction_score = (successful / total * 100) if total > 0 else 0
        
        return {
            'analyzer_results': results,
            'metadata': {
                'total_analyzers': total,
                'successful_analyzers': successful,
                'failed_analyzers': total - successful,
                'processing_time': processing_time,
                'video_duration': video_duration,
                'realtime_factor': realtime_factor,
                'reconstruction_score': reconstruction_score
            }
        }