#!/usr/bin/env python3
"""
Multiprocess GPU Executor using Registry Loader
Uses dynamic loading to avoid hardcoded analyzer imports
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

# Set spawn method at module level
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

logger = logging.getLogger(__name__)

def worker_process(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue, 
                  video_path: str, analyzer_configs: Dict[str, Dict], 
                  enable_caching: bool = True):
    """Worker process that runs analyzers with model caching"""
    try:
        # Set environment
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)
        
        # Import analyzers in worker to avoid serialization
        import sys
        sys.path.append('/home/user/tiktok_production')
        
        # Import registry loader - it handles all the dynamic loading
        from registry_loader import get_ml_analyzer
        
        loaded_analyzers = {}
        model_load_times = {}
        
        while True:
            try:
                task = task_queue.get(timeout=1)
                if task is None:  # Shutdown signal
                    break
                    
                analyzer_name = task
                
                # Load analyzer if needed (with caching)
                if analyzer_name not in loaded_analyzers:
                    try:
                        load_start = time.time()
                        
                        # Use registry loader to get analyzer
                        loaded_analyzers[analyzer_name] = get_ml_analyzer(analyzer_name)
                        
                        load_time = time.time() - load_start
                        model_load_times[analyzer_name] = load_time
                        
                        logger.info(f"Worker {gpu_id}: Loaded {analyzer_name} in {load_time:.1f}s")
                        
                        # Log GPU memory after loading
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated() / 1024**2
                            free = torch.cuda.mem_get_info()[0] / 1024**2
                            logger.info(f"Worker {gpu_id}: GPU memory - Allocated: {allocated:.1f}MB, Free: {free:.1f}MB")
                            
                    except Exception as e:
                        logger.error(f"Worker {gpu_id}: Failed to load {analyzer_name}: {e}")
                        logger.error(traceback.format_exc())
                        result_queue.put((analyzer_name, {"error": str(e)}))
                        continue
                else:
                    logger.debug(f"Worker {gpu_id}: Reusing cached {analyzer_name}")
                
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
                
                # Smart GPU cache management
                if not enable_caching:
                    # Clear GPU cache if caching disabled
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    # Only clear if memory is getting low
                    if torch.cuda.is_available():
                        free_memory = torch.cuda.mem_get_info()[0] / 1024**2  # MB
                        if free_memory < 5000:  # Less than 5GB free
                            logger.warning(f"Worker {gpu_id}: Low GPU memory ({free_memory:.1f}MB), clearing cache")
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
        # Cleanup on shutdown
        if enable_caching and 'loaded_analyzers' in locals():
            logger.info(f"Worker {gpu_id}: Cleaning up {len(loaded_analyzers)} cached models")
            for name, analyzer in loaded_analyzers.items():
                try:
                    if hasattr(analyzer, 'model'):
                        # Move to CPU before deletion
                        if hasattr(analyzer.model, 'cpu'):
                            analyzer.model.cpu()
                        del analyzer.model
                except:
                    pass
            loaded_analyzers.clear()
            
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Worker {gpu_id} shutting down")


class MultiprocessGPUExecutorRegistryCached:
    """Executor with persistent model caching for better performance"""
    
    def __init__(self, num_gpu_processes: int = 3, enable_caching: bool = True):
        self.num_gpu_processes = num_gpu_processes
        self.enable_caching = enable_caching
        logger.info(f"Initialized executor with {num_gpu_processes} GPU processes, caching={'enabled' if enable_caching else 'disabled'}")
        
    def execute_parallel(self, video_path: str, analyzer_list: List[str]) -> Dict[str, Any]:
        """Execute analysis with multiprocess GPU parallelization"""
        # Simply use the provided analyzer list
        all_tasks = analyzer_list
        
        logger.info(f"Starting analysis with {len(all_tasks)} analyzers across {self.num_gpu_processes} GPU processes")
        
        # Extract metadata first
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            fps = 30.0  # Default FPS
            frame_count = 0
            duration = 0
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Validate values
            if fps == 0 or frame_count == 0:
                logger.warning(f"Invalid video properties - FPS: {fps}, Frames: {frame_count}")
                # Try alternative method
                fps = 30.0  # Default to 30 FPS
                frame_count = 0
                # Count frames manually if needed
                if cap.isOpened():
                    temp_count = 0
                    while True:
                        ret, _ = cap.read()
                        if not ret:
                            break
                        temp_count += 1
                    frame_count = temp_count
                    duration = frame_count / fps
                    cap.release()
                    cap = cv2.VideoCapture(str(video_path))  # Reopen
            
        cap.release()
        
        metadata = {
            'video_path': str(video_path),
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'total_analyzers': len(all_tasks)
        }
        
        # Create queues
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # Start worker processes
        processes = []
        for gpu_id in range(self.num_gpu_processes):
            p = mp.Process(target=worker_process, args=(gpu_id, task_queue, result_queue, video_path, {}, self.enable_caching))
            p.start()
            processes.append(p)
            
        logger.info(f"Started {self.num_gpu_processes} GPU worker processes")
        
        # Submit tasks
        for task in all_tasks:
            task_queue.put(task)
            
        # Signal workers to shut down after tasks
        for _ in range(self.num_gpu_processes):
            task_queue.put(None)
        
        # Collect results
        results = {}
        completed = 0
        total_tasks = len(all_tasks)
        
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