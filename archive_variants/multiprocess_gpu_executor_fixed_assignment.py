#!/usr/bin/env python3
"""
Multiprocess GPU Executor with Fixed Worker Assignment
Ensures Worker 0 always processes qwen2_vl_temporal
"""

import torch
import torch.multiprocessing as mp
import time
import logging
import os
import gc
import signal
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
        # Prevent zombie processes
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        
        # Set environment
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)
        
        # Import analyzers in worker to avoid serialization
        import sys
        sys.path.append('/home/user/tiktok_production')
        sys.path.append('/home/user/tiktok_analyzer')
        
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


class MultiprocessGPUExecutorFixedAssignment:
    """Executor with fixed worker-task assignment to ensure Worker 0 gets qwen2_vl_temporal"""
    
    def __init__(self, num_gpu_processes: int = 3, enable_caching: bool = True):
        self.num_gpu_processes = num_gpu_processes
        self.enable_caching = enable_caching
        logger.info(f"Initialized executor with {num_gpu_processes} GPU processes, caching={'enabled' if enable_caching else 'disabled'}")
        
    def execute_parallel(self, video_path: str, analyzer_list: List[str]) -> Dict[str, Any]:
        """Execute analysis with fixed worker-task assignment"""
        
        # Separate analyzers by worker assignment
        worker_tasks = defaultdict(list)
        
        # Ensure qwen2_vl_temporal goes to Worker 0
        if 'qwen2_vl_temporal' in analyzer_list:
            worker_tasks[0].append('qwen2_vl_temporal')
            analyzer_list = [a for a in analyzer_list if a != 'qwen2_vl_temporal']
        
        # Distribute remaining tasks round-robin starting from Worker 1
        worker_id = 1
        for analyzer in analyzer_list:
            worker_tasks[worker_id % self.num_gpu_processes].append(analyzer)
            if worker_id % self.num_gpu_processes != 0:  # Skip Worker 0 for round-robin
                worker_id += 1
            else:
                worker_id = 1  # Jump back to Worker 1
        
        # Log task distribution
        for worker_id, tasks in worker_tasks.items():
            logger.info(f"Worker {worker_id} assigned: {', '.join(tasks)}")
        
        # Extract metadata first
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            fps = 30.0
            frame_count = 0
            duration = 0
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        metadata = {
            'video_path': str(video_path),
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'total_analyzers': sum(len(tasks) for tasks in worker_tasks.values())
        }
        
        # Create separate queues for each worker
        task_queues = {i: mp.Queue() for i in range(self.num_gpu_processes)}
        result_queue = mp.Queue()
        
        # Start worker processes
        processes = []
        for gpu_id in range(self.num_gpu_processes):
            p = mp.Process(
                target=worker_process, 
                args=(gpu_id, task_queues[gpu_id], result_queue, video_path, {}, self.enable_caching)
            )
            p.start()
            processes.append(p)
            
        logger.info(f"Started {self.num_gpu_processes} GPU worker processes with fixed assignment")
        
        # Submit tasks to specific workers
        total_tasks = 0
        for worker_id, tasks in worker_tasks.items():
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