#!/usr/bin/env python3
"""
Persistent Multiprocess GPU Executor with Model Caching
Workers stay alive between analyses for true model caching
"""

import torch
import torch.multiprocessing as mp
import time
import logging
import os
import gc
import traceback
from typing import Dict, List, Any, Tuple, Optional
from queue import Empty
import threading

# Set spawn method at module level
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

logger = logging.getLogger(__name__)

class PersistentWorkerProcess:
    """Persistent worker that keeps models loaded between analyses"""
    
    def __init__(self, gpu_id: int, command_queue: mp.Queue, result_queue: mp.Queue):
        self.gpu_id = gpu_id
        self.command_queue = command_queue
        self.result_queue = result_queue
        self.loaded_analyzers = {}
        self.running = True
        
    def run(self):
        """Main worker loop - stays alive between analyses"""
        try:
            # Set environment
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            torch.cuda.set_device(0)
            
            # Set GPU memory fraction based on worker ID
            if torch.cuda.is_available():
                if self.gpu_id == 0:
                    torch.cuda.set_per_process_memory_fraction(0.85)  # Qwen2-VL
                    logger.info(f"Worker {self.gpu_id}: Set memory to 85% for Qwen2-VL")
                else:
                    torch.cuda.set_per_process_memory_fraction(0.10)  # Light analyzers
                    logger.info(f"Worker {self.gpu_id}: Set memory to 10% for light analyzers")
                    
                torch.backends.cudnn.benchmark = True
            
            # Import after GPU setup
            import sys
            sys.path.append('/home/user/tiktok_production')
            from registry_loader import get_ml_analyzer
            
            logger.info(f"Worker {self.gpu_id}: Started and ready")
            
            while self.running:
                try:
                    # Wait for command (timeout allows periodic checks)
                    command = self.command_queue.get(timeout=1.0)
                    
                    if command is None:  # Shutdown signal
                        logger.info(f"Worker {self.gpu_id}: Received shutdown signal")
                        break
                        
                    cmd_type = command.get('type')
                    
                    if cmd_type == 'analyze':
                        analyzer_name = command['analyzer']
                        video_path = command['video_path']
                        task_id = command['task_id']
                        
                        # Check if this worker should handle this analyzer
                        if self.gpu_id == 0 and analyzer_name != 'qwen2_vl_temporal':
                            continue
                        elif self.gpu_id != 0 and analyzer_name == 'qwen2_vl_temporal':
                            continue
                        
                        # Load analyzer if needed (CACHED!)
                        if analyzer_name not in self.loaded_analyzers:
                            try:
                                logger.info(f"Worker {self.gpu_id}: Loading {analyzer_name}...")
                                load_start = time.time()
                                
                                analyzer = get_ml_analyzer(analyzer_name)
                                self.loaded_analyzers[analyzer_name] = analyzer
                                
                                load_time = time.time() - load_start
                                logger.info(f"Worker {self.gpu_id}: Loaded {analyzer_name} in {load_time:.1f}s")
                                
                                # Report GPU memory
                                if torch.cuda.is_available():
                                    allocated = torch.cuda.memory_allocated() / 1024**2
                                    logger.info(f"Worker {self.gpu_id}: GPU memory: {allocated:.1f}MB")
                            except Exception as e:
                                logger.error(f"Worker {self.gpu_id}: Failed to load {analyzer_name}: {e}")
                                self.result_queue.put({
                                    'task_id': task_id,
                                    'analyzer': analyzer_name,
                                    'result': {"error": str(e)}
                                })
                                continue
                        else:
                            logger.info(f"Worker {self.gpu_id}: Reusing cached {analyzer_name}")
                        
                        # Run analysis
                        try:
                            analyzer = self.loaded_analyzers[analyzer_name]
                            start_time = time.time()
                            
                            result = analyzer.analyze(video_path)
                            
                            analysis_time = time.time() - start_time
                            logger.info(f"Worker {self.gpu_id}: {analyzer_name} completed in {analysis_time:.1f}s")
                            
                            # DEBUG: Critical logging for Qwen2-VL result verification
                            if analyzer_name == 'qwen2_vl_temporal':
                                logger.info(f"Worker {self.gpu_id}: QWEN2-VL RESULT DEBUG:")
                                logger.info(f"  Result type: {type(result)}")
                                logger.info(f"  Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                                if isinstance(result, dict):
                                    segments = result.get('segments', [])
                                    logger.info(f"  Segments count: {len(segments)}")
                                    if segments:
                                        logger.info(f"  First segment: {segments[0]}")
                                        logger.info(f"  Last segment: {segments[-1]}")
                                    else:
                                        logger.error(f"Worker {self.gpu_id}: CRITICAL - Qwen2-VL result has EMPTY segments!")
                                else:
                                    logger.error(f"Worker {self.gpu_id}: CRITICAL - Qwen2-VL result is not a dict!")
                            
                            self.result_queue.put({
                                'task_id': task_id,
                                'analyzer': analyzer_name,
                                'result': result
                            })
                            
                        except Exception as e:
                            logger.error(f"Worker {self.gpu_id}: Analysis failed for {analyzer_name}: {e}")
                            logger.error(traceback.format_exc())
                            self.result_queue.put({
                                'task_id': task_id,
                                'analyzer': analyzer_name,
                                'result': {"error": str(e)}
                            })
                            
                    elif cmd_type == 'clear_cache':
                        # Clear specific or all models
                        model_name = command.get('model')
                        if model_name and model_name in self.loaded_analyzers:
                            del self.loaded_analyzers[model_name]
                            logger.info(f"Worker {self.gpu_id}: Cleared {model_name}")
                        elif not model_name:
                            self.loaded_analyzers.clear()
                            logger.info(f"Worker {self.gpu_id}: Cleared all models")
                        
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                except Empty:
                    # No command, continue loop
                    continue
                except Exception as e:
                    logger.error(f"Worker {self.gpu_id}: Error in main loop: {e}")
                    logger.error(traceback.format_exc())
                    
        except Exception as e:
            logger.error(f"Worker {self.gpu_id}: Fatal error: {e}")
            logger.error(traceback.format_exc())
        finally:
            logger.info(f"Worker {self.gpu_id}: Shutting down")
            # Cleanup
            self.loaded_analyzers.clear()
            torch.cuda.empty_cache()
            gc.collect()


class PersistentGPUExecutor:
    """Executor with persistent workers for true model caching"""
    
    def __init__(self, num_gpu_processes: int = 3):
        self.num_gpu_processes = num_gpu_processes
        self.command_queues = []
        self.result_queue = mp.Queue()
        self.processes = []
        self.started = False
        self._task_counter = 0
        self._lock = threading.Lock()
        
        logger.info(f"Initialized PersistentGPUExecutor with {num_gpu_processes} workers")
        
    def start_workers(self):
        """Start persistent worker processes"""
        if self.started:
            return
            
        logger.info("Starting persistent GPU workers...")
        
        for gpu_id in range(self.num_gpu_processes):
            cmd_queue = mp.Queue()
            self.command_queues.append(cmd_queue)
            
            # Create worker process
            worker = PersistentWorkerProcess(gpu_id, cmd_queue, self.result_queue)
            p = mp.Process(target=worker.run)
            p.daemon = False  # NOT daemon - we want to control shutdown
            p.start()
            self.processes.append(p)
            
        self.started = True
        logger.info(f"Started {self.num_gpu_processes} persistent GPU workers")
        
        # Wait a bit for workers to initialize
        time.sleep(2)
        
    def execute_parallel(self, video_path: str, analyzer_list: List[str]) -> Dict[str, Any]:
        """Execute analysis using persistent workers"""
        if not self.started:
            self.start_workers()
            
        logger.info(f"Starting analysis with {len(analyzer_list)} analyzers")
        
        # Extract metadata
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
        
        # Submit tasks to workers
        task_map = {}
        
        for analyzer in analyzer_list:
            with self._lock:
                task_id = f"task_{self._task_counter}"
                self._task_counter += 1
                
            task_map[task_id] = analyzer
            
            # Route to appropriate worker
            if analyzer == 'qwen2_vl_temporal':
                worker_id = 0  # Always to Worker 0
            else:
                # Distribute other tasks between Workers 1 and 2
                worker_id = 1 + (hash(analyzer) % 2)
                
            self.command_queues[worker_id].put({
                'type': 'analyze',
                'task_id': task_id,
                'analyzer': analyzer,
                'video_path': video_path
            })
            
        # Collect results
        results = {}
        completed = 0
        total_tasks = len(analyzer_list)
        
        while completed < total_tasks:
            try:
                result_data = self.result_queue.get(timeout=600)  # 10 min timeout
                
                task_id = result_data['task_id']
                analyzer_name = result_data['analyzer']
                result = result_data['result']
                
                # DEBUG: Critical logging for Qwen2-VL result verification at queue level
                if analyzer_name == 'qwen2_vl_temporal':
                    logger.info(f"MAIN EXECUTOR: QWEN2-VL RESULT RECEIVED:")
                    logger.info(f"  Result type: {type(result)}")
                    logger.info(f"  Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                    if isinstance(result, dict):
                        segments = result.get('segments', [])
                        logger.info(f"  Segments count in main executor: {len(segments)}")
                        if segments:
                            logger.info(f"  First segment received: {segments[0]}")
                        else:
                            logger.error(f"MAIN EXECUTOR: CRITICAL - Received EMPTY segments from Qwen2-VL!")
                
                results[analyzer_name] = result
                completed += 1
                
                logger.info(f"Progress: {completed}/{total_tasks} analyzers completed")
                
            except Exception as e:
                logger.error(f"Error collecting results: {e}")
                break
                
        # Add metadata
        results['metadata'] = metadata
        
        return results
        
    def shutdown(self):
        """Shutdown all workers gracefully"""
        if not self.started:
            return
            
        logger.info("Shutting down persistent workers...")
        
        # Send shutdown signal to all workers
        for queue in self.command_queues:
            queue.put(None)
            
        # Wait for processes to finish
        for p in self.processes:
            p.join(timeout=10)
            if p.is_alive():
                logger.warning(f"Force terminating process {p.pid}")
                p.terminate()
                p.join()
                
        self.started = False
        logger.info("All workers shut down")
        
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear model cache on all workers"""
        for queue in self.command_queues:
            queue.put({
                'type': 'clear_cache',
                'model': model_name
            })
            
    def __del__(self):
        """Cleanup on deletion"""
        self.shutdown()