#!/usr/bin/env python3
"""
Celery tasks for TikTok video processing pipeline
"""

from celery import Task, group, chain
from celery.utils.log import get_task_logger
from mass_processing.celery_config import app
from mass_processing.tiktok_downloader import TikTokDownloader
from mass_processing.queue_manager import QueueManager
import torch
import time
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List
import psutil
import GPUtil

# Add parent directory to path
sys.path.append('/home/user/tiktok_production')

logger = get_task_logger(__name__)

class GPUTask(Task):
    """Base task class for GPU processing with resource management"""
    
    _pipeline = None
    _gpu_id = None
    
    @property
    def pipeline(self):
        if self._pipeline is None:
            self._initialize_pipeline()
        return self._pipeline
        
    def _initialize_pipeline(self):
        """Initialize GPU pipeline with proper GPU selection"""
        from api.gpu_pipeline_v2 import GPUPipelineV2
        
        # Get available GPUs
        gpus = GPUtil.getGPUs()
        if gpus:
            # Select GPU with most free memory
            gpu = max(gpus, key=lambda g: g.memoryFree)
            self._gpu_id = gpu.id
            torch.cuda.set_device(self._gpu_id)
            logger.info(f"Initialized GPU pipeline on GPU {self._gpu_id}")
        
        self._pipeline = GPUPipelineV2()
        
    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        """Cleanup after task completion"""
        if self._pipeline:
            # Clear GPU cache
            torch.cuda.empty_cache()
            
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(f"Task {task_id} failed: {exc}")
        # Update failure metrics
        update_task_metrics(self.name, 'failed', str(exc))


@app.task(bind=True, base=GPUTask, max_retries=3)
def process_video_gpu_task(self, video_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process video using GPU pipeline
    
    Args:
        video_path: Path to video file
        metadata: Video metadata from downloader
        
    Returns:
        Dict with processing results
    """
    start_time = time.time()
    worker_id = f"gpu_worker_{self.request.hostname}"
    
    try:
        # Update worker status
        update_worker_status(worker_id, 'busy', f"Processing {metadata['tiktok_id']}")
        
        # Process video
        logger.info(f"Processing video: {video_path} on GPU {self._gpu_id}")
        
        # Get pipeline and process
        results = self.pipeline.analyze_video(video_path)
        
        # Add metadata to results
        results['video_metadata'] = metadata
        
        # Save to Supabase
        from api.supabase_integration import SupabaseAnalysisSaver
        
        video_id = SupabaseAnalysisSaver.save_complete_analysis(
            tiktok_url=metadata['tiktok_url'],
            all_results=results.get('analysis_results', {}),
            processing_times=results.get('processing_times', {}),
            file_path=video_path
        )
        
        # Update metrics
        processing_time = time.time() - start_time
        update_task_metrics('gpu_processing', 'success', None, processing_time)
        
        logger.info(f"GPU processing completed for {metadata['tiktok_id']} in {processing_time:.1f}s")
        
        return {
            'success': True,
            'video_id': video_id,
            'tiktok_id': metadata['tiktok_id'],
            'processing_time': processing_time,
            'analyzers_completed': len(results.get('analysis_results', {}))
        }
        
    except Exception as e:
        logger.error(f"GPU processing failed for {video_path}: {e}")
        
        # Retry or fallback to CPU
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60 * (self.request.retries + 1))
        else:
            # Fallback to CPU processing
            logger.info("Falling back to CPU processing")
            return process_video_cpu_task.apply_async(
                args=[video_path, metadata],
                queue='cpu_processing'
            )
            
    finally:
        update_worker_status(worker_id, 'idle', None)


@app.task(bind=True, max_retries=3)
def download_tiktok_task(self, url: str, priority: int = 5) -> Dict[str, Any]:
    """
    Download TikTok video
    
    Args:
        url: TikTok video URL
        priority: Processing priority
        
    Returns:
        Dict with download results
    """
    try:
        downloader = TikTokDownloader()
        result = downloader.download_video(url, priority)
        
        if result['success']:
            # Queue for processing based on priority
            if priority >= 8:
                task = process_video_gpu_task.apply_async(
                    args=[result['video_path'], result['metadata']],
                    queue='priority',
                    priority=priority
                )
            else:
                task = process_video_gpu_task.apply_async(
                    args=[result['video_path'], result['metadata']],
                    queue='gpu_processing',
                    priority=priority
                )
                
            logger.info(f"Queued {result['video_id']} for processing")
            
            # Mark URL as processed
            queue_manager = QueueManager()
            queue_manager.mark_url_processed(url)
            
            return result
            
        else:
            # Handle download failure
            error = result.get('error', 'Unknown error')
            
            # Check if we should retry
            if 'private' in error or 'removed' in error or '404' in error:
                # Don't retry for permanent failures
                logger.warning(f"Permanent failure for {url}: {error}")
                queue_manager = QueueManager()
                queue_manager.mark_url_failed(url, error, permanent=True)
                return result
                
            # Retry for temporary failures
            raise Exception(error)
            
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        
        if self.request.retries < self.max_retries:
            # Exponential backoff
            countdown = 30 * (2 ** self.request.retries)
            raise self.retry(exc=e, countdown=countdown)
        else:
            # Final failure
            queue_manager = QueueManager()
            queue_manager.mark_url_failed(url, str(e))
            raise


@app.task(bind=True)
def process_video_cpu_task(self, video_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process video using CPU-only analyzers
    Fallback for when GPU processing fails
    """
    from analyzers.cpu_batch_audio_environment import CPUBatchAudioEnvironment
    from analyzers.cpu_batch_scene_description import CPUBatchSceneDescription
    from analyzers.cpu_batch_temporal_flow import CPUBatchTemporalFlow
    
    start_time = time.time()
    worker_id = f"cpu_worker_{self.request.hostname}"
    
    try:
        update_worker_status(worker_id, 'busy', f"CPU Processing {metadata['tiktok_id']}")
        
        # Run CPU analyzers
        cpu_analyzers = {
            'audio_environment': CPUBatchAudioEnvironment(),
            'scene_description': CPUBatchSceneDescription(),
            'temporal_flow': CPUBatchTemporalFlow(),
        }
        
        results = {}
        processing_times = {}
        
        for name, analyzer in cpu_analyzers.items():
            try:
                analyzer_start = time.time()
                results[name] = analyzer.analyze(video_path)
                processing_times[name] = time.time() - analyzer_start
                logger.info(f"Completed {name} in {processing_times[name]:.1f}s")
            except Exception as e:
                logger.error(f"CPU analyzer {name} failed: {e}")
                results[name] = None
                
        # Save partial results
        from utils.supabase_client import supabase
        
        data = {
            'tiktok_url': metadata['tiktok_url'],
            'tiktok_id': metadata['tiktok_id'],
            'analysis_results': results,
            'processing_times': processing_times,
            'status': 'partial',
            'analyzer_type': 'cpu_only',
        }
        
        supabase.table('video_analysis').insert(data).execute()
        
        processing_time = time.time() - start_time
        update_task_metrics('cpu_processing', 'success', None, processing_time)
        
        return {
            'success': True,
            'tiktok_id': metadata['tiktok_id'],
            'processing_time': processing_time,
            'analyzers_completed': len([r for r in results.values() if r is not None])
        }
        
    finally:
        update_worker_status(worker_id, 'idle', None)


# Monitoring Tasks

@app.task
def cleanup_old_files():
    """Cleanup old video files and free up disk space"""
    from mass_processing.tiktok_downloader import TikTokDownloader
    
    downloader = TikTokDownloader()
    cleaned = downloader.cleanup_old_videos(days=7)
    
    # Check disk usage
    disk_usage = psutil.disk_usage('/')
    if disk_usage.percent > 80:
        # More aggressive cleanup
        cleaned += downloader.cleanup_old_videos(days=3)
        
    logger.info(f"Cleaned up {cleaned} old videos. Disk usage: {disk_usage.percent}%")
    return {'cleaned': cleaned, 'disk_usage': disk_usage.percent}


@app.task
def update_metrics():
    """Update system metrics in database"""
    from utils.supabase_client import supabase
    from mass_processing.queue_manager import QueueManager
    
    # Get queue stats
    queue_manager = QueueManager()
    queue_stats = queue_manager.get_queue_stats()
    
    # Get GPU stats
    gpus = GPUtil.getGPUs()
    avg_gpu_util = sum(gpu.load * 100 for gpu in gpus) / len(gpus) if gpus else 0
    avg_gpu_memory = sum(gpu.memoryUtil * 100 for gpu in gpus) / len(gpus) if gpus else 0
    
    # Get system stats
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    
    # Calculate throughput
    hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
    processed = supabase.table('video_analysis').select('id').gte('analyzed_at', hour_ago).execute()
    throughput = len(processed.data) if processed.data else 0
    
    # Save metrics
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'queue_download': queue_stats.get('download', 0),
        'queue_processing': queue_stats.get('processing', 0),
        'queue_priority': queue_stats.get('priority', 0),
        'queue_failed': queue_stats.get('failed', 0),
        'total_processed': queue_stats.get('processed', 0),
        'throughput_per_hour': throughput,
        'gpu_utilization': avg_gpu_util,
        'gpu_memory': avg_gpu_memory,
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
    }
    
    supabase.table('performance_metrics').insert(metrics).execute()
    logger.info(f"Updated metrics: {throughput} videos/hour, GPU: {avg_gpu_util:.1f}%")
    
    return metrics


@app.task
def worker_health_check():
    """Check health of all workers and restart if needed"""
    from utils.supabase_client import supabase
    import subprocess
    
    # Get all workers
    workers = supabase.table('worker_status').select('*').execute().data
    
    unhealthy_workers = []
    
    for worker in workers:
        last_heartbeat = datetime.fromisoformat(worker['last_heartbeat'])
        time_since_heartbeat = (datetime.now() - last_heartbeat).total_seconds()
        
        # Worker is unhealthy if no heartbeat for 2 minutes
        if time_since_heartbeat > 120:
            unhealthy_workers.append(worker)
            logger.warning(f"Worker {worker['id']} is unhealthy (last seen {time_since_heartbeat:.0f}s ago)")
            
    # Restart unhealthy workers
    if unhealthy_workers:
        # Send alert
        send_alert(f"{len(unhealthy_workers)} workers are unhealthy", unhealthy_workers)
        
        # Attempt to restart workers
        for worker in unhealthy_workers:
            if worker['worker_type'] == 'gpu':
                # Restart GPU worker
                gpu_id = worker['id'].split('_')[-1]
                subprocess.run([
                    'celery', '-A', 'mass_processing.celery_config', 
                    'worker', '-Q', 'gpu_processing,priority',
                    '-n', worker['id'], '--detach'
                ], env={'CUDA_VISIBLE_DEVICES': gpu_id})
                
    return {
        'total_workers': len(workers),
        'unhealthy_workers': len(unhealthy_workers),
        'restarted': len(unhealthy_workers)
    }


@app.task
def requeue_failed_tasks():
    """Requeue failed tasks that haven't exceeded retry limit"""
    queue_manager = QueueManager()
    requeued = queue_manager.requeue_failed(max_retries=3)
    
    logger.info(f"Requeued {requeued} failed tasks")
    return {'requeued': requeued}


@app.task
def optimize_gpu_scheduling():
    """Optimize GPU task scheduling based on current load"""
    from utils.supabase_client import supabase
    
    # Get GPU utilization
    gpus = GPUtil.getGPUs()
    
    for gpu in gpus:
        if gpu.load < 0.7:  # GPU underutilized
            # Check if there are pending tasks
            queue_manager = QueueManager()
            pending = queue_manager.get_queue_stats()
            
            if pending['gpu_processing'] > 0 or pending['priority'] > 0:
                logger.info(f"GPU {gpu.id} underutilized ({gpu.load*100:.1f}%), optimizing scheduling")
                
                # Increase worker concurrency temporarily
                # This would require dynamic worker management
                pass
                
    return {'optimized': True}


# Helper functions

def update_worker_status(worker_id: str, status: str, current_task: str = None):
    """Update worker status in database"""
    from utils.supabase_client import supabase
    
    # Get GPU stats if available
    gpu_stats = None
    if 'gpu' in worker_id:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_id = int(worker_id.split('_')[-1]) if '_' in worker_id else 0
            if gpu_id < len(gpus):
                gpu_stats = gpus[gpu_id]
    
    data = {
        'id': worker_id,
        'worker_type': 'gpu' if 'gpu' in worker_id else 'cpu',
        'status': status,
        'current_task': current_task,
        'gpu_utilization': gpu_stats.load * 100 if gpu_stats else 0,
        'memory_usage': gpu_stats.memoryUtil * 100 if gpu_stats else psutil.virtual_memory().percent,
        'last_heartbeat': datetime.now().isoformat()
    }
    
    supabase.table('worker_status').upsert(data).execute()


def update_task_metrics(task_type: str, status: str, error: str = None, processing_time: float = None):
    """Update task metrics for monitoring"""
    from utils.supabase_client import supabase
    
    data = {
        'task_type': task_type,
        'status': status,
        'error_message': error,
        'processing_time': processing_time,
        'timestamp': datetime.now().isoformat()
    }
    
    supabase.table('task_metrics').insert(data).execute()


def send_alert(message: str, details: Any = None):
    """Send alert via configured channels"""
    logger.error(f"ALERT: {message}")
    if details:
        logger.error(f"Details: {json.dumps(details, indent=2)}")
        
    # TODO: Implement Slack/Email notifications
    # For now, just log