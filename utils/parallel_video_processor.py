#!/usr/bin/env python3
"""
Parallel Video Processor for Maximum GPU Utilization
Processes multiple videos concurrently to achieve 90%+ GPU usage
"""

import asyncio
import torch
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
import os

logger = logging.getLogger(__name__)

@dataclass
class VideoJob:
    """Represents a video analysis job"""
    job_id: str
    video_path: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result_path: Optional[str] = None
    error: Optional[str] = None
    progress: float = 0.0

class ParallelVideoProcessor:
    """Processes multiple videos in parallel for maximum GPU utilization"""
    
    def __init__(self, max_concurrent_videos: int = 3):
        self.max_concurrent_videos = max_concurrent_videos
        self.active_jobs: Dict[str, VideoJob] = {}
        self.job_queue: List[VideoJob] = []
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_videos)
        
        # GPU memory allocation per video
        # With preloaded models, we only need memory for activations
        total_gpu_memory = 45000  # MB for RTX 8000
        preloaded_memory = 32000  # ~32GB for preloaded models
        reserved_memory = 3000    # Keep 3GB free
        available_for_videos = total_gpu_memory - preloaded_memory - reserved_memory
        self.memory_per_video = available_for_videos // max_concurrent_videos
        
        logger.info(f"ParallelVideoProcessor initialized:")
        logger.info(f"  - Max concurrent videos: {max_concurrent_videos}")
        logger.info(f"  - Memory per video: {self.memory_per_video} MB")
    
    async def add_video(self, video_path: str) -> str:
        """Add a video to the processing queue"""
        job_id = str(uuid.uuid4())
        job = VideoJob(
            job_id=job_id,
            video_path=video_path,
            status='pending'
        )
        
        self.job_queue.append(job)
        logger.info(f"Added video to queue: {job_id} - {video_path}")
        
        # Try to process immediately if slots available
        await self._process_queue()
        
        return job_id
    
    async def add_videos_batch(self, video_paths: List[str]) -> List[str]:
        """Add multiple videos to the queue"""
        job_ids = []
        for video_path in video_paths:
            job_id = await self.add_video(video_path)
            job_ids.append(job_id)
        return job_ids
    
    def get_job_status(self, job_id: str) -> Optional[VideoJob]:
        """Get the status of a specific job"""
        # Check active jobs
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        # Check queue
        for job in self.job_queue:
            if job.job_id == job_id:
                return job
        
        return None
    
    def get_all_jobs(self) -> Dict[str, List[VideoJob]]:
        """Get all jobs grouped by status"""
        result = {
            'processing': list(self.active_jobs.values()),
            'pending': self.job_queue.copy(),
            'completed': [],
            'failed': []
        }
        
        # Add completed/failed from active jobs
        for job in self.active_jobs.values():
            if job.status == 'completed':
                result['completed'].append(job)
            elif job.status == 'failed':
                result['failed'].append(job)
        
        return result
    
    async def _process_queue(self):
        """Process videos from the queue when slots are available"""
        while len(self.active_jobs) < self.max_concurrent_videos and self.job_queue:
            # Check GPU memory before starting new job
            if not self._check_gpu_memory():
                logger.warning("Insufficient GPU memory for new job")
                break
            
            # Get next job from queue
            job = self.job_queue.pop(0)
            job.status = 'processing'
            job.start_time = time.time()
            
            self.active_jobs[job.job_id] = job
            
            # Start processing in background
            asyncio.create_task(self._process_video(job))
    
    async def _process_video(self, job: VideoJob):
        """Process a single video"""
        try:
            logger.info(f"Starting processing job {job.job_id}: {job.video_path}")
            
            # Import here to avoid circular imports
            from api.stable_production_api_multiprocess import engine
            
            # Run analysis in thread pool to not block
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_analysis,
                job.video_path
            )
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_id = os.path.basename(job.video_path).split('.')[0]
            result_path = f"/home/user/tiktok_production/results/{video_id}_parallel_{timestamp}.json"
            
            # Use custom JSON encoder for numpy types
            import json
            from utils.json_encoder import NumpyEncoder
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2, cls=NumpyEncoder)
            
            # Update job
            job.status = 'completed'
            job.end_time = time.time()
            job.result_path = result_path
            job.progress = 100.0
            
            processing_time = job.end_time - job.start_time
            logger.info(f"Completed job {job.job_id} in {processing_time:.1f}s")
            
        except Exception as e:
            logger.error(f"Failed job {job.job_id}: {e}")
            job.status = 'failed'
            job.end_time = time.time()
            job.error = str(e)
        
        finally:
            # Remove from active jobs after a delay
            await asyncio.sleep(5)
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            
            # Process next video in queue
            await self._process_queue()
    
    def _run_analysis(self, video_path: str) -> Dict[str, Any]:
        """Run the actual analysis (blocking)"""
        # Import here to avoid circular imports
        import sys
        sys.path.append('/home/user/tiktok_production')
        
        from utils.multiprocess_gpu_executor_registry import MultiprocessGPUExecutorRegistry
        from registry_loader import ML_ANALYZERS
        from configs.gpu_groups_config import GPU_ANALYZER_GROUPS, DISABLED_ANALYZERS
        
        # Get active analyzers
        active_analyzers = []
        for group_name, analyzer_list in GPU_ANALYZER_GROUPS.items():
            for analyzer in analyzer_list:
                if analyzer not in DISABLED_ANALYZERS and analyzer in ML_ANALYZERS:
                    active_analyzers.append(analyzer)
        
        # Remove duplicates
        active_analyzers = list(dict.fromkeys(active_analyzers))
        
        # Create executor for this video
        executor = MultiprocessGPUExecutorRegistry(num_gpu_processes=3)
        
        # Run analysis
        results = executor.execute_parallel(video_path, active_analyzers)
        
        return results
    
    def _check_gpu_memory(self) -> bool:
        """Check if there's enough GPU memory for a new job"""
        if not torch.cuda.is_available():
            return True
        
        try:
            # With preloaded models, we need less memory per video
            # Models are already loaded, so we only need memory for activations
            free_memory = (torch.cuda.get_device_properties(0).total_memory - 
                          torch.cuda.memory_allocated()) / 1024**2
            
            # Only need ~5GB per additional video for activations
            required_memory = 5000  # 5GB for activations per video
            
            # Always allow at least one video
            if len(self.active_jobs) == 0:
                return True
            
            return free_memory > required_memory
        except:
            return True
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU statistics"""
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        used_memory = torch.cuda.memory_allocated() / 1024**2
        
        return {
            "gpu_available": True,
            "total_memory_mb": total_memory,
            "used_memory_mb": used_memory,
            "free_memory_mb": total_memory - used_memory,
            "utilization_percent": (used_memory / total_memory) * 100,
            "active_videos": len(self.active_jobs),
            "queued_videos": len(self.job_queue)
        }

# Global instance
parallel_processor = ParallelVideoProcessor(max_concurrent_videos=3)