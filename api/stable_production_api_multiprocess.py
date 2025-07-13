#!/usr/bin/env python3
"""
Stable Production API with Multiprocess GPU Parallelization
Achieves <3x realtime with >90% reconstruction score
"""

# CRITICAL: Set multiprocessing start method FIRST
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
# GPU Memory optimization - verhindert CUDA OOM
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
sys.path.append('/home/user/tiktok_production')

import torch
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

# Advanced GPU Parallelization - verhindert CPU oversubscription
import multiprocessing
import logging
cpu_count = multiprocessing.cpu_count()
gpu_workers = 3  # 3 GPU workers
threads_per_worker = max(1, cpu_count // gpu_workers)
torch.set_num_threads(threads_per_worker)
logger_init = logging.getLogger(__name__)
logger_init.info(f"Advanced parallelization: {threads_per_worker} threads per worker ({cpu_count} total CPUs)")

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import asyncio
import time
import json
import logging
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/user/tiktok_production/logs/stable_multiprocess_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import configurations
from configs.gpu_groups_config import GPU_ANALYZER_GROUPS, DISABLED_ANALYZERS, ANALYZER_TIMINGS
from registry_loader import ML_ANALYZERS
from utils.multiprocess_gpu_executor_registry_cached import MultiprocessGPUExecutorRegistryCached
from utils.output_normalizer import AnalyzerOutputNormalizer

# Model Pre-Loading imports
from utils.model_preloader import model_preloader
from utils.model_preloader_enhanced import enhanced_model_preloader
from utils.parallel_video_processor import parallel_processor
from utils.auto_cleanup import auto_cleanup

app = FastAPI(
    title="Stable Production API - Multiprocess Edition",
    description="Video analysis with true GPU parallelization",
    version="3.0"
)

class AnalyzeRequest(BaseModel):
    video_path: Optional[str] = None
    tiktok_url: Optional[str] = None
    creator_username: Optional[str] = None
    turbo_mode: bool = False

class AnalyzeResponse(BaseModel):
    status: str
    video_path: str
    processing_time: float
    successful_analyzers: int
    total_analyzers: int
    results_file: str
    error: Optional[str] = None

class BatchAnalyzeRequest(BaseModel):
    video_paths: List[str]
    max_concurrent: Optional[int] = 3

class BatchAnalyzeResponse(BaseModel):
    status: str
    job_ids: List[str]
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    video_path: str
    progress: float
    result_path: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

class ProductionEngine:
    def __init__(self):
        # Use multiprocess executor for true parallelization
        self.executor = MultiprocessGPUExecutorRegistryCached(enable_caching=True, num_gpu_processes=3)  # 3 GPU workers for better balance
        
        # Get active analyzers
        self.active_analyzers = []
        for group_name, analyzer_list in GPU_ANALYZER_GROUPS.items():
            for analyzer in analyzer_list:
                if analyzer not in DISABLED_ANALYZERS and analyzer in ML_ANALYZERS:
                    self.active_analyzers.append(analyzer)
        
        # Remove duplicates while preserving order
        seen = set()
        self.active_analyzers = [x for x in self.active_analyzers if not (x in seen or seen.add(x))]
        
        logger.info(f"🚀 Production Engine initialized with {len(self.active_analyzers)} active analyzers")
        logger.info("   Using multiprocess GPU parallelization")
        logger.info("   3 GPU processes for true parallel execution")
    
    async def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Analyze video using multiprocess parallelization"""
        start_time = time.time()
        
        # Validate video
        if not Path(video_path).exists():
            raise ValueError(f"Video not found: {video_path}")
        
        logger.info(f"🎬 Starting analysis: {video_path}")
        
        # Run analysis in executor to not block event loop
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None,
                self.executor.execute_parallel,
                video_path,
                self.active_analyzers  # All active analyzers
            )
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
        
        # Calculate metrics
        total_time = time.time() - start_time
        successful = sum(1 for k, v in results.items() 
                        if k != 'metadata' and isinstance(v, dict) and 'error' not in v)
        
        # Get video info from metadata
        duration = results.get('metadata', {}).get('duration', 0)
        realtime_factor = total_time / duration if duration > 0 else 0
        reconstruction_score = (successful / len(self.active_analyzers)) * 100
        
        logger.info(f"✅ Analysis complete in {total_time:.1f}s")
        logger.info(f"⚡ Realtime factor: {realtime_factor:.2f}x")
        logger.info(f"📊 Reconstruction score: {reconstruction_score:.1f}%")
        logger.info(f"🎯 Successful analyzers: {successful}/{len(self.active_analyzers)}")
        
        # Add performance metrics
        results['metadata']['analysis_time'] = total_time
        results['metadata']['realtime_factor'] = realtime_factor
        results['metadata']['reconstruction_score'] = reconstruction_score
        results['metadata']['successful_analyzers'] = successful
        results['metadata']['total_analyzers'] = len(self.active_analyzers)
        
        return results

# Global engine
engine = None

@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down...")
    if engine:
        # Executor handles cleanup automatically
        pass

@app.get("/")
async def root():
    return {
        "service": "Stable Production API - Multiprocess Edition",
        "status": "ready",
        "version": "3.0",
        "features": [
            "Process-based GPU parallelization",
            f"{len(engine.active_analyzers) if engine else 0} active analyzers",
            "<3x realtime performance",
            ">90% reconstruction score target",
            "Automatic GPU memory management"
        ]
    }

@app.get("/health")
async def health_check():
    gpu_info = {}
    try:
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory": {
                    "used_mb": torch.cuda.memory_allocated() / 1024**2,
                    "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
                    "utilization": f"{(torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100):.1f}%"
                }
            }
    except:
        pass
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gpu": gpu_info,
        "active_analyzers": len(engine.active_analyzers) if engine else 0,
        "parallelization": "multiprocess"
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_video(request: AnalyzeRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        # Handle TikTok URL download if needed
        video_path = request.video_path
        tiktok_metadata = None
        
        if request.tiktok_url and not request.video_path:
            # Import TikTok downloader
            from mass_processing.tiktok_downloader import TikTokDownloader
            downloader = TikTokDownloader()
            
            logger.info(f"📥 Downloading TikTok video: {request.tiktok_url}")
            download_result = downloader.download_video(request.tiktok_url)
            
            if not download_result or not download_result.get('success'):
                raise HTTPException(status_code=400, detail="Failed to download TikTok video")
            
            video_path = download_result['video_path']
            tiktok_metadata = download_result.get('metadata', {})
            logger.info(f"✅ Downloaded to: {video_path}")
            
            # Extract creator username from metadata if not provided
            if not request.creator_username and tiktok_metadata:
                request.creator_username = tiktok_metadata.get('username')
        
        elif not video_path:
            raise HTTPException(status_code=400, detail="Either video_path or tiktok_url is required")
        
        # Run analysis
        results = await engine.analyze_video(video_path)
        
        # Save results
        video_id = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/home/user/tiktok_production/results/{video_id}_multiprocess_{timestamp}.json"
        
        # Initialize normalizer
        normalizer = AnalyzerOutputNormalizer()
        
        # Prepare and normalize analyzer results
        analyzer_results = {}
        normalization_errors = []
        
        for analyzer_name, analyzer_data in results.items():
            if analyzer_name != 'metadata':
                # Normalize the output
                try:
                    normalized_data = normalizer.normalize(analyzer_name, analyzer_data)
                    analyzer_results[analyzer_name] = normalized_data
                    
                    # Validate normalized output
                    errors = normalizer.validate_output(analyzer_name, normalized_data)
                    if errors:
                        normalization_errors.extend(errors)
                        logger.warning(f"Validation errors for {analyzer_name}: {errors}")
                except Exception as e:
                    logger.error(f"Failed to normalize {analyzer_name}: {e}")
                    # Keep original data if normalization fails
                    analyzer_results[analyzer_name] = analyzer_data
        
        # Log normalization stats
        norm_stats = normalizer.get_normalization_stats()
        logger.info(f"📊 Normalization complete: {norm_stats['fields_normalized']} fields normalized across {len(norm_stats['analyzers_processed'])} analyzers")
        
        # Create full output with TikTok URL
        output_data = {
            "metadata": {
                "video_path": video_path,
                "video_filename": Path(video_path).name,
                "tiktok_url": request.tiktok_url,  # Include TikTok URL
                "creator_username": request.creator_username,  # Include creator
                "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "processing_time_seconds": results['metadata']['analysis_time'],
                "total_analyzers": results['metadata']['total_analyzers'],
                "successful_analyzers": results['metadata']['successful_analyzers'],
                "reconstruction_score": results['metadata']['reconstruction_score'],
                "realtime_factor": results['metadata']['realtime_factor'],
                "api_version": "3.0-multiprocess",
                "parallelization": "process-based"
            },
            "analyzer_results": analyzer_results
        }
        
        # Add TikTok metadata if provided
        if request.tiktok_url:
            output_data['metadata']['tiktok_url'] = request.tiktok_url
            
        if request.creator_username:
            output_data['metadata']['creator_username'] = request.creator_username
            
        # Extract video ID from path if it looks like a TikTok video
        video_name = Path(video_path).stem
        if video_name.isdigit() and len(video_name) > 15:  # TikTok IDs are long numbers
            output_data['metadata']['tiktok_video_id'] = video_name
            
        # Add TikTok metadata if available
        if tiktok_metadata:
            output_data['metadata']['tiktok_metadata'] = tiktok_metadata
            
        # Add video duration from results
        if 'duration' in results.get('metadata', {}):
            output_data['metadata']['duration'] = results['metadata']['duration']
        
        # Save
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {output_path}")
        
        # Auto cleanup after analysis
        if request.video_path:
            video_id = output_data['metadata'].get('tiktok_video_id', Path(request.video_path).stem)
        else:
            video_id = output_data['metadata'].get('tiktok_video_id', 'unknown')
        auto_cleanup.cleanup_after_analysis(video_id)
        
        return AnalyzeResponse(
            status="success",
            video_path=video_path,
            processing_time=results['metadata']['analysis_time'],
            successful_analyzers=results['metadata']['successful_analyzers'],
            total_analyzers=results['metadata']['total_analyzers'],
            results_file=output_path,
            error=None
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return AnalyzeResponse(
            status="error",
            video_path=request.video_path if request.video_path else "",
            processing_time=0,
            successful_analyzers=0,
            total_analyzers=len(engine.active_analyzers) if engine else 0,
            results_file="",
            error=str(e)
        )

@app.get("/analyzers")
async def list_analyzers():
    return {
        "total": len(engine.active_analyzers) if engine else 0,
        "active": engine.active_analyzers if engine else [],
        "disabled": list(DISABLED_ANALYZERS)
    }

@app.post("/analyze_batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(request: BatchAnalyzeRequest):
    """Analyze multiple videos in parallel for maximum GPU utilization"""
    try:
        # Validate all video paths
        for video_path in request.video_paths:
            if not Path(video_path).exists():
                raise HTTPException(status_code=400, detail=f"Video not found: {video_path}")
        
        # Set max concurrent if specified
        if request.max_concurrent and request.max_concurrent != parallel_processor.max_concurrent_videos:
            parallel_processor.max_concurrent_videos = min(request.max_concurrent, 3)  # Max 3 for safety
        
        # Add videos to processing queue
        job_ids = await parallel_processor.add_videos_batch(request.video_paths)
        
        return BatchAnalyzeResponse(
            status="success",
            job_ids=job_ids,
            message=f"Added {len(job_ids)} videos to processing queue"
        )
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a specific analysis job"""
    job = parallel_processor.get_job_status(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    processing_time = None
    if job.start_time and job.end_time:
        processing_time = job.end_time - job.start_time
    elif job.start_time:
        processing_time = time.time() - job.start_time
    
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        video_path=job.video_path,
        progress=job.progress,
        result_path=job.result_path,
        error=job.error,
        processing_time=processing_time
    )

@app.get("/jobs")
async def list_all_jobs():
    """List all jobs grouped by status"""
    jobs = parallel_processor.get_all_jobs()
    gpu_stats = parallel_processor.get_gpu_stats()
    
    return {
        "jobs": jobs,
        "gpu_stats": gpu_stats,
        "summary": {
            "processing": len(jobs['processing']),
            "pending": len(jobs['pending']),
            "completed": len(jobs['completed']),
            "failed": len(jobs['failed'])
        }
    }

# Pre-load models at startup
async def startup_preload():
    """Pre-load ALL analyzer models at API startup for maximum GPU utilization"""
    logger.info("Starting enhanced model pre-loading...")
    
    loop = asyncio.get_event_loop()
    
    # Run pre-loading in background
    def preload_models():
        try:
            # Use enhanced preloader to load ALL models
            logger.info("🚀 Loading ALL analyzer models for maximum GPU utilization...")
            preloaded_models = enhanced_model_preloader.preload_all_models()
            
            logger.info(f"✅ Preloaded {len(preloaded_models)} models successfully!")
            
            # Also load standard models for compatibility
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            import whisper
            
            # Ensure Qwen2-VL is in standard preloader too
            if 'qwen2_vl_temporal' in preloaded_models:
                logger.info("Registering Qwen2-VL in standard preloader...")
                model_preloader.get_model(
                    "Qwen/Qwen2-VL-7B-Instruct",
                    model_class=Qwen2VLForConditionalGeneration,
                    processor_class=AutoProcessor,
                    attn_implementation="eager"
                )
            
            # Log memory stats
            memory_stats = enhanced_model_preloader.get_memory_stats()
            logger.info(f"📊 GPU Memory Status:")
            logger.info(f"   Total: {memory_stats.get('total_mb', 0):.0f} MB")
            logger.info(f"   Used: {memory_stats.get('allocated_mb', 0):.0f} MB")
            logger.info(f"   Free: {memory_stats.get('free_mb', 0):.0f} MB")
            
        except Exception as e:
            logger.error(f"Enhanced model pre-loading failed: {e}")
            logger.info("Falling back to basic pre-loading...")
            
            # Fallback to basic preloading
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            import whisper
            
            try:
                model_preloader.get_model(
                    "Qwen/Qwen2-VL-7B-Instruct",
                    model_class=Qwen2VLForConditionalGeneration,
                    processor_class=AutoProcessor,
                    attn_implementation="eager"
                )
                logger.info("✅ Basic Qwen2-VL pre-loaded")
                
                model_preloader.get_model(
                    "whisper-base",
                    model_class=whisper.load_model,
                    model_kwargs={"name": "base", "device": "cuda", "download_root": "/home/user/.cache/whisper"}
                )
                logger.info("✅ Basic Whisper pre-loaded")
            except Exception as e2:
                logger.error(f"Basic pre-loading also failed: {e2}")
    
    # Run in executor to not block
    await loop.run_in_executor(None, preload_models)

# Startup event that initializes engine and pre-loads models
@app.on_event("startup")
async def startup_event():
    global engine
    logger.info("🚀 Stable Production API (Multiprocess) starting...")
    logger.info("📊 Target: <3x realtime, >90% reconstruction")
    logger.info("🔧 Using process-based GPU parallelization")
    
    # Initialize engine
    engine = ProductionEngine()
    
    # Pre-load models - DISABLED to free memory for analyzers
    # await startup_preload()
    logger.info("⚠️ Model preloading DISABLED to prevent CUDA OOM")
    
    # Force GPU memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    logger.info("✅ API is ready to accept requests!")

if __name__ == "__main__":
    logger.info("Starting Stable Production API with Multiprocess Parallelization...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        workers=1,  # Single worker, parallelization via multiprocessing
        loop="uvloop",
        access_log=False
    )