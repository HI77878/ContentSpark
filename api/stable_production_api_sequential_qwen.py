#!/usr/bin/env python3
"""
Stable Production API with Sequential Qwen2-VL Processing
Runs Qwen2-VL AFTER all other analyzers to avoid OOM
"""

# CRITICAL: Set multiprocessing start method FIRST
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
sys.path.append('/home/user/tiktok_production')

import torch
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

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
        logging.FileHandler('/home/user/tiktok_production/logs/sequential_qwen_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import configurations
from configs.gpu_groups_config import GPU_ANALYZER_GROUPS, DISABLED_ANALYZERS, ANALYZER_TIMINGS
from registry_loader import ML_ANALYZERS
from utils.multiprocess_gpu_executor_registry import MultiprocessGPUExecutorRegistry
from utils.output_normalizer import AnalyzerOutputNormalizer
from utils.model_preloader import model_preloader
from utils.auto_cleanup import auto_cleanup

app = FastAPI(
    title="Production API - Sequential Qwen2-VL",
    description="Runs Qwen2-VL separately after all other analyzers",
    version="4.0"
)

class AnalyzeRequest(BaseModel):
    video_path: str
    tiktok_url: Optional[str] = None
    creator_username: Optional[str] = None
    turbo_mode: bool = False

class ProductionEngineSequential:
    def __init__(self):
        # Use multiprocess executor for main analyzers
        self.executor = MultiprocessGPUExecutorRegistry(num_gpu_processes=4)
        
        # Get analyzers excluding qwen2_vl_temporal
        self.main_analyzers = []
        for group_name, analyzer_list in GPU_ANALYZER_GROUPS.items():
            for analyzer in analyzer_list:
                if analyzer not in DISABLED_ANALYZERS and analyzer in ML_ANALYZERS:
                    if analyzer != 'qwen2_vl_temporal':  # Exclude Qwen2-VL
                        self.main_analyzers.append(analyzer)
        
        # Remove duplicates
        seen = set()
        self.main_analyzers = [x for x in self.main_analyzers if not (x in seen or seen.add(x))]
        
        logger.info(f"🚀 Sequential Engine initialized:")
        logger.info(f"   Main analyzers: {len(self.main_analyzers)}")
        logger.info(f"   Qwen2-VL will run separately after main analysis")
        
        self.qwen_analyzer = None  # Will be loaded on demand
    
    def _load_qwen_analyzer(self):
        """Load Qwen2-VL analyzer when needed"""
        if self.qwen_analyzer is None:
            logger.info("Loading Qwen2-VL analyzer (DETAILED mode)...")
            from analyzers.qwen2_vl_video_analyzer_detailed import Qwen2VLVideoAnalyzerDetailed
            self.qwen_analyzer = Qwen2VLVideoAnalyzerDetailed()
            logger.info("✅ Qwen2-VL DETAILED loaded successfully")
    
    async def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Analyze video with sequential Qwen2-VL processing"""
        start_time = time.time()
        
        # Validate video
        if not Path(video_path).exists():
            raise ValueError(f"Video not found: {video_path}")
        
        logger.info(f"🎬 Starting sequential analysis: {video_path}")
        
        # PHASE 1: Run main analyzers in parallel
        logger.info("📊 Phase 1: Running 19 main analyzers in parallel...")
        phase1_start = time.time()
        
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None,
                self.executor.execute_parallel,
                video_path,
                self.main_analyzers
            )
        except Exception as e:
            logger.error(f"Phase 1 failed: {e}")
            raise
        
        phase1_time = time.time() - phase1_start
        logger.info(f"✅ Phase 1 complete in {phase1_time:.1f}s")
        
        # Clean GPU memory before Qwen2-VL
        logger.info("🧹 Cleaning GPU memory...")
        torch.cuda.empty_cache()
        gc.collect()
        
        # Wait a bit for memory to settle
        await asyncio.sleep(2)
        
        # PHASE 2: Run Qwen2-VL separately
        logger.info("📊 Phase 2: Running Qwen2-VL analyzer...")
        phase2_start = time.time()
        
        try:
            # Load Qwen2-VL if not loaded
            self._load_qwen_analyzer()
            
            # Run Qwen2-VL analysis
            qwen_result = await loop.run_in_executor(
                None,
                self.qwen_analyzer.analyze,
                video_path
            )
            
            # Add to results
            results['qwen2_vl_temporal'] = qwen_result
            
            phase2_time = time.time() - phase2_start
            logger.info(f"✅ Phase 2 complete in {phase2_time:.1f}s")
            
        except Exception as e:
            logger.error(f"Qwen2-VL analysis failed: {e}")
            # Continue without Qwen2-VL rather than failing everything
            results['qwen2_vl_temporal'] = {
                'error': str(e),
                'analyzer_name': 'qwen2_vl_temporal',
                'segments': [],
                'summary': {'total_segments': 0}
            }
        
        # Calculate final metrics
        total_time = time.time() - start_time
        successful = sum(1 for k, v in results.items() 
                        if k != 'metadata' and isinstance(v, dict) and 'error' not in v)
        
        # Get video info
        duration = results.get('metadata', {}).get('duration', 0)
        realtime_factor = total_time / duration if duration > 0 else 0
        reconstruction_score = (successful / (len(self.main_analyzers) + 1)) * 100  # +1 for Qwen
        
        logger.info(f"✅ Analysis complete in {total_time:.1f}s")
        logger.info(f"⚡ Realtime factor: {realtime_factor:.2f}x")
        logger.info(f"📊 Reconstruction score: {reconstruction_score:.1f}%")
        logger.info(f"🎯 Successful analyzers: {successful}/{len(self.main_analyzers) + 1}")
        
        # Add performance metrics
        results['metadata']['analysis_time'] = total_time
        results['metadata']['realtime_factor'] = realtime_factor
        results['metadata']['reconstruction_score'] = reconstruction_score
        results['metadata']['successful_analyzers'] = successful
        results['metadata']['total_analyzers'] = len(self.main_analyzers) + 1
        results['metadata']['phase1_time'] = phase1_time
        results['metadata']['phase2_time'] = phase2_time if 'phase2_time' in locals() else 0
        
        return results

# Global engine
engine = None

@app.get("/")
async def root():
    return {
        "service": "Production API - Sequential Qwen2-VL",
        "status": "ready",
        "version": "4.0",
        "features": [
            "Sequential Qwen2-VL processing",
            f"{len(engine.main_analyzers) if engine else 0} parallel analyzers",
            "1 sequential analyzer (Qwen2-VL)",
            "Memory-optimized workflow",
            "No OOM issues"
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
        "main_analyzers": len(engine.main_analyzers) if engine else 0,
        "sequential_analyzers": 1,  # Qwen2-VL
        "workflow": "two-phase"
    }

@app.post("/analyze")
async def analyze_video(request: AnalyzeRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        # Run analysis
        results = await engine.analyze_video(request.video_path)
        
        # Save results
        video_id = Path(request.video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/home/user/tiktok_production/results/{video_id}_sequential_{timestamp}.json"
        
        # Initialize normalizer
        normalizer = AnalyzerOutputNormalizer()
        
        # Normalize results
        analyzer_results = {}
        for analyzer_name, analyzer_data in results.items():
            if analyzer_name != 'metadata':
                try:
                    normalized_data = normalizer.normalize(analyzer_name, analyzer_data)
                    analyzer_results[analyzer_name] = normalized_data
                except Exception as e:
                    logger.error(f"Failed to normalize {analyzer_name}: {e}")
                    analyzer_results[analyzer_name] = analyzer_data
        
        # Create output
        output_data = {
            "metadata": {
                "video_path": request.video_path,
                "video_filename": Path(request.video_path).name,
                "tiktok_url": request.tiktok_url,
                "creator_username": request.creator_username,
                "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "processing_time_seconds": results['metadata']['analysis_time'],
                "total_analyzers": results['metadata']['total_analyzers'],
                "successful_analyzers": results['metadata']['successful_analyzers'],
                "reconstruction_score": results['metadata']['reconstruction_score'],
                "realtime_factor": results['metadata']['realtime_factor'],
                "api_version": "4.0-sequential",
                "workflow": "two-phase",
                "phase1_time": results['metadata']['phase1_time'],
                "phase2_time": results['metadata']['phase2_time']
            },
            "analyzer_results": analyzer_results
        }
        
        # Add video metadata
        if 'duration' in results.get('metadata', {}):
            output_data['metadata']['duration'] = results['metadata']['duration']
        
        # Save
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {output_path}")
        
        # Auto cleanup
        auto_cleanup.cleanup_after_analysis(video_id)
        
        return {
            "status": "success",
            "video_path": request.video_path,
            "processing_time": results['metadata']['analysis_time'],
            "successful_analyzers": results['metadata']['successful_analyzers'],
            "total_analyzers": results['metadata']['total_analyzers'],
            "results_file": output_path,
            "phase1_time": results['metadata']['phase1_time'],
            "phase2_time": results['metadata']['phase2_time'],
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "status": "error",
            "video_path": request.video_path,
            "processing_time": 0,
            "successful_analyzers": 0,
            "total_analyzers": len(engine.main_analyzers) + 1 if engine else 0,
            "results_file": "",
            "error": str(e)
        }

@app.get("/analyzers")
async def list_analyzers():
    return {
        "total": len(engine.main_analyzers) + 1 if engine else 0,
        "main_analyzers": engine.main_analyzers if engine else [],
        "sequential_analyzers": ["qwen2_vl_temporal"],
        "disabled": list(DISABLED_ANALYZERS)
    }

@app.on_event("startup")
async def startup_event():
    global engine
    logger.info("🚀 Sequential Qwen2-VL API starting...")
    logger.info("📊 Two-phase processing: Main analyzers → Qwen2-VL")
    
    # Initialize engine
    engine = ProductionEngineSequential()
    
    # Pre-load Whisper for speech transcription
    logger.info("Pre-loading Whisper model...")
    try:
        import whisper
        model_preloader.get_model(
            "whisper-base",
            model_class=whisper.load_model,
            model_kwargs={"name": "base", "device": "cuda", "download_root": "/home/user/.cache/whisper"}
        )
        logger.info("✅ Whisper pre-loaded successfully!")
    except Exception as e:
        logger.warning(f"Whisper pre-loading failed: {e}")
    
    # Force cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    logger.info("✅ API is ready for sequential processing!")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down...")
    if engine and engine.qwen_analyzer:
        # Cleanup Qwen model
        del engine.qwen_analyzer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    logger.info("Starting Sequential Qwen2-VL API...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8004,
        workers=1,
        loop="uvloop",
        access_log=False
    )