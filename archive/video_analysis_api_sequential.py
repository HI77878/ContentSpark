#!/usr/bin/env python3
"""
SAUBERE VIDEO ANALYSIS API
- TikTok Video downloaden
- Mit allen funktionierenden Analyzern analysieren
- Ergebnisse speichern
- Automatisches Cleanup
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import sys
import json
import time
import gc
import torch
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append('/home/user/tiktok_production')

from mass_processing.tiktok_downloader import TikTokDownloader
from ml_analyzer_registry_complete import ML_ANALYZERS
from configs.gpu_groups_config import DISABLED_ANALYZERS
from utils.cleanup_manager import cleanup_manager

app = FastAPI(title="Video Analysis API", version="1.0")

class AnalysisRequest(BaseModel):
    tiktok_url: str = None
    video_path: str = None  # Direkter Video-Pfad
    creator_username: str = None

class AnalysisResponse(BaseModel):
    success: bool
    message: str
    video_id: str = None
    results_file: str = None
    analysis_time: float = None
    total_analyzers: int = None
    successful_analyzers: int = None

def cleanup_gpu():
    """GPU Memory cleanup - uses cleanup manager"""
    return cleanup_manager.cleanup_gpu_memory()

def get_active_analyzers():
    """Get list of working analyzers"""
    active = []
    for name, analyzer_class in ML_ANALYZERS.items():
        if name not in DISABLED_ANALYZERS:
            active.append((name, analyzer_class))
    return active

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    gpu_memory = {}
    
    if gpu_available:
        gpu_memory = {
            "used_mb": torch.cuda.memory_allocated() / 1024**2,
            "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
            "utilization": f"{torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100:.1f}%"
        }
    
    active_analyzers = len(get_active_analyzers())
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gpu": {
            "gpu_available": gpu_available,
            "gpu_name": torch.cuda.get_device_name(0) if gpu_available else None,
            "gpu_memory": gpu_memory
        },
        "active_analyzers": active_analyzers
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(request: AnalysisRequest):
    """Main analysis endpoint"""
    
    try:
        print(f"🎬 Starting analysis for: {request.tiktok_url}")
        start_time = time.time()
        
        # 1. Get video path (download or use existing)
        if request.video_path and Path(request.video_path).exists():
            # Use existing video
            video_path = request.video_path
            video_id = Path(video_path).stem
            print(f"📁 Using existing video: {video_path}")
        elif request.tiktok_url:
            # Download video
            print("📥 Downloading video...")
            downloader = TikTokDownloader()
            download_result = downloader.download_video(request.tiktok_url)
            
            if not download_result or 'local_path' not in download_result:
                raise HTTPException(status_code=400, detail="Failed to download video")
            
            video_path = download_result['local_path']
            video_id = Path(video_path).stem
        else:
            raise HTTPException(status_code=400, detail="Either tiktok_url or video_path required")
        
        print(f"✅ Video downloaded: {video_path}")
        
        # 2. Get active analyzers
        active_analyzers = get_active_analyzers()
        print(f"🔧 Running {len(active_analyzers)} analyzers...")
        
        # 3. Run analysis
        results = {}
        successful = 0
        
        for analyzer_name, analyzer_class in active_analyzers:
            try:
                print(f"   Running {analyzer_name}...")
                analyzer = analyzer_class()
                result = analyzer.analyze(video_path)
                
                if result and ('segments' in result and result['segments']):
                    results[analyzer_name] = result
                    successful += 1
                    print(f"   ✅ {analyzer_name}: {len(result.get('segments', []))} segments")
                else:
                    results[analyzer_name] = {"error": "No segments generated"}
                    print(f"   ❌ {analyzer_name}: No segments")
                
                # Cleanup after each analyzer
                del analyzer
                cleanup_gpu()
                
            except Exception as e:
                results[analyzer_name] = {"error": str(e)}
                print(f"   ❌ {analyzer_name}: {str(e)}")
                cleanup_gpu()
        
        # 4. Save results
        analysis_time = time.time() - start_time
        
        final_results = {
            "metadata": {
                "video_path": video_path,
                "video_filename": Path(video_path).name,
                "tiktok_url": request.tiktok_url or "Direct video path",
                "creator_username": request.creator_username,
                "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "processing_time_seconds": analysis_time,
                "total_analyzers": len(active_analyzers),
                "successful_analyzers": successful,
                "api_version": "1.0-clean"
            },
            "analyzer_results": results
        }
        
        # Save to results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{video_id}_analysis_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved: {results_file}")
        
        # VOLLSTÄNDIGES CLEANUP nach jeder Analyse
        cleanup_success = cleanup_manager.full_cleanup()
        
        # Memory stats nach cleanup
        memory_stats = cleanup_manager.get_memory_stats()
        logger.info(f"Post-cleanup memory: GPU {memory_stats.get('gpu', {}).get('usage_percent', 0):.1f}%, System {memory_stats.get('system', {}).get('percent', 0):.1f}%")
        
        return AnalysisResponse(
            success=True,
            message=f"Analysis completed successfully",
            video_id=video_id,
            results_file=str(results_file),
            analysis_time=analysis_time,
            total_analyzers=len(active_analyzers),
            successful_analyzers=successful
        )
        
    except Exception as e:
        # Cleanup auch bei Fehlern
        cleanup_manager.full_cleanup()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Clean Video Analysis API...")
    print("📋 Available endpoints:")
    print("   GET  /health - Health check")
    print("   POST /analyze - Analyze TikTok video")
    
    uvicorn.run(app, host="0.0.0.0", port=8003)