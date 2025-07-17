#!/usr/bin/env python3
"""
EINE API - TikTok Analyzer Production API
Port 8000 - Keine anderen Ports!
"""

import sys
import os
import asyncio
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, '/home/user/tiktok_production')

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from single_workflow import TikTokProductionWorkflow

# Initialize FastAPI app
app = FastAPI(
    title="TikTok Analyzer Production API",
    description="Single endpoint for TikTok video analysis",
    version="1.0.0"
)

# Global workflow instance
workflow = TikTokProductionWorkflow()

# Request models
class AnalyzeRequest(BaseModel):
    url: str
    priority: int = 1

class AnalyzeResponse(BaseModel):
    status: str
    message: str
    result_path: str = None
    processing_time: float = None
    analyzers_successful: int = None
    analyzers_total: int = None

# Background task tracking
active_tasks = {}

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "TikTok Analyzer Production API",
        "version": "1.0.0",
        "port": 8000,
        "analyzers": len(workflow.analyzers),
        "endpoints": {
            "analyze": "POST /analyze",
            "health": "GET /health",
            "results": "GET /results",
            "result/{filename}": "GET /result/{filename}"
        }
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Analyze TikTok video
    
    Args:
        request: AnalyzeRequest with TikTok URL
        
    Returns:
        AnalyzeResponse with result path and metrics
    """
    try:
        print(f"\n🚀 API Request: {request.url}")
        
        # Validate URL
        if not request.url or not request.url.startswith('https://'):
            raise HTTPException(status_code=400, detail="Invalid TikTok URL")
        
        # Run workflow
        result = workflow.run(request.url)
        
        if result['status'] == 'success':
            return AnalyzeResponse(
                status="success",
                message="Video analyzed successfully",
                result_path=result['output_path'],
                processing_time=result['total_time'],
                analyzers_successful=result['analyzers_successful'],
                analyzers_total=result['analyzers_total']
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Analysis failed: {result.get('error', 'Unknown error')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Check if workflow is available
        analyzer_count = len(workflow.analyzers)
        
        # Check directories
        results_dir = Path("/home/user/tiktok_production/results")
        downloads_dir = Path("/home/user/tiktok_production/downloads")
        
        # Check GPU availability
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except:
            pass
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "analyzers_loaded": analyzer_count,
            "gpu_available": gpu_available,
            "directories": {
                "results": results_dir.exists(),
                "downloads": downloads_dir.exists()
            },
            "version": "1.0.0"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/results")
async def list_results():
    """List all analysis results"""
    try:
        results_dir = Path("/home/user/tiktok_production/results")
        
        if not results_dir.exists():
            return {"results": [], "count": 0}
        
        json_files = list(results_dir.glob("*.json"))
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        results = []
        for file_path in json_files:
            try:
                stat = file_path.stat()
                results.append({
                    "filename": file_path.name,
                    "size_mb": stat.st_size / 1024 / 1024,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except:
                continue
        
        return {
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/result/{filename}")
async def get_result(filename: str):
    """Download specific result file"""
    try:
        results_dir = Path("/home/user/tiktok_production/results")
        file_path = results_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Result file not found")
        
        if not file_path.suffix == '.json':
            raise HTTPException(status_code=400, detail="Only JSON files allowed")
        
        return FileResponse(
            str(file_path),
            media_type="application/json",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    try:
        results_dir = Path("/home/user/tiktok_production/results")
        downloads_dir = Path("/home/user/tiktok_production/downloads")
        
        # Count files
        json_files = list(results_dir.glob("*.json")) if results_dir.exists() else []
        video_files = list(downloads_dir.glob("*.mp4")) if downloads_dir.exists() else []
        
        # Calculate sizes
        total_results_size = sum(f.stat().st_size for f in json_files)
        total_downloads_size = sum(f.stat().st_size for f in video_files)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "results": {
                "count": len(json_files),
                "total_size_mb": total_results_size / 1024 / 1024
            },
            "downloads": {
                "count": len(video_files),
                "total_size_mb": total_downloads_size / 1024 / 1024
            },
            "analyzers": {
                "total": len(workflow.analyzers),
                "list": list(workflow.analyzers.keys())
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "available_endpoints": ["/", "/analyze", "/health", "/results", "/stats"]}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    print(f"🚀 TikTok Analyzer Production API starting...")
    print(f"📍 Port: 8000")
    print(f"🔍 Analyzers: {len(workflow.analyzers)}")
    print(f"📁 Results: /home/user/tiktok_production/results")
    print(f"⏰ Target: <3 minutes per video")
    print("="*50)

if __name__ == "__main__":
    # Configuration
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,  # Single worker for GPU management
        "reload": False,
        "log_level": "info"
    }
    
    print(f"Starting API server on port {config['port']}...")
    uvicorn.run(app, **config)