#!/usr/bin/env python3
"""
TikTok Analyzer API v2 - Extended Production API
Includes job management, search, export, and monitoring endpoints
"""

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import json
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import io
import psycopg2
from psycopg2.extras import RealDictCursor
import uuid

sys.path.append('/home/user/tiktok_production')

from api.stable_production_api_multiprocess import (
    NumpyEncoder, AnalyzeRequest, get_active_analyzers,
    ProductionEngine
)
from production_setup.batch_processor import BatchProcessor, JobPriority, JobStatus
from utils.gpu_monitor import gpu_monitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/user/tiktok_production/logs/api_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', 5432),
    'database': os.getenv('DB_NAME', 'tiktok_analyzer'),
    'user': os.getenv('DB_USER', 'user'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

app = FastAPI(
    title="TikTok Analyzer API v2",
    description="Extended API with job management, search, and export capabilities",
    version="2.0.0"
)

# Initialize components
production_engine = ProductionEngine()
batch_processor = BatchProcessor()

# Request/Response Models
class BatchAnalyzeRequest(BaseModel):
    urls: List[str] = Field(..., description="List of TikTok URLs to analyze")
    priority: Optional[str] = Field("normal", description="Priority: urgent, high, normal, low")
    analyzers: Optional[List[str]] = Field(None, description="Specific analyzers to run")

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    tiktok_url: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    processing_time: Optional[float]
    result_file: Optional[str]
    error_message: Optional[str]
    progress: Optional[float]

class SearchRequest(BaseModel):
    creator_username: Optional[str] = None
    hashtag: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    limit: int = Field(100, ge=1, le=1000)

class ExportFormat(str):
    CSV = "csv"
    JSON = "json"
    EXCEL = "xlsx"

# Database connection
def get_db_connection():
    """Get PostgreSQL connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.OperationalError:
        # Fallback to SQLite if PostgreSQL not available
        logger.warning("PostgreSQL not available, using SQLite fallback")
        return None

# Extended endpoints

@app.post("/analyze/batch", response_model=List[str])
async def analyze_batch(request: BatchAnalyzeRequest):
    """Submit multiple videos for analysis"""
    try:
        priority_map = {
            "urgent": JobPriority.URGENT,
            "high": JobPriority.HIGH,
            "normal": JobPriority.NORMAL,
            "low": JobPriority.LOW
        }
        
        priority = priority_map.get(request.priority, JobPriority.NORMAL)
        job_ids = batch_processor.add_batch(request.urls, priority)
        
        return job_ids
        
    except Exception as e:
        logger.error(f"Batch submission error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of a specific analysis job"""
    job_status = batch_processor.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Calculate progress
    progress = None
    if job_status['status'] == 'completed':
        progress = 100.0
    elif job_status['status'] == 'processing':
        progress = 50.0
    elif job_status['status'] == 'downloading':
        progress = 25.0
    elif job_status['status'] == 'pending':
        progress = 0.0
    
    return JobStatusResponse(
        job_id=job_id,
        status=job_status['status'],
        tiktok_url=job_status['tiktok_url'],
        created_at=job_status['created_at'],
        started_at=job_status.get('started_at'),
        completed_at=job_status.get('completed_at'),
        processing_time=job_status.get('processing_time'),
        result_file=job_status.get('result_file'),
        error_message=job_status.get('error_message'),
        progress=progress
    )

@app.get("/stats/performance")
async def get_performance_stats():
    """Get system performance statistics"""
    try:
        # Get current metrics
        gpu_stats = gpu_monitor.get_current_stats()
        
        # Get queue status
        queue_status = batch_processor.get_queue_status()
        
        # Get analyzer performance from monitoring DB
        analyzer_stats = {}
        monitoring_db = "/home/user/tiktok_production/monitoring.db"
        
        if os.path.exists(monitoring_db):
            import sqlite3
            conn = sqlite3.connect(monitoring_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT analyzer_name, 
                       AVG(success_count) as avg_success,
                       AVG(error_count) as avg_error,
                       AVG(avg_processing_time) as processing_time
                FROM analyzer_stats
                WHERE timestamp > datetime('now', '-1 hour')
                GROUP BY analyzer_name
            ''')
            
            for row in cursor.fetchall():
                analyzer_stats[row[0]] = {
                    'avg_success': row[1] or 0,
                    'avg_error': row[2] or 0,
                    'avg_processing_time': row[3] or 0
                }
            
            conn.close()
        
        return JSONResponse({
            'timestamp': datetime.now().isoformat(),
            'gpu': {
                'usage_percent': gpu_stats.get('gpu_usage', 0),
                'memory_used_gb': gpu_stats.get('memory_used', 0),
                'memory_total_gb': gpu_stats.get('memory_total', 0),
                'temperature': gpu_stats.get('temperature', 0)
            },
            'queue': queue_status,
            'analyzers': analyzer_stats,
            'system': {
                'active_analyzers': len(get_active_analyzers()),
                'api_version': '2.0.0',
                'multiprocess_enabled': True
            }
        })
        
    except Exception as e:
        logger.error(f"Performance stats error: {str(e)}")
        return JSONResponse({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

@app.post("/videos/search")
async def search_videos(request: SearchRequest):
    """Search analyzed videos with filters"""
    conn = get_db_connection()
    
    if not conn:
        # Fallback to file-based search
        results = []
        results_dir = Path("/home/user/tiktok_production/results")
        
        for json_file in results_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    metadata = data.get('metadata', {})
                    
                    # Apply filters
                    if request.creator_username:
                        if request.creator_username.lower() not in metadata.get('creator_username', '').lower():
                            continue
                    
                    if request.hashtag:
                        hashtags = metadata.get('hashtags', [])
                        if not any(request.hashtag.lower() in tag.lower() for tag in hashtags):
                            continue
                    
                    results.append({
                        'video_id': metadata.get('video_id'),
                        'tiktok_url': metadata.get('tiktok_url'),
                        'creator_username': metadata.get('creator_username'),
                        'duration': metadata.get('video_duration'),
                        'created_at': metadata.get('analysis_timestamp'),
                        'file_path': str(json_file)
                    })
                    
                    if len(results) >= request.limit:
                        break
                        
            except Exception as e:
                logger.error(f"Error reading {json_file}: {e}")
                
        return JSONResponse(results)
    
    # PostgreSQL search
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Build query
        query = """
            SELECT id, tiktok_url, video_id, creator_username, 
                   duration_seconds, created_at, result_file_path
            FROM video_analyses
            WHERE 1=1
        """
        params = []
        
        if request.creator_username:
            query += " AND creator_username ILIKE %s"
            params.append(f"%{request.creator_username}%")
        
        if request.hashtag:
            query += " AND %s = ANY(hashtags)"
            params.append(request.hashtag)
        
        if request.date_from:
            query += " AND created_at >= %s"
            params.append(request.date_from)
        
        if request.date_to:
            query += " AND created_at <= %s"
            params.append(request.date_to)
        
        if request.min_duration:
            query += " AND duration_seconds >= %s"
            params.append(request.min_duration)
        
        if request.max_duration:
            query += " AND duration_seconds <= %s"
            params.append(request.max_duration)
        
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(request.limit)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        return JSONResponse([dict(r) for r in results])
        
    finally:
        if conn:
            conn.close()

@app.get("/export/{video_id}")
async def export_analysis(
    video_id: str,
    format: ExportFormat = Query(ExportFormat.JSON)
):
    """Export analysis results in different formats"""
    # Try to find the analysis file
    results_dir = Path("/home/user/tiktok_production/results")
    result_file = None
    
    # Search by video ID
    for json_file in results_dir.glob(f"*{video_id}*.json"):
        result_file = json_file
        break
    
    if not result_file or not result_file.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Load data
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    if format == ExportFormat.JSON:
        # Return raw JSON
        return Response(
            content=json.dumps(data, indent=2, cls=NumpyEncoder),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={video_id}_analysis.json"
            }
        )
    
    elif format == ExportFormat.CSV:
        # Convert to CSV
        rows = []
        
        # Flatten analyzer results
        for analyzer_name, results in data.get('analyzer_results', {}).items():
            if 'segments' in results:
                for segment in results['segments']:
                    row = {
                        'analyzer': analyzer_name,
                        'timestamp': segment.get('timestamp', 0),
                        **{k: v for k, v in segment.items() if k != 'timestamp'}
                    }
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        return Response(
            content=csv_buffer.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={video_id}_analysis.csv"
            }
        )
    
    elif format == ExportFormat.EXCEL:
        # Convert to Excel with multiple sheets
        excel_buffer = io.BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Metadata sheet
            metadata_df = pd.DataFrame([data.get('metadata', {})])
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            # Analyzer results sheets
            for analyzer_name, results in data.get('analyzer_results', {}).items():
                if 'segments' in results and results['segments']:
                    df = pd.DataFrame(results['segments'])
                    # Truncate sheet name to Excel limit
                    sheet_name = analyzer_name[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        excel_buffer.seek(0)
        
        return Response(
            content=excel_buffer.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={video_id}_analysis.xlsx"
            }
        )

@app.get("/analyzers")
async def list_analyzers():
    """List all available analyzers with their status"""
    active_analyzers = get_active_analyzers()
    
    # Get performance stats if available
    analyzer_info = {}
    
    for analyzer in active_analyzers:
        analyzer_info[analyzer] = {
            'active': True,
            'description': f"{analyzer} analyzer",
            'performance': {
                'avg_processing_time': 0,
                'success_rate': 100
            }
        }
    
    return JSONResponse(analyzer_info)

@app.post("/maintenance/cleanup")
async def cleanup_old_results(days_to_keep: int = 30):
    """Clean up old analysis results"""
    try:
        results_dir = Path("/home/user/tiktok_production/results")
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        deleted_count = 0
        for json_file in results_dir.glob("*.json"):
            if json_file.stat().st_mtime < cutoff_date.timestamp():
                json_file.unlink()
                deleted_count += 1
        
        return JSONResponse({
            'status': 'success',
            'deleted_files': deleted_count,
            'cutoff_date': cutoff_date.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status"""
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'status': 'healthy',
        'components': {}
    }
    
    # Check API
    health_status['components']['api'] = {
        'status': 'healthy',
        'version': '2.0.0'
    }
    
    # Check GPU
    try:
        gpu_stats = gpu_monitor.get_current_stats()
        health_status['components']['gpu'] = {
            'status': 'healthy' if gpu_stats.get('gpu_usage', 0) < 95 else 'degraded',
            'usage': gpu_stats.get('gpu_usage', 0),
            'memory_percent': gpu_stats.get('memory_percent', 0)
        }
    except:
        health_status['components']['gpu'] = {'status': 'unhealthy'}
    
    # Check database
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            health_status['components']['database'] = {'status': 'healthy'}
        else:
            health_status['components']['database'] = {'status': 'degraded', 'note': 'Using fallback'}
    except:
        health_status['components']['database'] = {'status': 'unhealthy'}
    
    # Check queue
    queue_status = batch_processor.get_queue_status()
    health_status['components']['queue'] = {
        'status': 'healthy',
        'pending_jobs': queue_status.get('status_counts', {}).get('pending', 0),
        'active_jobs': queue_status.get('active_jobs', 0)
    }
    
    # Overall status
    component_statuses = [c['status'] for c in health_status['components'].values()]
    if any(s == 'unhealthy' for s in component_statuses):
        health_status['status'] = 'unhealthy'
    elif any(s == 'degraded' for s in component_statuses):
        health_status['status'] = 'degraded'
    
    return JSONResponse(health_status)

# Include original endpoints
from api.stable_production_api_multiprocess import analyze_video, health_check

app.add_api_route("/analyze", analyze_video, methods=["POST"])
app.add_api_route("/health", health_check, methods=["GET"])

# Start background tasks
@app.on_event("startup")
async def startup_event():
    logger.info("Starting TikTok Analyzer API v2...")
    
    # Start batch processor workers
    asyncio.create_task(batch_processor.start(num_workers=3))
    
    logger.info("API v2 startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API v2...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)