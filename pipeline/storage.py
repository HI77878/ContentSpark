#!/usr/bin/env python3
"""
Supabase Storage Integration
Handles uploading analysis results to Supabase
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class SupabaseStorage:
    """Handle storage of analysis results in Supabase"""
    
    def __init__(self, url: str = None, key: str = None):
        self.url = url or os.getenv("SUPABASE_URL", "")
        self.key = key or os.getenv("SUPABASE_KEY", "")
        self.client = None
        
        if self.url and self.key:
            try:
                from supabase import create_client, Client
                self.client: Client = create_client(self.url, self.key)
                logger.info("✅ Supabase client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase: {e}")
                logger.info("⚠️ Running in local mode - results will be saved to disk only")
    
    def store_analysis(self, analysis_result: Dict[str, Any]) -> bool:
        """
        Store analysis result in Supabase
        
        Args:
            analysis_result: Complete analysis result from processor
            
        Returns:
            Success status
        """
        if not self.client:
            logger.warning("Supabase not configured - saving locally only")
            return False
        
        try:
            # Prepare data for Supabase
            video_id = analysis_result.get("video_id", "unknown")
            
            # Main analysis record
            analysis_record = {
                "video_id": video_id,
                "processing_timestamp": analysis_result.get("processing_timestamp"),
                "processing_time_seconds": analysis_result.get("processing_time_seconds"),
                "video_duration_seconds": analysis_result.get("video_duration_seconds"),
                "realtime_factor": analysis_result.get("realtime_factor"),
                "total_analyzers": analysis_result["metadata"]["total_analyzers"],
                "successful_analyzers": analysis_result["metadata"]["successful_analyzers"],
                "failed_analyzers": analysis_result["metadata"]["failed_analyzers"],
                "analyzer_results": json.dumps(analysis_result.get("analyzer_results", {}))
            }
            
            # Insert main record
            result = self.client.table("video_analyses").insert(analysis_record).execute()
            analysis_id = result.data[0]["id"]
            
            logger.info(f"✅ Stored analysis {analysis_id} for video {video_id}")
            
            # Store individual analyzer results for easier querying
            self._store_analyzer_details(analysis_id, video_id, analysis_result.get("analyzer_results", {}))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store in Supabase: {e}")
            return False
    
    def _store_analyzer_details(self, analysis_id: int, video_id: str, analyzer_results: Dict[str, Any]):
        """Store individual analyzer results for detailed queries"""
        
        for analyzer_name, result in analyzer_results.items():
            try:
                # Count data points
                segment_count = 0
                if isinstance(result, dict):
                    for key in ["segments", "frames", "detections", "results"]:
                        if key in result and isinstance(result[key], list):
                            segment_count = len(result[key])
                            break
                
                # Store analyzer detail
                detail_record = {
                    "analysis_id": analysis_id,
                    "video_id": video_id,
                    "analyzer_name": analyzer_name,
                    "segment_count": segment_count,
                    "has_error": isinstance(result, dict) and ("error" in result or result.get("status") == "failed"),
                    "summary": json.dumps(result.get("summary", {})) if isinstance(result, dict) else None
                }
                
                self.client.table("analyzer_details").insert(detail_record).execute()
                
            except Exception as e:
                logger.warning(f"Failed to store details for {analyzer_name}: {e}")
    
    def get_analysis(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis for a video ID"""
        if not self.client:
            return None
        
        try:
            result = self.client.table("video_analyses")\
                .select("*")\
                .eq("video_id", video_id)\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()
            
            if result.data:
                analysis = result.data[0]
                # Parse JSON fields
                analysis["analyzer_results"] = json.loads(analysis["analyzer_results"])
                return analysis
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve analysis: {e}")
            return None
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get overall processing statistics"""
        if not self.client:
            return {"error": "Supabase not configured"}
        
        try:
            # Get total videos processed
            total_result = self.client.table("video_analyses").select("id", count="exact").execute()
            total_videos = total_result.count
            
            # Get average processing time
            stats_result = self.client.rpc("get_processing_stats").execute()
            
            if stats_result.data:
                stats = stats_result.data[0]
                return {
                    "total_videos_processed": total_videos,
                    "average_processing_time": stats.get("avg_processing_time", 0),
                    "average_realtime_factor": stats.get("avg_realtime_factor", 0),
                    "total_processing_hours": stats.get("total_processing_hours", 0),
                    "success_rate": stats.get("success_rate", 0)
                }
            
            return {"total_videos_processed": total_videos}
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

# SQL for Supabase tables (run in Supabase SQL editor):
"""
-- Main analysis table
CREATE TABLE video_analyses (
    id SERIAL PRIMARY KEY,
    video_id TEXT NOT NULL,
    processing_timestamp TIMESTAMP WITH TIME ZONE,
    processing_time_seconds FLOAT,
    video_duration_seconds FLOAT,
    realtime_factor FLOAT,
    total_analyzers INTEGER,
    successful_analyzers INTEGER,
    failed_analyzers INTEGER,
    analyzer_results JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Analyzer details for easier querying
CREATE TABLE analyzer_details (
    id SERIAL PRIMARY KEY,
    analysis_id INTEGER REFERENCES video_analyses(id),
    video_id TEXT NOT NULL,
    analyzer_name TEXT NOT NULL,
    segment_count INTEGER DEFAULT 0,
    has_error BOOLEAN DEFAULT FALSE,
    summary JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_video_analyses_video_id ON video_analyses(video_id);
CREATE INDEX idx_video_analyses_created_at ON video_analyses(created_at DESC);
CREATE INDEX idx_analyzer_details_analysis_id ON analyzer_details(analysis_id);
CREATE INDEX idx_analyzer_details_analyzer_name ON analyzer_details(analyzer_name);

-- Function for processing statistics
CREATE OR REPLACE FUNCTION get_processing_stats()
RETURNS TABLE (
    avg_processing_time FLOAT,
    avg_realtime_factor FLOAT,
    total_processing_hours FLOAT,
    success_rate FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        AVG(processing_time_seconds)::FLOAT as avg_processing_time,
        AVG(realtime_factor)::FLOAT as avg_realtime_factor,
        SUM(processing_time_seconds) / 3600::FLOAT as total_processing_hours,
        (AVG(CASE WHEN failed_analyzers = 0 THEN 1 ELSE 0 END) * 100)::FLOAT as success_rate
    FROM video_analyses;
END;
$$ LANGUAGE plpgsql;
"""