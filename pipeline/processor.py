#!/usr/bin/env python3
"""
Production Video Processor
Main processing pipeline for TikTok videos
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

sys.path.append('/home/user/tiktok_production')

from optimized_batch_processor import OptimizedBatchProcessor
from utils.json_serializer import safe_json_dump

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Main video processing pipeline"""
    
    def __init__(self):
        self.processor = OptimizedBatchProcessor()
        self.results_dir = Path("/home/user/tiktok_production/data/completed")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def process_video(self, video_path: str, video_id: str = None) -> Dict[str, Any]:
        """
        Process a video through all analyzers
        
        Args:
            video_path: Path to video file
            video_id: Optional video ID (TikTok ID)
            
        Returns:
            Analysis results with metadata
        """
        logger.info(f"🎬 Starting processing: {video_path}")
        start_time = time.time()
        
        # Validate video exists
        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            return {"error": "Video not found", "video_path": video_path}
        
        # Extract video ID if not provided
        if not video_id:
            video_id = Path(video_path).stem
        
        # Process with all analyzers
        try:
            logger.info("🔄 Running analyzers...")
            analysis_results = self.processor.process_video(video_path, video_url=video_id)
            
            # Calculate processing metrics
            elapsed_time = time.time() - start_time
            
            # Get video duration
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frames / fps if fps > 0 else 0
            cap.release()
            
            # Build complete result
            result = {
                "video_id": video_id,
                "video_path": video_path,
                "processing_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": elapsed_time,
                "video_duration_seconds": duration,
                "realtime_factor": elapsed_time / duration if duration > 0 else 0,
                "analyzer_results": analysis_results,
                "metadata": {
                    "total_analyzers": len(analysis_results),
                    "successful_analyzers": sum(1 for r in analysis_results.values() 
                                               if not isinstance(r, dict) or r.get("status") != "failed"),
                    "failed_analyzers": sum(1 for r in analysis_results.values() 
                                          if isinstance(r, dict) and r.get("status") == "failed")
                }
            }
            
            # Save results locally
            result_file = self.results_dir / f"{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            if safe_json_dump(result, str(result_file)):
                logger.info(f"💾 Results saved: {result_file}")
            
            # Log performance
            logger.info(f"✅ Processing complete!")
            logger.info(f"   Duration: {duration:.1f}s video in {elapsed_time:.1f}s")
            logger.info(f"   Performance: {elapsed_time/duration:.1f}x realtime")
            logger.info(f"   Analyzers: {result['metadata']['successful_analyzers']} successful, "
                       f"{result['metadata']['failed_analyzers']} failed")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "error": str(e),
                "video_path": video_path,
                "video_id": video_id,
                "processing_timestamp": datetime.now().isoformat()
            }
    
    def get_data_quality_report(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality of analyzer results"""
        if "analyzer_results" not in result:
            return {"error": "No analyzer results found"}
        
        quality_report = {
            "excellent": [],  # >20 data points
            "good": [],       # 5-20 data points
            "poor": [],       # 1-4 data points
            "failed": []      # 0 or error
        }
        
        for analyzer_name, data in result["analyzer_results"].items():
            # Count data points
            count = 0
            
            if isinstance(data, dict):
                if data.get("status") == "failed" or "error" in data:
                    quality_report["failed"].append((analyzer_name, "error"))
                    continue
                
                # Count segments/frames/detections
                for key in ["segments", "frames", "detections", "results"]:
                    if key in data and isinstance(data[key], list):
                        count = len(data[key])
                        break
                
                if count == 0 and "summary" in data:
                    count = 10  # Assume good if has summary
            
            # Categorize
            if count > 20:
                quality_report["excellent"].append((analyzer_name, count))
            elif count >= 5:
                quality_report["good"].append((analyzer_name, count))
            elif count >= 1:
                quality_report["poor"].append((analyzer_name, count))
            else:
                quality_report["failed"].append((analyzer_name, 0))
        
        # Calculate score
        total = sum(len(quality_report[cat]) for cat in quality_report)
        excellent = len(quality_report["excellent"])
        good = len(quality_report["good"])
        
        quality_score = ((excellent * 3 + good * 2 + len(quality_report["poor"])) / (total * 3)) * 100 if total > 0 else 0
        
        quality_report["summary"] = {
            "total_analyzers": total,
            "quality_score": quality_score,
            "target_achieved": quality_score >= 90
        }
        
        return quality_report