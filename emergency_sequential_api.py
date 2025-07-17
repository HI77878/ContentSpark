#!/usr/bin/env python3
"""
EMERGENCY SEQUENTIAL API - KEINE PARALLELISIERUNG!
ProcessPoolExecutor KOMPLETT ENTFERNT - alle Analyzer sequentiell
"""

import os
import sys
import json
import time
import gc
import logging
from datetime import datetime
from pathlib import Path

# Fix FFmpeg environment FIRST
os.environ['LD_LIBRARY_PATH'] = '/home/user/ffmpeg-install/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PATH'] = '/home/user/ffmpeg-install/bin:' + os.environ.get('PATH', '')
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "protocol_whitelist=file,http,https,tcp,tls"
os.environ['OPENCV_VIDEOIO_PRIORITY_BACKEND'] = "4"  # cv2.CAP_GSTREAMER
os.environ['OPENCV_FFMPEG_MULTITHREADED'] = "0"
os.environ['OPENCV_FFMPEG_DEBUG'] = "1"

# Add current directory to path
sys.path.insert(0, '/home/user/tiktok_production')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/user/tiktok_production/logs/emergency_sequential.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def analyze_video_sequential(video_path):
    """Führe ALLE Analyzer SEQUENTIELL aus - keine Parallelisierung"""
    
    # Import registry
    from ml_analyzer_registry_complete import ML_ANALYZERS
    
    results = {}
    successful_analyzers = 0
    
    logger.info(f"🚀 STARTING SEQUENTIAL ANALYSIS: {video_path}")
    logger.info(f"📊 Total Analyzers: {len(ML_ANALYZERS)}")
    
    start_time = time.time()
    
    for analyzer_name, analyzer_class in ML_ANALYZERS.items():
        analyzer_start = time.time()
        
        logger.info(f"\n🔄 Processing {analyzer_name}...")
        
        try:
            # Instantiate analyzer
            analyzer = analyzer_class()
            
            # Run analysis
            result = analyzer.analyze(video_path)
            
            # Count segments
            segments = len(result.get('segments', []))
            
            # Store result
            results[analyzer_name] = result
            
            # Log success
            elapsed = time.time() - analyzer_start
            logger.info(f"✅ {analyzer_name}: {segments} segments ({elapsed:.1f}s)")
            
            if segments > 0:
                successful_analyzers += 1
            
            # Force garbage collection
            del analyzer
            gc.collect()
            
        except Exception as e:
            elapsed = time.time() - analyzer_start
            logger.error(f"❌ {analyzer_name} failed: {str(e)} ({elapsed:.1f}s)")
            results[analyzer_name] = {"segments": [], "error": str(e)}
    
    total_time = time.time() - start_time
    success_rate = (successful_analyzers / len(ML_ANALYZERS)) * 100
    
    logger.info(f"\n🎯 SEQUENTIAL ANALYSIS COMPLETE")
    logger.info(f"📊 Success Rate: {successful_analyzers}/{len(ML_ANALYZERS)} = {success_rate:.1f}%")
    logger.info(f"⏱️ Total Time: {total_time:.1f}s")
    
    return {
        "results": results,
        "successful_analyzers": successful_analyzers,
        "total_analyzers": len(ML_ANALYZERS),
        "success_rate": success_rate,
        "processing_time": total_time
    }

def save_results(analysis_result, video_path):
    """Save analysis results to file"""
    
    # Create filename
    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{video_name}_sequential_{timestamp}.json"
    
    # Save to results directory
    results_dir = Path("/home/user/tiktok_production/results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / filename
    
    # Add metadata
    analysis_result.update({
        "video_path": str(video_path),
        "analysis_type": "sequential",
        "timestamp": timestamp,
        "results_file": str(results_file)
    })
    
    # Save JSON
    with open(results_file, 'w') as f:
        json.dump(analysis_result, f, indent=2, default=str)
    
    logger.info(f"💾 Results saved: {results_file}")
    return str(results_file)

def main():
    """Main function"""
    
    if len(sys.argv) != 2:
        print("Usage: python3 emergency_sequential_api.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Check if video exists
    if not Path(video_path).exists():
        logger.error(f"❌ Video file not found: {video_path}")
        sys.exit(1)
    
    logger.info(f"🎬 Video: {video_path}")
    
    # Run sequential analysis
    analysis_result = analyze_video_sequential(video_path)
    
    # Save results
    results_file = save_results(analysis_result, video_path)
    
    # Final summary
    logger.info(f"\n🚀 EMERGENCY SEQUENTIAL ANALYSIS COMPLETE")
    logger.info(f"📊 Success Rate: {analysis_result['success_rate']:.1f}%")
    logger.info(f"⏱️ Processing Time: {analysis_result['processing_time']:.1f}s")
    logger.info(f"💾 Results: {results_file}")
    
    # Print key metrics
    print(f"\n" + "="*60)
    print(f"🎯 SUCCESS RATE: {analysis_result['success_rate']:.1f}%")
    print(f"📊 SUCCESSFUL: {analysis_result['successful_analyzers']}/{analysis_result['total_analyzers']}")
    print(f"⏱️ TIME: {analysis_result['processing_time']:.1f}s")
    print(f"💾 RESULTS: {results_file}")
    print(f"="*60)

if __name__ == "__main__":
    main()