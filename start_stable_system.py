#!/usr/bin/env python3
"""
Stable startup script for the production system
Ensures all requirements are met before starting
"""

import os
import sys
import subprocess
import time
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_cuda():
    """Check CUDA availability"""
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available!")
        return False
    
    logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    return True

def fix_ffmpeg_env():
    """Apply FFmpeg pthread fix"""
    logger.info("🔧 Applying FFmpeg environment fix...")
    
    # Source the fix script
    fix_script = Path("/home/user/tiktok_production/fix_ffmpeg_env.sh")
    if fix_script.exists():
        # Export the environment variables directly
        os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"
        logger.info("✅ FFmpeg environment fixed")
    else:
        logger.warning("⚠️  fix_ffmpeg_env.sh not found, setting env vars manually")
        os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"

def check_dependencies():
    """Check all required dependencies"""
    required_modules = [
        'torch', 'cv2', 'transformers', 'whisper', 'easyocr', 
        'mediapipe', 'ultralytics', 'librosa', 'numpy', 'fastapi'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        logger.error(f"❌ Missing modules: {', '.join(missing)}")
        return False
    
    logger.info("✅ All dependencies available")
    return True

def check_models():
    """Check if key model files exist"""
    model_files = [
        'yolov8n.pt',
        'yolov8x.pt',
        'models'  # Transformers cache directory
    ]
    
    for model in model_files:
        path = Path(f"/home/user/tiktok_production/{model}")
        if not path.exists():
            logger.warning(f"⚠️  Model file/directory not found: {model}")
    
    return True

def kill_existing_api():
    """Kill any existing API processes"""
    logger.info("🔍 Checking for existing API processes...")
    
    # Kill processes on port 8003
    try:
        result = subprocess.run(
            ["lsof", "-ti:8003"], 
            capture_output=True, 
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                subprocess.run(["kill", "-9", pid])
                logger.info(f"   Killed process {pid}")
    except:
        pass

def start_api():
    """Start the stable production API"""
    logger.info("\n🚀 Starting Stable Production API...")
    
    # Change to the project directory
    os.chdir("/home/user/tiktok_production")
    
    # Set Python path
    if "/home/user/tiktok_production" not in sys.path:
        sys.path.insert(0, "/home/user/tiktok_production")
    
    # Import and run the API
    try:
        from api.stable_production_api import app
        import uvicorn
        
        logger.info("✅ API modules loaded successfully")
        logger.info("🌐 Starting server on http://0.0.0.0:8003")
        logger.info("\n" + "="*50)
        logger.info("System ready! Use the following command to test:")
        logger.info("curl -X POST http://localhost:8003/analyze -H 'Content-Type: application/json' -d '{\"video_path\": \"/home/user/tiktok_production/downloads/videos/7425998222721633569.mp4\"}'")
        logger.info("="*50 + "\n")
        
        # Run the server
        uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")
        
    except Exception as e:
        logger.error(f"❌ Failed to start API: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main startup sequence"""
    logger.info("=== TikTok Production System Startup ===")
    
    # Run all checks
    checks = [
        ("CUDA", check_cuda),
        ("Dependencies", check_dependencies),
        ("Models", check_models)
    ]
    
    for name, check_func in checks:
        if not check_func():
            logger.error(f"❌ {name} check failed!")
            sys.exit(1)
    
    # Apply fixes
    fix_ffmpeg_env()
    kill_existing_api()
    
    # Start the API
    if not start_api():
        logger.error("❌ Failed to start API")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure multiprocessing uses spawn
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()