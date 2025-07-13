# Troubleshooting Guide - TikTok Video Analysis System

## Overview

This comprehensive troubleshooting guide covers common issues, error resolution, and system optimization for the TikTok Video Analysis System. The guide is organized by error types and includes step-by-step solutions.

## Quick Diagnostic Commands

### System Health Check
```bash
# Complete system status check
cd /home/user/tiktok_production

# 1. Check GPU status
nvidia-smi

# 2. Check API health
curl http://localhost:8003/health

# 3. Check environment
echo $PYTORCH_CUDA_ALLOC_CONF

# 4. Check processes
ps aux | grep stable_production_api

# 5. Check logs
tail -f logs/api_optimized.log
```

### Emergency Recovery
```bash
# Emergency system restart
pkill -f stable_production_api
sudo nvidia-smi -i 0 -c DEFAULT
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py &
```

## GPU-Related Issues

### 1. CUDA Out of Memory (OOM)

#### Symptoms
```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB (GPU 0; 44.53 GiB total capacity)
```

#### Immediate Solutions
```bash
# Step 1: Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Step 2: Check GPU memory usage
nvidia-smi

# Step 3: Kill all GPU processes if necessary
sudo fuser -v /dev/nvidia*
sudo kill -9 <PID>  # Kill specific processes

# Step 4: Reset GPU to default mode
sudo nvidia-smi -i 0 -c DEFAULT
```

#### Long-term Solutions
```bash
# 1. Reduce batch sizes (edit configs/performance_config.py)
nano configs/performance_config.py

# Reduce these values by 50%:
OPTIMIZED_BATCH_SIZES = {
    'object_detection': 32,  # Reduced from 64
    'text_overlay': 8,       # Reduced from 16
    'background_segmentation': 4,  # Reduced from 8
}

# 2. Disable memory-intensive analyzers temporarily
nano configs/gpu_groups_config.py

# Add to DISABLED_ANALYZERS:
DISABLED_ANALYZERS = [
    # ... existing disabled analyzers
    'qwen2_vl_temporal',  # Temporarily disable if needed
]

# 3. Implement progressive batch size reduction
nano utils/gpu_memory_manager.py
```

```python
# Automatic batch size reduction on OOM
def handle_oom_error(analyzer_name):
    current_batch = OPTIMIZED_BATCH_SIZES.get(analyzer_name, 8)
    new_batch = max(1, current_batch // 2)
    OPTIMIZED_BATCH_SIZES[analyzer_name] = new_batch
    
    logger.warning(f"OOM detected: Reduced {analyzer_name} batch size to {new_batch}")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

#### Prevention
```bash
# 1. Monitor GPU memory before starting analysis
python3 -c "
import torch
free_mem = torch.cuda.mem_get_info()[0] / 1024**2
print(f'Free GPU memory: {free_mem:.0f} MB')
if free_mem < 8000:
    print('WARNING: Less than 8GB free - consider clearing cache')
"

# 2. Set up automatic monitoring
python3 monitoring_dashboard.py &
```

### 2. GPU Not Available

#### Symptoms
```
RuntimeError: No CUDA-capable device is detected
AssertionError: Torch not compiled with CUDA support
```

#### Diagnosis
```bash
# Check NVIDIA driver
nvidia-smi
# Should show GPU information

# Check CUDA installation
nvcc --version
# Should show CUDA version

# Check PyTorch CUDA support
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should print: CUDA available: True
```

#### Solutions
```bash
# 1. Reinstall NVIDIA drivers
sudo apt remove --purge nvidia-*
sudo apt autoremove
sudo apt install -y nvidia-driver-535
sudo reboot

# 2. Reinstall CUDA toolkit
sudo apt install -y cuda-toolkit-12-4

# 3. Reinstall PyTorch with CUDA support
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Check environment variables
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 3. Low GPU Utilization

#### Symptoms
- GPU utilization consistently below 20%
- Analysis takes longer than expected
- No improvement from optimizations

#### Diagnosis
```bash
# Check if optimizations are loaded
echo $PYTORCH_CUDA_ALLOC_CONF
# Should show: max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.9

# Check if model caching is working
grep -i "reusing cached" logs/api_optimized.log
# Should show cache hits after first analysis

# Check worker processes
ps aux | grep python | grep stable_production_api
# Should show multiple worker processes
```

#### Solutions
```bash
# 1. Ensure environment is properly sourced
source fix_ffmpeg_env.sh
echo "Environment loaded: $PYTORCH_CUDA_ALLOC_CONF"

# 2. Verify multiprocess API is being used
ps aux | grep stable_production_api_multiprocess
# Should show the multiprocess version

# 3. Check if MPS is needed
sudo nvidia-cuda-mps-control -d
echo "set_default_device_pinned_mem_limit 0 40000M" | sudo nvidia-cuda-mps-control

# 4. Restart with optimizations
pkill -f stable_production_api
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py &
```

## API and Service Issues

### 1. API Not Responding

#### Symptoms
```
curl: (7) Failed to connect to localhost port 8003: Connection refused
```

#### Diagnosis
```bash
# Check if API process is running
ps aux | grep stable_production_api

# Check port availability
lsof -i :8003
netstat -tulpn | grep 8003

# Check for errors in logs
tail -f logs/api_optimized.log
```

#### Solutions
```bash
# 1. Check if another process is using the port
sudo lsof -i :8003
# Kill conflicting process if needed

# 2. Start API manually to see errors
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py

# 3. Restart with proper environment
pkill -f stable_production_api
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py > logs/api_restart.log 2>&1 &

# 4. Check firewall settings
sudo ufw status
# Ensure port 8003 is not blocked
```

### 2. Slow API Response

#### Symptoms
- Requests timeout after 2+ minutes
- Processing takes 300+ seconds
- No cache benefits

#### Diagnosis
```bash
# Check system resources
htop  # CPU usage
free -h  # Memory usage
df -h  # Disk space

# Monitor during analysis
python3 monitoring_dashboard.py &
# Run analysis
curl -X POST "http://localhost:8003/analyze" -H "Content-Type: application/json" -d '{"video_path": "/path/to/test.mp4"}'
```

#### Solutions
```bash
# 1. Verify optimizations are active
grep "Model caching enabled" logs/api_optimized.log
grep "GPU optimization active" logs/api_optimized.log

# 2. Clear system caches
sync
echo 3 > /proc/sys/vm/drop_caches

# 3. Check for disk I/O issues
iotop  # Monitor disk usage during analysis

# 4. Restart with full optimization stack
./apply_optimizations.py
```

### 3. Memory Leaks

#### Symptoms
- System memory usage increases over time
- GPU memory not released after analysis
- System becomes unresponsive

#### Diagnosis
```bash
# Monitor memory usage over time
watch -n 5 'free -h && nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'

# Check for memory leaks in logs
grep -i "memory" logs/api_optimized.log
grep -i "leak" logs/api_optimized.log
```

#### Solutions
```bash
# 1. Force garbage collection
python3 -c "
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
print('Memory cleaned')
"

# 2. Restart API periodically
# Add to crontab: 0 */6 * * * /path/to/restart_api.sh

# 3. Implement automatic memory monitoring
nano utils/memory_monitor.py
```

```python
import psutil
import torch
import time

def monitor_memory():
    while True:
        # System memory
        mem = psutil.virtual_memory()
        if mem.percent > 90:
            print("High memory usage detected - triggering cleanup")
            gc.collect()
            
        # GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if gpu_mem > 0.9:
                print("High GPU memory usage - clearing cache")
                torch.cuda.empty_cache()
                
        time.sleep(60)  # Check every minute
```

## FFmpeg and Video Processing Issues

### 1. FFmpeg pthread Crashes

#### Symptoms
```
Assertion fctx->async_lock failed at libavcodec/pthread_frame.c:175
```

#### Solutions
```bash
# 1. ALWAYS source the environment fix
source fix_ffmpeg_env.sh

# Verify environment variables are set:
echo $OPENCV_FFMPEG_MULTITHREADED  # Should be 0
echo $OPENCV_VIDEOIO_PRIORITY_BACKEND  # Should be 4

# 2. If still occurring, try alternative FFmpeg settings
export OPENCV_FFMPEG_CAPTURE_OPTIONS="protocol_whitelist=file,http,https,tcp,tls,rtmp"
export OPENCV_FFMPEG_LOGLEVEL=16  # Reduce logging

# 3. Restart with clean environment
pkill -f stable_production_api
unset OPENCV_FFMPEG_MULTITHREADED
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py
```

### 2. Video Download Failures

#### Symptoms
- TikTok URLs fail to download
- yt-dlp errors
- Network timeout issues

#### Solutions
```bash
# 1. Update yt-dlp
pip install --upgrade yt-dlp

# 2. Test download manually
yt-dlp "https://www.tiktok.com/@username/video/123" -o "test_download.%(ext)s"

# 3. Configure network settings
nano mass_processing/download_config.py
```

```python
# Enhanced download configuration
DOWNLOAD_CONFIG = {
    'format': 'best[height<=1080]',
    'retries': 3,
    'fragment_retries': 3,
    'timeout': 30,
    'http_headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
}
```

### 3. Video Format Issues

#### Symptoms
- Unsupported video formats
- Corrupted video files
- Frame extraction failures

#### Solutions
```bash
# 1. Check video file integrity
ffprobe input_video.mp4

# 2. Convert to compatible format
ffmpeg -i input_video.mp4 -c:v libx264 -c:a aac -movflags +faststart output_video.mp4

# 3. Add format validation
nano utils/video_validator.py
```

```python
import cv2

def validate_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Cannot open video file"
            
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if frame_count <= 0 or fps <= 0:
            return False, "Invalid video properties"
            
        return True, "Video is valid"
    except Exception as e:
        return False, f"Validation error: {e}"
```

## Model and Analyzer Issues

### 1. Model Loading Failures

#### Symptoms
```
OSError: Unable to load weights from checkpoint
ConnectionError: Failed to download model
```

#### Solutions
```bash
# 1. Check internet connection and model cache
ls -la ~/.cache/huggingface/transformers/

# 2. Clear corrupted cache
rm -rf ~/.cache/huggingface/transformers/models--Qwen--Qwen2-VL-7B-Instruct

# 3. Manual model download
python3 -c "
from transformers import Qwen2VLForConditionalGeneration
model = Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')
print('Model downloaded successfully')
"

# 4. Check disk space
df -h ~/.cache/
```

### 2. Analyzer Failures

#### Symptoms
- Specific analyzers consistently fail
- Partial results returned
- Error messages in logs

#### Diagnosis
```bash
# Test individual analyzer
python3 -c "
from analyzers.gpu_batch_object_detection_yolo import GPUBatchObjectDetectionYOLO
analyzer = GPUBatchObjectDetectionYOLO()
result = analyzer.analyze('/path/to/test_video.mp4')
print('Analyzer test successful')
"

# Check analyzer configuration
python3 verify_analyzers.py
```

#### Solutions
```bash
# 1. Disable problematic analyzer temporarily
nano configs/gpu_groups_config.py
# Add to DISABLED_ANALYZERS list

# 2. Check analyzer dependencies
pip install --upgrade ultralytics  # For YOLO analyzers
pip install --upgrade easyocr      # For text overlay
pip install --upgrade whisper      # For speech transcription

# 3. Reset analyzer cache
python3 -c "
from utils.persistent_model_manager import PersistentModelManager
manager = PersistentModelManager()
manager.clear_cache()
print('Analyzer cache cleared')
"
```

### 3. Quality Issues

#### Symptoms
- Poor analysis results
- Low confidence scores
- Missing detections

#### Solutions
```bash
# 1. Check input video quality
ffprobe -v quiet -print_format json -show_format -show_streams input.mp4

# 2. Adjust quality settings
nano configs/performance_config.py

# Increase frame sampling for better quality:
FRAME_INTERVALS = {
    'object_detection': 5,   # Increased from 10
    'text_overlay': 15,      # Increased from 30
}

# 3. Enable high-quality mode
curl -X POST "http://localhost:8003/analyze" \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/video.mp4", "options": {"quality": "high"}}'
```

## Performance Issues

### 1. Cache Not Working

#### Symptoms
- No speed improvement on second analysis
- Models reload every time
- No cache hits in logs

#### Diagnosis
```bash
# Check if caching is enabled
grep -i "cache" logs/api_optimized.log

# Check cached model manager
python3 -c "
from utils.persistent_model_manager import PersistentModelManager
manager = PersistentModelManager()
print(f'Cached models: {len(manager.models)}')
"
```

#### Solutions
```bash
# 1. Ensure using multiprocess API
ps aux | grep stable_production_api_multiprocess

# 2. Verify cache configuration
python3 -c "
from utils.multiprocess_gpu_executor_registry_cached import MultiprocessGPUExecutorRegistryCached
executor = MultiprocessGPUExecutorRegistryCached()
print('Cached executor loaded successfully')
"

# 3. Restart with caching
pkill -f stable_production_api
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py
```

### 2. High CPU Usage

#### Symptoms
- CPU at 100% during analysis
- System becomes unresponsive
- Thermal throttling

#### Solutions
```bash
# 1. Check CPU governor
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# Should show "performance" for consistent performance

# 2. Limit CPU usage if needed
nano configs/system_config.py

# Add CPU limits:
CPU_CONFIG = {
    'max_cpu_percent': 80,
    'cpu_affinity': [0, 1, 2, 3],  # Use specific cores
}

# 3. Monitor CPU usage
htop  # Look for high CPU processes
```

### 3. I/O Bottlenecks

#### Symptoms
- High disk I/O wait times
- Slow frame extraction
- Storage full errors

#### Solutions
```bash
# 1. Monitor I/O
iotop -ao

# 2. Move temp files to faster storage
mkdir -p /tmp/tiktok_analysis
nano .env
# Set: TEMP_DIR=/tmp/tiktok_analysis

# 3. Clean up old files
find results/ -name "*.json" -mtime +7 -delete
find downloads/ -name "*.mp4" -mtime +1 -delete

# 4. Check disk space
df -h
```

## Specific Error Messages

### Error: "MPS client failed to connect"

#### Solution
```bash
# Stop MPS and use default mode
echo quit | sudo nvidia-cuda-mps-control
sudo nvidia-smi -i 0 -c DEFAULT

# Restart API
pkill -f stable_production_api
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py
```

### Error: "AssertionError: Expected tensor to be on cuda:0"

#### Solution
```bash
# Clear GPU state and restart
python3 -c "import torch; torch.cuda.empty_cache()"
pkill -f stable_production_api
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py
```

### Error: "No module named 'transformers'"

#### Solution
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt

# Check installation
python3 -c "import transformers; print(transformers.__version__)"
```

## System Maintenance

### Daily Maintenance
```bash
#!/bin/bash
# daily_maintenance.sh

# Clean GPU cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Clean system cache
sync
echo 1 > /proc/sys/vm/drop_caches

# Clean old logs
find logs/ -name "*.log" -mtime +7 -delete

# Check disk space
df -h | grep -E "(/$|/home)"

# Verify API health
curl -s http://localhost:8003/health | jq .status
```

### Weekly Maintenance
```bash
#!/bin/bash
# weekly_maintenance.sh

# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Python packages
pip install --upgrade -r requirements.txt

# Restart API with fresh state
pkill -f stable_production_api
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py &

# Run system health check
python3 test_optimizations.py
```

This troubleshooting guide provides comprehensive solutions for maintaining optimal performance of the TikTok Video Analysis System.