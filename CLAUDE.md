# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This codebase implements a **GPU-optimized TikTok video analysis system** that processes videos through 19 active ML analyzers to generate comprehensive film production documentation. The system achieves **1.5-0.8x realtime performance** (80-90% faster than baseline) through advanced GPU optimizations including model caching, memory pool management, and multiprocess GPU parallelization.

## System Architecture

### Core Processing Pipeline
```
TikTok URL → Downloader → Video File → API → Cached GPU Executor → Worker Processes → JSON Results
                                              ↓
                                    [19 ML Analyzers with model caching]
```

### GPU-Optimized Service Topology
- **Production API**: `api/stable_production_api_multiprocess.py` (Port 8003)
- **GPU Executor**: Uses cached model loading for 80-90% performance improvement
- **Workers**: 3 GPU processes (dedicated Qwen2-VL worker + 2 shared workers) + CPU pool
- **Model Caching**: Persistent models in GPU memory between analyses
- **Results Storage**: 2-3MB JSON files in `/home/user/tiktok_production/results/`

### Directory Structure
```
tiktok_production/
├── analyzers/                    # 130+ analyzer files (19 active, optimized versions)
├── api/                         # FastAPI servers (use stable_production_api_multiprocess.py)
├── configs/                     # GPU groups, performance configs, optimization settings
├── utils/                       # GPU management, model caching, parallel processing
├── mass_processing/             # TikTok downloader and batch processing
├── results/                     # Analysis output JSONs
├── logs/                        # Service logs
├── backups/                     # System backups for rollback
└── monitoring_dashboard.py      # Real-time GPU monitoring
```

## Essential Commands

### Environment Setup (CRITICAL)
```bash
# ALWAYS run before starting any services - fixes FFmpeg pthread issues
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh
```

### Starting the Optimized System
```bash
# Start GPU-optimized production API with model caching
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py &

# Optional: Enable MPS for maximum GPU utilization (requires sudo)
sudo ./start_mps.sh
```

### Video Analysis Commands
```bash
# Analyze TikTok video (with URL auto-download)
curl -X POST "http://localhost:8003/analyze" \
  -H "Content-Type: application/json" \
  -d '{"tiktok_url": "https://www.tiktok.com/@username/video/123"}'

# Analyze local video file
curl -X POST "http://localhost:8003/analyze" \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/video.mp4"}'

# Check system health and GPU status
curl http://localhost:8003/health
```

### Performance Testing and Monitoring
```bash
# Run optimization test to measure performance improvements
python3 test_optimizations.py

# Monitor GPU utilization during analysis
watch -n 1 nvidia-smi

# Real-time monitoring dashboard
python3 monitoring_dashboard.py

# Check baseline vs optimized performance
cat test_results.txt
```

### Mass Processing Commands
```bash
cd mass_processing

# Setup bulk processing
pip install -r requirements.txt
python3 init_db.py

# Add URLs to processing queue
python3 bulk_processor.py add-urls urls.txt --priority 5

# Start processing with workers
python3 bulk_processor.py process --workers 4

# Monitor processing status
python3 bulk_processor.py status

# Web dashboard
python3 dashboard.py  # Access at http://localhost:5000
```

## ML Analyzer Architecture

### Active Analyzers (17)
The system uses a carefully optimized set of analyzers distributed across GPU workers:

**GPU Worker 0 (Dedicated):**
- `qwen2_vl_temporal` - Qwen2-VL-7B video understanding (requires 16GB VRAM)

**GPU Worker 1 (Visual Analysis):**
- `object_detection` - YOLOv8x object detection
- `text_overlay` - EasyOCR text detection optimized for TikTok
- `background_segmentation` - SegFormer semantic segmentation
- `camera_analysis` - Camera movement tracking

**GPU Worker 2 (Detail Analysis):**
- `scene_segmentation`, `color_analysis`, `body_pose`, `age_estimation`
- `content_quality`, `eye_tracking`, `cut_analysis`

**CPU Workers (Audio/Metadata):**
- `speech_transcription` (Whisper), `audio_analysis`, `audio_environment`
- `speech_emotion`, `temporal_flow`, `speech_flow`

### Analyzer Base Classes
All analyzers inherit from `GPUBatchAnalyzer` with:
- `_load_model_impl()` - Lazy model loading with caching
- `analyze(video_path)` - Main analysis entry point
- `process_batch_gpu()` - GPU batch processing
- Automatic GPU memory management

### Configuration Files
- `ml_analyzer_registry_complete.py` - Central analyzer registry (30+ total, 19 active)
- `configs/gpu_groups_config.py` - GPU workload distribution and disabled analyzers
- `configs/performance_config.py` - Frame sampling rates and batch sizes
- `utils/multiprocess_gpu_executor_registry_cached.py` - Cached model executor

## GPU Optimization System

### Model Caching Architecture
The system implements persistent model caching that keeps ML models in GPU memory between analyses:

```python
# Model caching is automatic and transparent
# First analysis: Models load from disk (slower)
# Subsequent analyses: Models reused from GPU cache (much faster)
```

### Memory Pool Optimization
Environment variables in `fix_ffmpeg_env.sh`:
```bash
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.9'
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### Performance Metrics
- **Baseline Performance**: 394s processing time (8.02x realtime)
- **Optimized Performance**: 78s first analysis, 39s cached analysis
- **Improvement**: 80-90% faster, approaching realtime (0.8x-1.5x)
- **Quality**: 100% reconstruction score maintained, no data loss

## Development Workflows

### Testing Single Analyzers
```python
# Test individual analyzer
from analyzers.gpu_batch_object_detection_yolo import GPUBatchObjectDetectionYOLO
analyzer = GPUBatchObjectDetectionYOLO()
result = analyzer.analyze('/path/to/video.mp4')
```

### Adding New Analyzers
1. Create analyzer in `analyzers/` inheriting from `GPUBatchAnalyzer`
2. Register in `ml_analyzer_registry_complete.py`
3. Add to appropriate GPU group in `configs/gpu_groups_config.py`
4. Test with `python3 test_analyzer_quality_v2.py`

### Performance Optimization
```bash
# Validate system performance
python3 verify_analyzers.py

# Test with monitoring
python3 monitor_performance.py &
# Run analysis
# Stop monitoring with Ctrl+C

# Generate performance report
python3 analyze_performance.py
```

## System Administration

### Health Monitoring
```bash
# System health check with GPU stats
python3 -c "
import subprocess, requests
# Check GPU
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.free,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True)
print('GPU Status:', gpu.stdout)
# Check API
r = requests.get('http://localhost:8003/health')
print('API Status:', r.json())
"
```

### Backup and Rollback
```bash
# Create backup before changes
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p backups/$timestamp
cp -r configs/ utils/ fix_ffmpeg_env.sh backups/$timestamp/

# Rollback if needed
cp -r backups/[TIMESTAMP]/* .
pkill -f stable_production_api
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py
```

### Troubleshooting

#### GPU Out of Memory
```bash
# Check GPU memory
nvidia-smi

# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Restart with fresh GPU state
pkill -f python
sudo nvidia-smi -i 0 -c DEFAULT
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py
```

#### Performance Issues
```bash
# Check if FFmpeg environment is sourced
echo $PYTORCH_CUDA_ALLOC_CONF

# Verify model caching is working
grep -i "reusing cached" logs/api_optimized.log

# Test optimization improvements
python3 test_optimizations.py
```

#### API Not Responding
```bash
# Check processes
ps aux | grep stable_production_api

# Check port availability
lsof -i :8003

# Restart API
pkill -f stable_production_api
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py > logs/api_restart.log 2>&1 &
```

## Important Notes

### Critical Environment Setup
- **ALWAYS** run `source fix_ffmpeg_env.sh` before starting services
- Use `stable_production_api_multiprocess.py` (NOT `stable_production_api.py`)
- Model caching requires consistent GPU worker configuration

### Performance Optimization
- GPU Worker 0 is dedicated exclusively to Qwen2-VL (requires 16GB VRAM)
- Second analysis is significantly faster due to model caching
- Monitor GPU utilization with `nvidia-smi` during analysis
- Results are 2-3MB JSON files with comprehensive frame-by-frame data

### System Requirements
- NVIDIA Quadro RTX 8000 (44.5GB VRAM)
- Ubuntu 22.04 LTS with CUDA 12.4
- 40GB+ RAM, 100GB+ storage
- Python 3.10+ with specific ML model dependencies

### Current Status (July 2025)
- **19 active analyzers** (out of 30+ total available)
- **GPU-optimized** with 80-90% performance improvement
- **Production-ready** with model caching and memory optimization
- **Quality maintained** at 100% reconstruction score