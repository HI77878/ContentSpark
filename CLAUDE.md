# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

<<<<<<< HEAD
This codebase implements a **24-analyzer video analysis system** optimized for high GPU utilization. The system processes videos through real ML models to generate comprehensive film production documentation. After recent optimizations, the system achieved **100% analyzer success rate** (up from 60.9%) and **91% faster Qwen2-VL processing** (11s vs 128s).

## System Architecture

### Service Topology
- **Production API** (Port 8000): Main API server with multiprocess GPU parallelization
- **Clean Server API** (Port 8003): Alternative endpoint using stable_production_api.py
- **Results Storage**: JSON files in `/home/user/tiktok_production/results/`
- **GPU**: Quadro RTX 8000 with 44.5GB memory
- **Parallelization**: 3 GPU workers using multiprocessing.spawn

### Multiprocess Architecture
- **Worker Distribution**: 
  - Worker 0: Reserved for Qwen2-VL (16GB VRAM)
  - Worker 1-2: Handle remaining analyzers
- **CPU Thread Management**: 
  - Dynamic allocation: `threads_per_worker = cpu_count // gpu_workers`
  - Prevents CPU oversubscription
- **Memory Isolation**: Each worker has independent CUDA context

### Core Directory Structure
```
tiktok_production/
├── analyzers/          # 24 ML analyzers inheriting from GPUBatchAnalyzer
├── api/               # FastAPI server (stable_production_api_multiprocess.py)
├── configs/           # GPU groups, performance configurations
├── archive_variants/   # 50+ legacy analyzer versions (archived)
├── results/           # Analysis output JSONs
├── logs/             # Service logs
└── utils/            # Shared utilities including staged_gpu_executor
=======
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
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
```

## Essential Commands

### Environment Setup (CRITICAL)
```bash
# ALWAYS run before starting any services - fixes FFmpeg pthread issues
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh
```

<<<<<<< HEAD
### Starting the System
```bash
# Option 1: Use clean server script (RECOMMENDED)
./start_clean_server.sh

# Option 2: Start API manually
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py

# Option 3: Legacy start script
./start.sh
```

### Video Analysis
```bash
# Via API
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/video.mp4"}'

# Via CLI workflow
python3 single_workflow.py "https://www.tiktok.com/@username/video/123"

# Check API health
curl http://localhost:8000/health
```

### Testing and Monitoring
```bash
# Test individual analyzers
python3 quick_analyzer_test.py

# Performance test
python3 test_final_performance.py

# Direct Qwen2-VL test (shows optimal performance)
python3 test_qwen2vl_direct.py

# Monitor GPU
watch -n 1 nvidia-smi
nvidia-smi dmon -i 0 -s pucm -d 1

# Check logs
tail -f /home/user/tiktok_production/logs/stable_multiprocess_api.log
tail -f /home/user/tiktok_production/logs/stable_api.log  # Clean server logs
```

## Analyzer System Architecture

### Base Pattern
All analyzers inherit from `GPUBatchAnalyzer`:
- `_load_model_impl()` - Lazy model loading
- `analyze(video_path)` - Main entry point
- `process_batch_gpu(frames, frame_times)` - GPU batch processing
- Automatic GPU memory cleanup after processing

### 4-Stage GPU Execution (Optimized Architecture)
1. **Stage 1 (Heavy GPU - Runs Alone)**: 
   - `qwen2_vl_temporal` (~16GB VRAM reserved)
   - Optimized from 128s to 11s (91% improvement) via batch processing

2. **Stage 2 (Medium GPU - 6 concurrent)**: 
   - object_detection, background_segmentation, text_overlay
   - visual_effects, product_detection, face_emotion

3. **Stage 3 (Light GPU - 7 concurrent)**: 
   - camera_analysis, color_analysis, body_pose, age_estimation
   - content_quality, eye_tracking, composition_analysis

4. **Stage 4 (Fast/CPU - 10 concurrent)**: 
   - cut_analysis, scene_segmentation, speech_transcription
   - audio_analysis, audio_environment, speech_emotion
   - speech_rate, speech_flow, temporal_flow
   - cross_analyzer_intelligence (correlates all results)

### Active Analyzers (24 total)
- **Video Understanding**: qwen2_vl_temporal
- **Object/Scene**: object_detection, background_segmentation, text_overlay, visual_effects, product_detection
- **Person Analysis**: face_emotion, body_pose, eye_tracking, age_estimation
- **Content Analysis**: composition_analysis, content_quality, color_analysis
- **Audio (Fixed ProcessPool Issues)**: speech_transcription, audio_analysis, audio_environment, speech_emotion, speech_rate, speech_flow
- **Technical**: camera_analysis, cut_analysis, scene_segmentation
- **Narrative**: temporal_flow
- **Intelligence**: cross_analyzer_intelligence

## Key Configuration Files

- `ml_analyzer_registry_complete.py` - Central analyzer registry (24 analyzers)
- `configs/gpu_groups_config.py` - 4-stage GPU execution and memory management
- `configs/performance_config.py` - Frame sampling and batch sizes
- `api/stable_production_api_multiprocess.py` - Main API server with 3 GPU workers
- `utils/audio_processing_fix.py` - Fixed audio analyzers (removed ProcessPool conflicts)

## Known Issues and Solutions

### FFmpeg Assertion Error
- **Problem**: `Assertion fctx->async_lock failed at libavcodec/pthread_frame.c:175`
- **Solution**: Always run `source fix_ffmpeg_env.sh` before starting services

### Audio Analyzer Issues (FIXED)
- **Problem**: ProcessPoolExecutor conflicts with librosa/FFmpeg
- **Solution**: Audio analyzers now use forkserver method and run directly in Stage 4
- **Result**: 100% success rate for all audio analyzers

### Performance Achievements
- **Before Optimization**: ~14x realtime (140s for 10s video)
- **After Optimization**: ~3x realtime API, ~1.1x realtime direct
- **Qwen2-VL Optimization**: 91% faster (11s vs 128s in direct test)
- **GPU Utilization**: 15.5GB VRAM usage (efficient utilization)

## Development Workflow

### Adding New Analyzers
1. Create file in `analyzers/` inheriting from `GPUBatchAnalyzer`
2. Register in `ml_analyzer_registry_complete.py`
3. Add to appropriate stage in `configs/gpu_groups_config.py`
4. Configure frame sampling in `configs/performance_config.py`

### Testing Single Analyzer
```python
from analyzers.your_analyzer import YourAnalyzer
analyzer = YourAnalyzer()
result = analyzer.analyze('/path/to/video.mp4')
print(f'Found {len(result.get("segments", []))} segments')
```

## Current Status (July 2025)

- **Active Analyzers**: 24 (clean server MVP version)
- **Success Rate**: 100% (all 24 analyzers working!)
- **Processing Speed**: ~3x realtime API, ~1.1x realtime direct
- **API**: Port 8000 (stable_production_api_multiprocess.py)
- **Architecture**: Clean 4-stage GPU execution pipeline
- **Philosophy**: ONE API, ONE WORKFLOW, NO DUPLICATES

## Recent Optimizations

### Qwen2-VL Batch Processing (91% Improvement)
- All video segments processed in single GPU call
- Mixed precision with torch.cuda.amp.autocast
- Optimized resolution (512x384)
- Global model loading to avoid repeated initialization

### GPU Optimization Settings
```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
```

### Audio Analyzer Fix
- Removed ProcessPoolExecutor conflicts
- Implemented forkserver multiprocessing
- Direct execution in Stage 4
- Result: 100% success rate for all 6 audio analyzers

## Troubleshooting Common Issues

### Low GPU Utilization
- **Solution**: Ensure staged execution is working properly
- Check `nvidia-smi` during analysis for ~15GB VRAM usage
- Verify batch sizes in `configs/performance_config.py`

### Slow Analysis
- **API vs Direct**: API includes model loading overhead
- For optimal speed, use model preloading (experimental)
- Consider reducing frame intervals for faster processing

### Memory Issues
- **CUDA OOM**: Reduce batch sizes or disable concurrent analyzers
- Qwen2-VL reserved 16GB on Worker 0
- Monitor with `nvidia-smi` for memory spikes

## Quick Reference

### Check System Health
```bash
# API health
curl http://localhost:8000/health
curl http://localhost:8003/health  # Clean server

# GPU status
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

# Active analyzers
python3 -c "from ml_analyzer_registry_complete import ML_ANALYZERS; print(f'Active: {len(ML_ANALYZERS)} analyzers')"
```

### Quick Analysis Test
```bash
# Test with sample video
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/home/user/sample_videos/test.mp4"}'

# Direct Qwen2-VL test (fastest)
python3 test_qwen2vl_direct.py /path/to/video.mp4
```

### Performance Monitoring
```bash
# Real-time GPU monitoring during analysis
watch -n 0.5 'nvidia-smi | grep "MiB /" | head -3'

# Check analyzer timings
grep "completed in" logs/stable_multiprocess_api.log | tail -20
```

## Key Achievements
- **100% Success Rate**: All 24 analyzers working (up from 60.9%)
- **91% Faster Qwen2-VL**: 11s vs 128s in direct testing
- **Clean Architecture**: Single variant per analyzer, 4-stage execution
- **Production Ready**: Stable multiprocess API with error handling
=======
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
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
