# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

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
```

## Essential Commands

### Environment Setup (CRITICAL)
```bash
# ALWAYS run before starting any services - fixes FFmpeg pthread issues
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh
```

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