# TikTok Production System Summary

## System Status (July 4, 2025)
The TikTok video analysis production system is fully operational and optimized.

## Active Components

### 1. Core API
- **Main Service**: `api/stable_production_api.py` (Port 8003)
- **Status**: Running and healthy
- **Features**: 29 active ML analyzers with GPU optimization

### 2. Directory Structure
```
tiktok_production/
├── analyzers/          # 29 active ML analyzer implementations
├── api/               # Production API servers
├── configs/           # GPU groups and performance configurations
├── mass_processing/   # TikTok downloader and batch processing
├── pipeline/          # Core processing pipeline
├── utils/            # Multiprocess GPU executor
├── results/          # Analysis output JSONs
├── logs/             # Production logs (cleaned)
├── downloads/        # Downloaded TikTok videos
├── models/           # Cached ML models
└── CLAUDE.md         # System documentation
```

### 3. Essential Files in Root
- `ml_analyzer_registry_complete.py` - Central analyzer registry
- `fix_ffmpeg_env.sh` - FFmpeg environment fix script
- `tiktok_analyzer_workflow.py` - Main workflow script
- `system_config.json` - System configuration
- `cookies.txt` - TikTok download cookies
- `*.pt` - YOLO model weights

### 4. Active Services
- Stable Production API on port 8003
- 29 ML analyzers (6 disabled)
- GPU workload optimization with staged processing
- Frame caching system for efficiency

### 5. Key Commands
```bash
# Start system
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh
python3 api/stable_production_api.py &

# Analyze video
curl -X POST "http://localhost:8003/analyze" \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/video.mp4"}'

# Check health
curl http://localhost:8003/health
```

## Archived Components
The following components have been archived for reference but are not needed for production:

### Test Files (moved to `_archive_tests_20250704/`)
- All test scripts and validation files
- Test result JSONs
- Performance test scripts

### Utility Scripts (moved to `_archive_utils_20250704/`)
- `analysis_storage_complete.py` - Alternative storage system
- `batch_processor.py` - Batch processing utility
- `gpu_monitor.py` - GPU monitoring tool
- `monitoring_dashboard.py` - Web dashboard
- `parallel_analyzer_processor.py` - Alternative parallel processor
- `unified_production_system.py` - Alternative unified system
- Various other utility scripts

### Old Logs (moved to `logs/archive/test_logs_20250704/`)
- Development and test logs
- Old dated API logs
- Performance test logs

## Production Notes
1. The system uses `stable_production_api.py` as the main production API
2. Always run `source fix_ffmpeg_env.sh` before starting services
3. Results are saved to `/results/` directory
4. The system achieves <3x realtime processing with 85-95% GPU utilization
5. 29 analyzers are currently active, 6 are disabled

## System Health
- GPU: Quadro RTX 8000 with 44.5GB memory
- Active analyzers: 29 of 35 (83% active)
- Target processing: <3x realtime
- Output: 2-3MB JSON with comprehensive analysis

The production system is clean, organized, and ready for continued operation.