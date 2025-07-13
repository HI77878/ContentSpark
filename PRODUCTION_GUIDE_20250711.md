# TikTok Production System - Official Guide
**Version:** 2.0
**Date:** 2025-07-11
**Status:** Production Ready (75% operational, target 90%+)

## 🚨 CRITICAL: Read This First!

### The #1 Rule
**ALWAYS USE:** `api/stable_production_api_multiprocess.py`
**NEVER USE:** `api/production_api_simple.py` (causes 51% failure rate!)

### The #2 Rule
**ALWAYS RUN:** `source fix_ffmpeg_env.sh` before starting ANY service

## Quick Start

```bash
# 1. Fix environment (MANDATORY)
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh

# 2. Start production API
python3 api/stable_production_api_multiprocess.py &

# 3. Verify health
curl http://localhost:8003/health

# 4. Analyze video
curl -X POST "http://localhost:8003/analyze" \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/video.mp4"}'
```

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│           stable_production_api_multiprocess.py          │
│                    (Port 8003)                          │
├─────────────────────────────────────────────────────────┤
│                  4 GPU Worker Processes                  │
│  Worker 0: Qwen2-VL (Heavy)                            │
│  Worker 1: Detection (Object, Text, Cuts)              │
│  Worker 2: Visual (Background, Camera, Effects)        │
│  Worker 3: Mixed (8 analyzers)                         │
├─────────────────────────────────────────────────────────┤
│              8 CPU Workers (Audio/Metadata)             │
└─────────────────────────────────────────────────────────┘
```

### Analyzer Distribution

**Total:** 28 registered analyzers
**Active:** 20 in production
**Success Rate:** 75% (target: >90%)

| Worker | Analyzers | Type |
|--------|-----------|------|
| GPU 0 | qwen2_vl_temporal | Video understanding |
| GPU 1 | object_detection, text_overlay, cut_analysis | Detection |
| GPU 2 | background_segmentation, camera_analysis, visual_effects | Visual |
| GPU 3 | scene_segmentation, color_analysis, speech_rate, etc. | Mixed |
| CPU | speech_transcription, audio_analysis, etc. | Audio |

## Production Workflows

### 1. Single Video Analysis

```python
import requests

# Analyze local video
response = requests.post(
    "http://localhost:8003/analyze",
    json={"video_path": "/home/user/tiktok_production/test_video.mp4"}
)

result = response.json()
print(f"Success: {result['successful_analyzers']}/{result['total_analyzers']}")
print(f"Results: {result['results_file']}")
```

### 2. TikTok Video Download & Analysis

```python
from mass_processing.tiktok_downloader import TikTokDownloader

# Download TikTok video
downloader = TikTokDownloader()
video_info = downloader.download_video('https://www.tiktok.com/@user/video/123')

# Analyze downloaded video
if video_info and 'video_path' in video_info:
    response = requests.post(
        "http://localhost:8003/analyze",
        json={"video_path": video_info['video_path']}
    )
```

### 3. Batch Processing

```python
from mass_processing.bulk_processor import BulkProcessor

processor = BulkProcessor()
results = processor.process_directory("/home/user/videos/")
```

## Monitoring & Debugging

### GPU Monitoring
```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Detailed metrics
nvidia-smi dmon -i 0 -s pucm -d 1
```

### API Logs
```bash
# Main API log
tail -f logs/stable_multiprocess_api.log

# Check for errors
grep ERROR logs/stable_multiprocess_api.log | tail -20
```

### Health Checks
```bash
# Full health status
curl http://localhost:8003/health | jq

# Quick check
curl -s http://localhost:8003/health | jq '.status'
```

## Performance Metrics

### Current Performance
- **Processing Speed:** 3.54x realtime
- **Success Rate:** 75% (15/20 analyzers)
- **GPU Memory:** ~16GB used
- **Target:** <3x realtime, >90% success

### Optimization Tips
1. Ensure GPU memory stays under 85%
2. Use test video for quick validation
3. Monitor worker distribution
4. Check for analyzer errors in logs

## Troubleshooting

### Problem: Low Success Rate (<70%)
**Cause:** Wrong API running
**Fix:** 
```bash
pkill -f "production_api"
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py &
```

### Problem: FFmpeg Assertion Error
**Error:** `Assertion fctx->async_lock failed`
**Fix:** Always run `source fix_ffmpeg_env.sh`

### Problem: CUDA Re-initialization Error
**Cause:** Not using spawn method
**Fix:** Use stable_production_api_multiprocess.py (has spawn setup)

### Problem: Analyzer Not Working
**Check:**
1. Is it in DISABLED_ANALYZERS?
2. GPU memory available?
3. Dependencies installed?
4. Check specific error in logs

## API Endpoints

### POST /analyze
Analyze a video file
```json
{
  "video_path": "/absolute/path/to/video.mp4"
}
```

### GET /health
System health status
```json
{
  "status": "healthy",
  "active_analyzers": 20,
  "gpu": {
    "gpu_name": "Quadro RTX 8000",
    "memory_used_gb": 15.7,
    "memory_total_gb": 44.5
  }
}
```

### GET /analyzers
List all available analyzers
```json
{
  "analyzers": ["qwen2_vl_temporal", "object_detection", ...],
  "total": 20
}
```

## Output Format

Results are saved to `/home/user/tiktok_production/results/` as JSON:

```json
{
  "metadata": {
    "video_path": "...",
    "processing_time_seconds": 35.4,
    "successful_analyzers": 15,
    "realtime_factor": 3.54
  },
  "analyzer_results": {
    "qwen2_vl_temporal": {
      "segments": [...]
    },
    "object_detection": {
      "segments": [...]
    }
  }
}
```

## Best Practices

### DO:
- ✅ Always use stable_production_api_multiprocess.py
- ✅ Run fix_ffmpeg_env.sh before starting
- ✅ Monitor GPU usage during processing
- ✅ Check health endpoint regularly
- ✅ Use absolute paths for videos

### DON'T:
- ❌ Use production_api_simple.py
- ❌ Skip environment setup
- ❌ Process multiple videos simultaneously
- ❌ Ignore error logs
- ❌ Use relative paths

## Maintenance

### Daily Tasks
1. Check system health
2. Review error logs
3. Monitor GPU memory
4. Verify analyzer success rate

### Weekly Tasks
1. Clean old result files
2. Update analyzer configurations
3. Test disabled analyzers
4. Performance optimization

## Emergency Procedures

### System Down
```bash
# 1. Kill all Python processes
pkill -f python

# 2. Clear GPU memory
nvidia-smi --gpu-reset

# 3. Restart with fix
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py &
```

### Rollback Plan
```bash
# Revert to last known good state
git checkout stable-production
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py &
```

## Contact & Support

- **Documentation:** This file + CLAUDE.md
- **Logs:** `/home/user/tiktok_production/logs/`
- **Results:** `/home/user/tiktok_production/results/`
- **Test Video:** `/home/user/tiktok_production/test_video.mp4`

## Version History

- **v2.0** (2025-07-11): Fixed multiprocess API, 75% success rate
- **v1.0** (2025-07-08): Initial release, 100% success rate
- **Issue** (2025-07-10): Degraded to 51% due to wrong API

---

**Remember:** The difference between 51% and 75% success is using the RIGHT API!