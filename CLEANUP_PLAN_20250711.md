# Cleanup & Optimization Plan
**Generated:** 2025-07-11 06:05:00
**Priority:** High - System at 75% functionality

## Phase 1: Immediate Fixes (TODAY)

### 1.1 Restart Production API with Fixes
```bash
# Kill current API
pkill -f "stable_production_api_multiprocess.py"

# Apply environment fix and restart
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py &
```

### 1.2 Test Critical Analyzers
```bash
# Test face_emotion fix
curl -X POST "http://localhost:8003/analyze" \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/home/user/tiktok_production/test_video.mp4"}'
```

### 1.3 Update Documentation
- Update CLAUDE.md to specify stable_production_api_multiprocess.py as ONLY production API
- Add warning about production_api_simple.py

## Phase 2: API Cleanup

### 2.1 Archive Obsolete APIs
```bash
mkdir -p archived_apis/
mv api/production_api_simple.py archived_apis/
mv api/production_api_v2.py archived_apis/
mv api/stable_production_api.py archived_apis/  # Keep only multiprocess version
mv api/stable_production_api_optimized.py archived_apis/
mv api/stable_production_api_preload.py archived_apis/
mv api/stable_production_api_multiprocess_backup.py archived_apis/
```

### 2.2 Move Experimental APIs
```bash
mkdir -p experiments/ray_apis/
mv api/ray_*.py experiments/ray_apis/
```

### 2.3 Create API Documentation
```bash
cat > api/README.md << 'EOF'
# Production API

## IMPORTANT: Only use stable_production_api_multiprocess.py for production!

### Why?
- Proper multiprocessing with spawn method for CUDA
- GPU optimization and load balancing
- Output normalization
- 75%+ success rate

### Never use:
- production_api_simple.py - No multiprocessing, causes 51% failure rate
- Other variants - Experimental or obsolete

### Starting the API:
```bash
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh  # CRITICAL!
python3 api/stable_production_api_multiprocess.py
```
EOF
```

## Phase 3: Analyzer Optimization

### 3.1 Test Disabled Analyzers
Create test script to check each disabled analyzer:

```python
# test_disabled_analyzers.py
DISABLED = ['face_detection', 'emotion_detection', 'body_language', 
            'hand_gesture', 'gesture_recognition', 'facial_details', 
            'scene_description', 'depth_estimation', 'temporal_consistency', 
            'audio_visual_sync', 'trend_analysis', 'vid2seq', 
            'blip2_video_analyzer', 'auroracap_analyzer', 'composition_analysis', 
            'video_llava', 'tarsier_video_description', 
            'streaming_dense_captioning', 'product_detection', 'eye_tracking']

for analyzer in DISABLED:
    # Test each one individually
    pass
```

### 3.2 Fix body_pose Analyzer
- Check why it's not in test results
- Verify GPU memory requirements
- Test with single video

### 3.3 Consolidate Duplicate Analyzers
- face_emotion vs emotion_detection vs facial_details
- gesture_recognition vs hand_gesture
- scene_segmentation vs scene_description

## Phase 4: Code Consolidation

### 4.1 GPU Executor Cleanup
```bash
mkdir -p utils/archived/
# Keep only: multiprocess_gpu_executor_registry.py
mv utils/multiprocess_gpu_executor_*.py utils/archived/
mv utils/archived/multiprocess_gpu_executor_registry.py utils/
```

### 4.2 Workflow Consolidation
- Keep: tiktok_analyzer_workflow_final.py
- Archive others

## Phase 5: Performance Optimization

### 5.1 Target: <3x Realtime
Current: 3.54x realtime

Optimizations:
1. Increase batch sizes where possible
2. Enable TensorRT for compatible models
3. Optimize frame sampling rates
4. Use FP16 for all models

### 5.2 Target: >90% Success Rate
Current: 75% (15/20 analyzers)

Actions:
1. Fix face_emotion and body_pose
2. Re-enable lightweight analyzers
3. Improve error handling

## Phase 6: Monitoring & Maintenance

### 6.1 Create Health Check Script
```bash
# health_check.sh
#!/bin/bash
echo "=== API Health ==="
curl -s http://localhost:8003/health | jq

echo -e "\n=== GPU Status ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

echo -e "\n=== Active Analyzers ==="
curl -s http://localhost:8003/health | jq '.active_analyzers'
```

### 6.2 Create Daily Test Script
```bash
# daily_test.sh
#!/bin/bash
python3 test_fixed_system.py
if [ $? -eq 0 ]; then
    echo "✅ System healthy"
else
    echo "❌ System degraded - check logs"
fi
```

## Execution Timeline

**Day 1 (TODAY):**
- [ ] Phase 1: Immediate fixes
- [ ] Phase 2: API cleanup
- [ ] Update CLAUDE.md

**Day 2:**
- [ ] Phase 3: Analyzer optimization
- [ ] Test disabled analyzers

**Day 3:**
- [ ] Phase 4: Code consolidation
- [ ] Phase 5: Performance optimization

**Ongoing:**
- [ ] Phase 6: Monitoring setup
- [ ] Daily health checks

## Success Metrics

1. **API Success Rate:** >90% (from 75%)
2. **Realtime Factor:** <3x (from 3.54x)
3. **Code Reduction:** Remove 7+ duplicate APIs
4. **Documentation:** 100% accurate
5. **Analyzer Coverage:** 25+ active (from 20)

## Risk Mitigation

1. **Backup before changes:** Create full backup
2. **Test incrementally:** One change at a time
3. **Monitor GPU memory:** Stay under 85%
4. **Keep rollback plan:** Archive, don't delete

## Final Notes

The system degraded because the wrong API was running. This cleanup plan will:
1. Prevent future confusion
2. Improve performance
3. Increase reliability
4. Simplify maintenance

Priority is restoring >90% functionality while maintaining <3x realtime performance.