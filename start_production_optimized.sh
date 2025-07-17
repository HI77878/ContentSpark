#!/bin/bash
# start_production_optimized.sh

echo "Starting TikTok Analyzer Production System with Optimizations..."

# Environment Setup
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh

# GPU Optimization
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.5"
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable CUDA MPS for better GPU sharing (optional)
# export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
# export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
# nvidia-cuda-mps-control -d

# Kill any existing API
echo "Stopping existing API instances..."
pkill -f stable_production_api || true
sleep 2

# Start API with optimizations
echo "Starting API with Model Pre-Loading..."
nohup python3 -u api/stable_production_api_preload.py > logs/production_$(date +%Y%m%d_%H%M%S).log 2>&1 &
API_PID=$!

echo "API PID: $API_PID"
echo "Waiting for models to load (90 seconds)..."
sleep 90

# Health Check
if curl -s http://localhost:8003/health > /dev/null; then
    echo "✅ System is READY!"
    echo "API running on: http://localhost:8003"
    echo "Logs: logs/production_$(date +%Y%m%d_*)*.log"
    
    # Show health status
    echo ""
    echo "System Status:"
    curl -s http://localhost:8003/health | python3 -m json.tool
else
    echo "❌ System failed to start. Check logs."
    tail -50 logs/production_$(date +%Y%m%d_*)*.log
fi