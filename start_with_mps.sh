#!/bin/bash

# Script to start TikTok Analyzer with CUDA MPS for better GPU sharing

echo "🚀 Starting TikTok Analyzer with GPU Optimization..."

# Fix FFmpeg environment
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh

# Setup CUDA MPS directories
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY

# Stop any existing MPS daemon
echo quit | nvidia-cuda-mps-control 2>/dev/null || true

# Start CUDA MPS daemon
echo "Starting CUDA MPS daemon for better GPU sharing..."
nvidia-cuda-mps-control -d

# Set GPU memory fraction for better multi-process usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.7

# Enable CUDA optimization flags
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="7.5"  # For Quadro RTX 8000

# Start the optimized API
echo "Starting Stable Production API with multiprocess parallelization..."
python3 api/stable_production_api_multiprocess.py