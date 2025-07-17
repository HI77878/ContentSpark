#!/bin/bash
# TikTok Analyzer Production Starter
# Fixes FFmpeg environment and starts API

echo "🚀 Starting TikTok Analyzer Production System..."

# Change to production directory
cd /home/user/tiktok_production

# Source FFmpeg environment fixes
echo "🔧 Fixing FFmpeg environment..."
source fix_ffmpeg_env.sh

# Set GPU environment
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.5"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Start API
echo "🌐 Starting API on port 8000..."
python3 api/single_api.py