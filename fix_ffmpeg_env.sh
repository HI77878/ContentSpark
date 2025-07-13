#!/bin/bash
# FFmpeg pthread fix - MUSS vor jedem Start geladen werden
export OPENCV_FFMPEG_CAPTURE_OPTIONS="protocol_whitelist=file,http,https,tcp,tls"
export OPENCV_VIDEOIO_PRIORITY_BACKEND=4  # cv2.CAP_GSTREAMER
export OPENCV_FFMPEG_MULTITHREADED=0
export OPENCV_FFMPEG_DEBUG=1

# GPU Optimierungen
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TORCH_CUDA_ARCH_LIST="7.5"  # Für RTX 8000
export PYTHONPATH="/home/user/tiktok_production:$PYTHONPATH"

# GPU Optimization Settings for MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

# Memory Pool für weniger Fragmentierung
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.9'
export CUDA_LAUNCH_BLOCKING=0

# Optimale Threads für Xeon CPU
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "✅ FFmpeg Environment fixes loaded"