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

<<<<<<< HEAD
# Memory Pool für weniger Fragmentierung - OPTIMIERT für Qwen2-VL
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.9'
export CUDA_LAUNCH_BLOCKING=1  # Enable for better OOM debugging

# Prevent memory fragmentation in multiprocessing
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
export CUDA_CACHE_DISABLE=0
export TORCH_CUDA_LAZY_LOADING=0

# Additional optimizations for Model Sharing
export NCCL_BUFF_SIZE='16777216'  # 16MB NCCL Buffer for better GPU communication
export TORCH_CUDA_MEMORY_FRACTION=0.9  # Use 90% of GPU memory
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Enable MPS fallback
=======
# Memory Pool für weniger Fragmentierung
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.9'
export CUDA_LAUNCH_BLOCKING=0
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc

# Optimale Threads für Xeon CPU
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

<<<<<<< HEAD
# Qwen2-VL Specific Optimizations
export TOKENIZERS_PARALLELISM=false  # Avoid multiprocessing conflicts
export TRANSFORMERS_CACHE="/tmp/transformers_cache"  # Use faster temp storage
export HF_HOME="/tmp/huggingface_cache"  # Cache models in temp

echo "✅ FFmpeg Environment fixes loaded (Qwen2-VL optimized)"
=======
echo "✅ FFmpeg Environment fixes loaded"
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
