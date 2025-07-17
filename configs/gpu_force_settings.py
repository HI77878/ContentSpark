"""
GPU FORCE SETTINGS - DIESE DATEI WIRD BEI JEDEM START GELADEN
Zwingt ALLE Analyzer GPU zu nutzen - KEIN CPU FALLBACK!
"""

import os
import torch

# CUDA UMGEBUNGSVARIABLEN - PERSISTENT
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async operations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# PyTorch GPU Settings
if torch.cuda.is_available():
    # Memory fraction - use 95% of GPU memory
    torch.cuda.set_per_process_memory_fraction(0.95)
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set default tensor type
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    print(f"✅ GPU FORCE ACTIVE: {torch.cuda.get_device_name(0)}")
else:
    raise RuntimeError("NO GPU FOUND - SYSTEM CANNOT START!")

# Batch Sizes für maximale GPU Auslastung
GPU_BATCH_SIZES = {
    'blip2_video_analyzer': 16,  # War vid2seq
    'object_detection': 32,
    'face_detection': 32,
    'body_pose': 16,
    'emotion_detection': 32,
    'text_overlay': 8,
    'background_segmentation': 8,
    'scene_description': 16,
    'composition_analysis': 16,
}

# Frame Processing Settings
MAX_FRAMES_PER_BATCH = 32
USE_MIXED_PRECISION = True
FORCE_GPU_DECODE = True