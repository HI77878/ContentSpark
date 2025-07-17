#!/usr/bin/env python3
"""
Enhanced GPU Optimization Configuration
Optimiert für Quadro RTX 8000 mit 44.5GB Memory
"""

import torch
import os

# GPU Memory Management
CUDA_MEMORY_FRACTION = 0.90  # Use 90% of GPU memory
ENABLE_TF32 = True          # Enable TF32 for faster computation
ENABLE_CUDNN_BENCHMARK = True  # Auto-tune for best performance
ENABLE_MEMORY_PINNING = True   # Pin memory for faster transfers

# Batch Sizes per Analyzer (optimized for 44.5GB GPU)
OPTIMIZED_BATCH_SIZES = {
    # Heavy models - smaller batches
    'vid2seq': 1,
    'vid2seq_blip2': 1,
    'background_segmentation': 8,
    
    # Medium models
    'object_detection': 32,
    'face_detection': 32,
    'body_pose': 48,
    'emotion_detection': 16,
    
    # Light models - larger batches
    'text_overlay': 64,
    'eye_tracking': 48,
    'hand_gesture': 32,
    'age_estimation': 64,
    'facial_details': 32,
    'gesture_recognition': 48,
    
    # Analysis models
    'content_quality': 128,
    'composition_analysis': 64,
    'visual_effects': 96,
    'motion_vectors': 16,  # Process frame pairs
    'product_detection': 48,
    
    # Scene analysis
    'cut_analysis': 64,
    'scene_segmentation': 32,
    'camera_analysis': 32,
    
    # CPU-based (no GPU batch size)
    'audio_analysis': 1,
    'audio_environment': 1,
    'speech_emotion': 1,
    'speech_rate': 1,
    'sound_effects': 1,
    'scene_description': 8,
    'temporal_flow': 16,
    'body_language': 48,
    'speech_transcription': 1
}

# Frame Extraction Optimization
OPTIMIZED_FRAME_INTERVALS = {
    # Critical analyzers - high sampling rate
    'vid2seq': 30,  # Every second
    'object_detection': 30,
    'face_detection': 30,
    'body_pose': 30,
    
    # Medium priority
    'background_segmentation': 60,  # Every 2 seconds
    'emotion_detection': 15,
    'text_overlay': 10,
    'visual_effects': 15,
    'motion_vectors': 6,  # 5fps for smooth motion
    
    # Low priority
    'age_estimation': 90,
    'facial_details': 60,
    'content_quality': 15,
    'composition_analysis': 30,
    'product_detection': 15,
    
    # Scene-based
    'cut_analysis': 10,
    'scene_segmentation': 30,
    'camera_analysis': 60,
    
    # Body tracking
    'hand_gesture': 10,
    'eye_tracking': 15,
    'gesture_recognition': 20,
    'body_language': 15,
    
    # Scene description
    'scene_description': 180,
    'temporal_flow': 120
}

# GPU Stream Configuration
NUM_CUDA_STREAMS = 8  # Number of parallel GPU streams
STREAM_PRIORITY_HIGH = -1
STREAM_PRIORITY_NORMAL = 0

# Memory Pool Settings
MEMORY_POOL_SIZE_MB = 8192  # 8GB memory pool
ENABLE_MEMORY_POOL = True

def initialize_gpu_optimizations():
    """Apply all GPU optimizations at startup"""
    if torch.cuda.is_available():
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(CUDA_MEMORY_FRACTION)
        
        # Enable TF32 if supported
        if ENABLE_TF32 and torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmark
        if ENABLE_CUDNN_BENCHMARK:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Set environment variables
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        print(f"✅ GPU Optimizations Applied:")
        print(f"   - Memory Fraction: {CUDA_MEMORY_FRACTION * 100}%")
        print(f"   - TF32: {'Enabled' if ENABLE_TF32 else 'Disabled'}")
        print(f"   - cuDNN Benchmark: {'Enabled' if ENABLE_CUDNN_BENCHMARK else 'Disabled'}")
        print(f"   - Memory Pinning: {'Enabled' if ENABLE_MEMORY_PINNING else 'Disabled'}")

def get_optimal_batch_size(analyzer_name: str, available_memory_gb: float = None) -> int:
    """Get optimal batch size based on available GPU memory"""
    base_size = OPTIMIZED_BATCH_SIZES.get(analyzer_name, 32)
    
    if available_memory_gb is not None:
        # Adjust based on available memory
        if available_memory_gb < 10:
            return max(1, base_size // 4)
        elif available_memory_gb < 20:
            return max(1, base_size // 2)
        elif available_memory_gb > 30:
            return int(base_size * 1.5)
    
    return base_size

def create_cuda_streams(num_streams: int = NUM_CUDA_STREAMS):
    """Create multiple CUDA streams for parallel processing"""
    if not torch.cuda.is_available():
        return []
    
    streams = []
    for i in range(num_streams):
        priority = STREAM_PRIORITY_HIGH if i < 2 else STREAM_PRIORITY_NORMAL
        stream = torch.cuda.Stream(priority=priority)
        streams.append(stream)
    
    return streams

class GPUMemoryManager:
    """Manages GPU memory allocation and cleanup"""
    
    def __init__(self):
        self.allocated_tensors = []
        self.memory_pool = None
        
        if ENABLE_MEMORY_POOL and torch.cuda.is_available():
            # Pre-allocate memory pool
            try:
                self.memory_pool = torch.cuda.FloatTensor(
                    MEMORY_POOL_SIZE_MB * 1024 * 1024 // 4
                )
                torch.cuda.empty_cache()
                print(f"✅ Memory pool allocated: {MEMORY_POOL_SIZE_MB}MB")
            except:
                print("⚠️ Could not allocate memory pool")
    
    def pin_memory(self, tensor):
        """Pin tensor to GPU memory for faster transfers"""
        if ENABLE_MEMORY_PINNING and not tensor.is_pinned() and tensor.device.type == 'cpu':
            return tensor.pin_memory()
        return tensor
    
    def cleanup(self):
        """Clean up GPU memory"""
        self.allocated_tensors.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

# Performance Monitoring
class GPUPerformanceMonitor:
    """Monitor GPU performance metrics"""
    
    def __init__(self):
        self.start_events = {}
        self.end_events = {}
    
    def start_timing(self, name: str):
        """Start timing a GPU operation"""
        if torch.cuda.is_available():
            self.start_events[name] = torch.cuda.Event(enable_timing=True)
            self.end_events[name] = torch.cuda.Event(enable_timing=True)
            self.start_events[name].record()
    
    def end_timing(self, name: str) -> float:
        """End timing and return elapsed milliseconds"""
        if name in self.start_events and torch.cuda.is_available():
            self.end_events[name].record()
            torch.cuda.synchronize()
            elapsed = self.start_events[name].elapsed_time(self.end_events[name])
            del self.start_events[name]
            del self.end_events[name]
            return elapsed
        return 0.0
    
    def get_memory_stats(self) -> dict:
        """Get current GPU memory statistics"""
        if not torch.cuda.is_available():
            return {}
        
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'free_gb': (torch.cuda.get_device_properties(0).total_memory - 
                       torch.cuda.memory_allocated()) / 1024**3
        }

# Initialize on import
if torch.cuda.is_available():
    initialize_gpu_optimizations()