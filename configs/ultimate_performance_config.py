#!/usr/bin/env python3
"""
Ultimate Performance Configuration for <3x Realtime
Optimiert für maximale GPU-Auslastung und Parallelität
"""

# GPU Process Configuration
GPU_PROCESS_CONFIG = {
    'num_processes': 3,  # 3 parallel GPU processes
    'gpu_memory_fraction': 0.30,  # 30% per process = 90% total
    'batch_priority': {
        # Heavy models get their own process
        'process_0': ['video_llava', 'object_detection', 'product_detection'],
        'process_1': ['speech_transcription', 'text_overlay', 'background_segmentation'],
        'process_2': ['camera_analysis', 'visual_effects', 'scene_segmentation', 'others']
    }
}

# Optimized batch sizes for Ultimate analyzers
ULTIMATE_BATCH_SIZES = {
    # Heavy models - smaller batches
    'video_llava': 2,
    'object_detection': 4,
    'product_detection': 4,
    'background_segmentation': 4,
    
    # Medium models
    'text_overlay': 8,
    'camera_analysis': 8,
    'visual_effects': 8,
    'scene_segmentation': 8,
    'cut_analysis': 8,
    
    # Light models - larger batches
    'composition_analysis': 16,
    'color_analysis': 16,
    'content_quality': 16,
    'eye_tracking': 12,
    'age_estimation': 16,
    
    # Audio (CPU-based)
    'speech_transcription': 1,
    'speech_emotion': 1,
    'speech_rate': 1,
    'audio_analysis': 1,
    'audio_environment': 1,
    'sound_effects': 1,
    'temporal_flow': 1
}

# Optimized frame sampling for Ultimate analyzers
ULTIMATE_FRAME_SAMPLING = {
    # Heavy models - less frequent sampling
    'video_llava': {
        'interval': 120,  # Every 4 seconds
        'max_frames': 10,
        'min_frames': 5
    },
    'object_detection': {
        'interval': 60,   # Every 2 seconds
        'max_frames': 15,
        'min_frames': 8
    },
    'product_detection': {
        'interval': 90,   # Every 3 seconds
        'max_frames': 10,
        'min_frames': 5
    },
    
    # Medium models
    'text_overlay': {
        'interval': 30,   # Every 1 second
        'max_frames': 30,
        'min_frames': 15
    },
    'camera_analysis': {
        'interval': 30,
        'max_frames': 20,
        'min_frames': 10
    },
    'visual_effects': {
        'interval': 45,
        'max_frames': 15,
        'min_frames': 8
    },
    
    # Light models - more frequent
    'composition_analysis': {
        'interval': 30,
        'max_frames': 20,
        'min_frames': 10
    },
    'color_analysis': {
        'interval': 60,
        'max_frames': 15,
        'min_frames': 8
    },
    'eye_tracking': {
        'interval': 15,   # Every 0.5 seconds for precision
        'max_frames': 40,
        'min_frames': 20
    }
}

# Timeout configuration
ANALYZER_TIMEOUTS = {
    'video_llava': 60,
    'object_detection': 60,
    'speech_transcription': 45,
    'text_overlay': 60,
    'product_detection': 45,
    'background_segmentation': 60,
    'temporal_flow': 120,
    'default': 30
}

# Memory management
MEMORY_CONFIG = {
    'cleanup_threshold': 0.85,  # Clean GPU memory at 85% usage
    'batch_reduction_factor': 0.5,  # Reduce batch by 50% if OOM
    'max_retries': 2
}