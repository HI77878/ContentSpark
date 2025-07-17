# Clean Server GPU Groups Configuration - MVP Version
import torch

# Analyzer timings based on actual measurements
ANALYZER_TIMINGS = {
    'qwen2_vl_temporal': 60.0,
    'object_detection': 15.0,
    'background_segmentation': 18.0,
    'text_overlay': 25.0,
    'camera_analysis': 18.0,
    'visual_effects': 35.0,
    'color_analysis': 16.4,
    'speech_rate': 10.0,
    'composition_analysis': 13.6,
    'content_quality': 11.7,
    'eye_tracking': 10.4,
    'scene_segmentation': 10.6,
    'cut_analysis': 4.1,
    'age_estimation': 8.0,
    'face_emotion': 25.0,
    'body_pose': 20.0,
    'product_detection': 50.4,
    'speech_transcription': 4.5,
    'temporal_flow': 2.1,
    'speech_emotion': 1.6,
    'audio_environment': 0.5,
    'audio_analysis': 0.2,
    'speech_flow': 1.6,
    'cross_analyzer_intelligence': 2.0,
}

# Clean Server GPU Groups - 4 Stages for 1-Second Segments
GPU_ANALYZER_GROUPS = {
    # Stage 1: Heavy GPU (runs alone)
    'stage1_gpu_heavy': [
        'qwen2_vl_temporal',
    ],
    # Stage 2: Medium GPU models
    'stage2_gpu_medium': [
        'object_detection',
        'background_segmentation',
        'text_overlay',
        'visual_effects',
        'product_detection',
        'face_emotion',
    ],
    # Stage 3: Light GPU models
    'stage3_gpu_light': [
        'camera_analysis',
        'color_analysis',
        'body_pose',
        'age_estimation',
        'content_quality',
        'eye_tracking',
        'composition_analysis',
    ],
    # Stage 4: Temporal and CPU analysis
    'stage4_gpu_fast': [
        'cut_analysis',
        'scene_segmentation',
        'speech_transcription',
        'audio_analysis',
        'audio_environment',
        'speech_emotion',
        'speech_rate',
        'speech_flow',
        'temporal_flow',
        'cross_analyzer_intelligence',
    ]
}

# Disabled analyzers (not in MVP)
DISABLED_ANALYZERS = [
    # Legacy analyzers not in clean server
    'face_detection',
    'emotion_detection', 
    'body_language',
    'hand_gesture',
    'gesture_recognition',
    'facial_details',
    'scene_description',
    'depth_estimation',
    'temporal_consistency',
    'audio_visual_sync',
    'trend_analysis',
    'vid2seq',
    'blip2_video_analyzer',
    'auroracap_analyzer',
    'video_llava',
    'tarsier_video_description',
    'streaming_dense_captioning',
    'comment_cta_detection',
    'qwen2_vl_optimized',
    'sound_effects',
]

# GPU memory configuration for RTX 8000 (45GB VRAM)
GPU_MEMORY_CONFIG = {
    'cleanup_after_stage': False,  # Keep models in memory for faster processing
    'enable_model_preloading': False,  # DISABLED: Prevent CUDA OOM
    'prefetch_frames': False,  # DISABLED: Save memory
    'max_concurrent': {
        'stage1_gpu_heavy': 1,     # Qwen2-VL runs alone
        'stage2_gpu_medium': 6,    # Medium GPU models
        'stage3_gpu_light': 7,     # Light GPU models
        'stage4_gpu_fast': 10,     # Fast models
    },
    'memory_allocation': {  # Explicit memory allocation
        'qwen2_vl': 16000,  # 16GB RESERVED for Qwen2-VL on Worker 0
        'whisper': 3000,    # 3GB for Whisper
        'yolo': 600,        # 600MB for YOLO
        'default': 500      # 500MB default
    },
    'batch_sizes': {
        'qwen2_vl_temporal': 1,      # Single frame for stability
        'object_detection': 32,      # Good balance for YOLO
        'visual_effects': 8,         # Moderate
        'camera_analysis': 16,       # Moderate
        'text_overlay': 24,          # OCR can handle moderate batches
        'background_segmentation': 8,   # Segmentation needs moderate batches
        'composition_analysis': 32,  # Light model can handle more
        'color_analysis': 32,        # Light processing
        'scene_segmentation': 32,    # Light processing
        'cut_analysis': 32,          # Very light
        'age_estimation': 12,        # Moderate
        'face_emotion': 4,           # Heavy model needs small batches
        'body_pose': 16,             # YOLO pose moderate
        'eye_tracking': 16,          # MediaPipe moderate
        'content_quality': 24,       # Moderate
        'product_detection': 16,     # Moderate
        'default': 12                # Safe default
    },
    'memory_threshold': 0.95,    # Increased from 0.85 - more aggressive
    'enable_amp': True,
    'gpu_memory_fraction': 0.95  # Increased from 0.9 - use more GPU memory
}

def cleanup_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_frame_interval(analyzer_name):
    """Get frame interval for analyzer - Optimized for 1-second segments"""
    # Return frame interval based on analyzer type (targeting 1-second segments)
    if analyzer_name == 'qwen2_vl_temporal':
        return 30  # Process at 1 FPS for second-by-second descriptions
    elif analyzer_name in ['face_emotion', 'visual_effects', 'product_detection']:
        return 30  # 1 second for heavy analyzers
    elif analyzer_name in ['object_detection', 'text_overlay', 'background_segmentation']:
        return 15  # 0.5 seconds for medium analyzers
    elif analyzer_name in ['camera_analysis', 'body_pose', 'age_estimation']:
        return 15  # 0.5 seconds for moderate analyzers
    else:
        return 10  # 0.33 seconds for light analyzers