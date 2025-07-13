# Final GPU Groups Configuration for Production System
import torch

# Analyzer timings based on actual measurements + optimizations applied
ANALYZER_TIMINGS = {
    'qwen2_vl_temporal': 60.0,  # OPTIMIZED: From 110s with Flash Attention + optimizations
    'qwen2_vl_optimized': 3264.0,  # Memory-efficient but SLOW (48s per frame - NOT RECOMMENDED)
    'qwen2_vl_video': 40.0,  # Original Qwen2-VL (disabled due to OOM)
    'tarsier_video_description': 45.0,  # Tarsier 7B model (if enabled)
    'streaming_dense_captioning': 15.0,  # OPTIMIZED: Lazy loading + lighter models
    'product_detection': 50.4,
    'object_detection': 15.0,  # OPTIMIZED: From 25s with YOLOv8l + FP16 + batch 64
    'object_detection_tensorrt': 10.0,  # NEW: TensorRT version - 5x faster
    'background_segmentation': 18.0,  # OPTIMIZED: From 41.2s with smaller resolution + bigger batches
    'text_overlay': 25.0,  # OPTIMIZED: From 37.1s with batch OCR + deduplication 
    'camera_analysis': 18.0,  # OPTIMIZED: From 36.1s with sparse optical flow + downsampling
    'visual_effects': 35.0,  # ML-based version needs more time
    'color_analysis': 16.4,
    'speech_rate': 10.0,  # OPTIMIZED: From 14.1s with VAD + parallel processing
    'composition_analysis': 13.6,
    'content_quality': 11.7,
    'eye_tracking': 10.4,
    'scene_segmentation': 10.6,
    'cut_analysis': 4.1,
    'age_estimation': 8.0,  # InsightFace needs more time
    'face_emotion': 25.0,  # DeepFace with emotion analysis
    'body_pose': 20.0,  # YOLOv8x-pose
    'sound_effects': 5.9,
    'speech_transcription': 4.5,
    'temporal_flow': 2.1,
    'speech_emotion': 1.6,
    'audio_environment': 0.5,
    'audio_analysis': 0.2,
    'speech_flow': 1.6,
    'comment_cta_detection': 1.0,
    'cross_analyzer_intelligence': 2.0,  # Correlation analysis
}

# GPU analyzer groups for parallel execution
GPU_ANALYZER_GROUPS = {
    # CRITICAL: Worker 0 EXCLUSIVELY for Qwen2-VL (needs 16GB alone!)
    'gpu_worker_0': [
        'qwen2_vl_temporal',  # ISOLATED - Needs 16GB GPU memory alone
    ],
    # Worker 1: Object detection and visual analysis
    'gpu_worker_1': [
        'object_detection',         # Medium (5.8s avg)
        'text_overlay',             # Medium (12.2s avg)
        'background_segmentation',  # Medium (4.9s avg)
        'camera_analysis',          # Light (4.2s avg)
    ],
    # Worker 2: Remaining GPU analyzers
    'gpu_worker_2': [
        'scene_segmentation',       # Light (10.6s)
        'color_analysis',           # Light (4.4s avg)
        'body_pose',                # Medium (10.2s avg)
        'age_estimation',           # Light (8s)
        'content_quality',          # Light (11.7s)
        'eye_tracking',             # Light (4.9s)
        'cut_analysis',             # Light (4.1s)
    ],
    # CPU parallel processing
    'cpu_parallel': [
        'speech_transcription',  # 7.1s avg
        'audio_analysis',        # 5.1s avg
        'audio_environment',     # 0.5s
        'speech_emotion',        # 1.6s
        'temporal_flow',         # 2.1s
        'speech_flow',           # 5.6s avg
        'cross_analyzer_intelligence',  # 2s - MUST run LAST
    ]
}

# Disabled analyzers (not in production)
DISABLED_ANALYZERS = [
    'face_detection',  # Old analyzer, replaced by face_emotion
    'emotion_detection',  # Old analyzer, replaced by face_emotion
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
    'auroracap_analyzer',  # Experimentell - durch Video-LLaVA ersetzt
    'composition_analysis',  # Deaktiviert - liefert keine Daten
    'video_llava',  # Deaktiviert - halluziniert und beschreibt nicht temporal
    'tarsier_video_description',  # Deaktiviert - spezielle Setup erforderlich
    'streaming_dense_captioning',  # Deaktiviert - nur Platzhalter, kein echtes Model
    'product_detection',  # Deaktiviert - Qwen2-VL erkennt Produkte besser (50.4s gespart)
# 'eye_tracking',  # Reaktiviert für exakt 20 Analyzer
    'comment_cta_detection',  # Deaktiviert - Keine sinnvollen Daten bei Tests
    'qwen2_vl_optimized',  # DEAKTIVIERT - Nur qwen2_vl_temporal verwenden
    'streaming_dense_captioning',  # DEAKTIVIERT - Verursacht Probleme  
    'product_detection',  # DEAKTIVIERT - Qwen2-VL macht das bereits
    'speech_rate',  # DEAKTIVIERT - 2 zu viel für exakt 20
    'sound_effects',  # DEAKTIVIERT - 2 zu viel für exakt 20
    'face_emotion',  # DEAKTIVIERT - Zu langsam (169s+), blockiert GPU
    'visual_effects',  # DEAKTIVIERT - Zu langsam (117s+), verursacht Timeouts
]

# GPU memory configuration for RTX 8000 (45GB VRAM)
GPU_MEMORY_CONFIG = {
    'cleanup_after_stage': False,  # Keep models in memory for faster processing
    'enable_model_preloading': False,  # DISABLED: Prevent CUDA OOM
    'prefetch_frames': False,  # DISABLED: Save memory
    'max_concurrent': {
        'gpu_worker_0': 1,  # Qwen2-VL runs alone
        'gpu_worker_1': 4,  # Object detection group
        'gpu_worker_2': 7,  # Light analyzers group
        'cpu': 16           # CPU analyzers
    },
    'memory_allocation': {  # Explicit memory allocation
        'qwen2_vl': 16000,  # 16GB RESERVED for Qwen2-VL on Worker 0
        'whisper': 3000,    # 3GB for Whisper
        'yolo': 600,        # 600MB for YOLO
        'default': 500      # 500MB default
    },
    'batch_sizes': {
        'qwen2_vl_temporal': 3,      # Moderate for stability
        'qwen2_vl_optimized': 1,     # Optimized Qwen2-VL processes single frames only
        'qwen2_vl_video': 1,         # Original Qwen2-VL processes videos frame-by-frame
        'tarsier_video_description': 1,  # Tarsier processes videos frame-by-frame
        'streaming_dense_captioning': 1,  # Streaming processes videos sequentially
        'product_detection': 16,     # Moderate
        'object_detection': 32,      # Good balance for YOLO
        'object_detection_tensorrt': 64,  # TensorRT can handle larger batches
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
    """Get frame interval for analyzer - OPTIMIZED FOR GPU USAGE"""
    # Return frame interval based on analyzer type
    if analyzer_name in ['qwen2_vl_temporal', 'qwen2_vl_optimized', 'qwen2_vl_video', 'tarsier_video_description']:
        return 30  # Process at 1 FPS for second-by-second descriptions
    elif analyzer_name == 'streaming_dense_captioning':
        return 2  # Process at 15 FPS with dense overlapping
    elif analyzer_name in ['face_emotion', 'visual_effects']:
        return 45  # 1.5 seconds for very heavy analyzers
    elif analyzer_name in ['object_detection', 'text_overlay']:
        return 20  # Reduced from 90/60 - process more frames
    elif analyzer_name in ['camera_analysis', 'background_segmentation']:
        return 25  # Reduced from 60 - process more frames
    elif analyzer_name in ['body_pose', 'age_estimation']:
        return 15  # Reduced - these can handle more
    else:
        return 15  # Reduced from 30 - more frames for better accuracy