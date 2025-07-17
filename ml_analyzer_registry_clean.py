#!/usr/bin/env python3
"""CLEAN ML Analyzer Registry - Only Active Analyzers"""

import logging
logger = logging.getLogger(__name__)

# Import ONLY the active analyzers that exist
try:
    # Core analyzers
    from analyzers.background_segmentation_light import GPUBatchBackgroundSegmentationLight
    from analyzers.speech_transcription_ultimate import UltimateSpeechTranscription as EnhancedSpeechTranscription
    from analyzers.sound_effects_detector import SoundEffectsDetector
    
    # Object and scene analyzers
    from analyzers.gpu_batch_object_detection_yolo import GPUBatchObjectDetectionYOLO
    from analyzers.text_overlay_tiktok_fixed import TikTokTextOverlayAnalyzer
    from analyzers.camera_analysis_fixed import GPUBatchCameraAnalysisFixed
    from analyzers.gpu_batch_eye_tracking import GPUBatchEyeTracking
    from analyzers.gpu_batch_color_analysis import GPUBatchColorAnalysis
    from analyzers.composition_analysis_light import GPUBatchCompositionAnalysisLight
    from analyzers.age_estimation_light import GPUBatchAgeEstimationLight
    from analyzers.cut_analysis_fixed import CutAnalysisFixedAnalyzer
    from analyzers.scene_segmentation_fixed import SceneSegmentationFixedAnalyzer
    from analyzers.gpu_batch_content_quality_fixed import GPUBatchContentQualityFixed
    from analyzers.visual_effects_light_fixed import VisualEffectsLight
    from analyzers.gpu_batch_product_detection_light import GPUBatchProductDetectionLight
    
    # Audio analyzers
    from analyzers.audio_analysis_ultimate import UltimateAudioAnalysis as GPUBatchAudioAnalysisEnhanced
    from analyzers.audio_environment_enhanced import AudioEnvironmentEnhanced
    from analyzers.gpu_batch_speech_emotion import GPUBatchSpeechEmotion
    from analyzers.gpu_batch_speech_rate_enhanced import GPUBatchSpeechRateEnhanced as GPUBatchSpeechRate
    # from analyzers.gpu_batch_sound_effects_enhanced import GPUBatchSoundEffectsEnhanced  # Not in active list
    from analyzers.gpu_batch_speech_flow import GPUBatchSpeechFlow
    from analyzers.gpu_batch_comment_cta_detection import GPUBatchCommentCTADetection
    
    # CPU analyzers
    from analyzers.cpu_batch_temporal_flow import CPUBatchTemporalFlow
    
    # Video understanding analyzers
    from analyzers.streaming_dense_captioning_fixed import StreamingDenseCaptioningFixed as StreamingDenseCaptioningAnalyzer
    from analyzers.qwen2_vl_optimized_analyzer import Qwen2VLOptimizedAnalyzer
    from analyzers.qwen2_vl_temporal_fixed import Qwen2VLTemporalFixed

except ImportError as e:
    logger.error(f"Failed to import analyzers: {e}")
    raise

# ACTIVE ANALYZER MAPPING - 22 analyzers for full reconstruction
ML_ANALYZERS = {
    # Stage 1: Scene Understanding (Heavy GPU)
    'qwen2_vl_temporal': Qwen2VLTemporalFixed,  # Primary video understanding
    'qwen2_vl_optimized': Qwen2VLOptimizedAnalyzer,  # Memory-efficient version
    'streaming_dense_captioning': StreamingDenseCaptioningAnalyzer,  # Temporal descriptions
    'object_detection': GPUBatchObjectDetectionYOLO,  # YOLOv8x
    'background_segmentation': GPUBatchBackgroundSegmentationLight,  # SegFormer
    
    # Stage 2: Person Analysis (Medium GPU)
    'eye_tracking': GPUBatchEyeTracking,  # MediaPipe iris
    
    # Stage 3: Content Analysis (Light GPU)
    'text_overlay': TikTokTextOverlayAnalyzer,  # EasyOCR
    'speech_transcription': EnhancedSpeechTranscription,  # Whisper
    'color_analysis': GPUBatchColorAnalysis,  # Color extraction
    'composition_analysis': GPUBatchCompositionAnalysisLight,  # CLIP
    'visual_effects': VisualEffectsLight,  # Effect detection
    
    # Stage 4: Detail Analysis (Light GPU)
    'age_estimation': GPUBatchAgeEstimationLight,  # Age detection
    'content_quality': GPUBatchContentQualityFixed,  # Quality metrics
    'product_detection': GPUBatchProductDetectionLight,  # Product/brand detection
    
    # Stage 5: Temporal Analysis (GPU)
    'cut_analysis': CutAnalysisFixedAnalyzer,  # Scene cuts
    'scene_segmentation': SceneSegmentationFixedAnalyzer,  # Scene boundaries
    'camera_analysis': GPUBatchCameraAnalysisFixed,  # Camera movement
    
    # CPU Parallel - Audio and metadata
    'audio_analysis': GPUBatchAudioAnalysisEnhanced,  # Librosa
    'audio_environment': AudioEnvironmentEnhanced,  # Environment detection
    'speech_emotion': GPUBatchSpeechEmotion,  # Emotion from speech
    'speech_rate': GPUBatchSpeechRate,  # Speech metrics
    'sound_effects': SoundEffectsDetector,  # Sound effect detection
    'speech_flow': GPUBatchSpeechFlow,  # Speech emphasis and pauses
    'comment_cta_detection': GPUBatchCommentCTADetection,  # Comment references
    'temporal_flow': CPUBatchTemporalFlow,  # Narrative flow
}

def get_ml_analyzer(name, turbo_mode=False):
    """Get ML analyzer instance"""
    if name not in ML_ANALYZERS:
        raise ValueError(f"Unknown analyzer: {name}")
    
    try:
        analyzer = ML_ANALYZERS[name]()
        logger.info(f"Loaded analyzer: {name}")
        return analyzer
        
    except Exception as e:
        logger.error(f"Failed to load {name}: {e}")
        raise

def get_active_analyzers():
    """Get list of all active analyzers"""
    return list(ML_ANALYZERS.keys())

def get_analyzer_count():
    """Get total number of active analyzers"""
    return len(ML_ANALYZERS)

# For backward compatibility
ANALYZER_CAPABILITIES = {
    'qwen2_vl_temporal': {
        'description': 'Temporal video understanding with Qwen2-VL-7B',
        'reconstruction': 'scene_understanding',
        'fps': 1.0
    },
    'object_detection': {
        'description': 'Object detection with YOLOv8x',
        'reconstruction': 'object_presence',
        'fps': 3.0
    },
    'background_segmentation': {
        'description': 'Semantic segmentation of scene elements',
        'reconstruction': 'spatial_layout',
        'fps': 1.0
    }
}

def get_analyzer_info(name):
    """Get analyzer capability information"""
    return ANALYZER_CAPABILITIES.get(name, {})