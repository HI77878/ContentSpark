#!/usr/bin/env python3
"""
Registry Loader with Error Handling
Loads only the analyzers that actually exist
"""

import logging
import sys
import os

logger = logging.getLogger(__name__)

# Add paths
sys.path.append('/home/user/tiktok_production')
sys.path.append('/home/user/tiktok_production/analyzers')

def load_analyzer_safe(module_name, class_name):
    """Safely load an analyzer, return None if it fails"""
    try:
        module = __import__(f'analyzers.{module_name}', fromlist=[class_name])
        return getattr(module, class_name)
    except Exception as e:
        logger.warning(f"Could not load {module_name}.{class_name}: {e}")
        return None

# Build the analyzer registry dynamically
ML_ANALYZERS = {}

# Define which analyzers to load
ANALYZER_MAP = {
    # Core analyzers
    'background_segmentation': ('background_segmentation_light', 'GPUBatchBackgroundSegmentationLight'),
    'speech_transcription': ('speech_transcription_ultimate', 'UltimateSpeechTranscription'),
    'sound_effects': ('sound_effects_detector', 'SoundEffectsDetector'),
    
    # Object and scene analyzers
    'object_detection': ('gpu_batch_object_detection_yolo', 'GPUBatchObjectDetectionYOLO'),
    'text_overlay': ('text_overlay_tiktok_fixed', 'TikTokTextOverlayAnalyzer'),
    'camera_analysis': ('camera_analysis_fixed', 'GPUBatchCameraAnalysisFixed'),
    'eye_tracking': ('gpu_batch_eye_tracking', 'GPUBatchEyeTracking'),
    'color_analysis': ('gpu_batch_color_analysis', 'GPUBatchColorAnalysis'),
    'composition_analysis': ('composition_analysis_light', 'GPUBatchCompositionAnalysisLight'),
    'age_estimation': ('age_gender_insightface', 'AgeGenderInsightFace'),
    'face_emotion': ('face_emotion_mediapipe', 'FaceEmotionMediaPipe'),
    'body_pose': ('body_pose_yolov8', 'BodyPoseYOLOv8'),
    'cut_analysis': ('cut_analysis_fixed', 'CutAnalysisFixedAnalyzer'),
    'scene_segmentation': ('scene_segmentation_fixed', 'SceneSegmentationFixedAnalyzer'),
    'content_quality': ('gpu_batch_content_quality_fixed', 'GPUBatchContentQualityFixed'),
    'visual_effects': ('visual_effects_cv_based', 'VisualEffectsCVBased'),
    'product_detection': ('gpu_batch_product_detection_light', 'GPUBatchProductDetectionLight'),
    
    # Audio analyzers
    'audio_analysis': ('audio_analysis_ultimate', 'UltimateAudioAnalysis'),
    'audio_environment': ('audio_environment_enhanced', 'AudioEnvironmentEnhanced'),
    'speech_emotion': ('gpu_batch_speech_emotion', 'GPUBatchSpeechEmotion'),
    'speech_rate': ('gpu_batch_speech_rate_enhanced', 'GPUBatchSpeechRateEnhanced'),
    'speech_flow': ('gpu_batch_speech_flow', 'GPUBatchSpeechFlow'),
    'comment_cta_detection': ('gpu_batch_comment_cta_detection', 'GPUBatchCommentCTADetection'),
    
    # CPU analyzers
    'temporal_flow': ('narrative_analysis_wrapper', 'NarrativeAnalysisWrapper'),
    
    # Cross-analyzer intelligence - MUST RUN LAST
    'cross_analyzer_intelligence': ('cross_analyzer_intelligence', 'CrossAnalyzerIntelligence'),
    
    # Video understanding analyzers
    'streaming_dense_captioning': ('streaming_dense_captioning_fixed', 'StreamingDenseCaptioningFixed'),
    # 'qwen2_vl_optimized': ('qwen2_vl_optimized_analyzer', 'Qwen2VLOptimizedAnalyzer'),  # REMOVED from Clean Server MVP
    'qwen2_vl_temporal': ('qwen2_vl_video_analyzer', 'Qwen2VLVideoAnalyzer'),
    'qwen2_vl_ultra': ('qwen2_vl_ultra_detailed', 'Qwen2VLUltraDetailedAnalyzer'),
}

# Load all analyzers
for name, (module, class_name) in ANALYZER_MAP.items():
    analyzer_class = load_analyzer_safe(module, class_name)
    if analyzer_class:
        ML_ANALYZERS[name] = analyzer_class
        
logger.info(f"Successfully loaded {len(ML_ANALYZERS)} analyzers")

# For compatibility
def get_ml_analyzer(name, turbo_mode=False):
    """Get ML analyzer instance"""
    if name not in ML_ANALYZERS:
        raise ValueError(f"Unknown analyzer: {name}")
    
    try:
        analyzer = ML_ANALYZERS[name]()
        return analyzer
    except Exception as e:
        logger.error(f"Failed to instantiate {name}: {e}")
        raise