#!/usr/bin/env python3
"""CLEAN SERVER ML Analyzer Registry - MVP Version"""

import logging
logger = logging.getLogger(__name__)

# Import only the best variant of each analyzer
try:
    # Video Understanding
    from analyzers.qwen2_vl_temporal_analyzer import Qwen2VLTemporalAnalyzer
    
    # Object and Scene Analysis
    from analyzers.gpu_batch_object_detection_yolo import GPUBatchObjectDetectionYOLO
    from analyzers.background_segmentation_light import GPUBatchBackgroundSegmentationLight
    from analyzers.text_overlay_tiktok_fixed import TikTokTextOverlayAnalyzer
    from analyzers.composition_analysis_light import GPUBatchCompositionAnalysisLight
    from analyzers.visual_effects_cv_based import VisualEffectsCVBased
    from analyzers.gpu_batch_product_detection_light import GPUBatchProductDetectionLight
    
    # Person Analysis
    from analyzers.face_emotion_mediapipe import FaceEmotionMediaPipe
    from analyzers.body_pose_yolov8 import BodyPoseYOLOv8
    from analyzers.gpu_batch_eye_tracking import GPUBatchEyeTracking
    from analyzers.age_gender_insightface import AgeGenderInsightFace
    
    # Audio Analysis - Using Fixed Versions
    from utils.audio_processing_fix import AudioAnalysisUltimateFixed as UltimateSpeechTranscription
    from utils.audio_processing_fix import AudioAnalysisUltimateFixed as UltimateAudioAnalysis
    from utils.audio_processing_fix import AudioEnvironmentEnhancedFixed as AudioEnvironmentEnhanced
    from utils.audio_processing_fix import GPUBatchSpeechEmotionFixed as GPUBatchSpeechEmotion
    from utils.audio_processing_fix import GPUBatchSpeechRateEnhancedFixed as GPUBatchSpeechRateEnhanced
    from utils.audio_processing_fix import GPUBatchSpeechFlowFixed as GPUBatchSpeechFlow
    
    # Video Analysis
    from analyzers.camera_analysis_fixed import GPUBatchCameraAnalysisFixed
    from analyzers.gpu_batch_color_analysis import GPUBatchColorAnalysis
    from analyzers.gpu_batch_content_quality_fixed import GPUBatchContentQualityFixed
    from analyzers.cut_analysis_fixed import CutAnalysisFixedAnalyzer
    from analyzers.scene_segmentation_fixed import SceneSegmentationFixedAnalyzer
    from analyzers.temporal_flow_safe import TemporalFlowSafe as NarrativeAnalysisWrapper
    
    # Cross-Analyzer Intelligence - Safe Version
    from analyzers.cross_analyzer_intelligence_safe import CrossAnalyzerIntelligenceSafe as CrossAnalyzerIntelligence

except ImportError as e:
    logger.error(f"Failed to import analyzers: {e}")
    raise

# CLEAN SERVER ANALYZER MAPPING - 23 Best Analyzers Only
ML_ANALYZERS = {
    # Stage 1: Scene Understanding (Heavy GPU)
    'qwen2_vl_temporal': Qwen2VLTemporalAnalyzer,
    'object_detection': GPUBatchObjectDetectionYOLO,
    'background_segmentation': GPUBatchBackgroundSegmentationLight,
    
    # Stage 2: Person Analysis (Medium GPU)
    'face_emotion': FaceEmotionMediaPipe,
    'body_pose': BodyPoseYOLOv8,
    'eye_tracking': GPUBatchEyeTracking,
    'age_estimation': AgeGenderInsightFace,
    
    # Stage 3: Content Analysis (Light GPU)
    'text_overlay': TikTokTextOverlayAnalyzer,
    'speech_transcription': UltimateSpeechTranscription,
    'color_analysis': GPUBatchColorAnalysis,
    'composition_analysis': GPUBatchCompositionAnalysisLight,
    'visual_effects': VisualEffectsCVBased,
    'product_detection': GPUBatchProductDetectionLight,
    
    # Stage 4: Temporal Analysis (GPU)
    'cut_analysis': CutAnalysisFixedAnalyzer,
    'scene_segmentation': SceneSegmentationFixedAnalyzer,
    'camera_analysis': GPUBatchCameraAnalysisFixed,
    'content_quality': GPUBatchContentQualityFixed,
    
    # CPU Parallel - Audio and metadata
    'audio_analysis': UltimateAudioAnalysis,
    'audio_environment': AudioEnvironmentEnhanced,
    'speech_emotion': GPUBatchSpeechEmotion,
    'speech_rate': GPUBatchSpeechRateEnhanced,
    'speech_flow': GPUBatchSpeechFlow,
    'temporal_flow': NarrativeAnalysisWrapper,
    
    # Cross-Analyzer Intelligence - MUST RUN LAST
    'cross_analyzer_intelligence': CrossAnalyzerIntelligence,
}

# Analyzer capabilities for reconstruction
ANALYZER_CAPABILITIES = {
    'qwen2_vl_temporal': {
        'description': 'Advanced video understanding with temporal analysis',
        'reconstruction': 'scene_understanding',
        'fps': 0.5,
        'model': 'qwen2_vl_7b_instruct',
        'architecture': 'multimodal_llm'
    },
    'object_detection': {
        'description': 'Object detection and tracking with YOLOv8',
        'reconstruction': 'object_presence',
        'fps': 3.0,
        'model': 'yolov8',
        'architecture': 'cnn_detector'
    },
    'background_segmentation': {
        'description': 'Semantic segmentation of scene elements',
        'reconstruction': 'spatial_layout',
        'fps': 1.0,
        'model': 'segformer',
        'architecture': 'transformer_segmentation'
    },
    'face_emotion': {
        'description': 'Face detection and emotion recognition',
        'reconstruction': 'person_state',
        'fps': 2.0,
        'model': 'mediapipe_fer',
        'architecture': 'hybrid_detector'
    },
    'text_overlay': {
        'description': 'Text detection and OCR for TikTok content',
        'reconstruction': 'textual_content',
        'fps': 1.0,
        'model': 'easyocr',
        'architecture': 'ocr_detector'
    }
}

def get_ml_analyzer(name, turbo_mode=False):
    """Get ML analyzer instance"""
    
    if name not in ML_ANALYZERS:
        raise ValueError(f"Unknown analyzer: {name}")
    
    try:
        analyzer = ML_ANALYZERS[name]()
        
        # Log analyzer type
        capability = ANALYZER_CAPABILITIES.get(name, {})
        if capability:
            logger.info(f"Loaded {name}: {capability['description']}")
        
        return analyzer
        
    except Exception as e:
        logger.error(f"Failed to load {name}: {e}")
        raise

def get_analyzer_info(name):
    """Get analyzer capability information"""
    return ANALYZER_CAPABILITIES.get(name, {})

def get_reconstruction_analyzers():
    """Get analyzers critical for video reconstruction"""
    critical = []
    for name, info in ANALYZER_CAPABILITIES.items():
        if info.get('fps', 0) >= 1.0:
            critical.append(name)
    return critical

# Export for compatibility
__all__ = ['ML_ANALYZERS', 'get_ml_analyzer', 'get_analyzer_info', 'get_reconstruction_analyzers']