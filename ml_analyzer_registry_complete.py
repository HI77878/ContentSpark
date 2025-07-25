#!/usr/bin/env python3
<<<<<<< HEAD
"""CLEAN SERVER ML Analyzer Registry - MVP Version"""
=======
"""COMPLETE ML Analyzer Registry - No Fallbacks, Only Real ML Models"""
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc

import logging
logger = logging.getLogger(__name__)

<<<<<<< HEAD
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
=======
# Import ALL analyzers including new ones
try:
    # REAL ML ANALYZERS (No fallbacks)
    # from analyzers.vid2seq_real_fixed import Vid2SeqRealFixed as Vid2SeqScenicReal  # ARCHIVED - moved to archived_analyzers/
    # BLIP/BLIP2 REMOVED - Using real Video-LLaVA instead
    # from analyzers.emotion_detection_real import GPUBatchEmotionDetectionReal  # ARCHIVED
    # from analyzers.gpu_batch_facial_emotion_enhanced import GPUBatchFacialEmotionEnhanced  # ARCHIVED
    from analyzers.background_segmentation_light import GPUBatchBackgroundSegmentationLight
    # Motion vector analyzer removed - was causing 12 minute delays
    # from analyzers.speech_analysis_music_aware import GPUBatchSpeechAnalysisMusicAware  # ARCHIVED
    # Enhanced analyzers
    from analyzers.speech_transcription_ultimate import UltimateSpeechTranscription as EnhancedSpeechTranscription
    from analyzers.sound_effects_detector import SoundEffectsDetector
    
    # AuroraCap analyzer - REMOVED - unreliable
    # from analyzers.auroracap_analyzer import AuroraCapAnalyzer
    
    # Existing working analyzers
    # from analyzers.gpu_batch_face_detection_optimized import GPUBatchFaceDetectionOptimized  # ARCHIVED
    # from analyzers.face_emotion_deepface import FaceEmotionDeepFace  # BROKEN: TensorFlow incompatibility
    from analyzers.face_emotion_mediapipe import FaceEmotionMediaPipe  # NEW: MediaPipe + FER for face emotion
    from analyzers.gpu_batch_object_detection_yolo import GPUBatchObjectDetectionYOLO
    # from analyzers.text_overlay_enhanced import TextOverlayEnhancedAnalyzer  # DISABLED - Bad OCR
    # from analyzers.text_overlay_tesseract import TesseractTextOverlayAnalyzer  # DISABLED - Frame extraction bug
    from analyzers.text_overlay_tiktok_fixed import TikTokTextOverlayAnalyzer
    from analyzers.camera_analysis_fixed import GPUBatchCameraAnalysisFixed
    # from analyzers.gpu_batch_body_pose_yolo import GPUBatchBodyPoseYOLO  # ARCHIVED
    from analyzers.body_pose_yolov8 import BodyPoseYOLOv8  # NEW: YOLOv8x-pose for body language
    # from analyzers.gpu_batch_hand_gesture import GPUBatchHandGesture  # ARCHIVED
    from analyzers.gpu_batch_eye_tracking import GPUBatchEyeTracking
    from analyzers.gpu_batch_color_analysis import GPUBatchColorAnalysis
    from analyzers.composition_analysis_light import GPUBatchCompositionAnalysisLight
    # from analyzers.age_estimation_light import GPUBatchAgeEstimationLight  # REPLACED with InsightFace
    from analyzers.age_gender_insightface import AgeGenderInsightFace
    from analyzers.age_gender_mediapipe_gpu import AgeGenderMediaPipeGPU  # GPU-optimized version
    # from analyzers.gpu_batch_facial_details import GPUBatchFacialDetails  # ARCHIVED
    # from analyzers.gpu_batch_gesture_recognition_fixed import GPUBatchGestureRecognitionFixed  # ARCHIVED
    from analyzers.cut_analysis_fixed import CutAnalysisFixedAnalyzer
    from analyzers.scene_segmentation_fixed import SceneSegmentationFixedAnalyzer
    from analyzers.gpu_batch_content_quality_fixed import GPUBatchContentQualityFixed
    # from analyzers.visual_effects_light_fixed import VisualEffectsLight  # REPLACED with ML version
    # from analyzers.visual_effects_ml_advanced import VisualEffectsMLAdvanced  # ML models don't detect real effects
    from analyzers.visual_effects_cv_based import VisualEffectsCVBased  # CV-based for real effect detection
    from analyzers.gpu_batch_product_detection_light import GPUBatchProductDetectionLight
    
    # Audio analyzers
    from analyzers.audio_analysis_ultimate import UltimateAudioAnalysis as GPUBatchAudioAnalysisEnhanced
    # from analyzers.cpu_batch_audio_environment import CPUBatchAudioEnvironment  # ARCHIVED
    from analyzers.audio_environment_enhanced import AudioEnvironmentEnhanced
    from analyzers.gpu_batch_speech_emotion import GPUBatchSpeechEmotion
    from analyzers.gpu_batch_speech_rate_enhanced import GPUBatchSpeechRateEnhanced as GPUBatchSpeechRate
    # from analyzers.gpu_batch_sound_effects_enhanced import GPUBatchSoundEffectsEnhanced  # Removed - using SoundEffectsDetector
    from analyzers.gpu_batch_speech_flow import GPUBatchSpeechFlow
    from analyzers.gpu_batch_comment_cta_detection import GPUBatchCommentCTADetection
    
    # CPU analyzers
    # from analyzers.cpu_batch_scene_description import CPUBatchSceneDescription  # ARCHIVED
    # from analyzers.cpu_batch_temporal_flow import CPUBatchTemporalFlow  # REPLACED with narrative analysis
    # from analyzers.narrative_analysis_advanced import NarrativeAnalysisAdvanced
    from analyzers.narrative_analysis_wrapper import NarrativeAnalysisWrapper
    # from analyzers.gpu_batch_body_language import GPUBatchBodyLanguage  # ARCHIVED
    
    # Cross-Analyzer Intelligence
    from analyzers.cross_analyzer_intelligence import CrossAnalyzerIntelligence
    
    # Trend analysis - ARCHIVED
    # from analyzers.gpu_batch_trend_analysis_real_ml import GPUBatchTrendAnalysisRealML
    
    # NEW: LLaVA-NeXT Video for state-of-the-art video understanding
    # from analyzers.llava_next_video_analyzer import LLaVANextVideoAnalyzer  # ARCHIVED
    # from analyzers.llava_video_optimized import LLaVAVideoOptimized  # ARCHIVED
    # from analyzers.llava_video_frame_by_frame import LLaVAVideoFrameByFrame  # ARCHIVED
    # from analyzers.llava_video_ultra_detailed import LLaVAVideoUltraDetailed  # ARCHIVED
    # from analyzers.video_description_detailed import VideoDescriptionDetailed  # ARCHIVED
    # from analyzers.video_scene_narrator import VideoSceneNarrator  # ARCHIVED
    # from analyzers.video_comprehensive_description import VideoComprehensiveDescription  # ARCHIVED
    
    # NEW: Streaming Dense Video Captioning for temporal descriptions
    # from analyzers.streaming_dense_captioning_analyzer import StreamingDenseCaptioningAnalyzer  # BROKEN
    from analyzers.streaming_dense_captioning_fixed import StreamingDenseCaptioningFixed as StreamingDenseCaptioningAnalyzer
    
    # NEW: Tarsier Video Description Analyzer - ByteDance's state-of-the-art model
    # from analyzers.tarsier_video_description_analyzer import TarsierVideoDescriptionAnalyzer  # ARCHIVED
    
    # NEW: Qwen2-VL Video Analyzer - Alternative for second-by-second descriptions
    # from analyzers.qwen2_vl_video_analyzer import Qwen2VLVideoAnalyzer  # ARCHIVED
    
    # NEW: Qwen2-VL Optimized - Memory-efficient version with single-frame processing
    from analyzers.qwen2_vl_optimized_analyzer import Qwen2VLOptimizedAnalyzer
    
    # NEW: Qwen2-VL Temporal - Multi-frame analysis with accurate descriptions
    # from analyzers.qwen2_vl_temporal_analyzer import Qwen2VLTemporalAnalyzer  # ARCHIVED
    # KORREKTE VERSION - Proper video analysis
    from analyzers.qwen2_vl_video_analyzer import Qwen2VLVideoAnalyzer
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc

except ImportError as e:
    logger.error(f"Failed to import analyzers: {e}")
    raise

<<<<<<< HEAD
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
=======
# COMPLETE ANALYZER MAPPING - 30+ analyzers for full reconstruction
ML_ANALYZERS = {
    # Stage 1: Scene Understanding (Heavy GPU)
    'qwen2_vl_temporal': Qwen2VLVideoAnalyzer,  # KORREKTE Version für Video-Analyse
    'qwen2_vl_optimized': Qwen2VLOptimizedAnalyzer,  # Memory-efficient but slower (48s per video second)
    # 'qwen2_vl_video': Qwen2VLVideoAnalyzer,  # Original version - OOM issues
    # 'tarsier_video_description': TarsierVideoDescriptionAnalyzer,  # Tarsier - requires special setup
    'streaming_dense_captioning': StreamingDenseCaptioningAnalyzer,  # Real temporal descriptions at 15 FPS
    'object_detection': GPUBatchObjectDetectionYOLO,        # YOLOv8
    'background_segmentation': GPUBatchBackgroundSegmentationLight,  # SegFormer semantic segmentation
    
    # Stage 2: Person Analysis (Medium GPU)
    'face_emotion': FaceEmotionMediaPipe,                   # Face detection + emotion with MediaPipe + FER
    'body_pose': BodyPoseYOLOv8,                            # Body pose and gesture analysis
    # DISABLED for performance: 'face_detection', 'emotion_detection', 'hand_gesture'
    'eye_tracking': GPUBatchEyeTracking,                    # MediaPipe iris
    
    # Stage 3: Content Analysis (Light GPU)
    'text_overlay': TikTokTextOverlayAnalyzer,                  # FIXED: EasyOCR optimized for TikTok subtitles
    'speech_transcription': EnhancedSpeechTranscription,  # Enhanced with pitch/speed analysis
    'color_analysis': GPUBatchColorAnalysis,                # Color extraction
    'composition_analysis': GPUBatchCompositionAnalysisLight,  # CLIP
    'visual_effects': VisualEffectsCVBased,             # CV-based effect detection
    # 'motion_vectors': Removed - was causing 12 minute delays
    
    # Stage 4: Detail Analysis (Light GPU)
    'age_estimation': AgeGenderInsightFace,                 # Age+Gender with InsightFace
    # DISABLED: 'facial_details', 'gesture_recognition'
    'content_quality': GPUBatchContentQualityFixed,         # Quality metrics
    'product_detection': GPUBatchProductDetectionLight,          # Product/brand detection
    
    # Stage 5: Temporal Analysis (GPU)
    'cut_analysis': CutAnalysisFixedAnalyzer,           # Scene cuts
    'scene_segmentation': SceneSegmentationFixedAnalyzer,     # Scene boundaries (GPU)
    'camera_analysis': GPUBatchCameraAnalysisFixed,         # Camera movement
    
    # CPU Parallel - Audio and metadata
    'audio_analysis': GPUBatchAudioAnalysisEnhanced,        # Librosa
    'audio_environment': AudioEnvironmentEnhanced,          # Enhanced environment detection
    'speech_emotion': GPUBatchSpeechEmotion,                # Emotion from speech
    'speech_rate': GPUBatchSpeechRate,                     # Speech metrics
    'sound_effects': SoundEffectsDetector,                  # Advanced sound effect detection
    'speech_flow': GPUBatchSpeechFlow,                     # Speech emphasis and pauses
    'comment_cta_detection': GPUBatchCommentCTADetection,   # Comment references and CTAs
    # DISABLED: 'scene_description', 'body_language'
    'temporal_flow': NarrativeAnalysisWrapper,              # Advanced narrative analysis
    # 'trend_analysis': GPUBatchTrendAnalysisRealML,        # ARCHIVED - moved to archived_analyzers/
    
    # Cross-Analyzer Intelligence - MUST RUN LAST
    'cross_analyzer_intelligence': CrossAnalyzerIntelligence,  # Correlates all analyzer outputs
    
    # Experimental models removed - using LLaVA-NeXT as primary video analyzer
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
}

# Analyzer capabilities for reconstruction
ANALYZER_CAPABILITIES = {
<<<<<<< HEAD
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
=======
    'vid2seq': {
        'description': 'Frame-by-frame scene descriptions with Vid2Seq Scenic T5.1.1',
        'reconstruction': 'scene_understanding',
        'fps': 1.0,
        'model': 'vid2seq_scenic_real',
        'architecture': 'encoder_decoder_t5'
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
    },
    'background_segmentation': {
        'description': 'Semantic segmentation of scene elements',
        'reconstruction': 'spatial_layout',
<<<<<<< HEAD
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
=======
        'fps': 1.0
    },
    'motion_vectors': {
        'description': 'Optical flow for frame interpolation',
        'reconstruction': 'temporal_continuity',
        'fps': 6.0
    },
    'emotion_detection': {
        'description': 'Real emotion recognition on faces',
        'reconstruction': 'person_state',
        'fps': 2.0
    },
    'object_detection': {
        'description': 'Object detection and tracking',
        'reconstruction': 'object_presence',
        'fps': 3.0
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
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