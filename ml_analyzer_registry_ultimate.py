#!/usr/bin/env python3
"""
Ultimate ML Analyzer Registry - ALL Ultimate Versions
"""

# Import all Ultimate analyzers
from analyzers.text_overlay_ultimate_v2 import UltimateTextOverlayV2
from analyzers.speech_transcription_ultimate import UltimateSpeechTranscription
from analyzers.speech_emotion_ultimate import SpeechEmotionUltimate
from analyzers.video_llava_ultimate_fixed import VideoLLaVAUltimateFixed
from analyzers.object_detection_ultimate import UltimateObjectDetector
from analyzers.product_detection_ultimate import ProductDetectionUltimate
from analyzers.gesture_body_ultimate import UltimateGestureBodyAnalyzer
from analyzers.eye_tracking_ultimate import EyeTrackingUltimate
from analyzers.age_estimation_ultimate import AgeEstimationUltimate
from analyzers.camera_analysis_ultimate_v2 import CameraAnalysisUltimateV2
from analyzers.scene_segmentation_ultimate import UltimateSceneSegmentation
from analyzers.cut_analysis_ultimate import CutAnalysisUltimate
from analyzers.background_ultra_detailed import UltraDetailedBackgroundAnalyzer
from analyzers.visual_effects_ultimate import VisualEffectsUltimate
from analyzers.composition_analysis_ultimate import CompositionAnalysisUltimate
from analyzers.color_analysis_ultimate import ColorAnalysisUltimate
from analyzers.content_quality_ultimate import ContentQualityUltimate
from analyzers.audio_analysis_ultimate import UltimateAudioAnalysis
from analyzers.audio_environment_ultimate import AudioEnvironmentUltimate
from analyzers.sound_effects_ultimate import SoundEffectsUltimate
from analyzers.temporal_flow_ultimate import TemporalFlowUltimate

# Registry with all Ultimate analyzers
ML_ANALYZERS_ULTIMATE = {
    'text_overlay': UltimateTextOverlayV2,
    'speech_transcription': UltimateSpeechTranscription,
    'speech_emotion': SpeechEmotionUltimate,
    'video_llava': VideoLLaVAUltimateFixed,
    'object_detection': UltimateObjectDetector,
    'product_detection': ProductDetectionUltimate,
    'gesture_body': UltimateGestureBodyAnalyzer,
    'eye_tracking': EyeTrackingUltimate,
    'age_estimation': AgeEstimationUltimate,
    'camera_analysis': CameraAnalysisUltimateV2,
    'scene_segmentation': UltimateSceneSegmentation,
    'cut_analysis': CutAnalysisUltimate,
    'background_analysis': UltraDetailedBackgroundAnalyzer,
    'visual_effects': VisualEffectsUltimate,
    'composition_analysis': CompositionAnalysisUltimate,
    'color_analysis': ColorAnalysisUltimate,
    'content_quality': ContentQualityUltimate,
    'audio_analysis': UltimateAudioAnalysis,
    'audio_environment': AudioEnvironmentUltimate,
    'sound_effects': SoundEffectsUltimate,
    'temporal_flow': TemporalFlowUltimate,
}

# Get analyzer function
def get_ultimate_analyzer(name: str):
    """Get ultimate analyzer by name"""
    if name in ML_ANALYZERS_ULTIMATE:
        return ML_ANALYZERS_ULTIMATE[name]()
    else:
        raise ValueError(f"Unknown analyzer: {name}")

# Export
__all__ = ['ML_ANALYZERS_ULTIMATE', 'get_ultimate_analyzer']
