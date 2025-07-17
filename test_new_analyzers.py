#!/usr/bin/env python3
"""Test script for new analyzers"""

import os
import sys
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test video path
TEST_VIDEO = "/home/user/tiktok_production/test_video.mp4"

def test_analyzer(analyzer_name, analyzer_class):
    """Test a single analyzer"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {analyzer_name}")
    logger.info(f"{'='*60}")
    
    try:
        # Create analyzer instance
        analyzer = analyzer_class()
        logger.info(f"✅ {analyzer_name} initialized successfully")
        
        # Test analysis
        logger.info(f"Running analysis on test video...")
        result = analyzer.analyze(TEST_VIDEO)
        
        # Check results
        if result and isinstance(result, dict):
            segments = result.get('segments', [])
            logger.info(f"✅ Analysis completed with {len(segments)} segments")
            
            # Show sample output
            if segments:
                logger.info(f"Sample segment: {json.dumps(segments[0], indent=2)[:500]}...")
            
            # Check for metadata
            if 'metadata' in result:
                logger.info(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
            
            # For cross-analyzer, it needs other analyzer outputs
            if analyzer_name == 'cross_analyzer_intelligence':
                logger.info("Note: cross_analyzer_intelligence requires analyzer_outputs parameter")
                
            return True
        else:
            logger.error(f"❌ {analyzer_name} returned invalid result: {type(result)}")
            return False
            
    except Exception as e:
        logger.error(f"❌ {analyzer_name} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Check if test video exists
    if not os.path.exists(TEST_VIDEO):
        logger.error(f"Test video not found: {TEST_VIDEO}")
        logger.info("Please download a test video first")
        return
    
    # Import analyzers
    try:
        from analyzers.visual_effects_ml_advanced import VisualEffectsMLAdvanced
        from analyzers.age_gender_insightface import AgeGenderInsightFace
        from analyzers.face_emotion_deepface import FaceEmotionDeepFace
        from analyzers.body_pose_yolov8 import BodyPoseYOLOv8
        from analyzers.narrative_analysis_wrapper import NarrativeAnalysisWrapper
        from analyzers.cross_analyzer_intelligence import CrossAnalyzerIntelligence
        
        logger.info("✅ All analyzer imports successful")
    except ImportError as e:
        logger.error(f"Failed to import analyzers: {e}")
        return
    
    # Test each analyzer
    analyzers_to_test = [
        ('visual_effects', VisualEffectsMLAdvanced),
        ('age_estimation', AgeGenderInsightFace),
        ('face_emotion', FaceEmotionDeepFace),
        ('body_pose', BodyPoseYOLOv8),
        ('temporal_flow', NarrativeAnalysisWrapper),
        ('cross_analyzer_intelligence', CrossAnalyzerIntelligence),
    ]
    
    results = {}
    for name, analyzer_class in analyzers_to_test:
        success = test_analyzer(name, analyzer_class)
        results[name] = success
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Test Summary")
    logger.info(f"{'='*60}")
    
    total = len(results)
    successful = sum(1 for v in results.values() if v)
    
    for name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{name}: {status}")
    
    logger.info(f"\nTotal: {successful}/{total} analyzers passed")
    
    if successful == total:
        logger.info("\n🎉 All new analyzers are working correctly!")
    else:
        logger.warning(f"\n⚠️  {total - successful} analyzers failed")

if __name__ == "__main__":
    main()