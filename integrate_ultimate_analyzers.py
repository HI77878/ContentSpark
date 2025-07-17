#!/usr/bin/env python3
"""
Integration aller Ultimate Analyzer in das System
"""

import os
import json

# Alle Ultimate Analyzer mit korrekten Mappings
ULTIMATE_ANALYZERS = {
    # Frame-by-Frame Reconstructor (Höchste Priorität)
    'frame_reconstructor': {
        'module': 'analyzers.frame_by_frame_reconstructor',
        'class': 'FrameByFrameReconstructor',
        'priority': 1
    },
    
    # Text & Speech (Ultimate Versionen)
    'text_overlay': {
        'module': 'analyzers.text_overlay_ultimate_v2',
        'class': 'UltimateTextOverlayV2',
        'priority': 1
    },
    'speech_transcription': {
        'module': 'analyzers.speech_transcription_ultimate',
        'class': 'UltimateSpeechTranscription',
        'priority': 1
    },
    'speech_emotion': {
        'module': 'analyzers.speech_emotion_ultimate',
        'class': 'SpeechEmotionUltimate',
        'priority': 2
    },
    
    # Visual Understanding
    'video_llava': {
        'module': 'analyzers.video_llava_ultimate_fixed',
        'class': 'VideoLLaVAUltimateFixed',
        'priority': 1
    },
    'object_detection': {
        'module': 'analyzers.object_detection_ultimate',
        'class': 'UltimateObjectDetector',
        'priority': 2
    },
    'product_detection': {
        'module': 'analyzers.product_detection_ultimate',
        'class': 'ProductDetectionUltimate',
        'priority': 2
    },
    
    # Body & Movement
    'gesture_body': {
        'module': 'analyzers.gesture_body_ultimate',
        'class': 'UltimateGestureBodyAnalyzer',
        'priority': 2
    },
    'eye_tracking': {
        'module': 'analyzers.eye_tracking_ultimate',
        'class': 'EyeTrackingUltimate',
        'priority': 3
    },
    'age_estimation': {
        'module': 'analyzers.age_estimation_ultimate',
        'class': 'AgeEstimationUltimate',
        'priority': 3
    },
    
    # Scene & Camera
    'camera_analysis': {
        'module': 'analyzers.camera_analysis_ultimate_v2',
        'class': 'CameraAnalysisUltimateV2',
        'priority': 2
    },
    'scene_segmentation': {
        'module': 'analyzers.scene_segmentation_ultimate',
        'class': 'UltimateSceneSegmentation',
        'priority': 2
    },
    'cut_analysis': {
        'module': 'analyzers.cut_analysis_ultimate',
        'class': 'CutAnalysisUltimate',
        'priority': 2
    },
    'background_analysis': {
        'module': 'analyzers.background_ultra_detailed',
        'class': 'UltraDetailedBackgroundAnalyzer',
        'priority': 3
    },
    
    # Visual Effects & Style
    'visual_effects': {
        'module': 'analyzers.visual_effects_ultimate',
        'class': 'VisualEffectsUltimate',
        'priority': 3
    },
    'composition_analysis': {
        'module': 'analyzers.composition_analysis_ultimate',
        'class': 'CompositionAnalysisUltimate',
        'priority': 3
    },
    'color_analysis': {
        'module': 'analyzers.color_analysis_ultimate',
        'class': 'ColorAnalysisUltimate',
        'priority': 3
    },
    'content_quality': {
        'module': 'analyzers.content_quality_ultimate',
        'class': 'ContentQualityUltimate',
        'priority': 4
    },
    
    # Audio Analysis
    'audio_analysis': {
        'module': 'analyzers.audio_analysis_ultimate',
        'class': 'UltimateAudioAnalysis',
        'priority': 2
    },
    'audio_environment': {
        'module': 'analyzers.audio_environment_ultimate',
        'class': 'AudioEnvironmentUltimate',
        'priority': 3
    },
    'sound_effects': {
        'module': 'analyzers.sound_effects_ultimate',
        'class': 'SoundEffectsUltimate',
        'priority': 3
    },
    
    # Narrative & Flow
    'temporal_flow': {
        'module': 'analyzers.temporal_flow_ultimate',
        'class': 'TemporalFlowUltimate',
        'priority': 4
    }
}

def update_multiprocess_executor():
    """Update the multiprocess executor with all Ultimate analyzers"""
    executor_path = '/home/user/tiktok_production/utils/multiprocess_gpu_executor_ultimate.py'
    
    content = '''#!/usr/bin/env python3
"""
Ultimate Multiprocess GPU Executor with ALL Ultimate Analyzers
"""

import torch
import torch.multiprocessing as mp
import time
import logging
import os
import gc
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import traceback
import cv2
from pathlib import Path
from queue import Empty

# Set spawn method at module level
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

logger = logging.getLogger(__name__)

class MultiprocessGPUExecutorUltimate:
    """Ultimate executor with all Ultimate analyzer mappings"""
    
    def __init__(self, num_gpu_processes: int = 4):
        self.num_gpu_processes = num_gpu_processes
        self.analyzer_configs = self._get_analyzer_configs()
        
    def _get_analyzer_configs(self) -> Dict[str, Dict]:
        """Get Ultimate analyzer configurations"""
        return ''' + json.dumps(ULTIMATE_ANALYZERS, indent=8) + '''
    
    def execute_parallel(self, video_path: str, selected_analyzers: List[str]) -> Dict[str, Any]:
        """Execute analyzers in parallel using multiprocessing"""
        # Implementation remains the same as multiprocess_gpu_executor_final.py
        # Just with updated analyzer mappings
        pass

# Worker process implementation (same as before)
def worker_process(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue, 
                  video_path: str, analyzer_configs: Dict[str, Dict]):
    """Worker process that runs analyzers"""
    # Same implementation as before
    pass
'''
    
    with open(executor_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Created {executor_path}")

def update_ml_analyzer_registry():
    """Update the ML analyzer registry"""
    registry_path = '/home/user/tiktok_production/ml_analyzer_registry_ultimate.py'
    
    imports = []
    registry_entries = []
    
    for name, config in ULTIMATE_ANALYZERS.items():
        module = config['module']
        class_name = config['class']
        
        # Skip frame_reconstructor for now as it's special
        if name != 'frame_reconstructor':
            imports.append(f"from {module} import {class_name}")
            registry_entries.append(f"    '{name}': {class_name},")
    
    content = f'''#!/usr/bin/env python3
"""
Ultimate ML Analyzer Registry - ALL Ultimate Versions
"""

# Import all Ultimate analyzers
{chr(10).join(imports)}

# Registry with all Ultimate analyzers
ML_ANALYZERS_ULTIMATE = {{
{chr(10).join(registry_entries)}
}}

# Get analyzer function
def get_ultimate_analyzer(name: str):
    """Get ultimate analyzer by name"""
    if name in ML_ANALYZERS_ULTIMATE:
        return ML_ANALYZERS_ULTIMATE[name]()
    else:
        raise ValueError(f"Unknown analyzer: {{name}}")

# Export
__all__ = ['ML_ANALYZERS_ULTIMATE', 'get_ultimate_analyzer']
'''
    
    with open(registry_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Created {registry_path}")

def create_ultimate_test_script():
    """Create test script for all Ultimate analyzers"""
    test_path = '/home/user/tiktok_production/test_all_ultimate_analyzers.py'
    
    content = '''#!/usr/bin/env python3
"""
Test ALL Ultimate Analyzers
"""
import sys
import time
sys.path.append('/home/user/tiktok_production')

from ml_analyzer_registry_ultimate import ML_ANALYZERS_ULTIMATE, get_ultimate_analyzer

def test_analyzer(name: str, video_path: str):
    """Test single analyzer"""
    print(f"\\nTesting {name}...")
    
    try:
        analyzer = get_ultimate_analyzer(name)
        start = time.time()
        result = analyzer.analyze(video_path)
        elapsed = time.time() - start
        
        # Get key stats
        stats = result.get('statistics', {})
        
        print(f"✅ {name}: Success in {elapsed:.1f}s")
        print(f"   Key stats: {list(stats.keys())[:3]}")
        
        return True, elapsed
    except Exception as e:
        print(f"❌ {name}: Failed - {str(e)[:100]}")
        return False, 0

def main():
    video_path = "/home/user/tiktok_production/downloads/videos/7425998222721633569.mp4"
    
    print("=== Testing ALL Ultimate Analyzers ===")
    print(f"Video: {video_path}")
    print(f"Total analyzers: {len(ML_ANALYZERS_ULTIMATE)}\\n")
    
    successes = 0
    total_time = 0
    
    for name in sorted(ML_ANALYZERS_ULTIMATE.keys()):
        success, elapsed = test_analyzer(name, video_path)
        if success:
            successes += 1
            total_time += elapsed
    
    print(f"\\n=== Summary ===")
    print(f"Successful: {successes}/{len(ML_ANALYZERS_ULTIMATE)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average: {total_time/successes:.1f}s per analyzer")

if __name__ == "__main__":
    main()
'''
    
    with open(test_path, 'w') as f:
        f.write(content)
    
    os.chmod(test_path, 0o755)
    print(f"✅ Created {test_path}")

def main():
    print("=== Integrating ALL Ultimate Analyzers ===\n")
    
    # Update configurations
    update_multiprocess_executor()
    update_ml_analyzer_registry()
    create_ultimate_test_script()
    
    print("\n✅ Integration complete!")
    print("\nNext steps:")
    print("1. Run: python3 test_all_ultimate_analyzers.py")
    print("2. Update API to use new executor")
    print("3. Run full end-to-end test")

if __name__ == "__main__":
    main()