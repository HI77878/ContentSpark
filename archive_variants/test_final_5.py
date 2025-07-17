#!/usr/bin/env python3
"""Test the 5 problematic analyzers"""
import sys
import os
sys.path.append('/home/user/tiktok_production')

# FFmpeg global setzen
os.environ['LD_LIBRARY_PATH'] = '/home/user/ffmpeg-install/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PATH'] = '/home/user/ffmpeg-install/bin:' + os.environ.get('PATH', '')

test_video = "test_local_video.mp4"
problem_analyzers = [
    'background_segmentation',
    'audio_analysis', 
    'audio_environment',
    'speech_emotion',
    'cross_analyzer_intelligence'
]

# Import registry
from ml_analyzer_registry_complete import ML_ANALYZERS

results = {}

for analyzer_name in problem_analyzers:
    print(f"\n{'='*50}")
    print(f"Testing: {analyzer_name}")
    print(f"{'='*50}")
    
    try:
        # Get analyzer class from registry
        analyzer_class = ML_ANALYZERS.get(analyzer_name)
        if not analyzer_class:
            print(f"❌ {analyzer_name} not found in registry")
            results[analyzer_name] = "NOT_IN_REGISTRY"
            continue
            
        analyzer = analyzer_class()
        
        # Special handling for cross_analyzer
        if analyzer_name == 'cross_analyzer_intelligence':
            # Test with empty dict
            result = analyzer.analyze(test_video, {})
        else:
            result = analyzer.analyze(test_video)
            
        segments = result.get('segments', []) if isinstance(result, dict) else []
        print(f"✅ SUCCESS: {len(segments)} segments")
        results[analyzer_name] = f"SUCCESS: {len(segments)} segments"
        
        # Show sample output
        if segments and len(segments) > 0:
            print(f"Sample: {segments[0].get('description', segments[0])[:100]}...")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results[analyzer_name] = f"FAILED: {str(e)[:100]}"
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("FINAL RESULTS:")
print("="*70)
success_count = 0
for name, result in results.items():
    status = "✅" if "SUCCESS" in result else "❌"
    print(f"{status} {name}: {result}")
    if "SUCCESS" in result:
        success_count += 1

print(f"\nTotal: {success_count}/5 fixed")
print(f"Success Rate: {success_count/5*100:.1f}%")