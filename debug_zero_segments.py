#!/usr/bin/env python3
"""Debug 0-segment analyzers"""
import subprocess
import sys
import os
sys.path.append('/home/user/tiktok_production')

# FFmpeg Fix
os.environ['LD_LIBRARY_PATH'] = '/home/user/ffmpeg-install/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PATH'] = '/home/user/ffmpeg-install/bin:' + os.environ.get('PATH', '')

from ml_analyzer_registry_complete import ML_ANALYZERS

# Liste der 0-Segment Analyzer
zero_segment_analyzers = [
    'background_segmentation', 'audio_analysis', 'audio_environment',
    'speech_emotion', 'speech_flow', 'speech_transcription',
    'eye_tracking', 'cross_analyzer_intelligence'
]

test_video = "/home/user/tiktok_production/test_local_video.mp4"

for analyzer_name in zero_segment_analyzers:
    print(f"\n{'='*60}")
    print(f"DEBUGGING: {analyzer_name}")
    print(f"{'='*60}")
    
    # Test 1: Direct Import & Run
    try:
        # Get from registry
        analyzer_class = ML_ANALYZERS.get(analyzer_name)
        if not analyzer_class:
            print(f"❌ {analyzer_name} not in registry!")
            continue
            
        analyzer = analyzer_class()
        
        # Special handling for cross_analyzer
        if analyzer_name == 'cross_analyzer_intelligence':
            result = analyzer.analyze(test_video, {})
        else:
            result = analyzer.analyze(test_video)
        
        if result and 'segments' in result:
            seg_count = len(result['segments'])
            print(f"✅ DIRECT TEST: {seg_count} segments")
            if seg_count > 0:
                # Show sample
                first_seg = result['segments'][0]
                print(f"   Sample: {str(first_seg)[:100]}...")
        else:
            print(f"❌ DIRECT TEST: No segments key")
            print(f"   Keys: {list(result.keys()) if result else 'None'}")
            
    except Exception as e:
        print(f"❌ DIRECT TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Check in subprocess (multiprocessing simulation)
    print("\nTesting in subprocess...")
    
    cmd = f"""
import os
os.environ['LD_LIBRARY_PATH'] = '/home/user/ffmpeg-install/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PATH'] = '/home/user/ffmpeg-install/bin:' + os.environ.get('PATH', '')
os.environ['AUDIOREAD_FFDEC_PREFER'] = 'ffmpeg'

import sys
sys.path.append('/home/user/tiktok_production')

from ml_analyzer_registry_complete import ML_ANALYZERS

analyzer_class = ML_ANALYZERS.get('{analyzer_name}')
if analyzer_class:
    analyzer = analyzer_class()
    if '{analyzer_name}' == 'cross_analyzer_intelligence':
        result = analyzer.analyze('{test_video}', {{}})
    else:
        result = analyzer.analyze('{test_video}')
    seg_count = len(result.get('segments', []))
    print(f'SUBPROCESS: {{seg_count}} segments')
else:
    print('SUBPROCESS: Not in registry')
"""
    
    try:
        result = subprocess.run(
            [sys.executable, '-c', cmd],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.stdout:
            print(f"✅ {result.stdout.strip()}")
        if result.stderr:
            print(f"⚠️ STDERR: {result.stderr[:200]}")
        if result.returncode != 0:
            print(f"❌ Subprocess failed with code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print(f"❌ SUBPROCESS TIMEOUT after 60s")
    except Exception as e:
        print(f"❌ SUBPROCESS ERROR: {e}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Zero-segment analyzers need fixes in:")
print("1. FFmpeg/Librosa environment setup")
print("2. Multiprocessing context")
print("3. Cross-analyzer needs other results")