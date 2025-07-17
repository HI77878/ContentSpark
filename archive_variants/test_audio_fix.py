#!/usr/bin/env python3
"""Debug and fix audio analyzers"""
import sys
sys.path.append('/home/user/tiktok_production')

# FFmpeg Fix
import os
os.environ['LD_LIBRARY_PATH'] = '/home/user/ffmpeg-install/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PATH'] = '/home/user/ffmpeg-install/bin:' + os.environ.get('PATH', '')

# Try importing librosa directly
try:
    import librosa
    print("✅ Librosa imported successfully")
    
    # Test loading a simple audio file
    test_audio = "test_local_video.mp4"
    y, sr = librosa.load(test_audio, sr=22050, duration=5.0)
    print(f"✅ Loaded {len(y)/sr:.1f}s of audio at {sr}Hz")
    
except Exception as e:
    print(f"❌ Librosa error: {e}")
    import traceback
    traceback.print_exc()

# Now test the analyzers
from ml_analyzer_registry_complete import ML_ANALYZERS

# Test AudioEnvironmentEnhanced first (simpler)
print("\n" + "="*50)
print("Testing AudioEnvironmentEnhanced")
print("="*50)
try:
    AudioEnvClass = ML_ANALYZERS['audio_environment']
    analyzer = AudioEnvClass()
    print("✅ Initialized successfully")
    
    # Test analyze
    result = analyzer.analyze('test_local_video.mp4')
    segments = result.get('segments', [])
    print(f"✅ Analyzed: {len(segments)} segments")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test cross_analyzer_intelligence
print("\n" + "="*50)
print("Testing CrossAnalyzerIntelligence")
print("="*50)
try:
    CrossClass = ML_ANALYZERS['cross_analyzer_intelligence']
    analyzer = CrossClass()
    print("✅ Initialized successfully")
    
    # Test with empty dict
    result = analyzer.analyze('test_local_video.mp4', {})
    segments = result.get('segments', [])
    print(f"✅ Analyzed: {len(segments)} segments")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()