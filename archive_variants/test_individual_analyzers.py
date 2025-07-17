#!/usr/bin/env python3
"""Test analyzers individually to avoid segfault"""
import os
import subprocess
import json

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

results = {}

for analyzer_name in problem_analyzers:
    print(f"\n{'='*50}")
    print(f"Testing: {analyzer_name}")
    print(f"{'='*50}")
    
    # Test each analyzer in separate process to avoid segfaults
    test_script = f"""
import sys
sys.path.append('/home/user/tiktok_production')
import os
os.environ['LD_LIBRARY_PATH'] = '/home/user/ffmpeg-install/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PATH'] = '/home/user/ffmpeg-install/bin:' + os.environ.get('PATH', '')

from ml_analyzer_registry_complete import ML_ANALYZERS

try:
    analyzer_class = ML_ANALYZERS.get('{analyzer_name}')
    if analyzer_class:
        analyzer = analyzer_class()
        if '{analyzer_name}' == 'cross_analyzer_intelligence':
            result = analyzer.analyze('{test_video}', {{}})
        else:
            result = analyzer.analyze('{test_video}')
        segments = result.get('segments', []) if isinstance(result, dict) else []
        print(f'SUCCESS:{{len(segments)}}')
    else:
        print('NOT_FOUND')
except Exception as e:
    print(f'ERROR:{{str(e)[:100]}}')
"""
    
    try:
        # Run in subprocess to avoid segfaults
        proc = subprocess.run(['python3', '-c', test_script], 
                            capture_output=True, text=True, timeout=60)
        
        if proc.returncode == 0 and 'SUCCESS:' in proc.stdout:
            segments = int(proc.stdout.strip().split('SUCCESS:')[1])
            print(f"✅ SUCCESS: {segments} segments")
            results[analyzer_name] = f"SUCCESS: {segments} segments"
        elif 'NOT_FOUND' in proc.stdout:
            print(f"❌ Not found in registry")
            results[analyzer_name] = "NOT_IN_REGISTRY"
        else:
            error = proc.stderr[:200] if proc.stderr else proc.stdout[:200]
            print(f"❌ FAILED: {error}")
            results[analyzer_name] = f"FAILED: {error[:100]}"
            
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT after 60 seconds")
        results[analyzer_name] = "TIMEOUT"
    except Exception as e:
        print(f"❌ ERROR: {e}")
        results[analyzer_name] = f"ERROR: {str(e)[:100]}"

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

# Save results for API test
with open('test_results.json', 'w') as f:
    json.dump(results, f, indent=2)