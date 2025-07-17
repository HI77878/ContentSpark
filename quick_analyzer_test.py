#!/usr/bin/env python3
"""Quick test of all analyzers"""
import os
os.environ['LD_LIBRARY_PATH'] = '/home/user/ffmpeg-install/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PATH'] = '/home/user/ffmpeg-install/bin:' + os.environ.get('PATH', '')

import sys
sys.path.insert(0, '/home/user/tiktok_production')

from ml_analyzer_registry_complete import ML_ANALYZERS
from configs.gpu_groups_config import DISABLED_ANALYZERS

# Get active analyzers
active_analyzers = [name for name in ML_ANALYZERS.keys() if name not in DISABLED_ANALYZERS]

print(f"Testing {len(active_analyzers)} active analyzers:")
print(f"Disabled: {DISABLED_ANALYZERS}")
print(f"\nActive analyzers: {active_analyzers}")

# Quick test
test_video = "/home/user/tiktok_production/test_local_video.mp4"
results = {}

for i, analyzer_name in enumerate(active_analyzers):
    print(f"\n[{i+1}/{len(active_analyzers)}] Testing {analyzer_name}...")
    
    try:
        analyzer_class = ML_ANALYZERS[analyzer_name]
        analyzer = analyzer_class()
        
        # Special handling for cross_analyzer
        if analyzer_name == 'cross_analyzer_intelligence':
            # Provide dummy results
            dummy_results = {
                'object_detection': {'segments': [{'timestamp': 0}]},
                'text_overlay': {'segments': [{'timestamp': 0}]}
            }
            result = analyzer.analyze(dummy_results)
        else:
            result = analyzer.analyze(test_video)
        
        if result and 'segments' in result and len(result['segments']) > 0:
            print(f"✅ SUCCESS: {len(result['segments'])} segments")
            results[analyzer_name] = 'success'
        else:
            print(f"⚠️ WARNING: No segments")
            results[analyzer_name] = 'no_segments'
            
        # Cleanup
        del analyzer
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)[:100]}")
        results[analyzer_name] = f"failed: {str(e)[:50]}"

# Summary
print("\n" + "="*50)
print("SUMMARY:")
success = sum(1 for r in results.values() if r == 'success')
print(f"✅ Successful: {success}/{len(active_analyzers)}")
print(f"⚠️ No segments: {sum(1 for r in results.values() if r == 'no_segments')}")
print(f"❌ Failed: {sum(1 for r in results.values() if 'failed' in str(r))}")

# Show failures
print("\nFailures:")
for name, result in results.items():
    if 'failed' in str(result):
        print(f"  {name}: {result}")