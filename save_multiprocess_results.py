#!/usr/bin/env python3
"""Save multiprocess results with JSON serialization fix"""
import sys
sys.path.append('/home/user/tiktok_production')

from utils.simple_multiprocess_executor import SimpleMultiprocessExecutor
from configs.gpu_groups_config import DISABLED_ANALYZERS
from ml_analyzer_registry_complete import ML_ANALYZERS
import time
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)

def run_and_save():
    video_path = "/home/user/tiktok_production/test_videos/leon_schliebach_7446489995663117590.mp4"

    # Get ALL active analyzers
    all_analyzers = [name for name in ML_ANALYZERS.keys() if name not in DISABLED_ANALYZERS]
    
    print(f"Running ALL {len(all_analyzers)} active analyzers...")
    
    executor = SimpleMultiprocessExecutor(num_processes=4)

    start_time = time.time()
    results = executor.execute_parallel(video_path, all_analyzers)
    elapsed = time.time() - start_time
    
    print(f"\n✅ Completed in {elapsed:.2f}s")
    print(f"📊 Realtime factor: {results['metadata']['realtime_factor']:.2f}x")
    
    # Count successes
    success_count = sum(1 for a in all_analyzers if a in results and 'error' not in results[a])
    print(f"📈 Success rate: {success_count}/{len(all_analyzers)} ({success_count/len(all_analyzers)*100:.1f}%)")
    
    # Save with numpy encoder
    timestamp = int(time.time())
    output_file = f"/home/user/tiktok_production/results/multiprocess_all_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary
    print("\n📋 Analysis Summary:")
    for analyzer in all_analyzers:
        if analyzer in results:
            if 'error' in results[analyzer]:
                print(f"  ❌ {analyzer}")
            else:
                result = results[analyzer]
                if 'segments' in result:
                    print(f"  ✅ {analyzer}: {len(result['segments'])} segments")
                elif 'detections' in result:
                    print(f"  ✅ {analyzer}: {len(result['detections'])} detections")
                elif 'events' in result:
                    print(f"  ✅ {analyzer}: {len(result['events'])} events")
                else:
                    print(f"  ✅ {analyzer}")

if __name__ == '__main__':
    run_and_save()