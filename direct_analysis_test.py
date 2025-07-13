#!/usr/bin/env python3
"""
Direct video analysis test without API
"""
import os
import sys
import time
import json
from datetime import datetime

# Set environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "1"

sys.path.append('/home/user/tiktok_production')

# Import components
from utils.multiprocess_gpu_executor_final import MultiprocessGPUExecutorFinal
from ml_analyzer_registry_complete import ML_ANALYZERS
from configs.gpu_groups_config import DISABLED_ANALYZERS, GPU_ANALYZER_GROUPS

def main():
    video_path = "/home/user/tiktok_production/downloads/videos/7522589683939921165.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
        
    print("=== DIRECT VIDEO ANALYSIS TEST ===")
    print(f"Video: {video_path}")
    print(f"Size: {os.path.getsize(video_path) / 1024**2:.1f} MB")
    
    # Get active analyzers
    active_analyzers = []
    for group_name, analyzer_list in GPU_ANALYZER_GROUPS.items():
        for analyzer in analyzer_list:
            if analyzer not in DISABLED_ANALYZERS and analyzer in ML_ANALYZERS:
                if analyzer not in active_analyzers:
                    active_analyzers.append(analyzer)
    
    # Use top 21 analyzers
    selected_analyzers = active_analyzers[:21]
    print(f"\nActive analyzers: {len(selected_analyzers)}")
    print(f"Analyzers: {', '.join(selected_analyzers[:5])}...")
    
    # Initialize executor
    print("\nInitializing multiprocess executor...")
    executor = MultiprocessGPUExecutorFinal(num_gpu_processes=3)
    
    # Start timing
    start_time = time.time()
    print(f"\nStarting analysis at {datetime.now().strftime('%H:%M:%S')}...")
    
    try:
        # Run analysis
        results = executor.execute_parallel(video_path, selected_analyzers)
        
        # Calculate metrics
        total_time = time.time() - start_time
        video_duration = results.get('metadata', {}).get('duration', 68)  # 68s from earlier info
        realtime_factor = total_time / video_duration if video_duration > 0 else 0
        
        # Count successful analyzers
        successful = 0
        for k, v in results.items():
            if k != 'metadata' and isinstance(v, dict) and 'error' not in v:
                successful += 1
        
        print(f"\n✅ Analysis complete!")
        print(f"Time: {total_time:.1f} seconds")
        print(f"Video duration: {video_duration:.1f} seconds")
        print(f"Realtime factor: {realtime_factor:.2f}x")
        print(f"Successful analyzers: {successful}/{len(selected_analyzers)}")
        print(f"Reconstruction score: {(successful/len(selected_analyzers)*100):.1f}%")
        
        # Save results
        output_path = f"results/direct_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")
        
        # Check some specific results
        if 'video_llava' in results and 'segments' in results['video_llava']:
            print(f"\nBLIP-2 segments: {len(results['video_llava']['segments'])}")
            if results['video_llava']['segments']:
                first_desc = results['video_llava']['segments'][0].get('description', '')
                print(f"First description: {first_desc[:100]}...")
                
        return results
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()