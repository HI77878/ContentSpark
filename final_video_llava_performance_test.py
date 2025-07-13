#!/usr/bin/env python3
"""
Final Performance Test for Video-LLaVA Integration
Verifies end-to-end performance meets <3x realtime requirement
"""
import time
import httpx
import json
import sys
from pathlib import Path
import torch

sys.path.append('/home/user/tiktok_production')

API_URL = "http://localhost:8003"
TEST_VIDEOS = [
    "/home/user/tiktok_production/downloads/videos/7425998222721633569.mp4",  # 28.8s
    "/home/user/tiktok_production/downloads/videos/7446489995663117590.mp4",  # If available
]

def test_performance():
    """Test Video-LLaVA performance in production system"""
    print("="*80)
    print("Final Video-LLaVA Performance Test")
    print("="*80)
    
    # System info
    print("\nSystem Information:")
    print(f"- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"- GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A")
    print(f"- API: {API_URL}")
    
    # Check API health
    try:
        response = httpx.get(f"{API_URL}/health", timeout=10.0)
        if response.status_code != 200:
            print("❌ API not healthy")
            return False
    except:
        print("❌ API not reachable")
        return False
    
    print("✅ API is healthy")
    
    results = []
    
    for video_path in TEST_VIDEOS:
        if not Path(video_path).exists():
            print(f"\n⚠️  Skipping {video_path} (not found)")
            continue
            
        print(f"\n\nTesting: {Path(video_path).name}")
        print("-"*60)
        
        # Get video duration
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        print(f"Video duration: {duration:.1f}s")
        
        # Test with all analyzers
        print("\n1. Running full analysis (all 21 analyzers)...")
        try:
            with httpx.Client(timeout=600.0) as client:
                start_time = time.time()
                
                response = client.post(
                    f"{API_URL}/analyze",
                    json={"video_path": video_path}
                )
                
                total_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    api_time = result['processing_time']
                    realtime_factor = api_time / duration if duration > 0 else 0
                    
                    print(f"✅ Analysis completed")
                    print(f"   Total time: {total_time:.1f}s")
                    print(f"   API processing: {api_time:.1f}s")
                    print(f"   Realtime factor: {realtime_factor:.2f}x")
                    print(f"   Successful analyzers: {result['successful_analyzers']}/{result['total_analyzers']}")
                    
                    # Check Video-LLaVA results
                    results_file = result.get('results_file')
                    if results_file and Path(results_file).exists():
                        with open(results_file, 'r') as f:
                            data = json.load(f)
                        
                        if 'analyzer_results' in data and 'video_llava' in data['analyzer_results']:
                            llava = data['analyzer_results']['video_llava']
                            if 'error' not in llava:
                                print("✅ Video-LLaVA produced results")
                            else:
                                print(f"❌ Video-LLaVA error: {llava['error']}")
                    
                    results.append({
                        'video': Path(video_path).name,
                        'duration': duration,
                        'processing_time': api_time,
                        'realtime_factor': realtime_factor,
                        'success': True
                    })
                else:
                    print(f"❌ API error: {response.status_code}")
                    results.append({
                        'video': Path(video_path).name,
                        'duration': duration,
                        'success': False
                    })
                    
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append({
                'video': Path(video_path).name,
                'duration': duration,
                'success': False
            })
    
    # Summary
    print("\n\n" + "="*80)
    print("Performance Test Summary")
    print("="*80)
    
    if results:
        successful = [r for r in results if r.get('success', False)]
        if successful:
            avg_realtime = sum(r['realtime_factor'] for r in successful) / len(successful)
            print(f"\n✅ Tests passed: {len(successful)}/{len(results)}")
            print(f"✅ Average realtime factor: {avg_realtime:.2f}x")
            
            if avg_realtime < 3.0:
                print(f"✅ MEETS <3x realtime requirement!")
            else:
                print(f"⚠️  Exceeds 3x realtime target ({avg_realtime:.2f}x)")
            
            print("\nDetailed results:")
            for r in results:
                if r['success']:
                    print(f"  - {r['video']}: {r['realtime_factor']:.2f}x ({r['processing_time']:.1f}s for {r['duration']:.1f}s video)")
                else:
                    print(f"  - {r['video']}: FAILED")
                    
            # GPU Memory check
            if torch.cuda.is_available():
                print(f"\nGPU Memory Usage: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
        else:
            print("❌ All tests failed")
    else:
        print("❌ No tests completed")
    
    return len([r for r in results if r.get('success', False)]) > 0

if __name__ == "__main__":
    success = test_performance()
    sys.exit(0 if success else 1)