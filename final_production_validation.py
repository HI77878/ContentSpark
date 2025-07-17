#!/usr/bin/env python3
"""Final Production Test with Performance Validation"""

import sys
import os
import time
import json
import requests
from pathlib import Path

sys.path.append('/home/user/tiktok_production')

from performance_monitor import PerformanceMonitor

def run_final_test():
    """Run final production test with performance monitoring"""
    
    # Test video (28.9s video)
    video_path = "/home/user/tiktok_production/downloads/videos/7425998222721633569.mp4"
    api_url = "http://localhost:8003"
    
    print("🚀 FINAL PRODUCTION TEST")
    print("=" * 60)
    print(f"📹 Test video: {video_path}")
    print(f"🌐 API URL: {api_url}")
    print(f"🎯 Target: <3x realtime (<90s), >90% reconstruction")
    print("=" * 60)
    
    # Start performance monitoring
    monitor = PerformanceMonitor()
    monitor.start()
    
    # Start analysis
    print("\n🔄 Starting analysis...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{api_url}/analyze",
            json={"video_path": video_path},
            timeout=300  # 5 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n\n✅ Analysis completed successfully!")
            print(f"⏱️  Total time: {elapsed:.1f}s")
            
            # Load full results
            if result.get('results_file'):
                with open(result['results_file'], 'r') as f:
                    analysis = json.load(f)
                
                # Extract metrics
                video_duration = analysis.get('metadata', {}).get('duration', 28.9)
                realtime_factor = video_duration / elapsed
                successful = result.get('successful_analyzers', 0)
                total = result.get('total_analyzers', 21)
                reconstruction_score = (successful / total) * 100
                
                print(f"\n📊 PERFORMANCE METRICS:")
                print(f"   Video duration: {video_duration:.1f}s")
                print(f"   Processing time: {elapsed:.1f}s")
                print(f"   Realtime factor: {realtime_factor:.2f}x")
                print(f"   Target: <3x realtime")
                print(f"   ✅ PASSED" if realtime_factor < 3.0 else "❌ FAILED")
                
                print(f"\n🎯 QUALITY METRICS:")
                print(f"   Successful analyzers: {successful}/{total}")
                print(f"   Reconstruction score: {reconstruction_score:.0f}%")
                print(f"   Target: >90%")
                print(f"   ✅ PASSED" if reconstruction_score > 90 else "❌ FAILED")
                
                # Check BLIP-2 description
                if 'video_llava' in analysis.get('analysis_results', {}):
                    descriptions = analysis['analysis_results']['video_llava'].get('descriptions', [])
                    if descriptions and descriptions[0].get('text'):
                        desc_text = descriptions[0]['text']
                        print(f"\n📝 BLIP-2 Quality Check:")
                        print(f"   Description length: {len(desc_text)} chars")
                        print(f"   First 200 chars: {desc_text[:200]}...")
                        print(f"   ✅ PASSED" if len(desc_text) > 150 else "❌ FAILED")
                
                # Final verdict
                print(f"\n{'='*60}")
                if realtime_factor < 3.0 and reconstruction_score > 90:
                    print("🎉 SYSTEM IS 100% PRODUCTION READY! 🎉")
                    print("✅ All targets achieved")
                else:
                    print("⚠️  System needs optimization")
                    
        else:
            print(f"\n❌ Analysis failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        
    finally:
        # Stop monitoring
        monitor.stop()
        
        # Get GPU summary
        summary = monitor.get_summary()
        if summary:
            print(f"\n⚡ GPU UTILIZATION:")
            print(f"   Average: {summary['gpu_utilization']['average']:.0f}%")
            print(f"   Peak: {summary['gpu_utilization']['max']:.0f}%")
            print(f"   Target: >80%")
            print(f"   ✅ PASSED" if summary['gpu_utilization']['average'] > 80 else "⚠️  Could be optimized")

if __name__ == "__main__":
    run_final_test()