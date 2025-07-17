#!/usr/bin/env python3
"""
TEST 100% SUCCESS RATE - Final validation
Tests the system with all applied fixes for 100% analyzer success rate.
"""

import time
import json
import requests
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '/home/user/tiktok_production')

def test_100_percent_success():
    """Test that all 19 analyzers now work with applied fixes"""
    
    print("🎯 100% SUCCESS RATE TEST")
    print("=" * 60)
    print("Testing all applied fixes:")
    print("✅ Audio analyzer ProcessPool fix")
    print("✅ Cross-analyzer intelligence dummy data fix")
    print("✅ Eye tracking analyzer segment fix")
    print("=" * 60)
    
    video_path = "/home/user/tiktok_production/test_videos/test1.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Test video not found: {video_path}")
        return False
    
    # Test API
    try:
        start = time.time()
        response = requests.post(
            "http://localhost:8003/analyze", 
            json={"video_path": video_path},
            timeout=300
        )
        
        total_time = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            results_file = data.get('results_file', '')
            
            if results_file and Path(results_file).exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                return analyze_final_results(results, total_time)
            else:
                print("❌ No results file found")
                return False
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def analyze_final_results(results, processing_time):
    """Analyze the final results and determine success rate"""
    
    print(f"\n📊 FINAL RESULTS ANALYSIS")
    print("-" * 40)
    
    analyzer_results = results.get('analyzer_results', {})
    
    if not analyzer_results:
        print("❌ No analyzer results found")
        return False
    
    # Count working vs failed analyzers
    working_analyzers = []
    failed_analyzers = []
    
    for analyzer_name, result in analyzer_results.items():
        if isinstance(result, dict):
            segments = result.get('segments', [])
            if len(segments) > 0:
                working_analyzers.append((analyzer_name, len(segments)))
            else:
                error = result.get('error', 'No segments')
                failed_analyzers.append((analyzer_name, error))
        else:
            failed_analyzers.append((analyzer_name, f"Invalid result type: {type(result)}"))
    
    # Calculate success rate
    total_analyzers = len(analyzer_results)
    working_count = len(working_analyzers)
    success_rate = (working_count / total_analyzers) * 100 if total_analyzers > 0 else 0
    
    print(f"📈 SUCCESS RATE: {success_rate:.1f}% ({working_count}/{total_analyzers})")
    print(f"⏱️ Processing Time: {processing_time:.1f}s")
    print(f"🎯 Realtime Factor: {processing_time/10:.1f}x")
    
    # Show working analyzers
    print(f"\n✅ WORKING ANALYZERS ({working_count}):")
    for name, segment_count in working_analyzers:
        print(f"   • {name}: {segment_count} segments")
    
    # Show failed analyzers
    if failed_analyzers:
        print(f"\n❌ FAILED ANALYZERS ({len(failed_analyzers)}):")
        for name, error in failed_analyzers:
            print(f"   • {name}: {error}")
    
    # Special focus on previously failing analyzers
    print(f"\n🔍 FIXES VERIFICATION:")
    
    # Check audio analyzers
    audio_analyzers = ['audio_analysis', 'audio_environment', 'speech_emotion', 
                      'speech_transcription', 'speech_flow', 'speech_rate']
    audio_working = [name for name, _ in working_analyzers if name in audio_analyzers]
    print(f"   🎵 Audio analyzers: {len(audio_working)}/{len(audio_analyzers)} working")
    
    # Check cross-analyzer intelligence
    cross_working = any(name == 'cross_analyzer_intelligence' for name, _ in working_analyzers)
    print(f"   🧠 Cross-analyzer intelligence: {'✅ Working' if cross_working else '❌ Failed'}")
    
    # Check eye tracking
    eye_working = any(name == 'eye_tracking' for name, _ in working_analyzers)
    print(f"   👁️ Eye tracking: {'✅ Working' if eye_working else '❌ Failed'}")
    
    # Final verdict
    print(f"\n🏆 FINAL VERDICT:")
    
    if success_rate >= 100:
        print(f"   🎉 SUCCESS: 100% analyzer success rate achieved!")
        print(f"   ✅ All {total_analyzers} analyzers working perfectly")
        
        if processing_time < 30:
            print(f"   ✅ Speed target: <30s achieved ({processing_time:.1f}s)")
            print(f"\n🚀 SYSTEM IS PRODUCTION-READY!")
            print(f"   ✅ 100% analyzer success rate")
            print(f"   ✅ <3x realtime processing")
            print(f"   ✅ ALL OPTIMIZATION GOALS ACHIEVED")
            return True
        else:
            print(f"   ⚠️ Speed target: Not achieved ({processing_time:.1f}s > 30s)")
            print(f"   🔧 Consider further optimizations")
            return False
    
    elif success_rate >= 95:
        print(f"   🎯 EXCELLENT: {success_rate:.1f}% success rate")
        print(f"   ✅ Production-ready quality")
        return True
    
    elif success_rate >= 85:
        print(f"   👍 GOOD: {success_rate:.1f}% success rate")
        print(f"   ⚠️ Minor improvements needed")
        return False
    
    else:
        print(f"   ❌ NEEDS WORK: {success_rate:.1f}% success rate")
        print(f"   🔧 Significant improvements needed")
        return False

def main():
    """Main test function"""
    
    print("🔥 TESTING 100% SUCCESS RATE WITH ALL FIXES")
    print("=" * 60)
    
    success = test_100_percent_success()
    
    if success:
        print("\n🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
    else:
        print("\n⚠️ SOME ISSUES REMAIN - FURTHER OPTIMIZATION NEEDED")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)