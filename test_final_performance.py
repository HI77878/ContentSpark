#!/usr/bin/env python3
"""
FINAL PERFORMANCE TEST - Von 80% auf 100% Erfolgsrate
Ziel: <30s für 10s Video und >95% Analyzer-Erfolgsrate
"""

import time
import json
import requests
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '/home/user/tiktok_production')

def test_optimized_performance():
    """Test ob wir <30s für 10s Video erreichen"""
    
    print("🚀 FINAL PERFORMANCE TEST")
    print("="*60)
    print("🎯 TARGET: <30s für 10s Video + >95% Analyzer Success")
    print("="*60)
    
    video_path = "/home/user/tiktok_production/test_videos/test1.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Test video not found: {video_path}")
        return
    
    # Test 1: Full Pipeline via API
    print("\n1️⃣ TESTING FULL PIPELINE VIA API")
    print("-" * 40)
    
    try:
        start = time.time()
        response = requests.post(
            "http://localhost:8003/analyze", 
            json={"video_path": video_path},
            timeout=300  # 5 minute timeout
        )
        
        total_time = time.time() - start
        
        print(f"📊 API Response Status: {response.status_code}")
        print(f"⏱️ Total Processing Time: {total_time:.1f}s")
        print(f"🎯 Realtime Factor: {total_time/10:.1f}x")
        
        if response.status_code == 200:
            try:
                data = response.json()
                
                # Check basic response
                print(f"✅ API Response: {data.get('status', 'unknown')}")
                
                # Check results file
                results_file = data.get('results_file', '')
                if results_file and Path(results_file).exists():
                    print(f"📁 Results File: {results_file}")
                    
                    # Load and analyze results
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    analyze_results(results, total_time)
                    
                else:
                    print("❌ No results file found")
                    
            except json.JSONDecodeError:
                print("❌ Invalid JSON response")
                print(f"Response: {response.text[:200]}...")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (>5 minutes)")
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - Is API running?")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("\n" + "="*60)

def analyze_results(results, processing_time):
    """Analyze the results from the API"""
    
    print(f"\n2️⃣ ANALYZING RESULTS")
    print("-" * 40)
    
    # Get analyzer results
    analyzer_results = results.get('analyzer_results', {})
    
    if not analyzer_results:
        print("❌ No analyzer results found")
        return
    
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
    
    print(f"📊 ANALYZER PERFORMANCE:")
    print(f"   Total Analyzers: {total_analyzers}")
    print(f"   Working: {working_count}")
    print(f"   Failed: {len(failed_analyzers)}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    # Show working analyzers
    print(f"\n✅ WORKING ANALYZERS ({working_count}):")
    for name, segment_count in working_analyzers:
        print(f"   • {name}: {segment_count} segments")
    
    # Show failed analyzers
    if failed_analyzers:
        print(f"\n❌ FAILED ANALYZERS ({len(failed_analyzers)}):")
        for name, error in failed_analyzers:
            print(f"   • {name}: {error}")
    
    # Performance evaluation
    print(f"\n🎯 PERFORMANCE EVALUATION:")
    print(f"   Processing Time: {processing_time:.1f}s")
    print(f"   Realtime Factor: {processing_time/10:.1f}x")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    # Final verdict
    print(f"\n🏆 FINAL VERDICT:")
    if processing_time < 30:
        print(f"   ✅ Speed Target: <30s achieved ({processing_time:.1f}s)")
    else:
        print(f"   ❌ Speed Target: Failed ({processing_time:.1f}s > 30s)")
    
    if success_rate >= 95:
        print(f"   ✅ Success Target: >95% achieved ({success_rate:.1f}%)")
    else:
        print(f"   ❌ Success Target: Failed ({success_rate:.1f}% < 95%)")
    
    if processing_time < 30 and success_rate >= 95:
        print(f"\n🎉 SYSTEM IS PRODUCTION-READY!")
        print(f"   ✅ <3x realtime processing")
        print(f"   ✅ >95% analyzer success rate")
        print(f"   ✅ All major goals achieved")
    else:
        print(f"\n⚠️  SYSTEM NEEDS MORE OPTIMIZATION")
        if processing_time >= 30:
            print(f"   • Speed improvement needed")
        if success_rate < 95:
            print(f"   • Analyzer reliability improvement needed")

def test_individual_components():
    """Test individual components separately"""
    
    print("\n3️⃣ TESTING INDIVIDUAL COMPONENTS")
    print("-" * 40)
    
    video_path = "/home/user/tiktok_production/test_videos/test1.mp4"
    
    # Test Audio Analyzer (already fixed)
    print("🎵 Testing Audio Analyzer Fix...")
    try:
        from utils.audio_processing_fix import AudioAnalysisUltimateFixed
        analyzer = AudioAnalysisUltimateFixed()
        
        start = time.time()
        result = analyzer.analyze(video_path)
        elapsed = time.time() - start
        
        segments = len(result.get('segments', []))
        print(f"   ✅ Audio Analyzer: {segments} segments in {elapsed:.1f}s")
        
    except Exception as e:
        print(f"   ❌ Audio Analyzer failed: {e}")
    
    # Test Cross-Analyzer Intelligence
    print("🧠 Testing Cross-Analyzer Intelligence...")
    try:
        from analyzers.cross_analyzer_intelligence import CrossAnalyzerIntelligence
        analyzer = CrossAnalyzerIntelligence()
        
        # Mock some analyzer outputs
        mock_outputs = {
            'qwen2_vl_temporal': {
                'segments': [
                    {'start_time': 0, 'end_time': 2, 'description': 'Test scene'},
                    {'start_time': 2, 'end_time': 4, 'description': 'Another scene'}
                ]
            },
            'audio_analysis': {
                'segments': [
                    {'start_time': 0, 'end_time': 1, 'spectral_centroid': 1000}
                ]
            }
        }
        
        start = time.time()
        result = analyzer.analyze(mock_outputs)
        elapsed = time.time() - start
        
        segments = len(result.get('segments', []))
        print(f"   ✅ Cross-Analyzer: {segments} segments in {elapsed:.1f}s")
        
    except Exception as e:
        print(f"   ❌ Cross-Analyzer failed: {e}")

def main():
    """Main test function"""
    
    print("🔧 FINAL SYSTEM OPTIMIZATION TEST")
    print("From 80% to 100% Success Rate + <3x Realtime")
    print("=" * 60)
    
    # Test components
    test_individual_components()
    
    # Test full pipeline
    test_optimized_performance()
    
    print("\n🎯 TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()