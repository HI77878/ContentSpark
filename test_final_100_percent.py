#!/usr/bin/env python3
"""
FINALER 100% TEST - Alle Analyzer müssen funktionieren!
Nach Qwen2-VL Optimierung (11s statt 128s)
"""
import time
import json
import requests
import torch

print("🚀 FINALER 100% PIPELINE TEST")
print("="*70)
print("Nach erfolgreicher Qwen2-VL Optimierung: 11s statt 128s!")
print("="*70)

# GPU Status
print(f"\n📊 GPU Status:")
if torch.cuda.is_available():
    print(f"   Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB allocated")
    print(f"   Device: {torch.cuda.get_device_name()}")
    torch.cuda.empty_cache()

# Test full pipeline
video_path = "/home/user/tiktok_production/test_videos/test1.mp4"
print(f"\n🎬 Testing Full Pipeline with: {video_path}")
print("   Video: 10 seconds test pattern")
print("   Target: <30s processing time (3x realtime)")

# API Test
start_time = time.time()
print("\n⏳ Sending request to API...")

response = requests.post(
    "http://localhost:8003/analyze",
    json={"video_path": video_path},
    timeout=300
)

elapsed = time.time() - start_time

if response.status_code == 200:
    data = response.json()
    print(f"\n✅ API Response received in {elapsed:.1f}s")
    
    # Load detailed results
    if 'results_file' in data:
        with open(data['results_file'], 'r') as f:
            results = json.load(f)
        
        analyzer_results = results.get('analyzer_results', {})
        
        # Detailed analysis
        print(f"\n📊 ANALYZER RESULTS:")
        print("-"*70)
        
        working = []
        failed = []
        slow_analyzers = []
        
        for name, result in sorted(analyzer_results.items()):
            segments = len(result.get('segments', []))
            proc_time = result.get('metadata', {}).get('processing_time', 0)
            
            if segments > 0:
                working.append(name)
                status = "✅"
                
                # Check if slow
                if proc_time > 20:
                    slow_analyzers.append((name, proc_time))
                    status = "⚠️"
                    
                print(f"{status} {name}: {segments} segments ({proc_time:.1f}s)")
            else:
                failed.append(name)
                error = result.get('error', 'No segments')
                print(f"❌ {name}: FAILED - {error}")
        
        # Summary
        total = len(analyzer_results)
        success_rate = (len(working)/total)*100 if total > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"🎯 FINAL RESULTS:")
        print(f"   Total Analyzers: {total}")
        print(f"   Working: {len(working)}")
        print(f"   Failed: {len(failed)}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Total Time: {elapsed:.1f}s")
        print(f"   Realtime Factor: {elapsed/10:.1f}x")
        
        # Performance breakdown
        if 'processing_time' in data:
            print(f"\n⏱️  PERFORMANCE BREAKDOWN:")
            print(f"   API Processing: {data['processing_time']:.1f}s")
            
            # Find Qwen2-VL time
            qwen_time = analyzer_results.get('qwen2_vl_temporal', {}).get('metadata', {}).get('processing_time', 0)
            if qwen_time > 0:
                print(f"   Qwen2-VL: {qwen_time:.1f}s (was 128s)")
                print(f"   Other Analyzers: {data['processing_time'] - qwen_time:.1f}s")
        
        # Check goals
        print(f"\n🎯 GOAL CHECK:")
        if success_rate >= 95:
            print(f"   ✅ Success Rate: {success_rate:.1f}% >= 95%")
        else:
            print(f"   ❌ Success Rate: {success_rate:.1f}% < 95%")
            
        if elapsed < 30:
            print(f"   ✅ Performance: {elapsed:.1f}s < 30s (3x realtime)")
        else:
            print(f"   ❌ Performance: {elapsed:.1f}s >= 30s")
        
        # Final verdict
        if success_rate >= 95 and elapsed < 30:
            print(f"\n🎉🎉🎉 SYSTEM IST PRODUCTION READY! 🎉🎉🎉")
            print(f"   ✅ {success_rate:.1f}% Success Rate")
            print(f"   ✅ {elapsed/10:.1f}x Realtime")
            print(f"   ✅ Qwen2-VL optimiert auf {qwen_time:.1f}s")
        else:
            print(f"\n⚠️  Fast am Ziel!")
            if failed:
                print(f"   Noch zu fixen: {', '.join(failed)}")
            if slow_analyzers:
                print(f"   Langsame Analyzer:")
                for name, time in slow_analyzers:
                    print(f"     - {name}: {time:.1f}s")
                    
else:
    print(f"\n❌ API Error: {response.status_code}")
    print(response.text[:500])

# GPU Status nach Test
print(f"\n📊 GPU Status nach Test:")
if torch.cuda.is_available():
    print(f"   Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB allocated")
    torch.cuda.empty_cache()

print("="*70)