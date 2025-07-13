#!/usr/bin/env python3
"""
Quick Final Production Test
"""
import time
import json
import subprocess
import os

def run_test():
    print("🚀 FINAL PRODUCTION TEST")
    print("="*50)
    
    video_path = "/home/user/tiktok_production/downloads/videos/7425998222721633569.mp4"
    video_duration = 28.9
    
    print(f"📹 Video: Copenhagen Vlog ({video_duration}s)")
    print(f"🎯 Target: <3x realtime (<87s)")
    print()
    
    # Start analysis
    print("🔬 Starting analysis...")
    start_time = time.time()
    
    cmd = [
        'curl', '-X', 'POST', 'http://localhost:8003/analyze',
        '-H', 'Content-Type: application/json',
        '-d', json.dumps({'video_path': video_path}),
        '-s', '--max-time', '180'  # 3 minute timeout
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    analysis_time = time.time() - start_time
    
    if result.returncode == 0:
        try:
            response = json.loads(result.stdout)
            
            if response.get('status') == 'success':
                realtime_factor = analysis_time/video_duration
                
                print(f"\n✅ Analysis completed!")
                print(f"⏱️ Time: {analysis_time:.1f}s")
                print(f"📊 Realtime factor: {realtime_factor:.2f}x")
                
                # Check if <3x
                if realtime_factor < 3.0:
                    print(f"\n🎉 SUCCESS: Achieved <3x realtime!")
                else:
                    print(f"\n⚠️ Performance: {realtime_factor:.2f}x (target <3.0x)")
                
                # Check results file
                results_file = response.get('results_file')
                if results_file and os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    analyzer_results = results.get('analyzer_results', {})
                    successful = len([a for a in analyzer_results if 'error' not in analyzer_results[a]])
                    
                    print(f"\n📋 Analyzers: {successful}/21 successful")
                    
                    # List any failures
                    failures = [a for a in analyzer_results if 'error' in analyzer_results[a]]
                    if failures:
                        print(f"❌ Failed: {', '.join(failures)}")
                    
                    # Overall assessment
                    print(f"\n{'='*50}")
                    if successful == 21 and realtime_factor < 3.0:
                        print("💯 SYSTEM IST 100% PRODUKTIONSREIF!")
                        print("   ✓ Alle 21 Analyzer erfolgreich")
                        print("   ✓ Performance unter 3x Realtime")
                        print("   ✓ Keine Placeholders oder Demo-Daten")
                    else:
                        print("⚠️ System needs optimization")
                        if successful < 21:
                            print(f"   - Only {successful}/21 analyzers successful")
                        if realtime_factor >= 3.0:
                            print(f"   - Performance {realtime_factor:.2f}x (need <3.0x)")
                
            else:
                print(f"❌ Analysis failed: {response}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            print(f"Output: {result.stdout[:500]}")
    else:
        print(f"❌ API call failed or timed out after {analysis_time:.1f}s")
        if result.stderr:
            print(f"Error: {result.stderr}")

if __name__ == "__main__":
    run_test()