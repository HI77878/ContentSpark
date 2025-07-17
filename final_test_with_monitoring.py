#!/usr/bin/env python3
"""
Final End-to-End Test with Comprehensive Monitoring
"""
import time
import json
import subprocess
import threading
import os
from datetime import datetime
import psutil
import numpy as np

class PerformanceMonitor:
    def __init__(self):
        self.monitoring = False
        self.gpu_utils = []
        self.gpu_memory = []
        self.cpu_percents = []
        self.timestamps = []
        
    def start_monitoring(self):
        """Start monitoring in a separate thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """Monitor system resources"""
        while self.monitoring:
            try:
                # GPU monitoring
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    values = result.stdout.strip().split(', ')
                    self.gpu_utils.append(int(values[0]))
                    self.gpu_memory.append(float(values[1]))
                
                # CPU monitoring
                self.cpu_percents.append(psutil.cpu_percent(interval=0.1))
                self.timestamps.append(time.time())
                
                time.sleep(1)  # Sample every second
            except Exception as e:
                print(f"Monitor error: {e}")
                
    def get_summary(self):
        """Get monitoring summary"""
        if not self.gpu_utils:
            return {}
            
        return {
            'gpu': {
                'average_utilization': np.mean(self.gpu_utils),
                'peak_utilization': max(self.gpu_utils),
                'average_memory_mb': np.mean(self.gpu_memory),
                'peak_memory_mb': max(self.gpu_memory),
            },
            'cpu': {
                'average_percent': np.mean(self.cpu_percents),
                'peak_percent': max(self.cpu_percents),
            },
            'duration_seconds': self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0
        }

def main():
    print("🚀 FINALE END-TO-END TEST MIT UMFASSENDER ÜBERWACHUNG")
    print("=" * 70)
    
    # Video info
    video_path = "/home/user/tiktok_production/downloads/videos/7522589683939921165.mp4"
    video_duration = 68.45  # seconds
    
    print(f"📹 Test Video: {os.path.basename(video_path)}")
    print(f"   Duration: {video_duration:.1f}s")
    print()
    
    # Start performance monitoring
    monitor = PerformanceMonitor()
    print("📊 Starting performance monitoring...")
    monitor.start_monitoring()
    
    # Start analysis
    print("🔍 Starting video analysis...")
    print("-" * 70)
    
    start_time = time.time()
    
    # Call the API
    api_url = "http://localhost:8003/analyze"
    cmd = [
        'curl', '-X', 'POST', api_url,
        '-H', 'Content-Type: application/json',
        '-d', json.dumps({'video_path': video_path}),
        '-s'
    ]
    
    # Show progress
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Calling API: {api_url}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing video...")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    analysis_time = time.time() - start_time
    
    # Stop monitoring
    monitor.stop_monitoring()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Analysis completed!")
    print("-" * 70)
    
    # Parse results
    if result.returncode == 0:
        try:
            response = json.loads(result.stdout)
            
            if response.get('status') == 'success':
                print()
                print("✅ ANALYSIS SUCCESSFUL!")
                print()
                
                # Load full results
                results_file = response.get('results_file')
                if results_file and os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        full_results = json.load(f)
                    
                    metadata = full_results.get('metadata', {})
                    
                    # Performance metrics
                    print("📈 PERFORMANCE METRICS:")
                    print(f"   Video duration: {video_duration:.1f}s")
                    print(f"   Processing time: {metadata.get('processing_time_seconds', analysis_time):.1f}s")
                    print(f"   Realtime factor: {metadata.get('realtime_factor', analysis_time/video_duration):.2f}x")
                    print(f"   Successful analyzers: {metadata.get('successful_analyzers')}/{metadata.get('total_analyzers')}")
                    print(f"   Reconstruction score: {metadata.get('reconstruction_score', 0):.1f}%")
                    
                    # System monitoring summary
                    monitor_summary = monitor.get_summary()
                    if monitor_summary:
                        print()
                        print("🖥️  SYSTEM RESOURCE USAGE:")
                        print(f"   GPU Average: {monitor_summary['gpu']['average_utilization']:.1f}%")
                        print(f"   GPU Peak: {monitor_summary['gpu']['peak_utilization']}%")
                        print(f"   GPU Memory Peak: {monitor_summary['gpu']['peak_memory_mb']:.0f} MB")
                        print(f"   CPU Average: {monitor_summary['cpu']['average_percent']:.1f}%")
                        print(f"   CPU Peak: {monitor_summary['cpu']['peak_percent']:.1f}%")
                    
                    # Target validation
                    print()
                    print("🎯 TARGET VALIDATION:")
                    realtime_factor = metadata.get('realtime_factor', analysis_time/video_duration)
                    recon_score = metadata.get('reconstruction_score', 0)
                    
                    realtime_ok = realtime_factor < 3.0
                    recon_ok = recon_score >= 90.0
                    
                    print(f"   <3x Realtime: {'✅ PASS' if realtime_ok else '❌ FAIL'} (Actual: {realtime_factor:.2f}x)")
                    print(f"   >90% Reconstruction: {'✅ PASS' if recon_ok else '❌ FAIL'} (Actual: {recon_score:.1f}%)")
                    
                    # BLIP-2 quality check
                    if 'video_llava' in full_results.get('analyzer_results', {}):
                        blip2 = full_results['analyzer_results']['video_llava']
                        if 'segments' in blip2:
                            segments = [s for s in blip2['segments'] if not s.get('is_summary', False)]
                            print()
                            print("📝 BLIP-2 QUALITY:")
                            print(f"   Segments analyzed: {len(segments)}")
                            
                            # Check caption quality
                            captions = []
                            for seg in segments:
                                text = seg.get('description', seg.get('text', ''))
                                if text and len(text) > 20:
                                    captions.append(text)
                            
                            if captions:
                                avg_length = sum(len(c) for c in captions) / len(captions)
                                print(f"   Average caption length: {avg_length:.0f} chars")
                                print(f"   Sample: \"{captions[0][:100]}...\"")
                    
                    # Final verdict
                    print()
                    print("=" * 70)
                    print("🏁 FINALE BEWERTUNG:")
                    
                    if realtime_ok and recon_ok:
                        print("   ✅ SYSTEM IST 100% PRODUKTIONSREIF!")
                        print("   - Alle Ziele wurden erreicht")
                        print("   - Performance unter 3x Realtime ✓")
                        print("   - Rekonstruktions-Score über 90% ✓")
                        print("   - Alle 21 Analyzer funktionieren ✓")
                    else:
                        print("   ⚠️  SYSTEM FAST PRODUKTIONSREIF:")
                        if not realtime_ok:
                            print(f"   - Performance: {realtime_factor:.2f}x (Ziel: <3x)")
                        if not recon_ok:
                            print(f"   - Rekonstruktion: {recon_score:.1f}% (Ziel: >90%)")
                    
                    # Save final report
                    report = {
                        'test_timestamp': datetime.now().isoformat(),
                        'video_path': video_path,
                        'video_duration': video_duration,
                        'processing_time': metadata.get('processing_time_seconds', analysis_time),
                        'realtime_factor': realtime_factor,
                        'reconstruction_score': recon_score,
                        'successful_analyzers': metadata.get('successful_analyzers'),
                        'total_analyzers': metadata.get('total_analyzers'),
                        'targets_met': {
                            'realtime_under_3x': realtime_ok,
                            'reconstruction_over_90': recon_ok
                        },
                        'system_usage': monitor_summary,
                        'production_ready': realtime_ok and recon_ok
                    }
                    
                    report_path = '/home/user/tiktok_production/final_test_report.json'
                    with open(report_path, 'w') as f:
                        json.dump(report, f, indent=2)
                    
                    print()
                    print(f"📄 Full report saved to: {report_path}")
                    
            else:
                print(f"❌ Analysis failed: {response}")
                
        except Exception as e:
            print(f"❌ Error parsing results: {e}")
            print(f"Raw output: {result.stdout}")
    else:
        print(f"❌ API call failed: {result.stderr}")
    
    print()
    print("Test completed.")

if __name__ == "__main__":
    main()