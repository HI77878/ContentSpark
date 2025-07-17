#!/usr/bin/env python3
"""Monitor system performance during analysis"""
import subprocess
import time
import threading
import json
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.monitoring = True
        self.gpu_samples = []
        self.cpu_samples = []
        self.start_time = None
        
    def start(self):
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Performance monitoring started...")
        
    def stop(self):
        self.monitoring = False
        self.thread.join()
        duration = time.time() - self.start_time
        
        # Calculate stats
        avg_gpu = sum(self.gpu_samples) / len(self.gpu_samples) if self.gpu_samples else 0
        max_gpu = max(self.gpu_samples) if self.gpu_samples else 0
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        
        print(f"\n=== PERFORMANCE SUMMARY ===")
        print(f"Duration: {duration:.1f} seconds")
        print(f"GPU Average: {avg_gpu:.1f}%")
        print(f"GPU Peak: {max_gpu}%")
        print(f"CPU Average: {avg_cpu:.1f}%")
        print(f"Samples: {len(self.gpu_samples)}")
        
        return {
            'duration': duration,
            'gpu_avg': avg_gpu,
            'gpu_peak': max_gpu,
            'cpu_avg': avg_cpu
        }
        
    def _monitor_loop(self):
        while self.monitoring:
            try:
                # GPU monitoring
                gpu_result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True
                )
                if gpu_result.returncode == 0:
                    gpu_util = int(gpu_result.stdout.strip())
                    self.gpu_samples.append(gpu_util)
                    
                    # Print high GPU usage
                    if gpu_util > 50:
                        elapsed = time.time() - self.start_time
                        print(f"[{elapsed:6.1f}s] GPU: {gpu_util}%")
                
                # Simple CPU check
                load_result = subprocess.run(['uptime'], capture_output=True, text=True)
                if load_result.returncode == 0:
                    # Extract 1-minute load average
                    parts = load_result.stdout.split('load average:')
                    if len(parts) > 1:
                        load1 = float(parts[1].split(',')[0].strip())
                        cpu_percent = min(load1 * 100 / 12, 100)  # 12 threads
                        self.cpu_samples.append(cpu_percent)
                        
            except:
                pass
                
            time.sleep(1)

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.start()
    
    try:
        # Keep monitoring until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        stats = monitor.stop()
        
        # Save stats
        with open('performance_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
            print(f"\nStats saved to performance_stats.json")