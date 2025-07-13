#!/usr/bin/env python3
"""
Final Performance Monitor for Production Test
"""
import subprocess
import time
import threading
import json
from datetime import datetime
import numpy as np

class PerformanceMonitor:
    def __init__(self):
        self.monitoring = True
        self.gpu_samples = []
        self.cpu_samples = []
        self.memory_samples = []
        
    def start(self):
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        
    def stop(self):
        self.monitoring = False
        self.thread.join()
        return self.get_stats()
        
    def _monitor(self):
        while self.monitoring:
            try:
                # GPU monitoring
                gpu_result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True
                )
                if gpu_result.returncode == 0:
                    parts = gpu_result.stdout.strip().split(', ')
                    if len(parts) == 2:
                        self.gpu_samples.append(int(parts[0]))
                        self.memory_samples.append(int(parts[1]))
                
                # CPU monitoring
                cpu_result = subprocess.run(
                    ['top', '-bn1'], capture_output=True, text=True
                )
                if cpu_result.returncode == 0:
                    for line in cpu_result.stdout.split('\n'):
                        if '%Cpu' in line:
                            # Extract CPU usage
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if 'id,' in part and i > 0:
                                    idle = float(parts[i-1])
                                    cpu_used = 100.0 - idle
                                    self.cpu_samples.append(cpu_used)
                                    break
                            break
                            
            except Exception as e:
                print(f"Monitor error: {e}")
                
            time.sleep(0.5)
    
    def get_stats(self):
        stats = {
            'timestamp': datetime.now().isoformat(),
            'samples': len(self.gpu_samples)
        }
        
        if self.gpu_samples:
            stats['gpu'] = {
                'average': np.mean(self.gpu_samples),
                'peak': max(self.gpu_samples),
                'min': min(self.gpu_samples),
                'std_dev': np.std(self.gpu_samples)
            }
            
        if self.memory_samples:
            stats['gpu_memory_mb'] = {
                'average': np.mean(self.memory_samples),
                'peak': max(self.memory_samples)
            }
            
        if self.cpu_samples:
            stats['cpu'] = {
                'average': np.mean(self.cpu_samples),
                'peak': max(self.cpu_samples)
            }
            
        return stats

if __name__ == "__main__":
    print("Performance Monitor started...")
    monitor = PerformanceMonitor()
    monitor.start()
    
    try:
        # Wait for interrupt
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        stats = monitor.stop()
        
        print("\n=== PERFORMANCE STATISTICS ===")
        print(json.dumps(stats, indent=2))
        
        # Save to file
        with open('final_performance_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)