#!/usr/bin/env python3
"""Performance Monitor for TikTok Production System"""

import time
import subprocess
import threading
import json
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.gpu_stats = []
        self.running = False
        self.monitor_thread = None
        
    def start(self):
        """Start monitoring"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        print("🔍 Performance monitoring started...")
        
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("🛑 Performance monitoring stopped")
        
    def _monitor_loop(self):
        """Monitor GPU stats"""
        while self.running:
            try:
                # Get GPU stats
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    parts = result.stdout.strip().split(', ')
                    if len(parts) == 3:
                        stats = {
                            'timestamp': time.time(),
                            'gpu_utilization': float(parts[0]),
                            'memory_used': float(parts[1]),
                            'memory_total': float(parts[2]),
                            'memory_utilization': (float(parts[1]) / float(parts[2])) * 100
                        }
                        self.gpu_stats.append(stats)
                        
                        # Print current stats
                        print(f"\r⚡ GPU: {stats['gpu_utilization']:.0f}% | "
                              f"Memory: {stats['memory_used']:.0f}/{stats['memory_total']:.0f} MB "
                              f"({stats['memory_utilization']:.0f}%)", end='', flush=True)
                
            except Exception as e:
                print(f"\nError monitoring: {e}")
                
            time.sleep(1)
    
    def get_summary(self):
        """Get performance summary"""
        if not self.gpu_stats:
            return None
            
        gpu_utils = [s['gpu_utilization'] for s in self.gpu_stats]
        mem_utils = [s['memory_utilization'] for s in self.gpu_stats]
        
        return {
            'monitoring_duration': self.gpu_stats[-1]['timestamp'] - self.gpu_stats[0]['timestamp'],
            'samples': len(self.gpu_stats),
            'gpu_utilization': {
                'average': sum(gpu_utils) / len(gpu_utils),
                'max': max(gpu_utils),
                'min': min(gpu_utils)
            },
            'memory_utilization': {
                'average': sum(mem_utils) / len(mem_utils),
                'max': max(mem_utils),
                'min': min(mem_utils)
            }
        }

if __name__ == "__main__":
    # Test monitoring
    monitor = PerformanceMonitor()
    monitor.start()
    
    try:
        print("Monitoring for 10 seconds...")
        time.sleep(10)
    finally:
        monitor.stop()
        
    summary = monitor.get_summary()
    if summary:
        print("\n\n📊 Performance Summary:")
        print(json.dumps(summary, indent=2))