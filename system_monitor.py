#!/usr/bin/env python3
"""System monitoring during analysis"""
import psutil
import time
import threading
import json
from datetime import datetime

class SystemMonitor:
    def __init__(self):
        self.running = True
        self.stats = []
        
    def get_gpu_stats(self):
        """Get GPU stats using nvidia-smi"""
        try:
            import subprocess
            result = subprocess.check_output([
                'nvidia-smi', 
                '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ]).decode()
            
            values = result.strip().split(', ')
            return {
                'gpu_memory_mb': int(values[0]),
                'gpu_memory_total_mb': int(values[1]),
                'gpu_util': int(values[2]),
                'gpu_temp': int(values[3])
            }
        except:
            return {
                'gpu_memory_mb': 0,
                'gpu_memory_total_mb': 0,
                'gpu_util': 0,
                'gpu_temp': 0
            }
    
    def monitor(self):
        """Monitor system resources"""
        while self.running:
            gpu_stats = self.get_gpu_stats()
            
            stats = {
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().used / 1024**3,
                'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                'memory_percent': psutil.virtual_memory().percent,
                **gpu_stats
            }
            
            self.stats.append(stats)
            time.sleep(1)
    
    def start(self):
        """Start monitoring in background thread"""
        self.thread = threading.Thread(target=self.monitor)
        self.thread.daemon = True
        self.thread.start()
        return self
    
    def stop(self):
        """Stop monitoring and return stats"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        return self.stats
    
    def get_summary(self):
        """Get summary of monitoring stats"""
        if not self.stats:
            return {}
        
        cpu_values = [s['cpu_percent'] for s in self.stats]
        memory_values = [s['memory_gb'] for s in self.stats]
        gpu_util_values = [s['gpu_util'] for s in self.stats]
        gpu_memory_values = [s['gpu_memory_mb'] for s in self.stats]
        
        return {
            'duration_seconds': self.stats[-1]['timestamp'] - self.stats[0]['timestamp'],
            'samples': len(self.stats),
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory_gb': {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            },
            'gpu_util': {
                'avg': sum(gpu_util_values) / len(gpu_util_values),
                'max': max(gpu_util_values),
                'min': min(gpu_util_values)
            },
            'gpu_memory_mb': {
                'avg': sum(gpu_memory_values) / len(gpu_memory_values),
                'max': max(gpu_memory_values),
                'min': min(gpu_memory_values)
            }
        }
    
    def save_stats(self, filename):
        """Save stats to file"""
        with open(filename, 'w') as f:
            json.dump({
                'stats': self.stats,
                'summary': self.get_summary()
            }, f, indent=2)


if __name__ == "__main__":
    # Test monitoring
    print("Testing system monitor for 5 seconds...")
    monitor = SystemMonitor()
    monitor.start()
    
    time.sleep(5)
    
    stats = monitor.stop()
    print(f"Collected {len(stats)} samples")
    
    summary = monitor.get_summary()
    print("\nSummary:")
    print(f"CPU Average: {summary['cpu']['avg']:.1f}%")
    print(f"Memory Average: {summary['memory_gb']['avg']:.1f}GB")
    print(f"GPU Utilization Average: {summary['gpu_util']['avg']:.1f}%")
    print(f"GPU Memory Average: {summary['gpu_memory_mb']['avg']:.0f}MB")