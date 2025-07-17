#!/usr/bin/env python3
"""
Real-time Performance Monitor for Video Analysis
Monitors GPU, CPU, Memory, and Analyzer Performance
"""
import time
import psutil
import GPUtil
import torch
import threading
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class PerformanceMonitor:
    def __init__(self, log_dir="performance_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.monitoring = False
        self.stats = {
            'timestamps': [],
            'gpu_usage': [],
            'gpu_memory': [],
            'cpu_usage': [],
            'ram_usage': [],
            'analyzer_times': {},
            'total_time': 0
        }
        
        self.start_time = None
        self.monitor_thread = None
        
    def start(self):
        """Start monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        print("🔍 Performance monitoring started")
        
    def stop(self):
        """Stop monitoring and save results"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        self.stats['total_time'] = time.time() - self.start_time
        self._save_results()
        self._generate_report()
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            timestamp = time.time() - self.start_time
            
            # GPU stats
            if torch.cuda.is_available():
                gpu = GPUtil.getGPUs()[0]
                gpu_usage = gpu.load * 100
                gpu_memory = gpu.memoryUtil * 100
            else:
                gpu_usage = 0
                gpu_memory = 0
            
            # CPU stats
            cpu_usage = psutil.cpu_percent(interval=0.1)
            ram_usage = psutil.virtual_memory().percent
            
            # Store stats
            self.stats['timestamps'].append(timestamp)
            self.stats['gpu_usage'].append(gpu_usage)
            self.stats['gpu_memory'].append(gpu_memory)
            self.stats['cpu_usage'].append(cpu_usage)
            self.stats['ram_usage'].append(ram_usage)
            
            # Print live stats
            print(f"\r⚡ GPU: {gpu_usage:.1f}% | VRAM: {gpu_memory:.1f}% | CPU: {cpu_usage:.1f}% | RAM: {ram_usage:.1f}%", end='')
            
            time.sleep(0.5)  # Monitor every 500ms
    
    def log_analyzer(self, name: str, duration: float):
        """Log analyzer performance"""
        if name not in self.stats['analyzer_times']:
            self.stats['analyzer_times'][name] = []
        self.stats['analyzer_times'][name].append(duration)
    
    def _save_results(self):
        """Save monitoring results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw data
        json_path = self.log_dir / f"performance_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Generate plots
        self._generate_plots(timestamp)
        
    def _generate_plots(self, timestamp):
        """Generate performance plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # GPU Usage
        ax1.plot(self.stats['timestamps'], self.stats['gpu_usage'], 'b-')
        ax1.set_title('GPU Usage Over Time')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('GPU Usage (%)')
        ax1.grid(True)
        ax1.axhline(y=80, color='r', linestyle='--', label='Target 80%')
        
        # GPU Memory
        ax2.plot(self.stats['timestamps'], self.stats['gpu_memory'], 'g-')
        ax2.set_title('GPU Memory Usage Over Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('VRAM Usage (%)')
        ax2.grid(True)
        ax2.axhline(y=85, color='r', linestyle='--', label='Threshold 85%')
        
        # CPU Usage
        ax3.plot(self.stats['timestamps'], self.stats['cpu_usage'], 'r-')
        ax3.set_title('CPU Usage Over Time')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('CPU Usage (%)')
        ax3.grid(True)
        
        # Analyzer Times
        if self.stats['analyzer_times']:
            analyzers = list(self.stats['analyzer_times'].keys())
            times = [np.mean(self.stats['analyzer_times'][a]) for a in analyzers]
            
            ax4.barh(analyzers, times)
            ax4.set_title('Average Analyzer Execution Time')
            ax4.set_xlabel('Time (s)')
            ax4.grid(True, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.log_dir / f"performance_plot_{timestamp}.png", dpi=150)
        plt.close()
    
    def _generate_report(self):
        """Generate performance report"""
        print("\n\n" + "="*60)
        print("📊 PERFORMANCE ANALYSIS REPORT")
        print("="*60)
        
        # Overall stats
        print(f"\n⏱️  Total Processing Time: {self.stats['total_time']:.1f}s")
        
        # GPU stats
        if self.stats['gpu_usage']:
            avg_gpu = np.mean(self.stats['gpu_usage'])
            max_gpu = np.max(self.stats['gpu_usage'])
            avg_vram = np.mean(self.stats['gpu_memory'])
            max_vram = np.max(self.stats['gpu_memory'])
            
            print(f"\n🎮 GPU Performance:")
            print(f"   Average Usage: {avg_gpu:.1f}%")
            print(f"   Peak Usage: {max_gpu:.1f}%")
            print(f"   Average VRAM: {avg_vram:.1f}%")
            print(f"   Peak VRAM: {max_vram:.1f}%")
        
        # CPU stats
        if self.stats['cpu_usage']:
            avg_cpu = np.mean(self.stats['cpu_usage'])
            max_cpu = np.max(self.stats['cpu_usage'])
            
            print(f"\n💻 CPU Performance:")
            print(f"   Average Usage: {avg_cpu:.1f}%")
            print(f"   Peak Usage: {max_cpu:.1f}%")
        
        # Analyzer stats
        if self.stats['analyzer_times']:
            print(f"\n🔍 Analyzer Performance:")
            total_analyzer_time = 0
            for name, times in sorted(self.stats['analyzer_times'].items(), 
                                     key=lambda x: np.mean(x[1]), reverse=True):
                avg_time = np.mean(times)
                total_analyzer_time += avg_time
                print(f"   {name:30s}: {avg_time:6.1f}s")
            
            print(f"\n   Total Analyzer Time: {total_analyzer_time:.1f}s")
            print(f"   Parallelization Efficiency: {total_analyzer_time/self.stats['total_time']:.1f}x")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    # Test monitoring
    monitor = PerformanceMonitor()
    monitor.start()
    
    # Simulate work
    time.sleep(10)
    
    monitor.log_analyzer("test_analyzer", 5.2)
    monitor.stop()