#!/usr/bin/env python3
"""
Echtzeit-Monitoring Dashboard für GPU-Optimierungstest
"""

import time
import subprocess
import psutil
import threading
import json
from datetime import datetime
import os
import re

class RealtimeMonitor:
    def __init__(self):
        self.metrics = {
            "timestamps": [],
            "gpu_usage": [],
            "gpu_memory": [],
            "cpu_usage": [],
            "analyzer_times": {},
            "model_loads": [],
            "cache_hits": 0
        }
        self.monitoring = True
        self.start_time = time.time()
        
    def get_gpu_stats(self):
        """Holt GPU-Statistiken via nvidia-smi"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            parts = result.stdout.strip().split(", ")
            return {
                "usage": float(parts[0]),
                "memory_used": float(parts[1]),
                "memory_total": float(parts[2])
            }
        except:
            return {"usage": 0, "memory_used": 0, "memory_total": 45541}
    
    def monitor_system(self):
        """Sammelt System-Metriken jede Sekunde"""
        print("🔍 Starting real-time monitoring...")
        print("Time    | GPU Usage | GPU Memory | CPU Usage | Cache Hits")
        print("-" * 60)
        
        while self.monitoring:
            timestamp = datetime.now().isoformat()
            elapsed = time.time() - self.start_time
            
            # GPU Metriken
            gpu_stats = self.get_gpu_stats()
            gpu_usage = gpu_stats["usage"]
            gpu_memory = gpu_stats["memory_used"]
            
            # CPU Metriken
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Speichere Metriken
            self.metrics["timestamps"].append(timestamp)
            self.metrics["gpu_usage"].append(gpu_usage)
            self.metrics["gpu_memory"].append(gpu_memory)
            self.metrics["cpu_usage"].append(cpu_usage)
            
            # Live-Anzeige
            print(f"{elapsed:6.0f}s | {gpu_usage:8.1f}% | {gpu_memory:8.0f}MB | {cpu_usage:8.1f}% | {self.metrics['cache_hits']:8d}")
            
            time.sleep(1)
    
    def analyze_logs(self, log_file='api_optimized.log'):
        """Analysiert API-Logs für Analyzer-Performance"""
        if not os.path.exists(log_file):
            return
            
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
                # Suche nach Model-Load Zeiten
                load_matches = re.findall(r'Loaded (\w+) in ([\d.]+)s', content)
                for analyzer, load_time in load_matches:
                    self.metrics["model_loads"].append({
                        "analyzer": analyzer,
                        "load_time": float(load_time)
                    })
                
                # Suche nach Cache-Hits
                cache_hits = re.findall(r'Reusing cached (\w+)', content)
                self.metrics["cache_hits"] = len(cache_hits)
                
                # Suche nach Analyzer-Zeiten
                analyzer_matches = re.findall(r'(\w+) completed in ([\d.]+)s', content)
                for analyzer, time_str in analyzer_matches:
                    if analyzer not in self.metrics["analyzer_times"]:
                        self.metrics["analyzer_times"][analyzer] = []
                    self.metrics["analyzer_times"][analyzer].append(float(time_str))
        except:
            pass
    
    def start_monitoring(self):
        """Startet Monitoring in Background Thread"""
        self.thread = threading.Thread(target=self.monitor_system)
        self.thread.daemon = True
        self.thread.start()
    
    def stop_monitoring(self):
        """Stoppt Monitoring"""
        self.monitoring = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
    
    def get_summary(self):
        """Erstellt Zusammenfassung der Metriken"""
        if not self.metrics["gpu_usage"]:
            return "No metrics collected"
            
        avg_gpu = sum(self.metrics["gpu_usage"]) / len(self.metrics["gpu_usage"])
        max_gpu = max(self.metrics["gpu_usage"])
        avg_memory = sum(self.metrics["gpu_memory"]) / len(self.metrics["gpu_memory"])
        max_memory = max(self.metrics["gpu_memory"])
        
        return {
            "monitoring_duration": time.time() - self.start_time,
            "samples": len(self.metrics["gpu_usage"]),
            "gpu_utilization_avg": avg_gpu,
            "gpu_utilization_max": max_gpu,
            "gpu_memory_avg_mb": avg_memory,
            "gpu_memory_max_mb": max_memory,
            "cache_hits": self.metrics["cache_hits"],
            "model_loads": len(self.metrics["model_loads"]),
            "analyzer_times": self.metrics["analyzer_times"]
        }
    
    def save_metrics(self, filename='monitoring_metrics.json'):
        """Speichert alle Metriken in Datei"""
        summary = self.get_summary()
        summary["raw_metrics"] = self.metrics
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Metrics saved to {filename}")

# Global monitor instance
monitor = RealtimeMonitor()

def start_monitoring():
    """Start monitoring"""
    monitor.start_monitoring()

def stop_monitoring():
    """Stop monitoring and save results"""
    monitor.stop_monitoring()
    monitor.analyze_logs()
    return monitor.get_summary()

if __name__ == "__main__":
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print("\n🛑 Stopping monitoring...")
        summary = stop_monitoring()
        monitor.save_metrics()
        print(f"📊 Final summary: {json.dumps(summary, indent=2)}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    start_monitoring()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)