#!/usr/bin/env python3
"""
Test-Script für GPU-Optimierungen
Vergleicht Performance mit Baseline
"""

import time
import json
import requests
import subprocess
from datetime import datetime
import threading
import csv

class GPUMonitor:
    """Monitor GPU usage during test"""
    def __init__(self):
        self.running = True
        self.samples = []
        
    def start(self):
        """Start monitoring in background thread"""
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        
    def stop(self):
        """Stop monitoring"""
        self.running = False
        self.thread.join()
        
    def _monitor(self):
        """Monitor GPU usage"""
        while self.running:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", 
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True
                )
                parts = result.stdout.strip().split(", ")
                self.samples.append({
                    "time": time.time(),
                    "gpu_util": float(parts[0]),
                    "memory_mb": float(parts[1])
                })
            except:
                pass
            time.sleep(1)
    
    def get_stats(self):
        """Get GPU statistics"""
        if not self.samples:
            return {}
            
        gpu_utils = [s["gpu_util"] for s in self.samples]
        mem_uses = [s["memory_mb"] for s in self.samples]
        
        return {
            "avg_gpu_utilization": sum(gpu_utils) / len(gpu_utils),
            "max_gpu_utilization": max(gpu_utils),
            "avg_memory_mb": sum(mem_uses) / len(mem_uses),
            "max_memory_mb": max(mem_uses),
            "samples": len(self.samples)
        }

def run_optimization_test():
    """Run test with optimizations"""
    print("🚀 Testing GPU Optimizations...")
    
    # Load baseline for comparison
    try:
        with open("baseline_test.json", "r") as f:
            baseline = json.load(f)
        print(f"📊 Baseline loaded: {baseline['processing_time']:.1f}s processing time")
    except:
        baseline = None
        print("⚠️  No baseline found")
    
    # Test video
    test_video = "https://www.tiktok.com/@leon_schliebach/video/7446489995663117590"
    
    # Check API
    health = requests.get("http://localhost:8003/health")
    if health.status_code != 200:
        print("❌ API not healthy!")
        return
    
    print(f"✅ API healthy: {health.json()['status']}")
    
    # Start GPU monitoring
    monitor = GPUMonitor()
    monitor.start()
    
    # Run analysis
    print(f"📊 Starting analysis...")
    start_time = time.time()
    
    response = requests.post(
        "http://localhost:8003/analyze",
        json={"tiktok_url": test_video},
        timeout=1200
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Stop monitoring
    monitor.stop()
    gpu_stats = monitor.get_stats()
    
    # Process results
    if response.status_code == 200:
        result = response.json()
        
        # Calculate metrics
        total_analyzers = result.get("metadata", {}).get("total_analyzers", 18)
        successful = result.get("metadata", {}).get("successful_analyzers", 0)
        reconstruction_score = (successful / total_analyzers) * 100 if total_analyzers > 0 else 0
        realtime_factor = result.get("metadata", {}).get("realtime_factor", "N/A")
        
        # Create optimization results
        optimization_results = {
            "timestamp": datetime.now().isoformat(),
            "test_video": test_video,
            "processing_time_seconds": processing_time,
            "reconstruction_score": reconstruction_score,
            "successful_analyzers": successful,
            "total_analyzers": total_analyzers,
            "realtime_factor": realtime_factor,
            "gpu_stats": gpu_stats,
            "improvements": {}
        }
        
        # Calculate improvements if baseline exists
        if baseline:
            baseline_time = baseline.get("processing_time_seconds", baseline.get("processing_time", 0))
            if baseline_time > 0:
                time_improvement = ((baseline_time - processing_time) / baseline_time) * 100
                optimization_results["improvements"]["time_reduction_percent"] = time_improvement
                optimization_results["improvements"]["speedup_factor"] = baseline_time / processing_time
                
            baseline_gpu = baseline.get("gpu_utilization_avg", 1.4)
            if gpu_stats.get("avg_gpu_utilization", 0) > 0:
                gpu_improvement = gpu_stats["avg_gpu_utilization"] / baseline_gpu
                optimization_results["improvements"]["gpu_utilization_factor"] = gpu_improvement
        
        # Save results
        with open("optimization_test_results.json", "w") as f:
            json.dump(optimization_results, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 OPTIMIZATION TEST RESULTS")
        print("="*60)
        print(f"⏱️  Processing Time: {processing_time:.1f}s")
        print(f"🎯 Reconstruction Score: {reconstruction_score:.1f}%")
        print(f"⚡ Realtime Factor: {realtime_factor}")
        print(f"💻 GPU Utilization: {gpu_stats.get('avg_gpu_utilization', 0):.1f}% avg, {gpu_stats.get('max_gpu_utilization', 0):.1f}% max")
        print(f"💾 GPU Memory: {gpu_stats.get('avg_memory_mb', 0):.0f}MB avg, {gpu_stats.get('max_memory_mb', 0):.0f}MB max")
        
        if baseline and "improvements" in optimization_results:
            print("\n📈 IMPROVEMENTS vs BASELINE:")
            improvements = optimization_results["improvements"]
            if "time_reduction_percent" in improvements:
                print(f"⏱️  Time Reduction: {improvements['time_reduction_percent']:.1f}%")
                print(f"🚀 Speedup Factor: {improvements['speedup_factor']:.2f}x")
            if "gpu_utilization_factor" in improvements:
                print(f"💻 GPU Utilization: {improvements['gpu_utilization_factor']:.1f}x better")
        
        print("\n✅ Results saved to: optimization_test_results.json")
        
        # Save detailed GPU timeline
        with open("gpu_timeline.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "gpu_util", "memory_mb"])
            writer.writeheader()
            writer.writerows(monitor.samples)
        print("📊 GPU timeline saved to: gpu_timeline.csv")
        
    else:
        print(f"❌ Analysis failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    run_optimization_test()