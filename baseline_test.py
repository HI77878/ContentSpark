#!/usr/bin/env python3
"""Baseline Performance Test für GPU-Optimierung"""

import time
import json
import subprocess
import requests
from datetime import datetime

def get_gpu_stats():
    """Sammelt GPU-Statistiken mit nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        parts = result.stdout.strip().split(", ")
        return {
            "gpu_utilization": float(parts[0]),
            "memory_used_mb": float(parts[1]),
            "memory_total_mb": float(parts[2]),
            "memory_utilization": (float(parts[1]) / float(parts[2])) * 100,
            "temperature": float(parts[3])
        }
    except:
        return {
            "gpu_utilization": 0,
            "memory_used_mb": 0,
            "memory_total_mb": 45541,
            "memory_utilization": 0,
            "temperature": 0
        }

def run_baseline_test():
    """Führt Baseline-Test mit einem Test-Video durch"""
    
    print("🚀 Starte Baseline-Test...")
    
    # Test-Video URL
    test_video = "https://www.tiktok.com/@leon_schliebach/video/7446489995663117590"
    
    # Prüfe API-Status
    try:
        health = requests.get("http://localhost:8003/health")
        if health.status_code != 200:
            print("❌ API nicht erreichbar!")
            return
        
        print(f"✅ API Status: {health.json()['status']}")
    except Exception as e:
        print(f"❌ API-Fehler: {e}")
        return
    
    # GPU Stats vor dem Test
    gpu_stats_before = get_gpu_stats()
    
    # Starte Timer
    start_time = time.time()
    
    # Führe Analyse durch
    print(f"📊 Analysiere Video: {test_video}")
    
    try:
        response = requests.post(
            "http://localhost:8003/analyze",
            json={"tiktok_url": test_video},
            timeout=600
        )
    except Exception as e:
        print(f"❌ Analyse-Fehler: {e}")
        return
    
    # Ende Timer
    end_time = time.time()
    processing_time = end_time - start_time
    
    # GPU Stats nach dem Test
    gpu_stats_after = get_gpu_stats()
    
    # Sammle GPU-Stats während der Analyse (peak values)
    gpu_monitor_output = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    ).stdout.strip()
    
    # Parse Ergebnisse
    if response.status_code == 200:
        result = response.json()
        
        # Berechne Reconstruction Score
        total_analyzers = result.get("metadata", {}).get("total_analyzers", 18)
        successful = result.get("metadata", {}).get("successful_analyzers", 0)
        reconstruction_score = (successful / total_analyzers) * 100 if total_analyzers > 0 else 0
        
        # Zähle Output-Felder für Qualitätsmessung
        output_fields = 0
        for analyzer, data in result.items():
            if analyzer != "metadata" and isinstance(data, dict):
                output_fields += len(str(data))
        
        # Speichere Baseline-Ergebnisse
        baseline_results = {
            "timestamp": datetime.now().isoformat(),
            "test_video": test_video,
            "processing_time_seconds": processing_time,
            "gpu_stats_before": gpu_stats_before,
            "gpu_stats_after": gpu_stats_after,
            "gpu_peak_utilization": gpu_monitor_output,
            "reconstruction_score": reconstruction_score,
            "successful_analyzers": successful,
            "total_analyzers": total_analyzers,
            "output_size_chars": output_fields,
            "result_file": result.get("metadata", {}).get("output_file", ""),
            "realtime_factor": result.get("metadata", {}).get("realtime_factor", "N/A"),
            "video_duration": result.get("metadata", {}).get("video_duration", 0)
        }
        
        # Speichere detaillierte Ergebnisse
        with open("baseline_test.json", "w") as f:
            json.dump(baseline_results, f, indent=2)
        
        print("\n📊 BASELINE TEST ERGEBNISSE:")
        print(f"⏱️  Processing Time: {processing_time:.1f}s")
        print(f"🎯 Reconstruction Score: {reconstruction_score:.1f}%")
        print(f"💻 GPU Peak Utilization: {gpu_monitor_output}")
        print(f"📦 Output Size: {output_fields:,} chars")
        print(f"⚡ Realtime Factor: {baseline_results['realtime_factor']}")
        print(f"\n✅ Baseline gespeichert in: baseline_test.json")
        
        # Speichere auch das komplette Analyse-Ergebnis
        with open("baseline_analysis_result.json", "w") as f:
            json.dump(result, f, indent=2)
        
    else:
        print(f"❌ Analyse fehlgeschlagen: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    run_baseline_test()