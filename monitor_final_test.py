#!/usr/bin/env python3
"""
System Monitoring für finalen Produktionstest
"""
import psutil
import GPUtil
import time
import json
from datetime import datetime
import sys
import signal

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    global running
    print("\nMonitoring beendet")
    running = False

signal.signal(signal.SIGINT, signal_handler)

def monitor_system(duration=300):
    """Monitore System für duration Sekunden"""
    metrics = {
        "cpu": [],
        "gpu": [],
        "memory": [],
        "timestamps": [],
        "start_time": datetime.now().isoformat()
    }
    
    start = time.time()
    while running and (time.time() - start < duration):
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            
            # GPU
            gpus = GPUtil.getGPUs()
            gpu_load = gpus[0].load * 100 if gpus else 0
            gpu_memory = gpus[0].memoryUsed if gpus else 0
            gpu_memory_percent = (gpus[0].memoryUsed / gpus[0].memoryTotal * 100) if gpus else 0
            
            # Memory
            mem = psutil.virtual_memory()
            
            metrics["cpu"].append({
                "avg": sum(cpu_percent)/len(cpu_percent),
                "cores": cpu_percent
            })
            metrics["gpu"].append({
                "load": gpu_load,
                "memory_mb": gpu_memory,
                "memory_percent": gpu_memory_percent
            })
            metrics["memory"].append({
                "percent": mem.percent,
                "used_gb": mem.used / 1024**3,
                "available_gb": mem.available / 1024**3
            })
            metrics["timestamps"].append(datetime.now().isoformat())
            
            # Live output
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                  f"CPU: {sum(cpu_percent)/len(cpu_percent):.1f}% | "
                  f"GPU: {gpu_load:.1f}% ({gpu_memory:.0f}MB) | "
                  f"RAM: {mem.percent:.1f}%", end="", flush=True)
            
            time.sleep(1)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nFehler beim Monitoring: {e}")
            break
    
    # Berechne Durchschnittswerte
    if metrics["cpu"]:
        metrics["summary"] = {
            "duration": time.time() - start,
            "samples": len(metrics["cpu"]),
            "cpu_avg": sum(m["avg"] for m in metrics["cpu"]) / len(metrics["cpu"]),
            "cpu_max": max(m["avg"] for m in metrics["cpu"]),
            "gpu_avg_load": sum(m["load"] for m in metrics["gpu"]) / len(metrics["gpu"]) if metrics["gpu"] else 0,
            "gpu_max_load": max(m["load"] for m in metrics["gpu"]) if metrics["gpu"] else 0,
            "gpu_avg_memory_mb": sum(m["memory_mb"] for m in metrics["gpu"]) / len(metrics["gpu"]) if metrics["gpu"] else 0,
            "gpu_max_memory_mb": max(m["memory_mb"] for m in metrics["gpu"]) if metrics["gpu"] else 0,
            "memory_avg_percent": sum(m["percent"] for m in metrics["memory"]) / len(metrics["memory"]) if metrics["memory"] else 0,
            "end_time": datetime.now().isoformat()
        }
    
    return metrics

if __name__ == "__main__":
    print("🖥️  System Monitoring gestartet...")
    print("Drücke Ctrl+C zum Beenden\n")
    
    # Monitor für 5 Minuten oder bis Ctrl+C
    metrics = monitor_system(duration=300)
    
    # Speichere Ergebnisse
    with open("system_metrics_final_test.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n\n📊 Monitoring abgeschlossen!")
    print(f"Dauer: {metrics.get('summary', {}).get('duration', 0):.1f}s")
    print(f"CPU Durchschnitt: {metrics.get('summary', {}).get('cpu_avg', 0):.1f}%")
    print(f"GPU Durchschnitt: {metrics.get('summary', {}).get('gpu_avg_load', 0):.1f}%")
    print(f"GPU Max Memory: {metrics.get('summary', {}).get('gpu_max_memory_mb', 0):.0f}MB")
    print(f"\nDetails gespeichert in: system_metrics_final_test.json")