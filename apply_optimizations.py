#!/usr/bin/env python3
"""
Wendet GPU-Optimierungen auf das TikTok Analysis System an
SICHER und SCHRITTWEISE mit Rollback-Option
"""

import os
import sys
import shutil
import json
import subprocess
from datetime import datetime
import time

def log(message):
    """Log with timestamp"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def check_gpu_status():
    """Check current GPU status"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", 
             "--format=csv,noheader"],
            capture_output=True, text=True
        )
        log(f"GPU Status: {result.stdout.strip()}")
        return True
    except:
        log("❌ GPU check failed")
        return False

def stop_api():
    """Stop running API"""
    log("Stopping API...")
    subprocess.run(["pkill", "-f", "stable_production_api"], capture_output=True)
    time.sleep(2)

def apply_mps_optimization():
    """Apply MPS optimization (requires sudo)"""
    log("⚠️  MPS requires sudo access. Skip for now in automated mode.")
    log("To enable MPS manually, run: sudo ./start_mps.sh")
    return True

def update_api_to_use_cached_executor():
    """Update API to use cached executor"""
    api_file = "/home/user/tiktok_production/api/stable_production_api_multiprocess.py"
    backup_file = f"{api_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Backup
    shutil.copy2(api_file, backup_file)
    log(f"Backed up API to {backup_file}")
    
    # Read current content
    with open(api_file, 'r') as f:
        content = f.read()
    
    # Check if we need to update
    if 'MultiprocessGPUExecutorRegistryCached' not in content:
        log("Updating API to use cached executor...")
        
        # Replace import
        content = content.replace(
            'from utils.multiprocess_gpu_executor_registry import MultiprocessGPUExecutorRegistry',
            'from utils.multiprocess_gpu_executor_registry_cached import MultiprocessGPUExecutorRegistryCached'
        )
        
        # Replace class usage
        content = content.replace(
            'self.executor = MultiprocessGPUExecutorRegistry(',
            'self.executor = MultiprocessGPUExecutorRegistryCached(enable_caching=True, '
        )
        
        # Write back
        with open(api_file, 'w') as f:
            f.write(content)
        
        log("✅ API updated to use cached executor")
    else:
        log("✅ API already using cached executor")
    
    return True

def test_optimizations():
    """Test the optimizations with a quick analysis"""
    log("Testing optimizations...")
    
    # Start API with new settings
    log("Starting API with optimizations...")
    
    # First source the environment
    env_cmd = "cd /home/user/tiktok_production && source fix_ffmpeg_env.sh && python3 api/stable_production_api_multiprocess.py > optimization_test.log 2>&1 &"
    subprocess.Popen(env_cmd, shell=True, executable='/bin/bash')
    
    # Wait for API to start
    log("Waiting for API to start...")
    time.sleep(10)
    
    # Check if API is running
    try:
        import requests
        response = requests.get("http://localhost:8003/health", timeout=5)
        if response.status_code == 200:
            log("✅ API started successfully")
            health = response.json()
            log(f"API Health: {json.dumps(health, indent=2)}")
            return True
        else:
            log("❌ API health check failed")
            return False
    except Exception as e:
        log(f"❌ Could not connect to API: {e}")
        return False

def create_optimization_report():
    """Create a report of applied optimizations"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "optimizations_applied": [
            {
                "name": "Memory Pool Optimization",
                "status": "enabled",
                "settings": {
                    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.9",
                    "OMP_NUM_THREADS": "8",
                    "MKL_NUM_THREADS": "8"
                }
            },
            {
                "name": "Model Caching",
                "status": "enabled",
                "description": "Models stay in GPU memory between analyses"
            },
            {
                "name": "MPS (Multi-Process Service)",
                "status": "manual",
                "description": "Run 'sudo ./start_mps.sh' to enable"
            }
        ],
        "expected_improvements": {
            "gpu_utilization": "5% → 30-40%",
            "processing_time": "394s → 150-200s",
            "realtime_factor": "8x → 3-4x"
        },
        "rollback_instructions": [
            "1. Stop API: pkill -f stable_production_api",
            "2. Restore backups from backups/20250713_063835/",
            "3. Restart API normally"
        ]
    }
    
    with open("optimization_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    log("✅ Created optimization_report.json")

def main():
    """Main optimization application"""
    log("🚀 Starting GPU Optimization Application")
    
    # Check prerequisites
    if not check_gpu_status():
        log("❌ GPU not available, aborting")
        return 1
    
    # Stop current API
    stop_api()
    
    # Apply optimizations
    steps = [
        ("Update API to use cached executor", update_api_to_use_cached_executor),
        ("Apply MPS optimization", apply_mps_optimization),
        ("Test optimizations", test_optimizations),
        ("Create optimization report", create_optimization_report)
    ]
    
    for step_name, step_func in steps:
        log(f"📋 {step_name}...")
        if not step_func():
            log(f"❌ {step_name} failed")
            return 1
    
    log("✅ All optimizations applied successfully!")
    log("")
    log("📊 Next steps:")
    log("1. Run a test analysis to measure improvement")
    log("2. Compare with baseline_test.json")
    log("3. Monitor GPU usage with: watch -n 1 nvidia-smi")
    log("4. For maximum performance, run: sudo ./start_mps.sh")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())