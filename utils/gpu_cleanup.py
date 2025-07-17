#!/usr/bin/env python3
"""
GPU Cleanup Service - Automatisches Cleanup von Zombie GPU Prozessen
Erkennt und killt Prozesse die GPU Memory blockieren aber keine Aktivität zeigen
"""

import subprocess
import time
import logging
import re
import os
import signal
from datetime import datetime
from typing import List, Dict, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUCleanupService:
    def __init__(self, 
                 idle_threshold_seconds: int = 30,
                 memory_threshold_mb: int = 1000,
                 check_interval: int = 10):
        """
        Initialize GPU Cleanup Service
        
        Args:
            idle_threshold_seconds: Seconds a process must be idle before considered zombie
            memory_threshold_mb: Minimum memory usage to consider for cleanup
            check_interval: Seconds between cleanup checks
        """
        self.idle_threshold = idle_threshold_seconds
        self.memory_threshold_mb = memory_threshold_mb
        self.check_interval = check_interval
        self.process_tracking = {}  # Track process idle times
        
    def get_gpu_processes(self) -> List[Dict]:
        """Get list of all GPU processes with their stats"""
        try:
            # Run nvidia-smi to get process info
            cmd = [
                'nvidia-smi', 
                '--query-compute-apps=pid,process_name,used_gpu_memory,gpu_util',
                '--format=csv,noheader,nounits'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            processes = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    try:
                        process_info = {
                            'pid': int(parts[0]),
                            'name': parts[1],
                            'memory_mb': float(parts[2]),
                            'gpu_util': float(parts[3]) if parts[3] != '[N/A]' else 0.0
                        }
                        processes.append(process_info)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse line: {line} - {e}")
                        
            return processes
            
        except subprocess.CalledProcessError as e:
            logger.error(f"nvidia-smi failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting GPU processes: {e}")
            return []
    
    def is_zombie_process(self, process: Dict) -> bool:
        """Check if a process is a zombie (high memory, no GPU utilization)"""
        # Zombie criteria:
        # 1. Using more than threshold memory
        # 2. Zero or very low GPU utilization
        # 3. Has been idle for threshold time
        
        if process['memory_mb'] < self.memory_threshold_mb:
            return False
            
        if process['gpu_util'] > 5.0:  # Active if >5% GPU util
            return False
            
        # Check if it's a known ML framework process
        safe_processes = ['Xorg', 'gnome-shell', 'chrome', 'firefox']
        if any(safe in process['name'] for safe in safe_processes):
            return False
            
        return True
    
    def track_idle_time(self, pid: int) -> int:
        """Track how long a process has been idle"""
        current_time = time.time()
        
        if pid not in self.process_tracking:
            self.process_tracking[pid] = current_time
            return 0
            
        idle_time = current_time - self.process_tracking[pid]
        return int(idle_time)
    
    def cleanup_zombie_processes(self) -> List[int]:
        """Find and kill zombie GPU processes"""
        killed_pids = []
        
        try:
            processes = self.get_gpu_processes()
            
            # Clean up tracking for processes that no longer exist
            current_pids = {p['pid'] for p in processes}
            self.process_tracking = {
                pid: start_time 
                for pid, start_time in self.process_tracking.items() 
                if pid in current_pids
            }
            
            for process in processes:
                if self.is_zombie_process(process):
                    idle_time = self.track_idle_time(process['pid'])
                    
                    logger.info(
                        f"Zombie candidate: PID {process['pid']} ({process['name']}) - "
                        f"{process['memory_mb']:.0f}MB, {process['gpu_util']:.1f}% util, "
                        f"idle for {idle_time}s"
                    )
                    
                    if idle_time >= self.idle_threshold:
                        # Try graceful termination first
                        logger.warning(
                            f"Killing zombie process {process['pid']} ({process['name']}) - "
                            f"Used {process['memory_mb']:.0f}MB with 0% GPU for {idle_time}s"
                        )
                        
                        try:
                            # Try SIGTERM first
                            os.kill(process['pid'], signal.SIGTERM)
                            time.sleep(2)
                            
                            # Check if still exists, then SIGKILL
                            try:
                                os.kill(process['pid'], 0)  # Check if process exists
                                os.kill(process['pid'], signal.SIGKILL)
                                logger.warning(f"Had to force kill PID {process['pid']}")
                            except ProcessLookupError:
                                pass  # Process already terminated
                                
                            killed_pids.append(process['pid'])
                            del self.process_tracking[process['pid']]
                            
                        except Exception as e:
                            logger.error(f"Failed to kill PID {process['pid']}: {e}")
                else:
                    # Reset tracking for active processes
                    if process['pid'] in self.process_tracking:
                        del self.process_tracking[process['pid']]
                        
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            
        return killed_pids
    
    def get_gpu_memory_info(self) -> Dict:
        """Get current GPU memory statistics"""
        try:
            cmd = [
                'nvidia-smi',
                '--query-gpu=memory.total,memory.used,memory.free',
                '--format=csv,noheader,nounits'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            parts = result.stdout.strip().split(',')
            if len(parts) >= 3:
                return {
                    'total_mb': float(parts[0]),
                    'used_mb': float(parts[1]),
                    'free_mb': float(parts[2]),
                    'utilization_percent': (float(parts[1]) / float(parts[0])) * 100
                }
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            
        return {}
    
    def run_once(self) -> Dict:
        """Run cleanup once and return results"""
        start_time = time.time()
        
        # Get initial memory state
        memory_before = self.get_gpu_memory_info()
        
        # Cleanup zombie processes
        killed_pids = self.cleanup_zombie_processes()
        
        # Force CUDA cache cleanup if we killed anything
        if killed_pids:
            time.sleep(2)  # Wait for processes to fully terminate
            
            # Try to clear any remaining CUDA context
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass
        
        # Get final memory state
        memory_after = self.get_gpu_memory_info()
        
        # Calculate freed memory
        freed_mb = 0
        if memory_before and memory_after:
            freed_mb = memory_before.get('used_mb', 0) - memory_after.get('used_mb', 0)
        
        duration = time.time() - start_time
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'killed_pids': killed_pids,
            'freed_memory_mb': freed_mb,
            'memory_before': memory_before,
            'memory_after': memory_after,
            'duration_seconds': duration
        }
        
        if killed_pids:
            logger.info(
                f"Cleanup complete: Killed {len(killed_pids)} processes, "
                f"freed {freed_mb:.0f}MB GPU memory"
            )
        
        return result
    
    def run_continuous(self):
        """Run cleanup service continuously"""
        logger.info(
            f"GPU Cleanup Service started - "
            f"Idle threshold: {self.idle_threshold}s, "
            f"Memory threshold: {self.memory_threshold_mb}MB, "
            f"Check interval: {self.check_interval}s"
        )
        
        while True:
            try:
                self.run_once()
                time.sleep(self.check_interval)
            except KeyboardInterrupt:
                logger.info("GPU Cleanup Service stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in cleanup loop: {e}")
                time.sleep(self.check_interval)


def cleanup_gpu_memory():
    """Quick function to run cleanup once (for API integration)"""
    service = GPUCleanupService(
        idle_threshold_seconds=10,  # More aggressive for API
        memory_threshold_mb=500,
        check_interval=5
    )
    return service.run_once()


if __name__ == "__main__":
    # Run as standalone service
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU Cleanup Service')
    parser.add_argument('--idle-threshold', type=int, default=30,
                        help='Seconds before considering process zombie (default: 30)')
    parser.add_argument('--memory-threshold', type=int, default=1000,
                        help='Minimum MB to consider for cleanup (default: 1000)')
    parser.add_argument('--check-interval', type=int, default=10,
                        help='Seconds between checks (default: 10)')
    parser.add_argument('--once', action='store_true',
                        help='Run cleanup once and exit')
    
    args = parser.parse_args()
    
    service = GPUCleanupService(
        idle_threshold_seconds=args.idle_threshold,
        memory_threshold_mb=args.memory_threshold,
        check_interval=args.check_interval
    )
    
    if args.once:
        result = service.run_once()
        print(f"Cleanup result: {result}")
    else:
        service.run_continuous()