#!/usr/bin/env python3
"""
Cleanup Manager - Automatisches Aufräumen nach jeder Analyse
"""

import torch
import gc
import psutil
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class CleanupManager:
    """Verwaltet automatisches Cleanup für optimale Performance"""
    
    def __init__(self):
        self.cleanup_count = 0
        
    def cleanup_gpu_memory(self):
        """GPU Memory komplett aufräumen"""
        if torch.cuda.is_available():
            try:
                # Clear all cached memory
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Force garbage collection
                gc.collect()
                
                # Get memory stats
                allocated = torch.cuda.memory_allocated() / 1024**2
                cached = torch.cuda.memory_reserved() / 1024**2
                
                logger.info(f"GPU Cleanup: {allocated:.1f}MB allocated, {cached:.1f}MB cached")
                return True
                
            except Exception as e:
                logger.error(f"GPU cleanup error: {e}")
                return False
        return False
    
    def cleanup_system_memory(self):
        """System Memory aufräumen"""
        try:
            # Force Python garbage collection
            collected = gc.collect()
            
            # Get memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024**2
            
            logger.info(f"System Cleanup: {collected} objects collected, {memory_mb:.1f}MB used")
            return True
            
        except Exception as e:
            logger.error(f"System cleanup error: {e}")
            return False
    
    def cleanup_temporary_files(self, temp_dirs=None):
        """Temporäre Dateien aufräumen"""
        if temp_dirs is None:
            temp_dirs = ["/tmp", "/home/user/tmp"]
        
        cleaned_files = 0
        for temp_dir in temp_dirs:
            temp_path = Path(temp_dir)
            if temp_path.exists():
                try:
                    # Remove video analysis temp files
                    for pattern in ["*.mp4.tmp", "*.frame_*", "*_temp_*"]:
                        for file in temp_path.glob(pattern):
                            if file.is_file() and file.stat().st_mtime < time.time() - 3600:  # Älter als 1h
                                file.unlink()
                                cleaned_files += 1
                except Exception as e:
                    logger.warning(f"Temp cleanup in {temp_dir}: {e}")
        
        if cleaned_files > 0:
            logger.info(f"Cleaned {cleaned_files} temporary files")
        
        return cleaned_files
    
    def full_cleanup(self):
        """Vollständiges Cleanup nach Analyse"""
        logger.info("Starting full cleanup...")
        
        # GPU cleanup
        gpu_ok = self.cleanup_gpu_memory()
        
        # System memory cleanup
        sys_ok = self.cleanup_system_memory()
        
        # Temp files cleanup
        temp_files = self.cleanup_temporary_files()
        
        self.cleanup_count += 1
        
        logger.info(f"Cleanup #{self.cleanup_count} completed - GPU: {'✅' if gpu_ok else '❌'}, System: {'✅' if sys_ok else '❌'}, Temp: {temp_files} files")
        
        return gpu_ok and sys_ok
    
    def get_memory_stats(self):
        """Aktuelle Memory Statistics"""
        stats = {}
        
        # GPU Memory
        if torch.cuda.is_available():
            stats['gpu'] = {
                'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                'total_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2
            }
            stats['gpu']['usage_percent'] = (stats['gpu']['allocated_mb'] / stats['gpu']['total_mb']) * 100
        
        # System Memory
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            stats['system'] = {
                'rss_mb': memory_info.rss / 1024**2,
                'vms_mb': memory_info.vms / 1024**2,
                'percent': process.memory_percent()
            }
        except:
            stats['system'] = {'error': 'Could not get system memory stats'}
        
        return stats

# Global cleanup manager instance
cleanup_manager = CleanupManager()