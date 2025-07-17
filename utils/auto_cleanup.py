import os
import shutil
import psutil
import torch
import gc
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AutoCleanup:
    def __init__(self):
        self.temp_dirs = [
            '/tmp/gradio',
            '/tmp/transformers_cache',
            '/home/user/.cache/torch/hub/checkpoints'
        ]
        
    def cleanup_after_analysis(self, video_id):
        """Wird nach jeder Analyse automatisch aufgerufen"""
        logger.info(f"Starting cleanup for video {video_id}")
        
        # 1. GPU Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        # 2. Temp files cleanup
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    # Nur Dateien älter als 1 Stunde löschen
                    self._cleanup_old_files(temp_dir, hours=1)
                except Exception as e:
                    logger.warning(f"Could not clean {temp_dir}: {e}")
        
        # 3. Process cleanup - beende Zombie-Prozesse
        self._cleanup_zombie_processes()
        
        # 4. Log rotation
        self._rotate_logs()
        
        logger.info("Cleanup completed")
    
    def _cleanup_old_files(self, directory, hours=1):
        """Löscht Dateien älter als X Stunden"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if os.path.getmtime(file_path) < cutoff_time.timestamp():
                        os.remove(file_path)
                        logger.debug(f"Removed old file: {file_path}")
                except Exception as e:
                    logger.debug(f"Could not remove {file_path}: {e}")
    
    def _cleanup_zombie_processes(self):
        """Beendet Zombie-Prozesse"""
        cleaned = 0
        for proc in psutil.process_iter(['pid', 'status']):
            try:
                if proc.info['status'] == psutil.STATUS_ZOMBIE:
                    os.kill(proc.info['pid'], 9)
                    cleaned += 1
            except:
                pass
        if cleaned > 0:
            logger.info(f"Cleaned {cleaned} zombie processes")
    
    def _rotate_logs(self):
        """Komprimiert alte Logs"""
        log_dir = '/home/user/tiktok_production/logs'
        if os.path.exists(log_dir):
            # Komprimiere Logs größer als 100MB
            os.system(f'find {log_dir} -name "*.log" -size +100M -exec gzip {{}} \; 2>/dev/null')
            # Lösche gezippte Logs älter als 30 Tage
            os.system(f'find {log_dir} -name "*.log.gz" -mtime +30 -delete 2>/dev/null')

# Globale Instanz
auto_cleanup = AutoCleanup()