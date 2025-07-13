import torch
import psutil
import GPUtil

class DynamicBatchManager:
    """Dynamische Batch-Größen basierend auf verfügbarem GPU Memory"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu = GPUtil.getGPUs()[0]
            self.total_memory = self.gpu.memoryTotal
            self.base_batch_sizes = {
                'face_detection': 8,
                'object_detection': 4,
                'emotion_detection': 16,
                'body_pose': 6,
                'hand_gesture': 12,
                'text_overlay': 4,
                'vid2seq': 2,
                'scene_analysis': 8
            }
        
    def get_current_memory_usage(self):
        """Aktuelle GPU Memory Nutzung in MB"""
        if not self.gpu_available:
            return 0
        
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].memoryUsed
        return 0
    
    def get_optimal_batch_size(self, analyzer_name, default_size=4):
        """Berechne optimale Batch-Größe basierend auf verfügbarem Memory"""
        if not self.gpu_available:
            return 1
        
        current_usage = self.get_current_memory_usage()
        available_memory = self.total_memory - current_usage
        
        # Basis Batch-Größe
        base_size = self.base_batch_sizes.get(analyzer_name, default_size)
        
        # Skaliere basierend auf verfügbarem Memory
        if available_memory > 20000:  # > 20GB frei
            multiplier = 2.0
        elif available_memory > 10000:  # > 10GB frei
            multiplier = 1.5
        elif available_memory > 5000:  # > 5GB frei
            multiplier = 1.0
        elif available_memory > 2000:  # > 2GB frei
            multiplier = 0.75
        else:
            multiplier = 0.5
        
        optimal_size = int(base_size * multiplier)
        return max(1, optimal_size)
    
    def adjust_for_cpu_load(self, batch_size):
        """Reduziere Batch-Größe bei hoher CPU-Last"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        if cpu_percent > 90:
            return max(1, batch_size // 2)
        elif cpu_percent > 70:
            return max(1, int(batch_size * 0.75))
        
        return batch_size