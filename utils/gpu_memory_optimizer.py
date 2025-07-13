"""
GPU Memory Optimizer
Simple GPU memory management utilities
"""
import torch
import gc

class GPUMemoryOptimizer:
    """GPU memory optimization utilities"""
    
    @staticmethod
    def cleanup():
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
    @staticmethod
    def get_memory_info():
        """Get GPU memory information"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**2,
                'reserved': torch.cuda.memory_reserved() / 1024**2,
                'total': torch.cuda.get_device_properties(0).total_memory / 1024**2
            }
        return {'allocated': 0, 'reserved': 0, 'total': 0}
        
    @staticmethod
    def optimize_batch_size(model_memory_mb, available_memory_mb=None):
        """Calculate optimal batch size based on memory"""
        if available_memory_mb is None and torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
            used_memory = torch.cuda.memory_allocated() / 1024**2
            available_memory_mb = total_memory - used_memory
            
        if available_memory_mb and model_memory_mb:
            # Conservative estimate: use 80% of available memory
            safe_memory = available_memory_mb * 0.8
            return max(1, int(safe_memory / model_memory_mb))
        return 8  # Default batch size
    
    @staticmethod
    def pin_memory_batch(batch):
        """Pin memory for batch processing"""
        if torch.cuda.is_available() and isinstance(batch, torch.Tensor):
            return batch.pin_memory()
        return batch