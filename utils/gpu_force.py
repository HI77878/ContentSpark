"""GPU Force utilities"""
import torch
import os

# Force GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Import from parent
import sys
sys.path.append('/home/user/tiktok_production')
from gpu_force_config import GPUForce, DEVICE, force_gpu

def force_gpu_init():
    """Initialize GPU forcing"""
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return True
    return False

def get_gpu_info():
    """Get GPU information"""
    if torch.cuda.is_available():
        return {
            'name': torch.cuda.get_device_name(0),
            'memory_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2,
            'compute_capability': torch.cuda.get_device_capability(0)
        }
    return None

# Re-export for compatibility
__all__ = ['GPUForce', 'DEVICE', 'force_gpu', 'force_gpu_init', 'get_gpu_info']