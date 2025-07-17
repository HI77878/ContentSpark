"""
<<<<<<< HEAD
GPU Force Configuration - Ensures all models use GPU
"""

import os
import torch

# Force GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5'

def force_gpu():
    """Force GPU usage for all models"""
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        return True
    return False

# Auto-run on import
GPU_AVAILABLE = force_gpu()

# Global device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# GPUForce class for compatibility
class GPUForce:
    def __init__(self):
        self.device = DEVICE
        self.is_gpu = DEVICE.type == 'cuda'
    
    def force(self):
        return force_gpu()

if GPU_AVAILABLE:
    print(f"✅ GPU forced: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ No GPU available, using CPU")
=======
GPU Force Configuration
Simple GPU configuration for analyzers
"""
import torch
import os

# Force GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GPUForce:
    """Simple GPU force utility"""
    @staticmethod
    def enable():
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            return True
        return False

def force_gpu(device_id=0):
    """Force GPU usage"""
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        return torch.device('cuda')
    return torch.device('cpu')
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
