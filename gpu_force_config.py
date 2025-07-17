"""
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