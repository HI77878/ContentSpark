"""
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