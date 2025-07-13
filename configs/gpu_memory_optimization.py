import torch
import os

def optimize_gpu_memory():
    """Maximale GPU Performance durch Memory Pinning"""
    
    if torch.cuda.is_available():
        # Pin Memory für schnellere CPU->GPU Transfers
        torch.cuda.set_per_process_memory_fraction(0.9)  # 90% der GPU
        
        # Enable TF32 für A100/3090
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Persistent Mode
        os.system("nvidia-smi -pm 1")
        
        # Max Clock Speed
        os.system("nvidia-smi -lgc 1980")  # RTX 8000 max clock
        
        print("✅ GPU Memory optimiert für maximale Performance")

# Auto-execute on import
optimize_gpu_memory()