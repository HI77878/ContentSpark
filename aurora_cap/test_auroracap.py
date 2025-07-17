#!/usr/bin/env python3
"""
Test script to verify AuroraCap installation and basic functionality
"""
import torch
import sys
import os

def test_environment():
    """Test if environment is properly set up"""
    print("=== AuroraCap Environment Test ===\n")
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # PyTorch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test imports
    print("\nTesting imports...")
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
    
    try:
        import accelerate
        print(f"✓ Accelerate version: {accelerate.__version__}")
    except ImportError as e:
        print(f"✗ Accelerate import failed: {e}")
    
    try:
        import bitsandbytes
        print(f"✓ BitsAndBytes available")
    except ImportError as e:
        print(f"✗ BitsAndBytes import failed: {e}")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
    
    # Environment variables
    print("\nEnvironment variables:")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
    
    # Test model loading
    print("\nTesting model configuration...")
    try:
        from auroracap_inference_fixed import fix_rope_scaling_config
        print("✓ AuroraCap inference script can be imported")
        
        # Test rope_scaling fix
        test_config = {
            "rope_scaling": {
                "rope_type": "dynamic",
                "factor": 8.0,
                "original_max_position_embeddings": 4096,
                "unused_field": "remove_me"
            }
        }
        
        import tempfile
        import json
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_path = f.name
        
        fix_rope_scaling_config(temp_path)
        
        with open(temp_path, 'r') as f:
            fixed_config = json.load(f)
        
        os.unlink(temp_path)
        
        if len(fixed_config['rope_scaling']) == 2:
            print("✓ rope_scaling fix working correctly")
        else:
            print("✗ rope_scaling fix not working")
            
    except Exception as e:
        print(f"✗ Error testing model configuration: {e}")
    
    print("\n=== Environment test complete ===")

if __name__ == "__main__":
    test_environment()