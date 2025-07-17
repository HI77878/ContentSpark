#!/usr/bin/env python3
"""Fix InsightFace GPU Support"""

import subprocess
import sys

print("🔧 REPARIERE INSIGHTFACE GPU SUPPORT")
print("="*60)

# Check current ONNX runtime
print("1. Checking current onnxruntime installation...")
result = subprocess.run([sys.executable, "-m", "pip", "list", "|", "grep", "onnx"], 
                       shell=True, capture_output=True, text=True)
print(result.stdout)

# Check CUDA availability
print("\n2. Checking CUDA availability...")
try:
    import torch
    print(f"   PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"   Error: {e}")

# Check cudnn
print("\n3. Checking cuDNN...")
result = subprocess.run(["ldconfig", "-p", "|", "grep", "cudnn"], 
                       shell=True, capture_output=True, text=True)
print(result.stdout if result.stdout else "   No cuDNN found in ldconfig")

# Fix: Install onnxruntime-gpu
print("\n4. Installing onnxruntime-gpu...")
print("   Uninstalling CPU version first...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"], 
               capture_output=True)

print("   Installing GPU version...")
result = subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime-gpu"], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print("   ✅ onnxruntime-gpu installed successfully")
else:
    print(f"   ❌ Failed to install: {result.stderr}")

# Test ONNX GPU
print("\n5. Testing ONNX GPU support...")
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"   Available providers: {providers}")
    
    if 'CUDAExecutionProvider' in providers:
        print("   ✅ CUDA provider available!")
    else:
        print("   ❌ CUDA provider NOT available")
        
        # Try to find out why
        print("\n   Debugging CUDA issues...")
        import os
        
        # Check LD_LIBRARY_PATH
        import os
        ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        print(f"   LD_LIBRARY_PATH: {ld_path}")
        
        # Look for CUDA libraries
        cuda_paths = [
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/lib"
        ]
        
        for path in cuda_paths:
            if os.path.exists(path):
                print(f"\n   Checking {path}:")
                result = subprocess.run(["ls", "-la", path, "|", "grep", "-E", "(cudnn|cudart)"], 
                                      shell=True, capture_output=True, text=True)
                if result.stdout:
                    print(result.stdout[:500])
                    
except Exception as e:
    print(f"   Error testing: {e}")

print("\n" + "="*60)
print("FIXING CUDNN ISSUE...")
print("="*60)

# The error shows it's looking for libcudnn.so.9
# Let's create a symlink if we have cudnn.so.8
print("Creating cuDNN symlinks...")

cudnn_commands = [
    # Find existing cudnn
    "find /usr -name 'libcudnn.so*' 2>/dev/null | head -10",
    # Create symlinks if needed
    "sudo ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so.9 2>/dev/null || true",
    "sudo ln -sf /usr/local/cuda/lib64/libcudnn.so.8 /usr/local/cuda/lib64/libcudnn.so.9 2>/dev/null || true",
]

for cmd in cudnn_commands:
    print(f"\nRunning: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and "No such file" not in result.stderr:
        print(f"Error: {result.stderr}")

# Update LD_LIBRARY_PATH
print("\n6. Updating LD_LIBRARY_PATH...")
import os
cuda_lib_paths = [
    "/usr/local/cuda/lib64",
    "/usr/lib/x86_64-linux-gnu"
]

current_ld = os.environ.get('LD_LIBRARY_PATH', '')
new_paths = [p for p in cuda_lib_paths if p not in current_ld]

if new_paths:
    new_ld = ":".join(new_paths + [current_ld] if current_ld else new_paths)
    os.environ['LD_LIBRARY_PATH'] = new_ld
    print(f"   Updated LD_LIBRARY_PATH: {new_ld}")
    
    # Write to bashrc for persistence
    with open(os.path.expanduser("~/.bashrc"), "a") as f:
        f.write(f"\nexport LD_LIBRARY_PATH={new_ld}\n")
    print("   Added to ~/.bashrc")

# Final test
print("\n7. Final test after fixes...")
try:
    # Reimport to get new environment
    import importlib
    if 'onnxruntime' in sys.modules:
        del sys.modules['onnxruntime']
    
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"   Available providers now: {providers}")
    
    if 'CUDAExecutionProvider' in providers:
        print("\n   🎉 SUCCESS! CUDA provider is now available!")
    else:
        print("\n   ⚠️  CUDA still not available. May need to restart Python/API")
        
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*60)
print("RECOMMENDATION:")
print("Restart the API server to pick up the new ONNX GPU runtime!")
print("="*60)