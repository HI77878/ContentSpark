#!/bin/bash
# Install cuDNN for CUDA 12.1

echo "🔧 INSTALLIERE cuDNN für CUDA 12.1"
echo "============================================"

# Check CUDA version
echo "1. Checking CUDA version..."
nvcc --version | grep "release"

# Install cuDNN via apt (Ubuntu/Debian)
echo -e "\n2. Installing cuDNN..."

# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install cuDNN
echo -e "\n3. Installing cuDNN packages..."
sudo apt-get install -y libcudnn8 libcudnn8-dev libcudnn8-samples

# Alternative: Install specific version for CUDA 12
# sudo apt-get install -y libcudnn8=8.9.7.29-1+cuda12.2

# Verify installation
echo -e "\n4. Verifying cuDNN installation..."
ls -la /usr/lib/x86_64-linux-gnu/libcudnn*

# Update library cache
echo -e "\n5. Updating library cache..."
sudo ldconfig

# Check if libraries are found
echo -e "\n6. Checking ldconfig..."
ldconfig -p | grep cudnn

# Set up environment variables
echo -e "\n7. Setting up environment..."
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

echo -e "\n✅ cuDNN installation complete!"
echo "============================================"

# Test with Python
echo -e "\nTesting ONNX Runtime with CUDA..."
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'Available providers: {providers}')
if 'CUDAExecutionProvider' in providers:
    print('✅ CUDA provider available!')
else:
    print('❌ CUDA provider not available')
"

# Cleanup
rm -f cuda-keyring_1.1-1_all.deb