#!/bin/bash
# Fixed AuroraCap installation script with proper dependencies

set -e  # Exit on error

echo "=== AuroraCap Fixed Installation Script ==="
echo "This script installs AuroraCap with compatible dependencies"

# Set base directory
BASE_DIR="/home/user/tiktok_production/aurora_cap"
cd "$BASE_DIR"

# Create virtual environment
echo "1. Creating virtual environment..."
python3 -m venv aurora_venv
source aurora_venv/bin/activate

# Upgrade pip and essential tools
echo "2. Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Set CUDA environment variables
echo "3. Setting CUDA environment variables..."
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"  # Covering common architectures

# Install PyTorch first (matching our system)
echo "4. Installing PyTorch 2.2.2..."
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install transformers with specific version to avoid rope_scaling issues
echo "5. Installing compatible transformers..."
pip install transformers==4.37.2  # Version that handles rope_scaling properly

# Install other critical dependencies
echo "6. Installing core dependencies..."
pip install \
    accelerate==0.26.1 \
    bitsandbytes==0.42.0 \
    sentencepiece==0.1.99 \
    protobuf==3.20.3 \
    einops==0.7.0 \
    timm==0.9.12 \
    opencv-python==4.9.0.80 \
    Pillow==10.2.0 \
    numpy==1.24.4 \
    scipy==1.12.0 \
    huggingface-hub==0.20.3 \
    safetensors==0.4.2

# Clone aurora repository if not exists
if [ ! -d "aurora" ]; then
    echo "7. Cloning aurora repository..."
    git clone https://github.com/rese1f/aurora.git
else
    echo "7. Aurora repository already exists, pulling latest..."
    cd aurora && git pull && cd ..
fi

# Install xtuner without deepspeed to avoid CUDA_HOME issues
echo "8. Installing xtuner (without deepspeed)..."
cd aurora/src/xtuner
# Modify requirements to exclude deepspeed
pip install -e . --no-deps
pip install -r requirements/runtime.txt | grep -v deepspeed || true

# Create inference script directory
cd "$BASE_DIR"
mkdir -p scripts

# Write environment activation script
cat > activate_aurora.sh << 'EOF'
#!/bin/bash
# Activate AuroraCap environment

source /home/user/tiktok_production/aurora_cap/aurora_venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/home/user/tiktok_production/aurora_cap/aurora:$PYTHONPATH
export HF_HOME=/home/user/tiktok_production/aurora_cap/.cache
export TRANSFORMERS_CACHE=/home/user/tiktok_production/aurora_cap/.cache

echo "AuroraCap environment activated!"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
EOF

chmod +x activate_aurora.sh

echo "✅ Installation complete!"
echo ""
echo "To activate the environment:"
echo "  source $BASE_DIR/activate_aurora.sh"
echo ""
echo "Next steps:"
echo "1. Activate the environment"
echo "2. Run the inference script"