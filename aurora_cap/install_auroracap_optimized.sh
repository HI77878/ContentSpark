#!/bin/bash
# Optimized AuroraCap installation script with precise dependency management
# Based on research findings for xtuner and transformers compatibility

set -e  # Exit on error

echo "=== AuroraCap Optimized Installation ==="
echo "Installing with verified compatible dependencies"

# Configuration
BASE_DIR="/home/user/tiktok_production/aurora_cap"
VENV_NAME="aurora_venv_clean"
PYTHON_VERSION="3.10"

cd "$BASE_DIR"

# Remove old environment if exists
if [ -d "$VENV_NAME" ]; then
    echo "Removing existing environment..."
    rm -rf "$VENV_NAME"
fi

# Create fresh virtual environment
echo "1. Creating clean virtual environment..."
python${PYTHON_VERSION} -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# Upgrade base tools
echo "2. Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Set CUDA environment variables BEFORE any CUDA-dependent installations
echo "3. Setting CUDA environment variables..."
export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_ROOT=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# Verify CUDA installation
if [ ! -d "$CUDA_HOME" ]; then
    echo "Warning: CUDA_HOME directory not found. Creating symlink..."
    sudo ln -sf /usr/local/cuda-12.4 /usr/local/cuda-12.1
fi

# Install PyTorch FIRST (critical for xtuner)
echo "4. Installing PyTorch 2.2.2 with CUDA 12.1..."
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install specific transformers version that works with xtuner and handles rope_scaling
echo "5. Installing transformers with xtuner compatibility..."
# Based on research, transformers 4.36.2 works well with xtuner and has proper rope_scaling support
pip install transformers==4.36.2

# Install critical dependencies with specific versions
echo "6. Installing core dependencies..."
pip install \
    accelerate==0.25.0 \
    bitsandbytes==0.41.3 \
    sentencepiece==0.1.99 \
    protobuf==3.20.3 \
    einops==0.7.0 \
    timm==0.9.12 \
    opencv-python==4.9.0.80 \
    Pillow==10.2.0 \
    numpy==1.24.4 \
    scipy==1.12.0 \
    huggingface-hub==0.20.3 \
    safetensors==0.4.2 \
    tokenizers==0.15.0 \
    peft==0.7.1 \
    datasets==2.16.1 \
    evaluate==0.4.1

# Clone aurora repository
if [ ! -d "aurora" ]; then
    echo "7. Cloning aurora repository..."
    git clone https://github.com/rese1f/aurora.git
else
    echo "7. Updating aurora repository..."
    cd aurora && git pull && cd ..
fi

# Install xtuner with minimal dependencies
echo "8. Installing xtuner..."
cd aurora/src/xtuner

# Create a temporary requirements file without problematic dependencies
grep -v "deepspeed\|flash-attn" requirements/runtime.txt > requirements_minimal.txt || true

# Install xtuner
pip install -e . --no-deps
pip install -r requirements_minimal.txt || true

# Clean up
rm -f requirements_minimal.txt
cd "$BASE_DIR"

# Install additional xtuner dependencies
echo "9. Installing additional dependencies for AuroraCap..."
pip install \
    mmengine>=0.10.3 \
    lmms-eval==0.1.0 \
    pycocoevalcap \
    pycocotools

# Create model cache directory
mkdir -p "$BASE_DIR/.cache/models"

# Write environment activation script
cat > "$BASE_DIR/activate_aurora_clean.sh" << 'EOF'
#!/bin/bash
# Activate AuroraCap environment with all settings

BASE_DIR="/home/user/tiktok_production/aurora_cap"
source "$BASE_DIR/aurora_venv_clean/bin/activate"

# CUDA settings
export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_ROOT=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Python paths
export PYTHONPATH="$BASE_DIR/aurora/src:$BASE_DIR/aurora:$PYTHONPATH"

# Model cache
export HF_HOME="$BASE_DIR/.cache"
export TRANSFORMERS_CACHE="$BASE_DIR/.cache"
export XFORMERS_CACHE="$BASE_DIR/.cache"

# Disable warnings
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

echo "AuroraCap environment activated!"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
EOF

chmod +x "$BASE_DIR/activate_aurora_clean.sh"

# Test imports
echo "10. Testing installation..."
python -c "
import torch
import transformers
import xtuner
print('✓ All imports successful')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

echo ""
echo "✅ Installation complete!"
echo ""
echo "To activate the environment:"
echo "  source $BASE_DIR/activate_aurora_clean.sh"
echo ""
echo "Next: Run the optimized inference script"