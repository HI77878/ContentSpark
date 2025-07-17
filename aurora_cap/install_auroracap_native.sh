#!/bin/bash
# Install AuroraCap natively in a virtual environment

cd /home/user/tiktok_production/aurora_cap

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv aurora_env

# Activate it
source aurora_env/bin/activate

# Install dependencies with specific versions
echo "Installing dependencies..."
pip install --upgrade pip

# Install PyTorch first
pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu118

# Install compatible transformers and other deps
pip install \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    bitsandbytes==0.41.3 \
    sentencepiece \
    opencv-python \
    Pillow \
    numpy \
    scipy \
    einops \
    timm \
    peft==0.7.1 \
    huggingface-hub

# Clone aurora if not exists
if [ ! -d "aurora" ]; then
    git clone https://github.com/rese1f/aurora.git
fi

echo "Installation complete!"
echo "To run AuroraCap:"
echo "  source aurora_env/bin/activate"
echo "  python auroracap_native.py /path/to/video.mp4"