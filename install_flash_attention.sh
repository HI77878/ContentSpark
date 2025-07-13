#!/bin/bash
# Install Flash Attention 2 for Qwen2-VL optimization

echo "Installing Flash Attention 2..."

# Install ninja for faster compilation
pip install ninja

# Install flash-attn with CUDA 11.8 support
pip install flash-attn --no-build-isolation

# Alternative if above fails
# pip install flash-attn==2.5.8

echo "Flash Attention 2 installation complete!"

# Test import
python3 -c "import flash_attn; print('✅ Flash Attention 2 imported successfully')" || echo "❌ Flash Attention 2 import failed"