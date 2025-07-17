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
