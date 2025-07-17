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
