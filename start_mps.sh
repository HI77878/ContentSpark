#\!/bin/bash
# MPS (Multi-Process Service) Start Script für GPU-Optimierung
# WICHTIG: Muss als root/sudo ausgeführt werden

echo "🚀 Starting NVIDIA Multi-Process Service (MPS)..."

# Stoppe existierenden MPS falls vorhanden
echo "Stopping existing MPS if running..."
echo quit  < /dev/null |  sudo nvidia-cuda-mps-control 2>/dev/null || true
sleep 1

# Setze GPU zurück auf Default Mode
echo "Setting GPU to DEFAULT compute mode..."
sudo nvidia-smi -i 0 -c DEFAULT

# Starte MPS Server
echo "Starting MPS server..."
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY

# Setze GPU in EXCLUSIVE_PROCESS Mode für MPS
echo "Setting GPU to EXCLUSIVE_PROCESS mode..."
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS

# Starte MPS Control Daemon
sudo -E nvidia-cuda-mps-control -d

# Warte kurz
sleep 2

# Verifiziere MPS läuft
echo "Verifying MPS status..."
ps aux | grep nvidia-cuda-mps | grep -v grep

if ps aux | grep nvidia-cuda-mps | grep -v grep > /dev/null; then
    echo "✅ MPS successfully started!"
    echo ""
    echo "GPU Compute Mode:"
    nvidia-smi -q | grep "Compute Mode"
    echo ""
    echo "⚠️  WICHTIG: Starte API neu mit:"
    echo "   cd /home/user/tiktok_production"
    echo "   source fix_ffmpeg_env.sh"
    echo "   python3 api/stable_production_api_multiprocess.py"
else
    echo "❌ MPS failed to start!"
    echo "Reverting to DEFAULT mode..."
    sudo nvidia-smi -i 0 -c DEFAULT
fi
