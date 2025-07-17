#!/bin/bash

# Clean Server Start Script - MVP Version
# This script starts the TikTok Video Analyzer Clean Server

echo "🚀 Starting TikTok Video Analyzer Clean Server (MVP)"
echo "======================================================"

# 1. Environment Setup
echo "Setting up environment..."
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh

# 2. Check GPU Status
echo "Checking GPU status..."
nvidia-smi --query-gpu=name,memory.free,utilization.gpu --format=csv,noheader

# 3. Verify Dependencies
echo "Verifying critical dependencies..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 4. Start API Server
echo "Starting API server on port 8003..."
python3 api/stable_production_api_multiprocess.py &
API_PID=$!

# 5. Wait for server to start
echo "Waiting for server to initialize..."
sleep 5

# 6. Health Check
echo "Performing health check..."
curl -f http://localhost:8003/health > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Clean Server started successfully!"
    echo "✅ API available at http://localhost:8003"
    echo "✅ Health endpoint: http://localhost:8003/health"
    echo "✅ 23 analyzers configured in 4 GPU stages"
else
    echo "❌ Server health check failed"
    kill $API_PID
    exit 1
fi

# 7. Show usage
echo ""
echo "Usage Examples:"
echo "==============="
echo "curl -X POST http://localhost:8003/analyze -H 'Content-Type: application/json' -d '{\"video_path\": \"/path/to/video.mp4\"}'"
echo "python3 single_workflow.py 'https://www.tiktok.com/@username/video/123'"
echo ""
echo "Monitor with: watch -n 1 nvidia-smi"
echo "Logs: tail -f /home/user/tiktok_production/logs/stable_api.log"
echo ""
echo "Press Ctrl+C to stop the server"

# 8. Keep script running
wait $API_PID