#!/bin/bash
# Production Service Restart Script

echo "=========================================="
echo "TikTok Production System Service Manager"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="/home/user/tiktok_production"
cd $BASE_DIR

# Function to stop services
stop_services() {
    echo -e "${YELLOW}Stopping services...${NC}"
    
    # Kill API process
    pkill -f "stable_production_api"
    
    # Kill worker processes
    pkill -f "multiprocess_gpu_executor"
    
    # Clear GPU memory
    python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
    
    echo -e "${GREEN}Services stopped${NC}"
}

# Function to start services
start_services() {
    echo -e "${YELLOW}Starting services...${NC}"
    
    # Source FFmpeg fix
    source fix_ffmpeg_env.sh
    
    # Start API in background
    nohup python3 api/stable_production_api_multiprocess.py > logs/api_startup.log 2>&1 &
    API_PID=$!
    
    echo -e "${GREEN}API started with PID: $API_PID${NC}"
    
    # Wait for API to be ready
    echo "Waiting for API to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8003/health > /dev/null; then
            echo -e "${GREEN}API is ready!${NC}"
            break
        fi
        sleep 1
    done
}

# Function to check status
check_status() {
    echo -e "${YELLOW}Checking system status...${NC}"
    
    # Check API
    if curl -s http://localhost:8003/health > /dev/null; then
        echo -e "${GREEN}✓ API is running${NC}"
    else
        echo -e "${RED}✗ API is not responding${NC}"
    fi
    
    # Check GPU
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    
    # Check processes
    echo -e "\n${YELLOW}Running processes:${NC}"
    ps aux | grep -E "(stable_production_api|multiprocess_gpu)" | grep -v grep
}

# Main menu
case "$1" in
    start)
        start_services
        sleep 2
        check_status
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        sleep 2
        start_services
        sleep 2
        check_status
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac