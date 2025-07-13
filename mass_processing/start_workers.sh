#!/bin/bash
# Start all workers for TikTok mass processing system

echo "🚀 Starting TikTok Mass Processing System..."

# Configuration
REDIS_PORT=6379
FLOWER_PORT=5555
MONITOR_PORT=5000
LOG_DIR="/home/user/tiktok_production/logs/mass_processing"

# Create log directory
mkdir -p $LOG_DIR

# Function to check if service is running
check_service() {
    if pgrep -f "$1" > /dev/null; then
        echo "✅ $2 is already running"
        return 0
    else
        return 1
    fi
}

# Start Redis if not running
if ! check_service "redis-server" "Redis"; then
    echo "Starting Redis..."
    redis-server --port $REDIS_PORT --daemonize yes
    sleep 2
fi

# Kill existing Celery workers
echo "Stopping existing Celery workers..."
pkill -f "celery.*worker" || true
sleep 2

# Start download workers (2 workers for downloading)
echo "Starting download workers..."
for i in {1..2}; do
    celery -A mass_processing.celery_config worker \
        -Q download \
        -n download_worker_$i@%h \
        --loglevel=info \
        --logfile=$LOG_DIR/download_worker_$i.log \
        --detach
done

# Start GPU workers (1 per GPU)
echo "Starting GPU workers..."
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "Found $GPU_COUNT GPUs"

for i in $(seq 0 $((GPU_COUNT-1))); do
    CUDA_VISIBLE_DEVICES=$i celery -A mass_processing.celery_config worker \
        -Q gpu_processing,priority \
        -n gpu_worker_$i@%h \
        --loglevel=info \
        --logfile=$LOG_DIR/gpu_worker_$i.log \
        --pool=threads \
        --concurrency=1 \
        --detach
done

# Start CPU workers (4 workers for CPU tasks)
echo "Starting CPU workers..."
for i in {1..4}; do
    celery -A mass_processing.celery_config worker \
        -Q cpu_processing \
        -n cpu_worker_$i@%h \
        --loglevel=info \
        --logfile=$LOG_DIR/cpu_worker_$i.log \
        --concurrency=2 \
        --detach
done

# Start monitoring worker
echo "Starting monitoring worker..."
celery -A mass_processing.celery_config worker \
    -Q monitoring \
    -n monitoring_worker@%h \
    --loglevel=info \
    --logfile=$LOG_DIR/monitoring_worker.log \
    --detach

# Start Celery Beat for scheduled tasks
if ! check_service "celery.*beat" "Celery Beat"; then
    echo "Starting Celery Beat..."
    celery -A mass_processing.celery_config beat \
        --loglevel=info \
        --logfile=$LOG_DIR/celery_beat.log \
        --detach
fi

# Start Flower for monitoring
if ! check_service "flower" "Flower"; then
    echo "Starting Flower..."
    celery -A mass_processing.celery_config flower \
        --port=$FLOWER_PORT \
        --logfile=$LOG_DIR/flower.log \
        --detach &
fi

# Start monitoring dashboard
if ! check_service "flask.*monitoring" "Monitoring Dashboard"; then
    echo "Starting monitoring dashboard..."
    cd /home/user/tiktok_production/mass_processing
    python monitoring.py > $LOG_DIR/monitoring_dashboard.log 2>&1 &
fi

# Wait for services to start
sleep 5

# Check worker status
echo ""
echo "🔍 Checking worker status..."
celery -A mass_processing.celery_config inspect active_queues

echo ""
echo "✅ All services started!"
echo ""
echo "📊 Access points:"
echo "   - Monitoring Dashboard: http://localhost:$MONITOR_PORT"
echo "   - Flower (Celery): http://localhost:$FLOWER_PORT"
echo "   - Logs: $LOG_DIR"
echo ""
echo "💡 To add videos for processing:"
echo "   python bulk_processor.py -f urls.txt -p 5"
echo ""