#!/bin/bash
# Build and run Video-LLaVA Docker service

set -e

echo "=============================================="
echo "Building Video-LLaVA Docker Service"
echo "=============================================="

# Change to script directory
cd "$(dirname "$0")"

# Build the Docker image
echo "1. Building Docker image..."
docker-compose build --no-cache

# Start the service
echo "2. Starting Video-LLaVA service..."
docker-compose up -d

# Wait for service to be ready
echo "3. Waiting for service to start (this may take a few minutes)..."
max_attempts=60
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8004/health > /dev/null 2>&1; then
        echo "✅ Service is ready!"
        break
    fi
    echo -n "."
    sleep 5
    attempt=$((attempt + 1))
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Service failed to start within timeout"
    echo "Checking logs..."
    docker-compose logs --tail=50
    exit 1
fi

# Show service status
echo ""
echo "=============================================="
echo "Video-LLaVA Service Status"
echo "=============================================="
curl -s http://localhost:8004/health | python3 -m json.tool

echo ""
echo "=============================================="
echo "Service Information"
echo "=============================================="
echo "✅ Service URL: http://localhost:8004"
echo "✅ API Docs: http://localhost:8004/docs"
echo "✅ Container: video-llava-service"
echo ""
echo "Useful commands:"
echo "  - View logs: docker-compose logs -f"
echo "  - Stop service: docker-compose down"
echo "  - Restart service: docker-compose restart"
echo "=============================================="