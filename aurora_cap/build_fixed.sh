#!/bin/bash
# Build script for fixed AuroraCap Docker image

echo "Building AuroraCap with compatible dependencies..."
cd /home/user/tiktok_production/aurora_cap

# Build with fixed Dockerfile
docker build -f Dockerfile.fixed -t aurora-cap:fixed .

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Image size:"
    docker images aurora-cap:fixed
else
    echo "Build failed!"
    exit 1
fi