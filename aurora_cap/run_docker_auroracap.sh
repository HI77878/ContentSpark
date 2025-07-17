#!/bin/bash
# Docker wrapper script for AuroraCap

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <video_path> [output_dir]"
    echo "Example: $0 /path/to/video.mp4"
    exit 1
fi

VIDEO_PATH="$1"
VIDEO_FILENAME=$(basename "$VIDEO_PATH")
OUTPUT_DIR="${2:-/home/user/tiktok_production/aurora_cap/output}"

# Check if video exists
if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file not found: $VIDEO_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== Running AuroraCap Analysis ==="
echo "Video: $VIDEO_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

# Build Docker image if not exists
if ! docker images | grep -q "auroracap:latest"; then
    echo "Building Docker image..."
    cd /home/user/tiktok_production/aurora_cap
    docker build -f Dockerfile.auroracap -t auroracap:latest .
    if [ $? -ne 0 ]; then
        echo "Docker build failed!"
        exit 1
    fi
fi

# Run analysis
echo "Starting analysis..."
docker run --rm \
    --gpus all \
    -v "$VIDEO_PATH:/app/videos/input.mp4:ro" \
    -v "$OUTPUT_DIR:/app/output" \
    -v "/home/user/tiktok_production/aurora_cap/.cache:/app/.cache" \
    -e CUDA_VISIBLE_DEVICES=0 \
    auroracap:latest \
    python auroracap_inference_fixed.py /app/videos/input.mp4 /app/output

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Analysis completed successfully!"
    echo "Output files:"
    ls -la "$OUTPUT_DIR"/*aurora* 2>/dev/null || echo "No output files found"
else
    echo ""
    echo "❌ Analysis failed!"
    exit 1
fi