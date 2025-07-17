#!/bin/bash
# run_captioning.sh - Wrapper script to run BLIP-2 captioning via Docker

# Check if video path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <video_path>"
    exit 1
fi

VIDEO_PATH="$1"
VIDEO_FILENAME=$(basename "$VIDEO_PATH")
OUTPUT_DIR="/home/user/tiktok_production/aurora_cap/output"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Docker container
docker run --rm \
    --gpus all \
    -v "$VIDEO_PATH:/videos/input.mp4:ro" \
    -v "$OUTPUT_DIR:/app/output" \
    -e ANALYSIS_TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    blip2-captioner:v3 \
    python /app/video_captioning_script.py /videos/input.mp4

# Check if the analysis was successful
if [ $? -eq 0 ]; then
    echo "Analysis completed successfully!"
    echo "Output files:"
    ls -la "$OUTPUT_DIR/${VIDEO_FILENAME}_"*
else
    echo "Analysis failed!"
    exit 1
fi