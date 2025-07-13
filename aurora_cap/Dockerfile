# Dockerfile für AuroraCap (korrigierte Version basierend auf Repository-Struktur)

# Base image with PyTorch and CUDA
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone the AuroraCap repository
RUN git clone https://github.com/rese1f/aurora.git /app/aurora

# Install dependencies according to README
WORKDIR /app/aurora

# Install xtuner dependencies for aurora
RUN cd src/xtuner && pip install --no-cache-dir -e '.[all]'

# Install additional required packages
RUN pip install --no-cache-dir \
    opencv-python \
    pillow \
    av \
    einops \
    timm

# Create directories for videos and output
RUN mkdir -p /app/videos /app/output

# Copy our custom inference script
COPY ./auroracap_inference.py /app/aurora/auroracap_inference.py

# Set working directory back to aurora
WORKDIR /app/aurora

# Download the model weights (this will cache them in the image)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('wchai/AuroraCap-7B-VID-xtuner')"

# Default command
CMD ["python", "auroracap_inference.py"]