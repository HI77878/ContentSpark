# Dockerfile for Video-LLaVA Service
# Optimized for LLaVA-NeXT-Video-7B with GPU support

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
RUN pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
RUN pip3 install \
    transformers==4.47.0 \
    accelerate==0.25.0 \
    bitsandbytes==0.46.0 \
    sentencepiece==0.1.99 \
    protobuf==3.20.3 \
    av==14.4.0 \
    pillow==10.2.0 \
    numpy==1.24.3 \
    opencv-python==4.9.0.80 \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    pydantic==2.5.3 \
    httpx==0.26.0

# Install flash-attn for performance (optional, may fail on some systems)
RUN pip3 install flash-attn --no-build-isolation || echo "Flash attention not installed"

# Create working directory
WORKDIR /app

# Copy service files
COPY video_llava_service.py /app/
COPY preload_model.py /app/

# Create model cache directory
RUN mkdir -p /app/models

# Pre-download the model during build (optional - can be done at runtime)
# This significantly increases image size but improves startup time
# RUN python3 preload_model.py

# Expose API port
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Run the service
CMD ["python3", "-u", "video_llava_service.py"]