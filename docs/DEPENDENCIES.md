# Dependencies Documentation - TikTok Video Analysis System

## Overview

This document provides comprehensive information about all dependencies required for the TikTok Video Analysis System. The system relies on 17 active ML analyzers requiring specialized dependencies for GPU optimization, video processing, and advanced machine learning capabilities.

## Python Version Requirements

- **Required**: Python 3.10.x
- **Recommended**: Python 3.10.12
- **Not Supported**: Python 3.9 or earlier, Python 3.11+ (compatibility issues with some ML models)

## Core ML Framework Dependencies

### PyTorch Ecosystem
```bash
# Core PyTorch with CUDA support
torch>=2.0.0,<2.5.0          # Main ML framework
torchvision>=0.15.0,<0.20.0  # Computer vision operations  
torchaudio>=2.0.0,<2.5.0     # Audio processing
```

**Installation Notes:**
- MUST be installed with CUDA support for GPU optimization
- Install command: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- Verify CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

### HuggingFace Transformers
```bash
transformers>=4.30.0,<4.45.0 # Model loading and inference
accelerate>=0.20.0,<0.35.0   # Model acceleration
```

**Critical Models Loaded:**
- **Qwen2-VL-7B-Instruct**: 14GB download, requires 16GB VRAM
- **Whisper Large V3**: 1.5GB download
- **Various sentence transformers**: 500MB - 2GB each

## Video Processing Dependencies

### Core Video Processing
```bash
opencv-python>=4.8.0,<4.10.0          # Computer vision
opencv-contrib-python>=4.8.0,<4.10.0  # Additional modules
moviepy>=1.0.3,<1.1.0                 # Video editing
imageio>=2.31.0,<2.35.0               # Image I/O
av>=10.0.0,<12.0.0                    # PyAV for video
```

**System Requirements:**
- FFmpeg must be installed system-wide
- libsm6, libxext6, libxrender-dev for OpenCV
- Hardware acceleration support recommended

### Video Download
```bash
yt-dlp>=2023.7.6              # TikTok/YouTube downloader
requests>=2.31.0,<2.32.0     # HTTP requests
```

**Configuration:**
- yt-dlp requires regular updates for TikTok compatibility
- Network timeout configured for reliable downloads
- User-agent rotation for anti-bot protection

## Specialized ML Model Dependencies

### Computer Vision Models
```bash
ultralytics>=8.0.0,<8.3.0    # YOLOv8 object detection
mediapipe>=0.10.0,<0.11.0    # Google MediaPipe
insightface>=0.7.3,<0.8.0    # Face recognition
```

**Model Storage:**
- YOLOv8x weights: ~131MB (auto-downloaded)
- MediaPipe models: ~50MB total
- InsightFace models: ~200MB

### Text Recognition
```bash
easyocr>=1.7.0,<1.8.0        # OCR engine
pytesseract>=0.3.10,<0.4.0   # Tesseract wrapper
```

**Language Support:**
- EasyOCR: English, German, Spanish, French
- Optimized for TikTok subtitle detection
- GPU acceleration enabled

### Audio Processing
```bash
librosa>=0.10.0,<0.11.0      # Audio analysis
openai-whisper>=20230314     # Speech recognition
webrtcvad>=2.0.10,<2.1.0     # Voice activity detection
```

**Audio Capabilities:**
- 44.1kHz/48kHz sample rate support
- Multiple audio codec support
- Real-time processing optimization

## GPU and Performance Dependencies

### GPU Monitoring
```bash
nvidia-ml-py3>=7.352.0        # NVIDIA GPU monitoring
pynvml>=11.0.0               # GPU management
psutil>=5.9.0,<5.10.0        # System monitoring
```

**GPU Requirements:**
- NVIDIA driver 535.x or newer
- CUDA 12.4 or newer
- cuDNN 8.9 or newer
- 24GB+ VRAM recommended (16GB minimum)

### Performance Optimization
```bash
scikit-learn>=1.3.0,<1.4.0   # ML algorithms
numpy>=1.24.0,<1.26.0        # Optimized numerical computing
scipy>=1.10.0,<1.12.0        # Scientific computing
```

**Optimization Notes:**
- NumPy compiled with Intel MKL for performance
- SciPy with BLAS/LAPACK acceleration
- Memory pool optimization for large datasets

## Web Framework Dependencies

### FastAPI Stack
```bash
fastapi>=0.100.0,<0.105.0    # Modern web framework
uvicorn>=0.22.0,<0.25.0      # ASGI server
pydantic>=2.0.0,<2.6.0       # Data validation
```

**API Features:**
- Async request handling
- WebSocket support for real-time updates
- OpenAPI documentation generation
- Request/response validation

## Database and Storage

### Database Support
```bash
sqlalchemy>=2.0.0,<2.1.0     # ORM
psycopg2-binary>=2.9.6       # PostgreSQL
redis>=4.5.0,<5.1.0          # Caching
```

**Storage Configuration:**
- PostgreSQL for metadata and results
- Redis for caching and session management
- Local file storage for video files and analysis results

## Development and Testing

### Code Quality
```bash
pytest>=7.4.0,<7.5.0         # Testing framework
black>=23.0.0,<24.0.0        # Code formatting
flake8>=6.0.0,<6.1.0         # Linting
mypy>=1.4.0,<1.6.0           # Type checking
```

### Performance Profiling
```bash
memory-profiler>=0.60.0      # Memory profiling
line-profiler>=4.0.0         # Line profiling
py-spy>=0.3.14               # Sampling profiler
```

## System Dependencies (Non-Python)

### Required System Packages (Ubuntu/Debian)
```bash
# Video processing
sudo apt install -y ffmpeg libsm6 libxext6 libxrender-dev

# Audio support  
sudo apt install -y libasound2-dev portaudio19-dev

# GPU support
sudo apt install -y nvidia-driver-535 cuda-toolkit-12-4 libcudnn8-dev

# Build tools
sudo apt install -y build-essential cmake pkg-config

# Image processing
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev
```

### Required System Packages (CentOS/RHEL)
```bash
# Enable EPEL repository
sudo yum install -y epel-release

# Video processing
sudo yum install -y ffmpeg-devel opencv-devel

# Development tools
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake pkg-config

# GPU drivers (manual installation required)
# Download from NVIDIA website
```

## Installation Guide

### 1. System Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3.10 python3.10-venv python3.10-dev
sudo apt install -y ffmpeg libsm6 libxext6 libxrender-dev
```

### 2. Virtual Environment
```bash
# Create virtual environment
python3.10 -m venv /opt/tiktok_analysis/venv
source /opt/tiktok_analysis/venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. Core Dependencies
```bash
# Install PyTorch with CUDA support (CRITICAL)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4. All Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# Verify critical imports
python -c "
import torch, transformers, cv2, whisper, ultralytics
print('All critical dependencies imported successfully')
"
```

## Model Downloads and Caching

### Automatic Model Downloads
Models are downloaded automatically on first use:

1. **Qwen2-VL-7B-Instruct** (14GB)
   - Location: `~/.cache/huggingface/transformers/`
   - First load: ~15-30 minutes
   - Subsequent loads: ~10-15 seconds with caching

2. **YOLOv8x weights** (131MB)
   - Location: `~/.cache/ultralytics/`
   - Download: ~30 seconds
   - Load time: ~2 seconds

3. **Whisper Large V3** (1.5GB)
   - Location: `~/.cache/whisper/`
   - Download: ~5-10 minutes
   - Load time: ~5 seconds

### Pre-download Script
```python
# pre_download_models.py
import os
from transformers import Qwen2VLForConditionalGeneration
from ultralytics import YOLO
import whisper

def download_all_models():
    print("Downloading Qwen2-VL...")
    model = Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')
    
    print("Downloading YOLOv8...")
    yolo = YOLO('yolov8x.pt')
    
    print("Downloading Whisper...")
    whisper_model = whisper.load_model("large-v3")
    
    print("All models downloaded successfully!")

if __name__ == "__main__":
    download_all_models()
```

## Version Compatibility Matrix

| Component | Minimum | Recommended | Maximum | Notes |
|-----------|---------|-------------|---------|-------|
| Python | 3.10.0 | 3.10.12 | 3.10.x | ML model compatibility |
| CUDA | 12.0 | 12.4 | 12.6 | GPU optimization |
| PyTorch | 2.0.0 | 2.2.0 | 2.4.x | Model support |
| Transformers | 4.30.0 | 4.40.0 | 4.44.x | Qwen2-VL support |
| OpenCV | 4.8.0 | 4.9.0 | 4.9.x | Video processing |
| FastAPI | 0.100.0 | 0.104.0 | 0.104.x | API features |

## Memory Requirements

### GPU Memory (VRAM)
- **Minimum**: 16GB (basic operation)
- **Recommended**: 24GB (optimal performance)  
- **Enterprise**: 48GB (multiple concurrent analyses)

### System Memory (RAM)
- **Minimum**: 32GB
- **Recommended**: 64GB
- **Enterprise**: 128GB

### Storage Requirements
- **Base installation**: 10GB
- **Model cache**: 20GB
- **Working space**: 100GB+
- **Long-term storage**: 1TB+ for results

## Troubleshooting Dependencies

### Common Installation Issues

#### CUDA Not Found
```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Add to PATH if needed
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### PyTorch CPU-Only Installation
```bash
# Uninstall CPU version
pip uninstall torch torchvision torchaudio

# Install CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### OpenCV Import Errors
```bash
# Install system dependencies
sudo apt install -y libglib2.0-0 libgl1-mesa-glx

# Reinstall OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python
```

#### Model Download Failures
```bash
# Clear corrupted cache
rm -rf ~/.cache/huggingface/transformers/
rm -rf ~/.cache/whisper/

# Set proxy if needed
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
```

## Security Considerations

### Model Integrity
- All models downloaded from verified sources
- Checksum validation for critical models
- No remote model loading during inference

### Dependency Security
- Regular security updates for all packages
- Pinned versions for stability
- Vulnerability scanning recommended

### Network Security
- TikTok downloads through secure channels
- No external API calls during analysis
- Local processing only

This dependency system ensures reliable, secure, and high-performance operation of the TikTok Video Analysis System.