# Production Deployment Guide - TikTok Video Analysis System

## Overview

This guide covers production deployment of the GPU-optimized TikTok Video Analysis System. The system is designed for high-performance video analysis with near-realtime processing capabilities.

## System Requirements

### Hardware Requirements

#### Minimum Requirements
- **GPU**: NVIDIA RTX 3090 (24GB VRAM) or equivalent
- **CPU**: 8-core processor (Intel i7-9700K or AMD Ryzen 7 3700X)
- **RAM**: 32GB DDR4
- **Storage**: 500GB NVMe SSD
- **Network**: 1Gbps connection for TikTok downloads

#### Recommended Requirements
- **GPU**: NVIDIA Quadro RTX 8000 (48GB VRAM) or RTX A6000 (48GB)
- **CPU**: 16-core processor (Intel i9-12900K or AMD Ryzen 9 5950X)
- **RAM**: 64GB DDR4
- **Storage**: 1TB NVMe SSD + 2TB HDD for results storage
- **Network**: 10Gbps connection

#### Enterprise Requirements
- **Multi-GPU**: 2x RTX A100 (80GB VRAM each)
- **CPU**: 32-core Xeon or EPYC processor
- **RAM**: 128GB DDR4/DDR5
- **Storage**: 2TB NVMe SSD + 10TB NAS for long-term storage
- **Network**: 25Gbps+ with load balancing

### Software Requirements

#### Operating System
- **Primary**: Ubuntu 22.04 LTS (recommended)
- **Alternative**: Ubuntu 20.04 LTS, CentOS 8, RHEL 8
- **Not Supported**: Windows (due to GPU optimization dependencies)

#### CUDA and Drivers
```bash
# NVIDIA Driver (minimum version 535.x)
nvidia-driver-535

# CUDA Toolkit 12.4+
cuda-toolkit-12-4

# cuDNN 8.9+
libcudnn8-dev
```

#### Python Environment
- **Python**: 3.10.x (required for optimal ML model compatibility)
- **pip**: Latest version
- **Virtual Environment**: Recommended for isolation

## Installation Guide

### 1. System Preparation

#### Update System
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y software-properties-common apt-transport-https ca-certificates
```

#### Install NVIDIA Drivers and CUDA
```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install NVIDIA drivers
sudo apt install -y nvidia-driver-535 nvidia-dkms-535

# Install CUDA toolkit
sudo apt install -y cuda-toolkit-12-4

# Install cuDNN
sudo apt install -y libcudnn8-dev

# Reboot to load drivers
sudo reboot
```

#### Verify GPU Installation
```bash
# Check NVIDIA driver
nvidia-smi

# Verify CUDA
nvcc --version

# Test GPU compute capability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Application Installation

#### Clone Repository
```bash
# Create application directory
sudo mkdir -p /opt/tiktok_analysis
sudo chown $USER:$USER /opt/tiktok_analysis
cd /opt/tiktok_analysis

# Clone repository (replace with actual repository)
git clone [REPOSITORY_URL] .

# Alternative: Upload deployment package
# tar -xzf tiktok_analysis_system.tar.gz
```

#### Python Environment Setup
```bash
# Install Python 3.10
sudo apt install -y python3.10 python3.10-venv python3.10-dev

# Create virtual environment
python3.10 -m venv /opt/tiktok_analysis/venv
source /opt/tiktok_analysis/venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Install Dependencies
```bash
# Install from requirements.txt
pip install -r requirements.txt

# Critical packages installed:
# - torch>=2.0.0 (with CUDA support)
# - transformers>=4.30.0
# - fastapi>=0.100.0
# - ultralytics>=8.0.0
# - whisper>=1.1.10
# - opencv-python>=4.8.0
# - [50+ additional ML dependencies]
```

#### Download ML Models
```bash
# Models are downloaded automatically on first run
# Pre-download for faster startup:
python3 -c "
from transformers import Qwen2VLForConditionalGeneration
model = Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')
print('Qwen2-VL downloaded successfully')
"

# Other models download automatically:
# - YOLOv8x weights (~131MB)
# - Whisper Large V3 (~1.5GB)
# - Additional models (~5GB total)
```

### 3. Configuration

#### Environment Configuration
```bash
# Copy and customize environment file
cp .env.example .env

# Edit configuration
nano .env
```

```bash
# .env file contents
# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.9'

# System Configuration
MAX_CONCURRENT_VIDEOS=1
GPU_MEMORY_LIMIT=0.9
API_PORT=8003
API_HOST=0.0.0.0

# Storage Configuration
RESULTS_DIR=/opt/tiktok_analysis/results
TEMP_DIR=/tmp/tiktok_analysis
LOG_DIR=/opt/tiktok_analysis/logs

# TikTok Download Configuration
DOWNLOAD_DIR=/opt/tiktok_analysis/downloads
MAX_VIDEO_DURATION=300
DOWNLOAD_TIMEOUT=30

# Performance Configuration
ENABLE_MODEL_CACHING=true
ENABLE_GPU_OPTIMIZATION=true
MAX_PROCESSING_TIME=600
```

#### Directory Structure Setup
```bash
# Create required directories
mkdir -p /opt/tiktok_analysis/{results,logs,downloads,backups,temp}

# Set permissions
chown -R $USER:$USER /opt/tiktok_analysis
chmod -R 755 /opt/tiktok_analysis
```

#### GPU Configuration
```bash
# Ensure critical environment setup
source fix_ffmpeg_env.sh

# Verify GPU configuration
python3 -c "
import torch
print(f'GPU available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU name: {torch.cuda.get_device_name(0)}')
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"
```

### 4. Service Configuration

#### Create Systemd Service
```bash
sudo nano /etc/systemd/system/tiktok-analysis.service
```

```ini
[Unit]
Description=TikTok Video Analysis System
After=network.target
Wants=network.target

[Service]
Type=simple
User=tiktok-analysis
Group=tiktok-analysis
WorkingDirectory=/opt/tiktok_analysis
Environment=PATH=/opt/tiktok_analysis/venv/bin
ExecStartPre=/bin/bash -c 'source /opt/tiktok_analysis/fix_ffmpeg_env.sh'
ExecStart=/opt/tiktok_analysis/venv/bin/python api/stable_production_api_multiprocess.py
ExecStop=/bin/kill -TERM $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=tiktok-analysis

# Resource limits
LimitNOFILE=65536
LimitMEMLOCK=infinity

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/tiktok_analysis

[Install]
WantedBy=multi-user.target
```

#### Create Dedicated User
```bash
# Create system user for security
sudo useradd -r -s /bin/false -d /opt/tiktok_analysis tiktok-analysis
sudo chown -R tiktok-analysis:tiktok-analysis /opt/tiktok_analysis
```

#### Enable and Start Service
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable tiktok-analysis

# Start service
sudo systemctl start tiktok-analysis

# Check status
sudo systemctl status tiktok-analysis

# View logs
sudo journalctl -u tiktok-analysis -f
```

### 5. Load Balancing and High Availability

#### Nginx Configuration
```bash
# Install Nginx
sudo apt install -y nginx

# Create configuration
sudo nano /etc/nginx/sites-available/tiktok-analysis
```

```nginx
upstream tiktok_analysis {
    # Single server for GPU optimization
    server 127.0.0.1:8003 max_fails=3 fail_timeout=30s;
    
    # Future: Multiple servers with load balancing
    # server 127.0.0.1:8004 max_fails=3 fail_timeout=30s;
    # server 127.0.0.1:8005 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/tiktok-analysis.crt;
    ssl_certificate_key /etc/ssl/private/tiktok-analysis.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    # Large file uploads for videos
    client_max_body_size 1G;
    client_body_timeout 300s;
    
    # Proxy settings
    proxy_read_timeout 600s;
    proxy_connect_timeout 60s;
    proxy_send_timeout 600s;
    
    location / {
        proxy_pass http://tiktok_analysis;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Long timeout for video processing
        proxy_read_timeout 600s;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://tiktok_analysis/health;
        access_log off;
    }
    
    # Static files (if needed)
    location /static/ {
        alias /opt/tiktok_analysis/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/tiktok-analysis /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 6. Monitoring and Logging

#### System Monitoring
```bash
# Install monitoring tools
sudo apt install -y htop iotop nethogs

# GPU monitoring
pip install nvidia-ml-py3

# Create monitoring script
nano /opt/tiktok_analysis/monitor.py
```

```python
#!/usr/bin/env python3
import time
import psutil
import nvidia_ml_py3 as nvml
import json
from datetime import datetime

def monitor_system():
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    
    while True:
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics
        gpu_util = nvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_memory = nvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': disk.percent,
            'gpu_utilization': gpu_util.gpu,
            'gpu_memory_percent': (gpu_memory.used / gpu_memory.total) * 100,
            'gpu_temperature': gpu_temp
        }
        
        # Log metrics
        with open('/opt/tiktok_analysis/logs/system_metrics.json', 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        time.sleep(60)  # Log every minute

if __name__ == '__main__':
    monitor_system()
```

#### Logging Configuration
```bash
# Configure logrotate
sudo nano /etc/logrotate.d/tiktok-analysis
```

```
/opt/tiktok_analysis/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 tiktok-analysis tiktok-analysis
    postrotate
        systemctl reload tiktok-analysis
    endscript
}
```

### 7. Security Configuration

#### Firewall Setup
```bash
# Install UFW
sudo apt install -y ufw

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH
sudo ufw allow ssh

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow internal API port (if needed)
# sudo ufw allow from 192.168.1.0/24 to any port 8003

# Enable firewall
sudo ufw enable
```

#### SSL Certificate Setup
```bash
# Install Certbot for Let's Encrypt
sudo apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add line: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 8. Backup and Recovery

#### Automated Backup Script
```bash
nano /opt/tiktok_analysis/backup.sh
```

```bash
#!/bin/bash
BACKUP_DIR="/opt/backups/tiktok-analysis"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$DATE.tar.gz"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup application
tar -czf $BACKUP_FILE \
    --exclude='/opt/tiktok_analysis/venv' \
    --exclude='/opt/tiktok_analysis/downloads' \
    --exclude='/opt/tiktok_analysis/temp' \
    /opt/tiktok_analysis

# Backup database (if applicable)
# mysqldump -u root -p database_name > $BACKUP_DIR/db_$DATE.sql

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "backup_*.tar.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_FILE"
```

```bash
# Make executable and schedule
chmod +x /opt/tiktok_analysis/backup.sh

# Add to crontab
crontab -e
# Add line: 0 2 * * * /opt/tiktok_analysis/backup.sh
```

## Performance Tuning

### 1. GPU Optimization

#### NVIDIA MPS Setup
```bash
# Enable MPS for maximum GPU utilization
sudo nano /etc/systemd/system/nvidia-mps.service
```

```ini
[Unit]
Description=NVIDIA Multi-Process Service
After=nvidia-persistenced.service

[Service]
Type=forking
ExecStart=/usr/bin/nvidia-cuda-mps-control -d
ExecStop=/usr/bin/echo quit | /usr/bin/nvidia-cuda-mps-control
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable nvidia-mps
sudo systemctl start nvidia-mps
```

### 2. System Optimization

#### Kernel Parameters
```bash
# Optimize for ML workloads
sudo nano /etc/sysctl.d/99-tiktok-analysis.conf
```

```
# Memory management
vm.swappiness=10
vm.vfs_cache_pressure=50

# Network optimization
net.core.rmem_max=134217728
net.core.wmem_max=134217728
net.ipv4.tcp_rmem=4096 87380 134217728
net.ipv4.tcp_wmem=4096 65536 134217728

# File limits
fs.file-max=2097152
```

#### CPU Governor
```bash
# Set performance governor
sudo apt install -y cpufrequtils
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl restart cpufrequtils
```

## Monitoring and Alerting

### 1. Health Checks

#### Automated Health Check
```bash
nano /opt/tiktok_analysis/health_check.py
```

```python
#!/usr/bin/env python3
import requests
import sys
import time

def check_health():
    try:
        response = requests.get('http://localhost:8003/health', timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'healthy':
                print("✅ Service healthy")
                return True
        print("❌ Service unhealthy")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == '__main__':
    if not check_health():
        sys.exit(1)
```

### 2. Alerting Setup

#### Email Alerts
```bash
# Install mailutils
sudo apt install -y mailutils

# Configure alerts
nano /opt/tiktok_analysis/alert.sh
```

```bash
#!/bin/bash
SERVICE_NAME="tiktok-analysis"
EMAIL="admin@your-domain.com"

if ! systemctl is-active --quiet $SERVICE_NAME; then
    echo "Service $SERVICE_NAME is down" | mail -s "Service Alert" $EMAIL
    systemctl restart $SERVICE_NAME
fi
```

## Scaling and Load Distribution

### Horizontal Scaling

#### Multi-GPU Setup
```python
# configs/multi_gpu_config.py
MULTI_GPU_CONFIG = {
    'gpu_worker_0_gpu0': ['qwen2_vl_temporal'],           # GPU 0
    'gpu_worker_1_gpu0': ['object_detection'],           # GPU 0
    'gpu_worker_0_gpu1': ['background_segmentation'],    # GPU 1
    'gpu_worker_1_gpu1': ['camera_analysis'],            # GPU 1
}

# Environment
export CUDA_VISIBLE_DEVICES=0,1
```

#### Multiple Server Deployment
```bash
# Server 1: Primary processing
# Server 2: Backup/overflow processing
# Server 3: Analytics and monitoring

# Load balancer configuration for multiple servers
upstream tiktok_analysis {
    server server1.internal:8003 weight=3;
    server server2.internal:8003 weight=2;
    server server3.internal:8003 weight=1 backup;
}
```

## Troubleshooting

### Common Issues

#### 1. GPU Out of Memory
```bash
# Check GPU usage
nvidia-smi

# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Restart service
sudo systemctl restart tiktok-analysis
```

#### 2. Service Won't Start
```bash
# Check logs
sudo journalctl -u tiktok-analysis -n 100

# Check configuration
source /opt/tiktok_analysis/fix_ffmpeg_env.sh
python3 /opt/tiktok_analysis/api/stable_production_api_multiprocess.py

# Check permissions
sudo chown -R tiktok-analysis:tiktok-analysis /opt/tiktok_analysis
```

#### 3. Performance Issues
```bash
# Check system resources
htop
nvidia-smi

# Monitor I/O
iotop

# Check network
netstat -tulpn | grep 8003
```

This deployment guide ensures reliable, secure, and performant operation of the TikTok Video Analysis System in production environments.