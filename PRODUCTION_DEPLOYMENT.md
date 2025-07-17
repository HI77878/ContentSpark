# TikTok Video Analyzer - Production Deployment Guide

## Overview

This guide covers the deployment and operation of the TikTok Video Analyzer production system, which processes videos through 22 ML analyzers achieving <3x realtime performance with 95%+ reconstruction capability.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │   API v2     │  │  Monitoring  │  │ Batch Processor │ │
│  │  Port 8004   │  │  Port 5000   │  │   Workers x3    │ │
│  └──────────────┘  └──────────────┘  └─────────────────┘ │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │              Production Engine                        │ │
│  │         22 Active ML Analyzers (GPU)                 │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │ PostgreSQL  │  │   Results    │  │ Quality Monitor │ │
│  │  Database   │  │  Storage     │  │   (Auto QA)     │ │
│  └─────────────┘  └──────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Ubuntu 20.04 or later
- NVIDIA GPU (Quadro RTX 8000 or similar with 40GB+ VRAM)
- CUDA 11.8+
- Python 3.10+
- PostgreSQL 14+ (optional, falls back to SQLite)
- 100GB+ free disk space
- Systemd for service management

## Installation Steps

### 1. System Setup

```bash
# Clone repository
cd /home/user
git clone https://github.com/your-org/tiktok_production.git
cd tiktok_production

# Install system dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv ffmpeg postgresql postgresql-contrib nginx

# Create Python environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install NVIDIA drivers and CUDA (if not already installed)
# Follow NVIDIA's official guide for your system
```

### 2. Database Setup (PostgreSQL)

```bash
# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE tiktok_analyzer;
CREATE USER tiktok_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE tiktok_analyzer TO tiktok_user;
EOF

# Apply schema
psql -U tiktok_user -d tiktok_analyzer -f production_setup/database_schema.sql
```

### 3. Environment Configuration

Create `/home/user/tiktok_production/.env`:

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=tiktok_analyzer
DB_USER=tiktok_user
DB_PASSWORD=secure_password

# GPU
CUDA_VISIBLE_DEVICES=0
TF_FORCE_GPU_ALLOW_GROWTH=true
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# FFmpeg fixes
OPENCV_FFMPEG_CAPTURE_OPTIONS=protocol_whitelist=file,http,https,tcp,tls
OPENCV_VIDEOIO_PRIORITY_BACKEND=4
OPENCV_FFMPEG_MULTITHREADED=0
```

### 4. Service Installation

```bash
# Copy systemd service
sudo cp production_setup/tiktok-analyzer.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start service
sudo systemctl enable tiktok-analyzer
sudo systemctl start tiktok-analyzer

# Check status
sudo systemctl status tiktok-analyzer
```

### 5. Monitoring Setup

```bash
# Create monitoring service
sudo tee /etc/systemd/system/tiktok-monitoring.service << EOF
[Unit]
Description=TikTok Analyzer Monitoring Dashboard
After=network.target

[Service]
Type=simple
User=user
WorkingDirectory=/home/user/tiktok_production
ExecStart=/home/user/tiktok_production/venv/bin/python production_setup/monitoring_dashboard.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start monitoring
sudo systemctl enable tiktok-monitoring
sudo systemctl start tiktok-monitoring
```

### 6. Quality Monitor Setup

```bash
# Create quality monitor service
sudo tee /etc/systemd/system/tiktok-quality.service << EOF
[Unit]
Description=TikTok Analyzer Quality Monitor
After=tiktok-analyzer.service

[Service]
Type=simple
User=user
WorkingDirectory=/home/user/tiktok_production
ExecStart=/home/user/tiktok_production/venv/bin/python production_setup/quality_monitor.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start quality monitor
sudo systemctl enable tiktok-quality
sudo systemctl start tiktok-quality
```

## Usage Guide

### Starting/Stopping Services

```bash
# Main API
sudo systemctl start tiktok-analyzer
sudo systemctl stop tiktok-analyzer
sudo systemctl restart tiktok-analyzer

# Check logs
sudo journalctl -u tiktok-analyzer -f

# Monitoring
sudo systemctl start tiktok-monitoring
sudo systemctl stop tiktok-monitoring

# Quality Monitor
sudo systemctl start tiktok-quality
sudo systemctl stop tiktok-quality
```

### API Endpoints

#### Core Endpoints (Port 8003/8004)

```bash
# Analyze single video
curl -X POST http://localhost:8004/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/video.mp4",
    "tiktok_url": "https://www.tiktok.com/@user/video/123"
  }'

# Batch analysis
curl -X POST http://localhost:8004/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://www.tiktok.com/@user1/video/123",
      "https://www.tiktok.com/@user2/video/456"
    ],
    "priority": "high"
  }'

# Check job status
curl http://localhost:8004/analyze/status/{job_id}

# Search videos
curl -X POST http://localhost:8004/videos/search \
  -H "Content-Type: application/json" \
  -d '{
    "creator_username": "username",
    "limit": 10
  }'

# Export results
curl http://localhost:8004/export/{video_id}?format=csv -o results.csv

# System health
curl http://localhost:8004/health/detailed
```

### Batch Processing

```bash
# Start batch processor
cd /home/user/tiktok_production
python3 production_setup/batch_processor.py start -w 3

# Add videos from file
python3 production_setup/batch_processor.py add -f video_urls.txt -p high

# Check queue status
python3 production_setup/batch_processor.py queue

# Check specific job
python3 production_setup/batch_processor.py status -j job_123456
```

### Monitoring Access

- **Dashboard**: http://localhost:5000
- **Metrics API**: http://localhost:5000/api/metrics/current
- **Alerts**: http://localhost:5000/api/alerts

## Performance Tuning

### GPU Optimization

```python
# In configs/gpu_groups_config.py
GPU_MEMORY_CONFIG = {
    'max_concurrent': {
        'stage1': 3,  # Heavy models
        'stage2': 4,  # Medium models
        'stage3': 5,  # Light models
        'stage4': 8,  # Fast models
        'cpu': 12     # CPU analyzers
    },
    'batch_sizes': {
        'qwen2_vl_temporal': 3,
        'object_detection': 32,
        'text_overlay': 32,
        # Adjust based on your GPU
    }
}
```

### Analyzer Timing Adjustments

```python
# In configs/gpu_groups_config.py
ANALYZER_TIMINGS = {
    'qwen2_vl_temporal': 60.0,  # Reduce if too slow
    'object_detection': 25.0,
    # Update based on your performance
}
```

### System Resources

```bash
# Increase file descriptors
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# GPU persistence mode
sudo nvidia-smi -pm 1

# Set GPU clock speeds (optional)
sudo nvidia-smi -ac 5001,1590  # Memory,Graphics clocks
```

## Troubleshooting

### Common Issues

#### 1. FFmpeg pthread error
```bash
# Always run before starting services
source /home/user/tiktok_production/fix_ffmpeg_env.sh
```

#### 2. GPU Memory Issues
```bash
# Check GPU usage
nvidia-smi

# Clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache()"

# Restart service
sudo systemctl restart tiktok-analyzer
```

#### 3. Slow Processing
```bash
# Check analyzer performance
curl http://localhost:8004/stats/performance

# Reduce batch sizes in gpu_groups_config.py
# Disable problematic analyzers
```

#### 4. Database Connection Failed
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -U tiktok_user -d tiktok_analyzer -c "SELECT 1;"

# Falls back to SQLite automatically
```

### Log Locations

- Main API: `/home/user/tiktok_production/logs/stable_multiprocess_api.log`
- Batch Processor: `/home/user/tiktok_production/logs/batch_processor.log`
- Quality Monitor: `/home/user/tiktok_production/logs/quality_monitor.log`
- System logs: `sudo journalctl -u tiktok-analyzer`

### Emergency Recovery

```bash
# Full system restart
sudo systemctl stop tiktok-analyzer tiktok-monitoring tiktok-quality
sudo nvidia-smi -r  # Reset GPU
sudo systemctl start tiktok-analyzer tiktok-monitoring tiktok-quality

# Clear stuck jobs
sqlite3 /home/user/tiktok_production/batch_jobs.db \
  "UPDATE video_jobs SET status='pending' WHERE status IN ('processing', 'downloading');"
```

## Maintenance

### Daily Tasks

1. Check monitoring dashboard for alerts
2. Review quality reports in `/home/user/tiktok_production/quality_report_*.json`
3. Monitor disk space for results storage

### Weekly Tasks

1. Clean old results: `curl -X POST http://localhost:8004/maintenance/cleanup?days_to_keep=30`
2. Update analyzer statistics: Check `/api/analyzers/stats`
3. Review error logs for patterns

### Monthly Tasks

1. Update system packages: `sudo apt update && sudo apt upgrade`
2. Update Python dependencies: `pip install -U -r requirements.txt`
3. Optimize PostgreSQL: `sudo -u postgres vacuumdb -a -z`
4. Archive old results to cold storage

## Security Considerations

1. **API Authentication**: Implement API keys or OAuth2 for production
2. **Network Security**: Use reverse proxy (nginx) with SSL
3. **Database Security**: Use strong passwords, limit connections
4. **File Permissions**: Ensure proper ownership of results files
5. **Input Validation**: The system validates TikTok URLs automatically

## Scaling Guide

### Horizontal Scaling

1. **Multiple GPU Nodes**: Deploy workers on multiple GPU servers
2. **Load Balancer**: Use nginx/HAProxy to distribute requests
3. **Shared Storage**: Use NFS/S3 for results storage
4. **Message Queue**: Replace SQLite queue with Redis/RabbitMQ

### Vertical Scaling

1. **GPU Upgrade**: More VRAM allows larger batch sizes
2. **CPU Cores**: More cores for parallel CPU analyzers
3. **RAM**: 64GB+ recommended for large batches
4. **NVMe Storage**: Faster video loading and results writing

## Monitoring Metrics

Key metrics to track:

- **Processing Time**: Target <3x realtime
- **GPU Utilization**: Target 85-95%
- **Queue Size**: Should stay <100 for real-time processing
- **Error Rate**: Should be <5% per analyzer
- **Duplicate Rate**: Should be 0% for Qwen2-VL
- **Disk Usage**: Monitor results directory growth

Access metrics at:
- Real-time: http://localhost:5000
- Historical: Query PostgreSQL database
- Alerts: Check quality_alerts.json

## Support

For issues:
1. Check logs in `/home/user/tiktok_production/logs/`
2. Review monitoring dashboard for system status
3. Check quality alerts for degradation
4. Consult troubleshooting section
5. File issues at: https://github.com/your-org/tiktok_production/issues

---

**Production Ready**: The system is now configured for 24/7 operation with automatic recovery, monitoring, and quality assurance.