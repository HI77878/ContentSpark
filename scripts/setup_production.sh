#!/bin/bash
# Final production setup script

echo "==================================="
echo "TikTok Analyzer Production Setup"
echo "==================================="

# Check if running as user
if [ "$EUID" -eq 0 ]; then 
   echo "Please run as regular user, not root"
   exit 1
fi

# Setup systemd service
echo "Setting up systemd service..."
if [ -f /etc/systemd/system/tiktok-analyzer.service ]; then
    echo "✓ Service file already exists"
else
    echo "Installing service file (requires sudo)..."
    sudo cp /home/user/tiktok_production/tiktok-analyzer.service /etc/systemd/system/
    sudo chmod 644 /etc/systemd/system/tiktok-analyzer.service
fi

# Reload and enable service
echo "Enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable tiktok-analyzer.service

# Setup cron jobs
echo ""
echo "Setting up cron jobs..."
crontab -l > /tmp/current_cron 2>/dev/null || true
if grep -q "tiktok_analyzer" /tmp/current_cron; then
    echo "✓ Cron jobs already installed"
else
    echo "Installing cron jobs..."
    cat /home/user/tiktok_production/cron/tiktok_analyzer_cron >> /tmp/current_cron
    crontab /tmp/current_cron
    echo "✓ Cron jobs installed"
fi
rm /tmp/current_cron

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p /home/user/tiktok_production/logs/archive
mkdir -p /home/user/tiktok_production/results
mkdir -p /home/user/tiktok_production/downloads/videos

# Set permissions
chmod +x /home/user/tiktok_production/scripts/*.sh
chmod +x /home/user/tiktok_production/monitoring/*.py

# Test configuration
echo ""
echo "Testing configuration..."
source /home/user/tiktok_production/fix_ffmpeg_env.sh

# Check if API is already running
if curl -s http://localhost:8003/health > /dev/null 2>&1; then
    echo "✓ API is already running"
else
    echo "Starting API service..."
    sudo systemctl start tiktok-analyzer
    sleep 10
fi

# Final status check
echo ""
echo "==================================="
echo "Production Setup Status"
echo "==================================="

# Service status
if systemctl is-active --quiet tiktok-analyzer; then
    echo "✓ Systemd service: ACTIVE"
else
    echo "✗ Systemd service: INACTIVE"
fi

# API health
if curl -s http://localhost:8003/health > /dev/null 2>&1; then
    echo "✓ API health: OK"
else
    echo "✗ API health: FAILED"
fi

# Cron status
if crontab -l | grep -q "tiktok_analyzer"; then
    echo "✓ Cron jobs: INSTALLED"
else
    echo "✗ Cron jobs: NOT INSTALLED"
fi

# GPU status
if nvidia-smi > /dev/null 2>&1; then
    echo "✓ GPU available: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
else
    echo "✗ GPU: NOT AVAILABLE"
fi

echo "==================================="
echo ""
echo "Setup complete! System is ready for production."
echo ""
echo "Important commands:"
echo "  Status:  sudo systemctl status tiktok-analyzer"
echo "  Logs:    sudo journalctl -u tiktok-analyzer -f"
echo "  Monitor: python3 monitoring/system_monitor.py"