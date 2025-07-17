#!/bin/bash
# Setup systemd service for TikTok Analyzer

echo "Setting up TikTok Analyzer systemd service..."

# Copy service file
sudo cp /home/user/tiktok_production/tiktok-analyzer.service /etc/systemd/system/

# Set correct permissions
sudo chmod 644 /etc/systemd/system/tiktok-analyzer.service

# Create log directory if not exists
mkdir -p /home/user/tiktok_production/logs

# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable tiktok-analyzer.service

# Start service
sudo systemctl start tiktok-analyzer.service

# Check status
sleep 3
sudo systemctl status tiktok-analyzer.service

echo ""
echo "Service setup complete!"
echo "Commands:"
echo "  Start:   sudo systemctl start tiktok-analyzer"
echo "  Stop:    sudo systemctl stop tiktok-analyzer"
echo "  Status:  sudo systemctl status tiktok-analyzer"
echo "  Logs:    sudo journalctl -u tiktok-analyzer -f"