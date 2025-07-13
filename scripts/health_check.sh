#!/bin/bash
# Health check with auto-restart capability

API_URL="http://localhost:8003/health"
MAX_RETRIES=3
RETRY_DELAY=10

echo "[$(date)] Starting health check..."

# Function to check API health
check_api_health() {
    curl -s -f -m 5 $API_URL > /dev/null
    return $?
}

# Try health check with retries
for i in $(seq 1 $MAX_RETRIES); do
    if check_api_health; then
        echo "[$(date)] API health check passed"
        exit 0
    else
        echo "[$(date)] API health check failed (attempt $i/$MAX_RETRIES)"
        
        if [ $i -lt $MAX_RETRIES ]; then
            sleep $RETRY_DELAY
        fi
    fi
done

# Health check failed, attempt restart
echo "[$(date)] API is not responding after $MAX_RETRIES attempts. Attempting restart..."

# Check if using systemd
if systemctl is-active --quiet tiktok-analyzer; then
    echo "[$(date)] Restarting via systemd..."
    sudo systemctl restart tiktok-analyzer
    sleep 20
    
    # Verify restart
    if check_api_health; then
        echo "[$(date)] API successfully restarted via systemd"
        exit 0
    fi
else
    # Manual restart
    echo "[$(date)] Restarting manually..."
    cd /home/user/tiktok_production
    ./scripts/restart_services.sh restart
    sleep 20
    
    # Verify restart
    if check_api_health; then
        echo "[$(date)] API successfully restarted manually"
        exit 0
    fi
fi

echo "[$(date)] ERROR: Failed to restart API"
exit 1