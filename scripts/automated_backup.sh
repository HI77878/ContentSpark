#!/bin/bash
# Automated Backup Script for Cron Jobs
# Runs daily backup with error handling and notifications

set -e

# Configuration
SCRIPT_DIR="$(dirname "$0")"
LOG_FILE="/home/user/tiktok_production/logs/automated_backup_$(date +%Y%m%d).log"
LOCK_FILE="/tmp/tiktok_backup.lock"
MAX_RUNTIME=3600  # 1 hour timeout

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging with timestamp
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "${RED}FATAL ERROR: $1${NC}"
    send_error_notification "$1"
    cleanup
    exit 1
}

# Cleanup function
cleanup() {
    # Remove lock file
    rm -f "$LOCK_FILE"
    
    # Kill any hanging processes
    pkill -f "backup_system.sh" 2>/dev/null || true
}

# Check if another backup is running
check_lock() {
    if [[ -f "$LOCK_FILE" ]]; then
        local lock_pid=$(cat "$LOCK_FILE")
        if kill -0 "$lock_pid" 2>/dev/null; then
            error_exit "Another backup process is already running (PID: $lock_pid)"
        else
            log "${YELLOW}Removing stale lock file${NC}"
            rm -f "$LOCK_FILE"
        fi
    fi
    
    # Create lock file
    echo $$ > "$LOCK_FILE"
}

# Send error notification
send_error_notification() {
    local error_msg="$1"
    
    # Log error
    log "${RED}Backup failed: $error_msg${NC}"
    
    # Email notification (if configured)
    if command -v mail &> /dev/null && [[ -n "${BACKUP_ADMIN_EMAIL:-}" ]]; then
        echo "TikTok Analysis System backup failed at $(date): $error_msg" | \
        mail -s "Backup Failed - $(hostname)" "$BACKUP_ADMIN_EMAIL"
    fi
    
    # Webhook notification (if configured)
    if [[ -n "${BACKUP_WEBHOOK_URL:-}" ]]; then
        curl -X POST "$BACKUP_WEBHOOK_URL" \
             -H "Content-Type: application/json" \
             -d "{\"text\":\"🚨 TikTok Analysis backup failed on $(hostname): $error_msg\"}" \
             2>/dev/null || true
    fi
}

# Send success notification
send_success_notification() {
    local backup_info="$1"
    
    log "${GREEN}Backup completed successfully${NC}"
    
    # Email notification (if configured)
    if command -v mail &> /dev/null && [[ -n "${BACKUP_ADMIN_EMAIL:-}" ]]; then
        echo "TikTok Analysis System backup completed successfully at $(date): $backup_info" | \
        mail -s "Backup Successful - $(hostname)" "$BACKUP_ADMIN_EMAIL"
    fi
    
    # Webhook notification (if configured)
    if [[ -n "${BACKUP_WEBHOOK_URL:-}" ]]; then
        curl -X POST "$BACKUP_WEBHOOK_URL" \
             -H "Content-Type: application/json" \
             -d "{\"text\":\"✅ TikTok Analysis backup completed on $(hostname): $backup_info\"}" \
             2>/dev/null || true
    fi
}

# Check system health before backup
check_system_health() {
    log "${YELLOW}Checking system health...${NC}"
    
    # Check disk space (need at least 10GB free)
    local available_gb=$(df /tmp | awk 'NR==2 {print int($4/1024/1024)}')
    if [[ $available_gb -lt 10 ]]; then
        error_exit "Insufficient disk space: ${available_gb}GB available, need 10GB"
    fi
    
    # Check if system is running
    if pgrep -f "stable_production_api" > /dev/null; then
        log "${GREEN}TikTok Analysis API is running${NC}"
    else
        log "${YELLOW}TikTok Analysis API is not running${NC}"
    fi
    
    # Check GPU status
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            log "${GREEN}GPU is accessible${NC}"
        else
            log "${YELLOW}GPU check failed${NC}"
        fi
    fi
    
    # Check B2 authorization
    if ! b2 account get &> /dev/null; then
        error_exit "B2 not authorized. Please run 'b2 account authorize'"
    fi
    
    log "${GREEN}System health check passed${NC}"
}

# Cleanup old local logs
cleanup_old_logs() {
    log "${YELLOW}Cleaning up old logs...${NC}"
    
    # Delete backup logs older than 30 days
    find "/home/user/tiktok_production/logs" -name "backup_*.log" -mtime +30 -delete 2>/dev/null || true
    find "/home/user/tiktok_production/logs" -name "automated_backup_*.log" -mtime +30 -delete 2>/dev/null || true
    
    # Delete temporary files older than 7 days
    find /tmp -name "backup_*" -type d -mtime +7 -exec rm -rf {} \; 2>/dev/null || true
    
    log "${GREEN}Log cleanup completed${NC}"
}

# Backup type selection based on day
get_backup_type() {
    local day_of_week=$(date +%u)  # 1=Monday, 7=Sunday
    local day_of_month=$(date +%d)
    
    # Full backup on Sundays or 1st of month
    if [[ $day_of_week -eq 7 ]] || [[ $day_of_month -eq 01 ]]; then
        echo "full"
    else
        echo "configs"
    fi
}

# Run backup
run_backup() {
    local backup_type="$1"
    
    log "${YELLOW}Starting $backup_type backup...${NC}"
    
    # Set timeout for backup process
    timeout "$MAX_RUNTIME" "$SCRIPT_DIR/backup_system.sh" 2>&1 | tee -a "$LOG_FILE"
    local backup_exit_code=${PIPESTATUS[0]}
    
    if [[ $backup_exit_code -eq 0 ]]; then
        log "${GREEN}Backup process completed successfully${NC}"
        return 0
    elif [[ $backup_exit_code -eq 124 ]]; then
        error_exit "Backup timed out after $MAX_RUNTIME seconds"
    else
        error_exit "Backup process failed with exit code $backup_exit_code"
    fi
}

# Performance optimization before backup
optimize_for_backup() {
    log "${YELLOW}Optimizing system for backup...${NC}"
    
    # Sync filesystem
    sync
    
    # Clear system caches to free memory
    echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || true
    
    # Temporarily stop non-critical services to free resources
    systemctl stop bluetooth 2>/dev/null || true
    systemctl stop cups 2>/dev/null || true
    
    log "${GREEN}System optimization completed${NC}"
}

# Restore services after backup
restore_services() {
    log "${YELLOW}Restoring services...${NC}"
    
    # Restart services that were stopped
    systemctl start bluetooth 2>/dev/null || true
    systemctl start cups 2>/dev/null || true
    
    log "${GREEN}Services restored${NC}"
}

# Main function
main() {
    # Start logging
    mkdir -p "$(dirname "$LOG_FILE")"
    log "${GREEN}=== Starting Automated Backup ===${NC}"
    log "Hostname: $(hostname)"
    log "Date: $(date)"
    log "Script: $0"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Check for lock file
    check_lock
    
    # Load environment variables if config file exists
    if [[ -f "/home/user/tiktok_production/.env" ]]; then
        source "/home/user/tiktok_production/.env"
    fi
    
    # System health check
    check_system_health
    
    # Cleanup old logs
    cleanup_old_logs
    
    # Determine backup type
    backup_type=$(get_backup_type)
    log "${YELLOW}Backup type: $backup_type${NC}"
    
    # Optimize system for backup
    optimize_for_backup
    
    # Run the backup
    if run_backup "$backup_type"; then
        # Success notification
        backup_info="Type: $backup_type, Host: $(hostname), Date: $(date)"
        send_success_notification "$backup_info"
        
        # Restore services
        restore_services
        
        log "${GREEN}=== Automated Backup Complete ===${NC}"
    else
        # Error already handled in run_backup
        restore_services
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    "test")
        # Test mode - just run health checks
        log "${YELLOW}Running in test mode${NC}"
        check_system_health
        log "${GREEN}Test completed${NC}"
        ;;
    "force-full")
        # Force full backup regardless of schedule
        log "${YELLOW}Forcing full backup${NC}"
        backup_type="full"
        main
        ;;
    "configs-only")
        # Force config backup only
        log "${YELLOW}Forcing config-only backup${NC}"
        backup_type="configs"
        main
        ;;
    "")
        # Normal operation
        main
        ;;
    *)
        echo "Usage: $0 [test|force-full|configs-only]"
        echo "  test        - Run health checks only"
        echo "  force-full  - Force full backup"
        echo "  configs-only - Force config backup only"
        echo "  (no args)   - Normal scheduled backup"
        exit 1
        ;;
esac