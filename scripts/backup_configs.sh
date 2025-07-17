#!/bin/bash
# Configuration Files Backup Script for TikTok Analysis System
# Quick backup of just configuration files

set -e

# Colors and configuration
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Backup configuration
SYSTEM_DIR="/home/user/tiktok_production"
BACKUP_NAME="tiktok-configs-$(date +%Y%m%d_%H%M%S)"
B2_BUCKET="tiktok-analysis-backup"
TEMP_DIR="/tmp/backup_$BACKUP_NAME"
LOG_FILE="/home/user/tiktok_production/logs/backup_configs_$(date +%Y%m%d).log"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "${RED}ERROR: $1${NC}"
    cleanup
    exit 1
}

# Cleanup function
cleanup() {
    if [[ -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
    fi
}

# Main backup process
main() {
    log "${GREEN}=== Starting Configuration Backup ===${NC}"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Check B2 CLI
    if ! command -v b2 &> /dev/null; then
        error_exit "B2 CLI not found"
    fi
    
    # Create backup directory
    mkdir -p "$TEMP_DIR"
    
    log "${BLUE}Backing up configuration files...${NC}"
    
    # Backup configuration files
    if [[ -d "$SYSTEM_DIR/configs" ]]; then
        cp -r "$SYSTEM_DIR/configs" "$TEMP_DIR/"
    fi
    
    # Backup critical system files
    cp "$SYSTEM_DIR/fix_ffmpeg_env.sh" "$TEMP_DIR/" 2>/dev/null || true
    cp "$SYSTEM_DIR/ml_analyzer_registry_complete.py" "$TEMP_DIR/" 2>/dev/null || true
    cp "$SYSTEM_DIR/requirements.txt" "$TEMP_DIR/" 2>/dev/null || true
    cp "$SYSTEM_DIR/.env" "$TEMP_DIR/" 2>/dev/null || true
    
    # Create metadata
    cat > "$TEMP_DIR/backup_info.json" << EOF
{
    "backup_type": "configurations",
    "backup_name": "$BACKUP_NAME",
    "backup_date": "$(date -Iseconds)",
    "hostname": "$(hostname)"
}
EOF
    
    # Compress
    cd "$(dirname "$TEMP_DIR")"
    ARCHIVE_NAME="$BACKUP_NAME.tar.gz"
    tar -czf "$ARCHIVE_NAME" "$(basename "$TEMP_DIR")"
    
    # Upload to B2
    if b2 file upload "$B2_BUCKET" "$ARCHIVE_NAME" "config_backups/$ARCHIVE_NAME"; then
        log "${GREEN}Configuration backup uploaded successfully${NC}"
        rm -f "$ARCHIVE_NAME"
    else
        error_exit "Failed to upload configuration backup"
    fi
    
    log "${GREEN}=== Configuration Backup Complete ===${NC}"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi