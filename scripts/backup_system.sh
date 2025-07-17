#!/bin/bash
# Complete System Backup Script for TikTok Analysis System
# Backs up entire system to Backblaze B2

set -e

# Colors and configuration
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Backup configuration
SYSTEM_DIR="/home/user/tiktok_production"
BACKUP_NAME="tiktok-analysis-system-$(date +%Y%m%d_%H%M%S)"
B2_BUCKET="tiktok-analysis-backup"
TEMP_DIR="/tmp/backup_$BACKUP_NAME"
LOG_FILE="/home/user/tiktok_production/logs/backup_$(date +%Y%m%d).log"

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
        log "${YELLOW}Cleaning up temporary files...${NC}"
        rm -rf "$TEMP_DIR"
    fi
}

# Check prerequisites
check_prerequisites() {
    log "${BLUE}Checking prerequisites...${NC}"
    
    # Check B2 CLI
    if ! command -v b2 &> /dev/null; then
        error_exit "B2 CLI not found. Please install and configure B2 CLI first."
    fi
    
    # Check B2 authorization
    if ! b2 account get &> /dev/null; then
        error_exit "B2 not authorized. Please run 'b2 account authorize' first."
    fi
    
    # Check bucket access
    if ! b2 bucket list | grep -q "$B2_BUCKET"; then
        error_exit "B2 bucket '$B2_BUCKET' not found or not accessible."
    fi
    
    # Check disk space
    REQUIRED_SPACE_GB=20
    AVAILABLE_SPACE_GB=$(df /tmp | awk 'NR==2 {print int($4/1024/1024)}')
    if [[ $AVAILABLE_SPACE_GB -lt $REQUIRED_SPACE_GB ]]; then
        error_exit "Insufficient disk space. Need ${REQUIRED_SPACE_GB}GB, have ${AVAILABLE_SPACE_GB}GB"
    fi
    
    log "${GREEN}Prerequisites check passed${NC}"
}

# Create backup directory structure
create_backup_structure() {
    log "${BLUE}Creating backup structure...${NC}"
    
    mkdir -p "$TEMP_DIR"/{system,configs,utils,docs,scripts,logs,results_sample}
    
    log "${GREEN}Backup structure created at $TEMP_DIR${NC}"
}

# Backup system files
backup_system_files() {
    log "${BLUE}Backing up system files...${NC}"
    
    # Critical system files
    cp "$SYSTEM_DIR/ml_analyzer_registry_complete.py" "$TEMP_DIR/system/"
    cp "$SYSTEM_DIR/fix_ffmpeg_env.sh" "$TEMP_DIR/system/"
    cp "$SYSTEM_DIR/requirements.txt" "$TEMP_DIR/system/"
    cp "$SYSTEM_DIR/README_COMPLETE.md" "$TEMP_DIR/system/"
    cp "$SYSTEM_DIR/CLAUDE.md" "$TEMP_DIR/system/"
    
    # API files
    if [[ -d "$SYSTEM_DIR/api" ]]; then
        cp -r "$SYSTEM_DIR/api" "$TEMP_DIR/system/"
    fi
    
    # Analyzer files (selective backup - exclude large models)
    if [[ -d "$SYSTEM_DIR/analyzers" ]]; then
        mkdir -p "$TEMP_DIR/system/analyzers"
        find "$SYSTEM_DIR/analyzers" -name "*.py" -exec cp {} "$TEMP_DIR/system/analyzers/" \;
    fi
    
    log "${GREEN}System files backed up${NC}"
}

# Backup configuration files
backup_configs() {
    log "${BLUE}Backing up configuration files...${NC}"
    
    if [[ -d "$SYSTEM_DIR/configs" ]]; then
        cp -r "$SYSTEM_DIR/configs"/* "$TEMP_DIR/configs/"
    fi
    
    # Environment configuration
    if [[ -f "$SYSTEM_DIR/.env" ]]; then
        cp "$SYSTEM_DIR/.env" "$TEMP_DIR/configs/"
    fi
    
    log "${GREEN}Configuration files backed up${NC}"
}

# Backup utilities
backup_utils() {
    log "${BLUE}Backing up utility files...${NC}"
    
    if [[ -d "$SYSTEM_DIR/utils" ]]; then
        cp -r "$SYSTEM_DIR/utils"/* "$TEMP_DIR/utils/"
    fi
    
    log "${GREEN}Utility files backed up${NC}"
}

# Backup documentation
backup_documentation() {
    log "${BLUE}Backing up documentation...${NC}"
    
    if [[ -d "$SYSTEM_DIR/docs" ]]; then
        cp -r "$SYSTEM_DIR/docs"/* "$TEMP_DIR/docs/"
    fi
    
    log "${GREEN}Documentation backed up${NC}"
}

# Backup scripts
backup_scripts() {
    log "${BLUE}Backing up scripts...${NC}"
    
    if [[ -d "$SYSTEM_DIR/scripts" ]]; then
        cp -r "$SYSTEM_DIR/scripts"/* "$TEMP_DIR/scripts/"
    fi
    
    # Backup optimization files
    for file in apply_optimizations.py test_optimizations.py monitoring_dashboard.py; do
        if [[ -f "$SYSTEM_DIR/$file" ]]; then
            cp "$SYSTEM_DIR/$file" "$TEMP_DIR/scripts/"
        fi
    done
    
    log "${GREEN}Scripts backed up${NC}"
}

# Backup logs (recent only)
backup_logs() {
    log "${BLUE}Backing up recent logs...${NC}"
    
    if [[ -d "$SYSTEM_DIR/logs" ]]; then
        # Only backup logs from last 7 days
        find "$SYSTEM_DIR/logs" -name "*.log" -mtime -7 -exec cp {} "$TEMP_DIR/logs/" \;
        
        # Always include the latest test results
        if [[ -f "$SYSTEM_DIR/test_results.txt" ]]; then
            cp "$SYSTEM_DIR/test_results.txt" "$TEMP_DIR/logs/"
        fi
    fi
    
    log "${GREEN}Recent logs backed up${NC}"
}

# Backup sample results
backup_sample_results() {
    log "${BLUE}Backing up sample analysis results...${NC}"
    
    if [[ -d "$SYSTEM_DIR/results" ]]; then
        # Backup only the 3 most recent result files (they can be large)
        find "$SYSTEM_DIR/results" -name "*.json" -type f -printf '%T@ %p\n' | \
        sort -nr | head -3 | cut -d' ' -f2- | \
        while read -r file; do
            cp "$file" "$TEMP_DIR/results_sample/"
        done
    fi
    
    log "${GREEN}Sample results backed up${NC}"
}

# Create backup metadata
create_backup_metadata() {
    log "${BLUE}Creating backup metadata...${NC}"
    
    cat > "$TEMP_DIR/backup_info.json" << EOF
{
    "backup_name": "$BACKUP_NAME",
    "backup_date": "$(date -Iseconds)",
    "system_version": "$(git -C "$SYSTEM_DIR" describe --tags --always 2>/dev/null || echo 'unknown')",
    "git_commit": "$(git -C "$SYSTEM_DIR" rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "system_info": {
        "hostname": "$(hostname)",
        "os": "$(uname -a)",
        "python_version": "$(python3 --version)",
        "gpu_info": "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU')"
    },
    "backup_contents": [
        "system_files",
        "configuration_files", 
        "utility_scripts",
        "documentation",
        "backup_scripts",
        "recent_logs",
        "sample_results"
    ],
    "restore_instructions": "Extract backup and run restore_backup.sh"
}
EOF
    
    log "${GREEN}Backup metadata created${NC}"
}

# Create restore script
create_restore_script() {
    log "${BLUE}Creating restore script...${NC}"
    
    cat > "$TEMP_DIR/restore_backup.sh" << 'EOF'
#!/bin/bash
# Restore script for TikTok Analysis System backup

set -e

RESTORE_DIR="/home/user/tiktok_production"
BACKUP_DIR="$(dirname "$0")"

echo "=== TikTok Analysis System Restore ==="
echo "Backup created: $(jq -r '.backup_date' "$BACKUP_DIR/backup_info.json")"
echo "System version: $(jq -r '.system_version' "$BACKUP_DIR/backup_info.json")"
echo

read -p "This will overwrite existing files. Continue? (y/N): " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Restore cancelled"
    exit 0
fi

echo "Restoring system files..."
mkdir -p "$RESTORE_DIR"

# Restore system files
cp -r "$BACKUP_DIR/system"/* "$RESTORE_DIR/"

# Restore configs
mkdir -p "$RESTORE_DIR/configs"
cp -r "$BACKUP_DIR/configs"/* "$RESTORE_DIR/configs/"

# Restore utils
mkdir -p "$RESTORE_DIR/utils"
cp -r "$BACKUP_DIR/utils"/* "$RESTORE_DIR/utils/"

# Restore docs
mkdir -p "$RESTORE_DIR/docs"
cp -r "$BACKUP_DIR/docs"/* "$RESTORE_DIR/docs/"

# Restore scripts
mkdir -p "$RESTORE_DIR/scripts"
cp -r "$BACKUP_DIR/scripts"/* "$RESTORE_DIR/scripts/"

# Set permissions
chmod +x "$RESTORE_DIR/fix_ffmpeg_env.sh"
chmod +x "$RESTORE_DIR/scripts"/*.sh
chmod +x "$RESTORE_DIR/api"/*.py

echo "Restore complete!"
echo "Next steps:"
echo "1. Install Python dependencies: pip install -r requirements.txt"
echo "2. Configure environment: source fix_ffmpeg_env.sh"
echo "3. Start system: python3 api/stable_production_api_multiprocess.py"
EOF
    
    chmod +x "$TEMP_DIR/restore_backup.sh"
    
    log "${GREEN}Restore script created${NC}"
}

# Compress backup
compress_backup() {
    log "${BLUE}Compressing backup...${NC}"
    
    cd "$(dirname "$TEMP_DIR")"
    ARCHIVE_NAME="$BACKUP_NAME.tar.gz"
    
    tar -czf "$ARCHIVE_NAME" "$(basename "$TEMP_DIR")"
    
    ARCHIVE_SIZE=$(du -h "$ARCHIVE_NAME" | cut -f1)
    log "${GREEN}Backup compressed to $ARCHIVE_NAME (${ARCHIVE_SIZE})${NC}"
    
    echo "$PWD/$ARCHIVE_NAME"
}

# Upload to B2
upload_to_b2() {
    local archive_path="$1"
    log "${BLUE}Uploading backup to Backblaze B2...${NC}"
    
    local b2_path="backups/$(basename "$archive_path")"
    
    if b2 file upload "$B2_BUCKET" "$archive_path" "$b2_path"; then
        log "${GREEN}Successfully uploaded to B2: $b2_path${NC}"
        
        # Create latest symlink
        echo "$b2_path" > "/tmp/latest_backup.txt"
        b2 file upload "$B2_BUCKET" "/tmp/latest_backup.txt" "backups/latest.txt"
        rm -f "/tmp/latest_backup.txt"
        
        # Cleanup old backups (keep last 30)
        cleanup_old_backups
        
        # Remove local archive
        rm -f "$archive_path"
        
        return 0
    else
        error_exit "Failed to upload backup to B2"
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    log "${BLUE}Cleaning up old backups...${NC}"
    
    # List all backup files, sort by date, keep only last 30
    b2 file list-file-names "$B2_BUCKET" --prefix "backups/tiktok-analysis-system-" --json | \
    jq -r '.files[] | select(.fileName | endswith(".tar.gz")) | .fileName' | \
    sort | head -n -30 | \
    while read -r old_backup; do
        if [[ -n "$old_backup" ]]; then
            log "${YELLOW}Deleting old backup: $old_backup${NC}"
            FILE_ID=$(b2 file list-file-names "$B2_BUCKET" --prefix "$old_backup" --json | jq -r '.files[0].fileId')
            b2 file delete-file-version "$B2_BUCKET" "$old_backup" "$FILE_ID"
        fi
    done
}

# Send notification
send_notification() {
    local status="$1"
    local message="$2"
    
    # Log the notification
    log "${GREEN}Backup $status: $message${NC}"
    
    # You can add email notification here if needed
    # echo "$message" | mail -s "TikTok Analysis Backup $status" admin@example.com
}

# Main backup process
main() {
    log "${GREEN}=== Starting TikTok Analysis System Backup ===${NC}"
    log "Backup name: $BACKUP_NAME"
    log "Target bucket: $B2_BUCKET"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run backup steps
    check_prerequisites
    create_backup_structure
    backup_system_files
    backup_configs
    backup_utils
    backup_documentation
    backup_scripts
    backup_logs
    backup_sample_results
    create_backup_metadata
    create_restore_script
    
    # Compress and upload
    archive_path=$(compress_backup)
    upload_to_b2 "$archive_path"
    
    # Success notification
    send_notification "COMPLETED" "System backup completed successfully: $BACKUP_NAME"
    
    log "${GREEN}=== Backup Process Complete ===${NC}"
    log "Backup uploaded to: B2://$B2_BUCKET/backups/$BACKUP_NAME.tar.gz"
    log "Log file: $LOG_FILE"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi