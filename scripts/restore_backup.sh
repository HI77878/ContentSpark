#!/bin/bash
# Restore Script for TikTok Analysis System Backups

set -e

# Colors and configuration
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
B2_BUCKET="tiktok-analysis-backup"
RESTORE_DIR="/home/user/tiktok_production"
TEMP_DIR="/tmp/restore_$(date +%s)"

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
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

# List available backups
list_backups() {
    log "${BLUE}Available backups:${NC}"
    
    if ! b2 file list-file-names "$B2_BUCKET" --prefix "backups/tiktok-analysis-system-" --json | \
         jq -r '.files[] | select(.fileName | endswith(".tar.gz")) | "\(.uploadTimestamp) \(.fileName)"' | \
         sort -nr | head -20 | nl; then
        error_exit "Failed to list backups from B2"
    fi
}

# Get latest backup
get_latest_backup() {
    b2 file download-by-name "$B2_BUCKET" "backups/latest.txt" "/tmp/latest_backup.txt" 2>/dev/null || {
        # Fallback: get most recent backup
        b2 file list-file-names "$B2_BUCKET" --prefix "backups/tiktok-analysis-system-" --json | \
        jq -r '.files[] | select(.fileName | endswith(".tar.gz")) | .fileName' | \
        sort | tail -1
        return
    }
    
    cat "/tmp/latest_backup.txt"
    rm -f "/tmp/latest_backup.txt"
}

# Download and extract backup
download_backup() {
    local backup_file="$1"
    
    log "${BLUE}Downloading backup: $backup_file${NC}"
    
    mkdir -p "$TEMP_DIR"
    local archive_path="$TEMP_DIR/$(basename "$backup_file")"
    
    if ! b2 file download-by-name "$B2_BUCKET" "$backup_file" "$archive_path"; then
        error_exit "Failed to download backup from B2"
    fi
    
    log "${BLUE}Extracting backup...${NC}"
    cd "$TEMP_DIR"
    tar -xzf "$(basename "$backup_file")"
    
    # Find extracted directory
    local extract_dir=$(find "$TEMP_DIR" -maxdepth 1 -type d -name "tiktok-analysis-system-*")
    if [[ -z "$extract_dir" ]]; then
        error_exit "Could not find extracted backup directory"
    fi
    
    echo "$extract_dir"
}

# Validate backup
validate_backup() {
    local backup_dir="$1"
    
    log "${BLUE}Validating backup...${NC}"
    
    # Check for backup info
    if [[ ! -f "$backup_dir/backup_info.json" ]]; then
        error_exit "Invalid backup: missing backup_info.json"
    fi
    
    # Display backup information
    log "${GREEN}Backup Information:${NC}"
    jq -r '
        "Backup Date: " + .backup_date + "\n" +
        "System Version: " + .system_version + "\n" +
        "Hostname: " + .system_info.hostname + "\n" +
        "OS: " + .system_info.os
    ' "$backup_dir/backup_info.json"
    
    # Check for essential files
    local essential_files=(
        "system/ml_analyzer_registry_complete.py"
        "system/fix_ffmpeg_env.sh"
        "configs"
        "utils"
        "restore_backup.sh"
    )
    
    for file in "${essential_files[@]}"; do
        if [[ ! -e "$backup_dir/$file" ]]; then
            error_exit "Invalid backup: missing $file"
        fi
    done
    
    log "${GREEN}Backup validation passed${NC}"
}

# Create system backup before restore
create_restore_backup() {
    if [[ -d "$RESTORE_DIR" ]]; then
        log "${YELLOW}Creating backup of current system before restore...${NC}"
        
        local current_backup="/tmp/pre_restore_backup_$(date +%s).tar.gz"
        cd "$(dirname "$RESTORE_DIR")"
        tar -czf "$current_backup" "$(basename "$RESTORE_DIR")" 2>/dev/null || true
        
        if [[ -f "$current_backup" ]]; then
            log "${GREEN}Current system backed up to: $current_backup${NC}"
            echo "$current_backup" > "/tmp/restore_rollback_path.txt"
        fi
    fi
}

# Perform restore
perform_restore() {
    local backup_dir="$1"
    local restore_type="$2"
    
    log "${BLUE}Performing $restore_type restore...${NC}"
    
    # Create target directory if it doesn't exist
    mkdir -p "$RESTORE_DIR"
    
    case "$restore_type" in
        "full")
            # Stop any running services
            pkill -f stable_production_api || true
            
            # Restore system files
            log "Restoring system files..."
            cp -r "$backup_dir/system"/* "$RESTORE_DIR/"
            
            # Restore configs
            log "Restoring configurations..."
            cp -r "$backup_dir/configs"/* "$RESTORE_DIR/configs/"
            
            # Restore utils
            log "Restoring utilities..."
            mkdir -p "$RESTORE_DIR/utils"
            cp -r "$backup_dir/utils"/* "$RESTORE_DIR/utils/"
            
            # Restore docs
            log "Restoring documentation..."
            mkdir -p "$RESTORE_DIR/docs"
            cp -r "$backup_dir/docs"/* "$RESTORE_DIR/docs/"
            
            # Restore scripts
            log "Restoring scripts..."
            mkdir -p "$RESTORE_DIR/scripts"
            cp -r "$backup_dir/scripts"/* "$RESTORE_DIR/scripts/"
            ;;
            
        "configs")
            log "Restoring configurations only..."
            cp -r "$backup_dir/configs"/* "$RESTORE_DIR/configs/"
            cp "$backup_dir/system/fix_ffmpeg_env.sh" "$RESTORE_DIR/" 2>/dev/null || true
            ;;
            
        "utils")
            log "Restoring utilities only..."
            cp -r "$backup_dir/utils"/* "$RESTORE_DIR/utils/"
            ;;
    esac
    
    # Set correct permissions
    chmod +x "$RESTORE_DIR/fix_ffmpeg_env.sh" 2>/dev/null || true
    chmod +x "$RESTORE_DIR/scripts"/*.sh 2>/dev/null || true
    find "$RESTORE_DIR" -name "*.py" -exec chmod +x {} \; 2>/dev/null || true
    
    log "${GREEN}Restore completed successfully${NC}"
}

# Post-restore setup
post_restore_setup() {
    log "${BLUE}Running post-restore setup...${NC}"
    
    # Check Python dependencies
    if [[ -f "$RESTORE_DIR/requirements.txt" ]]; then
        log "${YELLOW}Note: You may need to reinstall Python dependencies:${NC}"
        log "  pip install -r $RESTORE_DIR/requirements.txt"
    fi
    
    # Environment setup reminder
    log "${YELLOW}Remember to configure environment before starting:${NC}"
    log "  cd $RESTORE_DIR"
    log "  source fix_ffmpeg_env.sh"
    log "  python3 api/stable_production_api_multiprocess.py"
    
    # Check GPU status
    if command -v nvidia-smi &> /dev/null; then
        log "${BLUE}Current GPU status:${NC}"
        nvidia-smi --query-gpu=name,memory.free --format=csv,noheader || true
    fi
}

# Interactive restore menu
interactive_restore() {
    echo -e "${GREEN}=== TikTok Analysis System Restore ===${NC}"
    echo
    
    # List available backups
    list_backups
    echo
    
    # Get user choice
    echo "Restore options:"
    echo "1) Latest backup (full system)"
    echo "2) Choose specific backup (full system)"
    echo "3) Latest backup (configs only)"
    echo "4) Choose specific backup (configs only)"
    echo "5) Exit"
    echo
    
    read -p "Select option (1-5): " choice
    
    case "$choice" in
        1)
            local backup_file=$(get_latest_backup)
            if [[ -z "$backup_file" ]]; then
                error_exit "No backups found"
            fi
            backup_dir=$(download_backup "$backup_file")
            validate_backup "$backup_dir"
            
            echo -e "${YELLOW}This will restore the full system. Continue? (y/N):${NC}"
            read -r confirm
            if [[ $confirm =~ ^[Yy]$ ]]; then
                create_restore_backup
                perform_restore "$backup_dir" "full"
                post_restore_setup
            fi
            ;;
            
        2)
            list_backups
            echo
            read -p "Enter backup number to restore: " backup_num
            
            backup_file=$(b2 file list-file-names "$B2_BUCKET" --prefix "backups/tiktok-analysis-system-" --json | \
                         jq -r '.files[] | select(.fileName | endswith(".tar.gz")) | .fileName' | \
                         sort | sed -n "${backup_num}p")
            
            if [[ -z "$backup_file" ]]; then
                error_exit "Invalid backup selection"
            fi
            
            backup_dir=$(download_backup "$backup_file")
            validate_backup "$backup_dir"
            
            echo -e "${YELLOW}This will restore the full system. Continue? (y/N):${NC}"
            read -r confirm
            if [[ $confirm =~ ^[Yy]$ ]]; then
                create_restore_backup
                perform_restore "$backup_dir" "full"
                post_restore_setup
            fi
            ;;
            
        3)
            local backup_file=$(get_latest_backup)
            if [[ -z "$backup_file" ]]; then
                error_exit "No backups found"
            fi
            backup_dir=$(download_backup "$backup_file")
            validate_backup "$backup_dir"
            
            echo -e "${YELLOW}This will restore configurations only. Continue? (y/N):${NC}"
            read -r confirm
            if [[ $confirm =~ ^[Yy]$ ]]; then
                perform_restore "$backup_dir" "configs"
            fi
            ;;
            
        4)
            list_backups
            echo
            read -p "Enter backup number for config restore: " backup_num
            
            backup_file=$(b2 file list-file-names "$B2_BUCKET" --prefix "backups/tiktok-analysis-system-" --json | \
                         jq -r '.files[] | select(.fileName | endswith(".tar.gz")) | .fileName' | \
                         sort | sed -n "${backup_num}p")
            
            if [[ -z "$backup_file" ]]; then
                error_exit "Invalid backup selection"
            fi
            
            backup_dir=$(download_backup "$backup_file")
            validate_backup "$backup_dir"
            
            echo -e "${YELLOW}This will restore configurations only. Continue? (y/N):${NC}"
            read -r confirm
            if [[ $confirm =~ ^[Yy]$ ]]; then
                perform_restore "$backup_dir" "configs"
            fi
            ;;
            
        5)
            echo "Restore cancelled"
            exit 0
            ;;
            
        *)
            error_exit "Invalid selection"
            ;;
    esac
}

# Main function
main() {
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Check prerequisites
    if ! command -v b2 &> /dev/null; then
        error_exit "B2 CLI not found. Please install B2 CLI first."
    fi
    
    if ! b2 account get &> /dev/null; then
        error_exit "B2 not authorized. Please run 'b2 account authorize' first."
    fi
    
    # Check if running interactively
    if [[ $# -eq 0 ]]; then
        interactive_restore
    else
        # Command line arguments (for automation)
        case "$1" in
            "latest")
                backup_file=$(get_latest_backup)
                backup_dir=$(download_backup "$backup_file")
                validate_backup "$backup_dir"
                create_restore_backup
                perform_restore "$backup_dir" "full"
                post_restore_setup
                ;;
            "list")
                list_backups
                ;;
            *)
                echo "Usage: $0 [latest|list]"
                echo "  latest - Restore latest backup automatically"
                echo "  list   - List available backups"
                echo "  (no args) - Interactive restore menu"
                exit 1
                ;;
        esac
    fi
    
    log "${GREEN}=== Restore Process Complete ===${NC}"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi