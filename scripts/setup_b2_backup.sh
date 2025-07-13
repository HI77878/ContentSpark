#!/bin/bash
# Backblaze B2 Backup Setup Script for TikTok Analysis System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="/home/user/tiktok_production"
B2_BUCKET_NAME="tiktok-analysis-backup"
CONFIG_FILE="$HOME/.b2_config"

echo -e "${GREEN}=== Backblaze B2 Backup Setup ===${NC}"

# Check if B2 CLI is installed
if ! command -v b2 &> /dev/null; then
    echo -e "${RED}Error: B2 CLI is not installed${NC}"
    echo "Please install B2 CLI first"
    exit 1
fi

echo -e "${YELLOW}B2 CLI version:${NC}"
b2 version

# Function to setup B2 credentials
setup_credentials() {
    echo -e "${YELLOW}Setting up B2 credentials...${NC}"
    
    # Check if credentials are already configured
    if b2 account get 2>/dev/null | grep -q "account_id"; then
        echo -e "${GREEN}B2 account already authorized${NC}"
        b2 account get
        return 0
    fi
    
    echo "Please enter your Backblaze B2 credentials:"
    read -p "Application Key ID: " APP_KEY_ID
    read -s -p "Application Key: " APP_KEY
    echo
    
    # Authorize account
    echo "Authorizing B2 account..."
    if echo -e "$APP_KEY_ID\n$APP_KEY" | b2 account authorize; then
        echo -e "${GREEN}Successfully authorized B2 account${NC}"
        
        # Save credentials to config file (encrypted)
        echo "APP_KEY_ID=$APP_KEY_ID" > "$CONFIG_FILE"
        echo "APP_KEY=$APP_KEY" >> "$CONFIG_FILE"
        chmod 600 "$CONFIG_FILE"
        
        return 0
    else
        echo -e "${RED}Failed to authorize B2 account${NC}"
        return 1
    fi
}

# Function to create or verify bucket
setup_bucket() {
    echo -e "${YELLOW}Setting up B2 bucket...${NC}"
    
    # Check if bucket exists
    if b2 bucket list | grep -q "$B2_BUCKET_NAME"; then
        echo -e "${GREEN}Bucket '$B2_BUCKET_NAME' already exists${NC}"
    else
        echo "Creating bucket '$B2_BUCKET_NAME'..."
        if b2 bucket create "$B2_BUCKET_NAME" allPrivate; then
            echo -e "${GREEN}Successfully created bucket '$B2_BUCKET_NAME'${NC}"
        else
            echo -e "${RED}Failed to create bucket '$B2_BUCKET_NAME'${NC}"
            return 1
        fi
    fi
    
    # Set lifecycle rules for automatic cleanup
    echo "Setting up lifecycle rules..."
    cat > /tmp/lifecycle_rules.json << EOF
{
    "fileNamePrefix": "logs/",
    "daysFromHidingToDeleting": 90,
    "daysFromUploadingToHiding": 30
}
EOF
    
    b2 bucket update "$B2_BUCKET_NAME" --lifecycle-rules /tmp/lifecycle_rules.json
    rm /tmp/lifecycle_rules.json
    
    echo -e "${GREEN}Bucket setup complete${NC}"
}

# Function to test backup
test_backup() {
    echo -e "${YELLOW}Testing backup functionality...${NC}"
    
    # Create test file
    TEST_FILE="/tmp/b2_test_$(date +%s).txt"
    echo "B2 backup test - $(date)" > "$TEST_FILE"
    
    # Upload test file
    if b2 file upload "$B2_BUCKET_NAME" "$TEST_FILE" "test/backup_test.txt"; then
        echo -e "${GREEN}Test upload successful${NC}"
        
        # Download test file
        if b2 file download-by-name "$B2_BUCKET_NAME" "test/backup_test.txt" "/tmp/b2_download_test.txt"; then
            echo -e "${GREEN}Test download successful${NC}"
            
            # Clean up test files
            b2 file delete-file-version "$B2_BUCKET_NAME" "test/backup_test.txt" $(b2 file list-file-names "$B2_BUCKET_NAME" --prefix "test/backup_test.txt" --json | jq -r '.files[0].fileId')
            rm -f "$TEST_FILE" "/tmp/b2_download_test.txt"
            
            echo -e "${GREEN}B2 backup system is working correctly${NC}"
            return 0
        else
            echo -e "${RED}Test download failed${NC}"
            return 1
        fi
    else
        echo -e "${RED}Test upload failed${NC}"
        return 1
    fi
}

# Main setup process
main() {
    echo "Starting B2 backup setup..."
    
    # Setup credentials
    if ! setup_credentials; then
        echo -e "${RED}Failed to setup B2 credentials${NC}"
        exit 1
    fi
    
    # Setup bucket
    if ! setup_bucket; then
        echo -e "${RED}Failed to setup B2 bucket${NC}"
        exit 1
    fi
    
    # Test backup
    if ! test_backup; then
        echo -e "${RED}Backup test failed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}=== B2 Backup Setup Complete ===${NC}"
    echo "You can now use the backup scripts to backup your TikTok Analysis System"
    echo
    echo "Available commands:"
    echo "  ./backup_system.sh      - Full system backup"
    echo "  ./backup_configs.sh     - Configuration files only"
    echo "  ./backup_results.sh     - Analysis results only"
    echo "  ./restore_backup.sh     - Restore from backup"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi