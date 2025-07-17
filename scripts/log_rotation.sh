#!/bin/bash
# Log rotation script for TikTok Analyzer

LOG_DIR="/home/user/tiktok_production/logs"
ARCHIVE_DIR="$LOG_DIR/archive"
MAX_AGE_DAYS=7
MAX_SIZE_MB=100

echo "Starting log rotation..."

# Create archive directory
mkdir -p $ARCHIVE_DIR

# Function to rotate a log file
rotate_log() {
    local logfile=$1
    local basename=$(basename $logfile)
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Check if file exists and size
    if [ -f "$logfile" ]; then
        size_mb=$(du -m "$logfile" | cut -f1)
        
        if [ $size_mb -gt $MAX_SIZE_MB ]; then
            echo "Rotating $basename (${size_mb}MB)"
            
            # Compress and move to archive
            gzip -c "$logfile" > "$ARCHIVE_DIR/${basename}_${timestamp}.gz"
            
            # Truncate original file
            > "$logfile"
            
            echo "Rotated $basename"
        fi
    fi
}

# Rotate all log files
for logfile in $LOG_DIR/*.log; do
    rotate_log "$logfile"
done

# Clean old archives
echo "Cleaning old archives (older than $MAX_AGE_DAYS days)..."
find $ARCHIVE_DIR -name "*.gz" -mtime +$MAX_AGE_DAYS -delete

# Display summary
echo "Log rotation complete!"
echo "Active logs:"
ls -lh $LOG_DIR/*.log 2>/dev/null | tail -5

echo ""
echo "Archive size:"
du -sh $ARCHIVE_DIR 2>/dev/null