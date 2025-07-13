# Backup System Documentation - TikTok Video Analysis System

## Overview

The TikTok Video Analysis System includes a comprehensive backup solution using Backblaze B2 cloud storage. The backup system ensures data protection, disaster recovery, and system continuity through automated, incremental, and configurable backup strategies.

## Backup Architecture

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Backup System Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │  Local System   │    │  Backup Scripts │    │ Backblaze   │ │
│  │                 │    │                 │    │     B2      │ │
│  │ • System Files  │───▶│ • Full Backup   │───▶│             │ │
│  │ • Configs       │    │ • Config Backup │    │ • Encrypted │ │
│  │ • Documentation │    │ • Restore       │    │ • Versioned │ │
│  │ • Scripts       │    │ • Automated     │    │ • Lifecycle │ │
│  │ • Logs          │    │                 │    │             │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Backup Schedule                          │ │
│  │                                                             │ │
│  │  Daily:     Configuration backups                          │ │
│  │  Weekly:    Full system backups (Sundays)                  │ │
│  │  Monthly:   Complete system snapshots (1st of month)       │ │
│  │  On-demand: Manual backups and emergency backups           │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Backup Scripts

### 1. Full System Backup (`backup_system.sh`)

Comprehensive backup of the entire TikTok Analysis System.

**Contents:**
- System files (analyzers, API, registry)
- Configuration files
- Utility scripts
- Documentation
- Recent logs (last 7 days)
- Sample analysis results
- Backup metadata and restore scripts

**Usage:**
```bash
cd /home/user/tiktok_production/scripts
./backup_system.sh
```

**Output:**
- Compressed archive (~50-200MB)
- Uploaded to B2: `backups/tiktok-analysis-system-YYYYMMDD_HHMMSS.tar.gz`
- Automatic cleanup (keeps last 30 backups)

### 2. Configuration Backup (`backup_configs.sh`)

Quick backup of critical configuration files only.

**Contents:**
- All files in `configs/` directory
- Environment configuration (`.env`)
- Core system files (`fix_ffmpeg_env.sh`, `ml_analyzer_registry_complete.py`)
- Requirements file

**Usage:**
```bash
./backup_configs.sh
```

**Output:**
- Small archive (~5-10MB)
- Uploaded to B2: `config_backups/tiktok-configs-YYYYMMDD_HHMMSS.tar.gz`

### 3. Automated Backup (`automated_backup.sh`)

Cron-compatible script for scheduled backups with error handling.

**Features:**
- Intelligent scheduling (configs daily, full weekly)
- System health checks before backup
- Lock file management (prevents concurrent runs)
- Email/webhook notifications
- Automatic cleanup of old logs
- Performance optimization during backup

**Usage:**
```bash
# Manual run
./automated_backup.sh

# Test mode (health checks only)
./automated_backup.sh test

# Force full backup
./automated_backup.sh force-full

# Config only backup
./automated_backup.sh configs-only
```

### 4. Restore System (`restore_backup.sh`)

Interactive and automated restore functionality.

**Features:**
- Interactive menu for backup selection
- Automatic latest backup restore
- Partial restore options (configs only, utils only)
- Pre-restore system backup
- Validation of backup integrity
- Post-restore setup guidance

**Usage:**
```bash
# Interactive restore
./restore_backup.sh

# Restore latest backup automatically
./restore_backup.sh latest

# List available backups
./restore_backup.sh list
```

## Setup and Configuration

### 1. Initial Setup

#### Install B2 CLI
```bash
# Download and install B2 CLI
wget https://github.com/Backblaze/B2_Command_Line_Tool/releases/latest/download/b2-linux -O /tmp/b2
sudo mv /tmp/b2 /usr/local/bin/b2
sudo chmod +x /usr/local/bin/b2
```

#### Configure B2 Account
```bash
# Run setup script
cd /home/user/tiktok_production/scripts
chmod +x setup_b2_backup.sh
./setup_b2_backup.sh
```

**Setup Process:**
1. Prompts for Backblaze B2 credentials
2. Creates bucket `tiktok-analysis-backup`
3. Sets up lifecycle rules
4. Tests upload/download functionality
5. Configures automatic cleanup

### 2. Automated Scheduling

#### Cron Configuration
```bash
# Edit crontab
crontab -e

# Add backup schedule
# Daily config backup at 2 AM
0 2 * * * /home/user/tiktok_production/scripts/automated_backup.sh configs-only

# Weekly full backup on Sundays at 3 AM  
0 3 * * 0 /home/user/tiktok_production/scripts/automated_backup.sh force-full

# Monthly full backup on 1st at 4 AM
0 4 1 * * /home/user/tiktok_production/scripts/automated_backup.sh force-full
```

#### Systemd Timer (Alternative)
```bash
# Create systemd service
sudo tee /etc/systemd/system/tiktok-backup.service << EOF
[Unit]
Description=TikTok Analysis System Backup
After=network.target

[Service]
Type=oneshot
User=user
ExecStart=/home/user/tiktok_production/scripts/automated_backup.sh
WorkingDirectory=/home/user/tiktok_production
EOF

# Create systemd timer
sudo tee /etc/systemd/system/tiktok-backup.timer << EOF
[Unit]
Description=Run TikTok Analysis backup daily
Requires=tiktok-backup.service

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Enable and start timer
sudo systemctl enable tiktok-backup.timer
sudo systemctl start tiktok-backup.timer
```

### 3. Notification Configuration

#### Email Notifications
```bash
# Install mail utilities
sudo apt install -y mailutils

# Configure environment variables
echo "BACKUP_ADMIN_EMAIL=admin@yourdomain.com" >> /home/user/tiktok_production/.env
```

#### Webhook Notifications (Slack/Discord)
```bash
# Add webhook URL to environment
echo "BACKUP_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL" >> /home/user/tiktok_production/.env
```

## Backup Types and Retention

### Backup Types

#### 1. Full System Backup
- **Frequency**: Weekly (Sundays) + Monthly (1st)
- **Size**: 50-200MB
- **Contents**: Complete system state
- **Retention**: 30 versions (~6 months)

#### 2. Configuration Backup
- **Frequency**: Daily
- **Size**: 5-10MB  
- **Contents**: Critical config files only
- **Retention**: 90 versions (~3 months)

#### 3. Emergency Backup
- **Frequency**: On-demand
- **Size**: Variable
- **Contents**: Pre-change system state
- **Retention**: 10 versions

### Retention Policies

#### Automatic Cleanup Rules
```json
{
  "full_backups": {
    "retention_count": 30,
    "lifecycle_rules": {
      "days_to_hide": 30,
      "days_to_delete": 90
    }
  },
  "config_backups": {
    "retention_count": 90,
    "lifecycle_rules": {
      "days_to_hide": 7,
      "days_to_delete": 90
    }
  },
  "logs": {
    "retention_count": 365,
    "lifecycle_rules": {
      "days_to_hide": 30,
      "days_to_delete": 365
    }
  }
}
```

## Restore Procedures

### 1. Interactive Restore

```bash
cd /home/user/tiktok_production/scripts
./restore_backup.sh
```

**Process:**
1. Lists available backups with dates
2. Allows selection of specific backup
3. Shows backup information and validation
4. Creates pre-restore backup of current system
5. Performs restore with progress indication
6. Provides post-restore setup instructions

### 2. Emergency Restore

```bash
# Quick restore of latest backup
./restore_backup.sh latest

# Restore configs only (minimal downtime)
./restore_backup.sh latest configs
```

### 3. Disaster Recovery

**Complete System Recovery:**
1. Fresh system installation
2. Install B2 CLI and configure credentials
3. Download and run restore script
4. Restore latest full backup
5. Install Python dependencies
6. Verify system functionality

**Recovery Script:**
```bash
#!/bin/bash
# Disaster recovery script

# Install B2 CLI
wget https://github.com/Backblaze/B2_Command_Line_Tool/releases/latest/download/b2-linux -O /usr/local/bin/b2
chmod +x /usr/local/bin/b2

# Configure B2 (requires manual input)
b2 account authorize

# Download restore script
b2 file download-by-name tiktok-analysis-backup backups/restore_backup.sh ./restore_backup.sh
chmod +x restore_backup.sh

# Run recovery
./restore_backup.sh latest
```

## Monitoring and Alerts

### Backup Health Monitoring

#### Daily Health Check
```bash
#!/bin/bash
# Check backup health

# Verify last backup date
LAST_BACKUP=$(b2 file list-file-names tiktok-analysis-backup --prefix "backups/" --json | \
               jq -r '.files[0].uploadTimestamp')

LAST_BACKUP_AGE=$(( ($(date +%s) - $LAST_BACKUP/1000) / 86400 ))

if [[ $LAST_BACKUP_AGE -gt 2 ]]; then
    echo "WARNING: Last backup is $LAST_BACKUP_AGE days old"
    # Send alert
fi
```

#### Backup Size Monitoring
```bash
# Monitor backup size trends
b2 file list-file-names tiktok-analysis-backup --prefix "backups/" --json | \
jq -r '.files[] | "\(.uploadTimestamp) \(.size)"' | \
sort -nr | head -10
```

### Error Handling

#### Common Issues and Solutions

**B2 Authorization Failed:**
```bash
# Re-authorize B2 account
b2 account clear
b2 account authorize
```

**Backup Timeout:**
```bash
# Increase timeout in automated_backup.sh
MAX_RUNTIME=7200  # 2 hours
```

**Insufficient Disk Space:**
```bash
# Clean up temporary files
find /tmp -name "backup_*" -type d -mtime +1 -exec rm -rf {} \;

# Clean old logs
find /home/user/tiktok_production/logs -name "*.log" -mtime +30 -delete
```

**Large Backup Size:**
```bash
# Exclude large files from backup
# Edit backup_system.sh and add exclusions:
tar --exclude='*.mp4' --exclude='*.mov' -czf backup.tar.gz source/
```

## Security Considerations

### Encryption
- All backups are encrypted in transit (HTTPS/TLS)
- B2 server-side encryption enabled
- Sensitive configuration files protected

### Access Control
- B2 application keys with minimal permissions
- Backup scripts require specific user permissions
- Lock files prevent concurrent execution

### Compliance
- Data retention policies configurable
- Audit logs for all backup operations
- Geographic data storage in specified B2 regions

## Performance Optimization

### Backup Speed
- Compression using gzip (-9 for maximum compression)
- Parallel uploads for large files
- Incremental backup detection
- Temporary file cleanup

### Resource Usage
- System optimization before backup
- Non-critical service suspension
- Memory cache clearing
- CPU priority adjustment

### Network Optimization
- Retry logic for failed uploads
- Chunked upload for large files
- Bandwidth throttling options
- Progress monitoring

## Cost Management

### B2 Storage Costs
- **Storage**: $0.005/GB/month
- **Download**: $0.01/GB
- **API Calls**: Class B (list/download) - $0.004/10,000

### Estimated Monthly Costs
```
Full backups:    30 × 200MB = 6GB     = $0.03/month
Config backups:  90 × 10MB  = 900MB   = $0.005/month
Logs:           365 × 5MB   = 1.8GB   = $0.009/month
Total storage:                8.7GB   = $0.044/month

Download (restore): ~1GB/month        = $0.01/month
API calls: ~1000/month                = $0.0004/month

Total estimated cost: ~$0.055/month
```

### Cost Optimization
- Lifecycle rules for automatic deletion
- Compression to reduce storage size
- Retention period optimization
- Monitoring of storage usage

## Maintenance

### Weekly Tasks
- Review backup logs for errors
- Check backup completion status
- Verify B2 storage quota usage
- Test restore functionality

### Monthly Tasks
- Perform disaster recovery test
- Review and update retention policies
- Audit backup file sizes and trends
- Update backup documentation

### Quarterly Tasks
- Full disaster recovery drill
- Review and optimize backup scripts
- Evaluate storage costs and usage
- Update backup security measures

This backup system provides enterprise-grade data protection for the TikTok Video Analysis System with minimal maintenance overhead and cost-effective cloud storage.