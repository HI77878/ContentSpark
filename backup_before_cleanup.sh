#!/bin/bash
# backup_before_cleanup.sh
BACKUP_DIR="/home/user/tiktok_cleanup_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Liste kritische Dateien
echo "Sichere kritische Konfigurationen..."
cp -r /home/user/tiktok_production/configs $BACKUP_DIR/
cp -r /home/user/tiktok_production/analyzers/*.py $BACKUP_DIR/
cp /home/user/tiktok_production/api/stable_production_api_multiprocess.py $BACKUP_DIR/
cp /home/user/tiktok_production/registry_loader.py $BACKUP_DIR/
cp /home/user/tiktok_production/ml_analyzer_registry_complete.py $BACKUP_DIR/
echo "Backup erstellt in: $BACKUP_DIR"