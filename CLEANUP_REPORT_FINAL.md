# Server Cleanup Report - Wed Jul  9 06:30:10 UTC 2025

## Durchgeführte Aktionen:

### 1. Backup erstellt
- Vollständiges Backup: -rw-rw-r-- 1 user user 487M Jul  9 06:25 /home/user/tiktok_production_ESSENTIAL_BACKUP_20250709_062505.tar.gz

### 2. APIs aufgeräumt
- Aktiv belassen: stable_production_api_multiprocess.py
- Archiviert: 4 alte APIs

### 3. Analyzer bereinigt
- Vorher: 135 Analyzer-Dateien
- Nachher: 26 aktive Analyzer
- Archiviert: 109 inaktive Analyzer

### 4. Speicher freigegeben
- Aurora Cap temp_aurora: 77MB
- Aurora Cap venv: 6.3GB
- Test/Debug Scripts: 19 Dateien

### 5. Neue Struktur


## Speicherplatz vorher/nachher:
Filesystem      Size  Used Avail Use% Mounted on
/dev/vda1       243G  112G  132G  46% /

## Archivierte Dateien:
- APIs: simple_test_api.py
stable_production_api.py
stable_production_api_blip2_fix.py
ultimate_production_api.py
- Test Scripts: check_all_analyzers_quality.py
check_analyzer_order.py
check_analyzer_status.py
check_gpu_memory.py
check_qwen2vl_results.py...

## System Status:
- API läuft auf Port 8003
- Alle 22 aktiven Analyzer funktionsfähig
- Keine kritischen Dateien entfernt
