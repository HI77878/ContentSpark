# 📋 Übergabe an das Betriebsteam - TikTok Video Analysis System

## System-Übersicht

**System**: TikTok Video Analysis API  
**Status**: ✅ PRODUKTIONSREIF  
**Performance**: 2.99x Realtime (Ziel: <3x erreicht)  
**Verfügbarkeit**: 24/7 mit Auto-Restart  

## Kritische Informationen

### Zugriff
- **API Endpoint**: http://localhost:8003
- **Health Check**: http://localhost:8003/health
- **Basis-Verzeichnis**: /home/user/tiktok_production

### Systemd Service
```bash
# Status prüfen
sudo systemctl status tiktok-analyzer

# Neustart
sudo systemctl restart tiktok-analyzer

# Logs
sudo journalctl -u tiktok-analyzer -f
```

## Tägliche Wartungsaufgaben

### 1. Morning Check (täglich 9:00)
```bash
# Quick health check
curl http://localhost:8003/health

# GPU Status
nvidia-smi

# Letzte Fehler prüfen
tail -n 50 /home/user/tiktok_production/logs/stable_multiprocess_api.log | grep ERROR
```

### 2. Monitoring Dashboard
```bash
# System-Metriken anzeigen
python3 /home/user/tiktok_production/monitoring/system_monitor.py
```

### 3. Speicherplatz prüfen
```bash
# Results Verzeichnis
du -sh /home/user/tiktok_production/results/

# Logs
du -sh /home/user/tiktok_production/logs/
```

## Notfall-Prozeduren

### Problem: API reagiert nicht

1. **Health Check**:
   ```bash
   curl -v http://localhost:8003/health
   ```

2. **Automatischer Restart** (läuft alle 10 Min via Cron):
   ```bash
   /home/user/tiktok_production/scripts/health_check.sh
   ```

3. **Manueller Restart**:
   ```bash
   sudo systemctl restart tiktok-analyzer
   ```

### Problem: GPU Out of Memory

1. **GPU Status prüfen**:
   ```bash
   nvidia-smi
   ```

2. **GPU Cache leeren**:
   ```bash
   python3 -c "import torch; torch.cuda.empty_cache()"
   ```

3. **Service neu starten**:
   ```bash
   sudo systemctl restart tiktok-analyzer
   ```

### Problem: Disk Space

1. **Alte Logs löschen**:
   ```bash
   /home/user/tiktok_production/scripts/log_rotation.sh
   ```

2. **Alte Results archivieren**:
   ```bash
   find /home/user/tiktok_production/results/ -mtime +30 -type f -delete
   ```

## Automatisierte Wartung

### Cron Jobs (bereits eingerichtet)
- **Health Check**: Alle 10 Minuten mit Auto-Restart
- **System Monitoring**: Alle 5 Minuten
- **Log Rotation**: Täglich um 2:00 Uhr
- **GPU Cleanup**: Stündlich

### Alerts einrichten (empfohlen)
```bash
# Beispiel für Email-Alert bei API-Ausfall
# In health_check.sh hinzufügen:
if [ $? -ne 0 ]; then
    echo "API Down at $(date)" | mail -s "ALERT: TikTok API Down" ops@company.com
fi
```

## Performance-Überwachung

### Wichtige Metriken
- **GPU Auslastung**: Sollte 40-70% sein
- **GPU Memory**: Sollte <90% bleiben
- **API Response Time**: Sollte <1s sein
- **Realtime Factor**: Sollte <3x bleiben

### Performance-Report
```bash
# Wöchentlicher Report (Sonntags 3:00 via Cron)
tail -n 1000 /home/user/tiktok_production/logs/system_metrics.jsonl | \
  jq -s '[.[] | select(.gpu != null)] | 
    {avg_gpu: (map(.gpu.gpu_utilization) | add/length),
     avg_memory: (map(.gpu.memory_percent) | add/length)}'
```

## Backup-Strategie

### Täglich sichern
1. **Results** (wichtig!):
   ```bash
   rsync -av /home/user/tiktok_production/results/ /backup/tiktok_results_$(date +%Y%m%d)/
   ```

2. **Konfiguration**:
   ```bash
   tar -czf /backup/tiktok_config_$(date +%Y%m%d).tar.gz \
     /home/user/tiktok_production/configs/ \
     /home/user/tiktok_production/*.md
   ```

## Wartungsfenster

### Empfohlene Zeiten
- **Beste Zeit**: Dienstag/Donnerstag 3:00-5:00 Uhr
- **Updates**: Immer mit Backup vorher
- **GPU-Treiber**: Quartalsweise prüfen

### Update-Prozedur
1. Backup erstellen
2. Service stoppen
3. Updates durchführen
4. Service starten
5. Health Check durchführen

## Kontakte & Eskalation

### Support-Level
1. **Level 1**: Restart-Prozeduren (Ops Team)
2. **Level 2**: Config-Änderungen (DevOps)
3. **Level 3**: Code-Änderungen (Dev Team)

### Wichtige Logs
- **API Log**: `/logs/stable_multiprocess_api.log`
- **System Monitor**: `/logs/system_monitor.log`
- **Health Checks**: `/logs/health_check.log`
- **Systemd**: `journalctl -u tiktok-analyzer`

## Quick Reference Card

```bash
# Service Management
systemctl start/stop/restart/status tiktok-analyzer

# Health Check
curl http://localhost:8003/health

# GPU Status
nvidia-smi

# Logs anzeigen
tail -f /home/user/tiktok_production/logs/stable_multiprocess_api.log

# Manueller Restart
cd /home/user/tiktok_production && ./scripts/restart_services.sh restart

# System-Überwachung
python3 monitoring/system_monitor.py
```

## Wichtige Hinweise

⚠️ **IMMER** vor manuellen Starts:
```bash
source /home/user/tiktok_production/fix_ffmpeg_env.sh
```

⚠️ **AuroraCap** ist experimentell - NICHT für Produktion verwenden

⚠️ **GPU Memory** bei >90% → Service-Restart erforderlich

✅ **BLIP-2** ist der primäre Video-Analyzer - sehr zuverlässig

## Übergabe-Checkliste

- [x] Systemd Service eingerichtet und aktiviert
- [x] Monitoring & Logging konfiguriert
- [x] Auto-Restart Mechanismen aktiv
- [x] Dokumentation vollständig
- [x] Backup-Strategie definiert
- [x] Notfall-Prozeduren dokumentiert
- [x] Performance-Metriken definiert

**Das System ist bereit für den 24/7 Produktivbetrieb!**

Bei Fragen: Siehe PROJECT_FINAL_SUMMARY.md und DEPLOYMENT_GUIDE.md