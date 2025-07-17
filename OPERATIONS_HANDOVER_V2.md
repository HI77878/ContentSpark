# Operations Handover - TikTok Video Analysis System V2

## Executive Summary

Übergabe des produktionsreifen TikTok Video Analysis Systems an das Operations-Team. Das System nutzt **Video-LLaVA** als primären Video-Analyzer und erreicht eine Performance von 3.15x Realtime mit 21 ML-Analyzern.

## Kritische Informationen

### ⚠️ WICHTIGSTE REGEL
**IMMER** vor jedem Start ausführen:
```bash
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh
```
Ohne dies gibt es FFmpeg-Fehler!

### Primärer Video-Analyzer: Video-LLaVA
- **Status**: ✅ Aktiv und verifiziert funktionsfähig
- **Modell**: LLaVA-NeXT-Video-7B (4-bit quantisiert)
- **Performance**: 10s Analyse pro Video
- **GPU-Speicher**: 3.8GB

### Deaktivierte Analyzer (NICHT verwenden)
- ❌ **BLIP-2**: 3+ Minuten Ladezeit, inkompatibel
- ❌ **AuroraCap**: Experimentell, durch Video-LLaVA ersetzt
- ❌ **Vid2Seq**: Archiviert, nicht produktionsreif

## Standard Operating Procedures

### System Start
```bash
# 1. Terminal/Screen Session öffnen
screen -S tiktok-api

# 2. Zum Projektverzeichnis
cd /home/user/tiktok_production

# 3. FFmpeg Fix (KRITISCH!)
source fix_ffmpeg_env.sh

# 4. API starten
python3 api/stable_production_api_multiprocess.py

# 5. Screen detachen: Ctrl+A, D
```

### System Stop
```bash
# Prozess finden
ps aux | grep stable_production_api

# Graceful shutdown
kill -SIGTERM <PID>

# Warten auf Shutdown (max 30s)
# Falls hängt: kill -9 <PID>
```

### Health Monitoring

#### Quick Health Check
```bash
curl http://localhost:8003/health | python3 -m json.tool
```

Erwartete Antwort:
```json
{
    "status": "healthy",
    "timestamp": "...",
    "gpu": {
        "gpu_available": true,
        "gpu_name": "Quadro RTX 8000",
        "gpu_memory": {...}
    },
    "active_analyzers": 21,
    "parallelization": "multiprocess"
}
```

#### GPU Monitoring
```bash
# Echtzeit GPU-Auslastung
watch -n 1 nvidia-smi

# Detaillierte GPU-Metriken
nvidia-smi dmon -i 0 -s pucm -d 1
```

#### Log Monitoring
```bash
# Live API Logs
tail -f /home/user/tiktok_production/logs/stable_multiprocess_api.log

# Fehler suchen
grep ERROR /home/user/tiktok_production/logs/stable_multiprocess_api.log | tail -20

# Video-LLaVA spezifisch
grep -i "video_llava\|llava" logs/stable_multiprocess_api.log | tail -50
```

## Tägliche Wartung

### Morgen-Checkliste
1. **API Health Check**
   ```bash
   curl http://localhost:8003/health
   ```

2. **GPU-Status**
   ```bash
   nvidia-smi
   # GPU-Speicher sollte < 20GB sein im Idle
   ```

3. **Log-Größe prüfen**
   ```bash
   du -sh /home/user/tiktok_production/logs/*
   # Logs > 1GB können rotiert werden
   ```

4. **Speicherplatz**
   ```bash
   df -h /home/user/tiktok_production
   # Mindestens 20GB frei halten
   ```

### Log Rotation
```bash
# Alte Logs archivieren (wöchentlich)
cd /home/user/tiktok_production/logs
mkdir -p archive/$(date +%Y%m%d)
mv *.log archive/$(date +%Y%m%d)/
touch stable_multiprocess_api.log

# Alte Ergebnisse löschen (> 30 Tage)
find /home/user/tiktok_production/results -name "*.json" -mtime +30 -delete
```

## Performance Monitoring

### Key Metrics
- **Realtime Factor**: Sollte < 3.5x bleiben
  - Aktuell: 3.15x (akzeptabel)
  - Warnung bei > 4x
  - Kritisch bei > 5x

- **Erfolgsrate**: Sollte > 95% sein
  - Check: Successful vs Total Analyzers in API Response

- **GPU-Auslastung**: 
  - Idle: < 1GB
  - Während Analyse: 15-20GB normal
  - Peak (mit Video-LLaVA): bis 25GB

### Performance Test
```bash
cd /home/user/tiktok_production
python3 final_video_llava_performance_test.py
```

## Troubleshooting Guide

### Problem: API startet nicht
```bash
# 1. Ports prüfen
sudo netstat -tlnp | grep 8003

# 2. Alte Prozesse killen
pkill -f stable_production_api

# 3. FFmpeg Fix nicht vergessen!
source fix_ffmpeg_env.sh

# 4. Neu starten
python3 api/stable_production_api_multiprocess.py
```

### Problem: Video-LLaVA Fehler
```bash
# 1. GPU-Speicher prüfen
nvidia-smi

# 2. Modell-Cache prüfen
ls -la ~/.cache/huggingface/hub/models--llava-hf*

# 3. Cache löschen wenn korrupt
rm -rf ~/.cache/huggingface/hub/models--llava-hf*

# 4. API neu starten (lädt Modell neu)
```

### Problem: Hohe Latenz (> 5x Realtime)
1. **GPU-Auslastung prüfen**
   ```bash
   nvidia-smi dmon -i 0 -s u -d 1
   ```

2. **Aktive Worker prüfen**
   ```bash
   ps aux | grep python | grep -c multiprocess
   # Sollten 3-4 Worker sein
   ```

3. **Temporär einzelne Analyzer deaktivieren**
   - In Request: `"analyzers": ["video_llava", "speech_transcription", ...]`

### Problem: Speicher voll
```bash
# 1. Große Logs finden
find /home/user/tiktok_production -name "*.log" -size +1G

# 2. Alte Results löschen
find /home/user/tiktok_production/results -name "*.json" -mtime +7 -delete

# 3. Docker Images aufräumen (falls Docker genutzt)
docker system prune -a
```

## Notfall-Prozeduren

### System-Neustart (Clean Restart)
```bash
# 1. Alle Prozesse stoppen
pkill -f stable_production_api
pkill -f python3

# 2. GPU-Speicher clearen
sudo nvidia-smi --gpu-reset

# 3. 30 Sekunden warten
sleep 30

# 4. Normal starten
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py
```

### Fallback ohne Video-LLaVA
Falls Video-LLaVA Probleme macht:
```bash
# In gpu_groups_config.py temporär hinzufügen:
DISABLED_ANALYZERS.append('video_llava')

# API neu starten
# WARNUNG: Reduziert Analyse-Qualität erheblich!
```

## Wartungsfenster

### Wöchentlich (Sonntag Nacht)
1. Log-Rotation
2. Cache-Cleanup
3. Performance-Test
4. System-Updates (optional)

### Monatlich
1. Modell-Updates prüfen
2. Vollständiges Backup
3. Speicherplatz-Analyse
4. Performance-Trend Review

## Kontakte und Eskalation

### Level 1: Operations Team
- Basis-Troubleshooting
- Log-Monitoring
- Restart-Prozeduren

### Level 2: DevOps
- Performance-Optimierung
- Docker-Konfiguration
- System-Updates

### Level 3: ML Engineering
- Analyzer-Updates
- Modell-Probleme
- Architektur-Änderungen

## Wichtige Dateien und Pfade

```
/home/user/tiktok_production/
├── api/stable_production_api_multiprocess.py  # Haupt-API
├── configs/gpu_groups_config.py               # Analyzer-Konfiguration
├── ml_analyzer_registry_complete.py           # Analyzer-Registry
├── logs/stable_multiprocess_api.log          # Haupt-Logdatei
├── results/                                   # Analyse-Ergebnisse
├── fix_ffmpeg_env.sh                         # KRITISCHES Startup-Script
└── final_video_llava_performance_test.py     # Performance-Test
```

## Abschließende Hinweise

1. **Video-LLaVA ist kritisch** - ohne funktioniert das System, aber mit deutlich reduzierter Qualität
2. **FFmpeg-Fix ist PFLICHT** - niemals vergessen!
3. **3.15x Realtime ist akzeptabel** - keine Panik wenn leicht über 3x
4. **GPU-Monitoring wichtig** - bei Problemen zuerst GPU prüfen
5. **Logs sind dein Freund** - im Zweifel immer Logs checken

Das System ist stabil und produktionsreif. Mit dieser Dokumentation sollte das Operations-Team in der Lage sein, das System zuverlässig zu betreiben.

---
Übergeben am: 07. Juli 2025
Version: 2.0 (mit Video-LLaVA)
Erstellt von: ML Engineering Team