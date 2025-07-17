# 📋 TikTok Video Analysis - Deployment Guide

## Quick Start

### 1. Umgebung vorbereiten
```bash
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh  # KRITISCH: Immer zuerst ausführen!
```

### 2. System starten
```bash
./scripts/restart_services.sh start
```

### 3. Status prüfen
```bash
./scripts/restart_services.sh status
```

## Vollständige Installation

### Systemanforderungen
- GPU: NVIDIA mit min. 24GB VRAM (getestet: Quadro RTX 8000)
- RAM: 32GB minimum
- Python: 3.8+
- CUDA: 11.8+

### Schritt-für-Schritt

1. **Repository klonen**
   ```bash
   git clone <repository>
   cd tiktok_production
   ```

2. **Abhängigkeiten installieren**
   ```bash
   pip install -r requirements.txt
   ```

3. **FFmpeg-Umgebung einrichten**
   ```bash
   source fix_ffmpeg_env.sh
   ```

4. **Modelle herunterladen** (automatisch beim ersten Start)

5. **Service starten**
   ```bash
   python3 api/stable_production_api_multiprocess.py
   ```

## Konfiguration

### Primärer Video-Analyzer (BLIP-2)

**Datei**: `ml_analyzer_registry_complete.py`
```python
ML_ANALYZERS = {
    'blip2': BLIP2VideoCaptioningOptimized,  # Primärer Video-Analyzer
    ...
}
```

**Performance-Tuning**: `configs/gpu_groups_config.py`
```python
# Frame-Sampling anpassen
def get_frame_interval(analyzer_name):
    if analyzer_name == 'blip2':
        return 90  # Alle 3 Sekunden (für <3x Realtime)
```

### GPU-Gruppen

**Optimale Konfiguration für Quadro RTX 8000**:
- Stage 1: 2 concurrent (Heavy models)
- Stage 2: 3 concurrent (Medium models)
- Stage 3: 4 concurrent (Light models)
- CPU: 8 concurrent (Audio/Metadata)

## API Verwendung

### Video analysieren
```bash
curl -X POST "http://localhost:8003/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/video.mp4",
    "analyzers": ["blip2", "object_detection", "speech_transcription"]
  }'
```

### Alle Analyzer verwenden
```bash
curl -X POST "http://localhost:8003/analyze" \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/video.mp4"}'
```

## Monitoring

### Live GPU-Überwachung
```bash
watch -n 1 nvidia-smi
```

### System-Monitoring starten
```bash
python3 monitoring/system_monitor.py
```

### Logs überwachen
```bash
# API Logs
tail -f logs/stable_multiprocess_api.log

# System Metrics
tail -f logs/system_metrics.jsonl | jq .
```

## Troubleshooting

### Problem: FFmpeg Assertion Error
```bash
Assertion fctx->async_lock failed at libavcodec/pthread_frame.c:175
```
**Lösung**: Immer `source fix_ffmpeg_env.sh` vor dem Start ausführen

### Problem: GPU Out of Memory
**Lösung**: 
1. Batch-Größen in `gpu_groups_config.py` reduzieren
2. Weniger Analyzer parallel ausführen
3. GPU-Cache leeren: `torch.cuda.empty_cache()`

### Problem: API reagiert nicht
**Lösung**:
```bash
./scripts/restart_services.sh restart
```

## Systemd Service (Optional)

### Service-Datei erstellen
```bash
sudo nano /etc/systemd/system/tiktok-analyzer.service
```

```ini
[Unit]
Description=TikTok Video Analysis API
After=network.target

[Service]
Type=simple
User=user
WorkingDirectory=/home/user/tiktok_production
Environment="PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStartPre=/bin/bash -c 'source /home/user/tiktok_production/fix_ffmpeg_env.sh'
ExecStart=/usr/bin/python3 /home/user/tiktok_production/api/stable_production_api_multiprocess.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Service aktivieren
```bash
sudo systemctl daemon-reload
sudo systemctl enable tiktok-analyzer
sudo systemctl start tiktok-analyzer
```

## Backup & Wartung

### Tägliches Backup
```bash
# Results backup
rsync -av results/ /backup/tiktok_results/

# Logs rotation
find logs/ -name "*.log" -mtime +7 -delete
```

### GPU-Cache bereinigen
```bash
python3 -c "import torch; torch.cuda.empty_cache()"
```

## Performance-Optimierung

### BLIP-2 für maximale Geschwindigkeit
1. Frame-Interval erhöhen (90 → 120)
2. Max Frames reduzieren (120 → 60)
3. 4-bit Quantisierung testen

### GPU-Auslastung erhöhen
1. Mehr Worker-Prozesse (3 → 4)
2. Batch-Größen erhöhen
3. Concurrent Analyzer erhöhen

## Sicherheit

### API absichern
```python
# In stable_production_api_multiprocess.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Nur lokale Zugriffe
    allow_methods=["POST", "GET"],
)
```

### Ressourcen-Limits
```python
# Max Video-Größe
MAX_VIDEO_SIZE = 1024 * 1024 * 500  # 500MB

# Timeout für Analysen
ANALYSIS_TIMEOUT = 300  # 5 Minuten
```

## Support

Bei Problemen:
1. Logs prüfen (`logs/`)
2. GPU-Status prüfen (`nvidia-smi`)
3. System-Monitor ausführen
4. Service neu starten

**Wichtig**: AuroraCap ist als experimentell markiert und sollte NICHT in Produktion verwendet werden. Nutzen Sie BLIP-2 für alle Video-Beschreibungen.