# Deployment Guide - TikTok Video Analysis System V2

## Übersicht

Diese Anleitung beschreibt das Deployment des TikTok Video Analysis Systems mit Video-LLaVA als primärem Video-Analyzer.

## System-Voraussetzungen

### Hardware
- **GPU**: NVIDIA GPU mit mindestens 16GB VRAM
  - Getestet: Quadro RTX 8000 (44.5GB)
  - Minimum: RTX 3090 (24GB) oder RTX 4090 (24GB)
- **RAM**: 64GB empfohlen (32GB minimum)
- **CPU**: 16+ Cores für optimale Parallelisierung
- **Storage**: 100GB+ für Modell-Cache und Ergebnisse

### Software
- **OS**: Ubuntu 20.04 oder 22.04 LTS
- **Python**: 3.10.x
- **CUDA**: 12.1 (kompatibel mit PyTorch 2.2.2)
- **Docker**: 20.10+ (optional für Video-LLaVA Service)

## Installation

### 1. System-Vorbereitung

```bash
# System-Updates
sudo apt update && sudo apt upgrade -y

# Entwicklungstools
sudo apt install -y git build-essential python3.10 python3.10-dev python3-pip

# FFmpeg (WICHTIG für Video-Verarbeitung)
sudo apt install -y ffmpeg

# NVIDIA Docker (optional)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-docker2
```

### 2. Repository Setup

```bash
# Clone Repository (falls noch nicht vorhanden)
cd /home/user
git clone <repository-url> tiktok_production
cd tiktok_production

# Python-Umgebung
python3.10 -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Modell-Downloads

Video-LLaVA wird automatisch beim ersten Start heruntergeladen. Pre-Download möglich:

```bash
python3 -c "
from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration
print('Downloading Video-LLaVA model...')
processor = AutoProcessor.from_pretrained('llava-hf/LLaVA-NeXT-Video-7B-hf')
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    'llava-hf/LLaVA-NeXT-Video-7B-hf',
    torch_dtype='auto',
    device_map='auto'
)
print('✅ Model downloaded successfully')
"
```

### 4. Konfiguration überprüfen

```bash
# Verifiziere GPU
nvidia-smi

# Teste Python-Imports
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Überprüfe FFmpeg
ffmpeg -version
```

## Deployment-Schritte

### 1. FFmpeg-Umgebung vorbereiten

**KRITISCH**: Immer vor dem Start ausführen!

```bash
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh
```

### 2. API-Server starten

```bash
# Produktion (Hintergrund)
nohup python3 api/stable_production_api_multiprocess.py > logs/api_production.log 2>&1 &

# Oder mit Screen/Tmux für bessere Kontrolle
screen -S tiktok-api
python3 api/stable_production_api_multiprocess.py
# Detach mit Ctrl+A, D
```

### 3. Deployment verifizieren

```bash
# Health Check
curl http://localhost:8003/health | python3 -m json.tool

# Test-Analyse
curl -X POST http://localhost:8003/analyze \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/test/video.mp4"}'
```

## Video-LLaVA Spezifika

### Modell-Eigenschaften
- **Modell**: LLaVA-NeXT-Video-7B
- **Quantisierung**: 4-bit (optimal für Performance)
- **Ladezeit**: ~14 Sekunden pro Worker
- **GPU-Speicher**: 3.8GB
- **Analyse-Zeit**: ~10 Sekunden pro Video

### Konfiguration in ml_analyzer_registry_complete.py
```python
ML_ANALYZERS = {
    'video_llava': LLaVAVideoOptimized,  # Primärer Video-Analyzer
    # ... andere Analyzer
}
```

### Deaktivierte Analyzer
In `configs/gpu_groups_config.py`:
```python
DISABLED_ANALYZERS = [
    # ... andere
    'blip2_video_analyzer',     # Ersetzt durch Video-LLaVA
    'auroracap_analyzer',       # Ersetzt durch Video-LLaVA
]
```

## Optional: Docker Service für Video-LLaVA

Für bessere Isolation und Pre-Loading:

```bash
cd /home/user/tiktok_production/docker/video_llava
./build_and_run.sh

# Verifiziere Service
curl http://localhost:8004/health
```

## Systemd Service (Empfohlen für Produktion)

```bash
# Service-Datei erstellen
sudo nano /etc/systemd/system/tiktok-analyzer.service
```

Inhalt:
```ini
[Unit]
Description=TikTok Video Analyzer API
After=network.target

[Service]
Type=simple
User=user
WorkingDirectory=/home/user/tiktok_production
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="OMP_NUM_THREADS=1"
ExecStartPre=/bin/bash -c 'source /home/user/tiktok_production/fix_ffmpeg_env.sh'
ExecStart=/usr/bin/python3 /home/user/tiktok_production/api/stable_production_api_multiprocess.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Service aktivieren
sudo systemctl daemon-reload
sudo systemctl enable tiktok-analyzer
sudo systemctl start tiktok-analyzer
sudo systemctl status tiktok-analyzer
```

## Monitoring

### Logs
```bash
# API Logs
tail -f /home/user/tiktok_production/logs/stable_multiprocess_api.log

# Systemd Logs
sudo journalctl -u tiktok-analyzer -f
```

### Performance
```bash
# GPU-Auslastung
watch -n 1 nvidia-smi

# API-Metriken
curl http://localhost:8003/health
```

## Troubleshooting

### Problem: FFmpeg Assertion Error
**Lösung**: Immer `source fix_ffmpeg_env.sh` vor dem Start

### Problem: Video-LLaVA lädt nicht
**Lösung**: 
1. GPU-Speicher prüfen: `nvidia-smi`
2. Modell-Cache löschen: `rm -rf ~/.cache/huggingface/hub/models--llava-hf*`
3. Neu herunterladen

### Problem: Performance > 3x Realtime
**Lösung**:
1. GPU-Auslastung prüfen
2. Worker-Anzahl in `stable_production_api_multiprocess.py` anpassen
3. Batch-Größen in `gpu_groups_config.py` optimieren

## Skalierung

### Horizontal (mehrere Instanzen)
```bash
# Instanz 1
PORT=8003 python3 api/stable_production_api_multiprocess.py

# Instanz 2
PORT=8004 python3 api/stable_production_api_multiprocess.py

# Load Balancer (nginx) davor
```

### Vertikal (mehr GPU-Worker)
In `api/stable_production_api_multiprocess.py`:
```python
self.executor = MultiprocessGPUExecutorFinal(num_gpu_processes=4)  # Erhöhen
```

## Wartung

### Model-Updates
```bash
# Cache löschen
rm -rf ~/.cache/huggingface/hub/models--llava-hf*

# Modell neu laden (passiert automatisch)
```

### Log-Rotation
```bash
# Crontab
0 0 * * * find /home/user/tiktok_production/logs -name "*.log" -mtime +7 -delete
```

## Checkliste für Produktion

- [ ] FFmpeg-Environment gesetzt
- [ ] GPU verfügbar und ausreichend Speicher
- [ ] Alle Python-Dependencies installiert
- [ ] API startet ohne Fehler
- [ ] Health-Check erfolgreich
- [ ] Test-Video erfolgreich analysiert
- [ ] Video-LLaVA produziert Ergebnisse
- [ ] Performance < 3.5x Realtime
- [ ] Logs werden geschrieben
- [ ] Systemd-Service konfiguriert

---
Stand: 07. Juli 2025
Version: 2.0 (Video-LLaVA Integration)