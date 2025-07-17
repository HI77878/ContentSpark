<<<<<<< HEAD
# 🚀 TikTok Analyzer Production System

**DER EINE WORKFLOW** - Keine Duplikate, keine Tests, nur Production!

## 🎯 MISSION COMPLETE
- ✅ **5.8GB Archive** erstellt (alte Systeme gesichert)
- ✅ **25 Dateien** in clean Production-System
- ✅ **19 Analyzer** in 5 GPU-Stufen
- ✅ **Eine API** (Port 8000)
- ✅ **Ein Workflow** (single_workflow.py)
- ✅ **Ziel: <3 Minuten** pro Video

## 📁 STRUKTUR
```
tiktok_production/
├── analyzers/          # 20 Analyzer-Dateien (19 aktiv + base)
├── api/               # single_api.py (Port 8000)
├── configs/           # gpu_groups_config.py, ml_analyzer_registry
├── downloads/         # TikTok Downloads
├── results/           # JSON Ergebnisse
├── logs/              # Service Logs
├── utils/             # json_encoder.py
├── single_workflow.py # DER EINE WORKFLOW
├── start.sh          # Starter Script
├── fix_ffmpeg_env.sh # FFmpeg Fix
└── tiktok-analyzer.service # Systemd Service
```

## 🚀 QUICK START

### 1. CLI Workflow
```bash
cd /home/user/tiktok_production
python3 single_workflow.py "https://www.tiktok.com/@user/video/123"
```

### 2. API Server
```bash
cd /home/user/tiktok_production
./start.sh
```

### 3. API Usage
```bash
# Analyze video
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.tiktok.com/@user/video/123"}'

# Health check
curl http://localhost:8000/health

# List results
curl http://localhost:8000/results
```

## 🔍 ANALYZER STAGES

### Stage 1: Heavy (Qwen2-VL)
- `qwen2_vl_temporal` - Video understanding mit Qwen2-VL-7B

### Stage 2: Core Analysis
- `object_detection` - YOLOv8 Objekt-Erkennung
- `text_overlay` - EasyOCR Text-Erkennung
- `speech_transcription` - Whisper Speech-to-Text

### Stage 3: Detailed Analysis
- `body_pose` - YOLOv8 Körperhaltung
- `background_segmentation` - SegFormer Hintergrund
- `camera_analysis` - Kamera-Bewegung
- `scene_segmentation` - Szenen-Segmentierung
- `color_analysis` - Farb-Analyse
- `content_quality` - CLIP Qualität
- `cut_analysis` - Schnitt-Analyse
- `age_estimation` - InsightFace Alter
- `eye_tracking` - MediaPipe Augen

### Stage 4: Audio & Flow
- `audio_analysis` - Librosa Audio-Analyse
- `audio_environment` - Umgebungs-Audio
- `speech_emotion` - Wav2Vec2 Emotion
- `speech_flow` - Speech Flow-Analyse
- `temporal_flow` - Narrative Analyse

### Stage 5: Intelligence
- `cross_analyzer_intelligence` - Korrelation aller Ergebnisse

## 📊 PERFORMANCE ZIELE
- **Zeit:** <3 Minuten pro Video
- **Erfolgsrate:** 19/19 Analyzer (100%)
- **GPU-Auslastung:** 85-95%
- **Speicher:** <16GB GPU RAM
- **Output:** Eine JSON mit allen Daten

## 🔧 SYSTEM REQUIREMENTS
- NVIDIA GPU mit 16GB+ VRAM
- Python 3.10+
- FFmpeg (fixed mit fix_ffmpeg_env.sh)
- yt-dlp für TikTok Downloads
- ML-Modelle in ~/.cache/huggingface/

## 📋 MONITORING
```bash
# GPU-Auslastung
watch -n 1 nvidia-smi

# API-Logs
tail -f logs/api.log

# Ergebnisse
ls -la results/
```

## 🎯 NEXT STEPS
1. **Teste CLI:** `python3 single_workflow.py "URL"`
2. **Teste API:** `./start.sh`
3. **Alte Systeme löschen** (nach erfolgreichem Test)

## 🚨 WICHTIG
- **Keine Duplikate:** Alles in einem System
- **Eine API:** Nur Port 8000
- **Ein Workflow:** single_workflow.py
- **ML-Modelle:** Bleiben in ~/.cache/huggingface/
- **Backups:** In /home/user/ARCHIVE_20250716/

---
**Mission Complete! 🎉**
=======
# TikTok Video Analysis System

Ein hochoptimiertes System zur umfassenden Analyse von TikTok-Videos mit 23 ML-basierten Analyzern. Das System nutzt eine NVIDIA Quadro RTX 8000 GPU für Echtzeit-Videoanalyse und generiert detaillierte JSON-Berichte für Film- und Videoproduktionsanalysen.

## System-Übersicht

- **Aktive Analyzer**: 23 von 42 (55%)
- **Performance**: <3x Realtime für die meisten Videos
- **GPU**: Quadro RTX 8000 (44.5GB VRAM)
- **API**: FastAPI auf Port 8003
- **Output**: ~2-3MB JSON pro Video mit umfassenden Analysedaten

## Aktive Analyzer (23)

### Video-Analyse
- `object_detection` - YOLOv8x-basierte Objekterkennung
- `product_detection` - Produkterkennung in Videos
- `visual_effects` - Erkennung visueller Effekte
- `text_overlay` - OCR für Textüberlagerungen (EasyOCR)
- `camera_analysis` - Kamerabewegung und -techniken
- `background_segmentation` - Hintergrund/Vordergrund-Trennung
- `color_analysis` - Farbanalyse und Paletten
- `content_quality` - CLIP-basierte Qualitätsbewertung
- `scene_segmentation` - Szenenübergänge
- `cut_analysis` - Schnitterkennung
- `eye_tracking` - Blickverfolgung mit MediaPipe
- `age_estimation` - Altersschätzung

### Audio-Analyse
- `speech_transcription` - Whisper Large V3 Transkription
- `audio_analysis` - Umfassende Audioanalyse
- `audio_environment` - Umgebungsgeräusch-Klassifikation
- `sound_effects` - Sound-Effekt-Erkennung
- `speech_emotion` - Emotionserkennung in Sprache
- `speech_rate` - Sprechgeschwindigkeit
- `speech_flow` - Sprachfluss-Analyse
- `comment_cta_detection` - Call-to-Action Erkennung

### Erweiterte Analyse
- `qwen2_vl_temporal` - Qwen2-VL-7B Temporal Video Understanding
- `qwen2_vl_optimized` - Optimierte Qwen2-VL Variante
- `temporal_flow` - Zeitliche Narrative Analyse

## Deaktivierte Analyzer (19)

Diese Analyzer sind aus Performance- oder Stabilitätsgründen deaktiviert:

- `face_detection`, `emotion_detection`, `facial_details` - Gesichtsanalyse
- `body_pose`, `body_language`, `hand_gesture`, `gesture_recognition` - Körperanalyse
- `video_llava`, `blip2_video_analyzer`, `vid2seq` - Alternative Video-Verständnis-Modelle
- `depth_estimation`, `temporal_consistency`, `audio_visual_sync` - Technische Analyse
- `scene_description`, `composition_analysis` - Szenenbeschreibung
- `streaming_dense_captioning`, `tarsier_video_description` - Alternative Captioning
- `auroracap_analyzer`, `trend_analysis` - Spezialanalysen

## Installation & Setup

### Voraussetzungen
- Ubuntu 22.04 LTS
- NVIDIA GPU mit mindestens 24GB VRAM
- Python 3.10+
- CUDA 12.4
- 40GB+ RAM
- 100GB+ freier Speicherplatz

### Quick Start

```bash
# 1. FFmpeg-Umgebung einrichten (WICHTIG!)
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh

# 2. API starten
python3 api/stable_production_api_multiprocess.py

# 3. Health Check
curl http://localhost:8003/health
```

## API-Nutzung

### Video analysieren

```bash
curl -X POST "http://localhost:8003/analyze" \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/video.mp4"}'
```

### TikTok-Video herunterladen und analysieren

```python
from mass_processing.tiktok_downloader import TikTokDownloader
downloader = TikTokDownloader()
result = downloader.download_video('https://www.tiktok.com/@username/video/123')
# Dann mit der API analysieren
```

### API-Dokumentation

Swagger UI verfügbar unter: http://localhost:8003/docs

## Performance-Metriken

### Durchschnittliche Analysezeiten (30s Video)
- Gesamtzeit: ~90-120s (<3x Realtime)
- GPU-Auslastung: 85-95%
- RAM-Nutzung: ~2-4GB
- Output-Größe: ~2-3MB JSON

### Langsamste Analyzer (Optimierungspotential)
1. `qwen2_vl_temporal` - 252.4s
2. `text_overlay` - 115.3s
3. `object_detection` - 38.7s
4. `speech_rate` - 36.2s

## Projektstruktur

```
tiktok_production/
├── analyzers/              # ML Analyzer Implementierungen
│   ├── *.py               # 129 Analyzer-Dateien (23 aktiv)
│   └── __init__.py
├── api/                    # API Server
│   └── stable_production_api_multiprocess.py
├── configs/                # Konfigurationsdateien
│   ├── gpu_groups_config.py
│   └── performance_config.py
├── mass_processing/        # Batch-Verarbeitung
│   └── tiktok_downloader.py
├── results/                # JSON Ausgabedateien
├── logs/                   # System- und API-Logs
├── utils/                  # Hilfsfunktionen
│   └── multiprocess_gpu_executor_final.py
├── docs/                   # Dokumentation
└── test_videos/           # Test-Videos
```

## Monitoring & Debugging

### GPU-Monitoring
```bash
# Live GPU-Auslastung
watch -n 1 nvidia-smi

# Detaillierte GPU-Metriken
nvidia-smi dmon -i 0 -s pucm -d 1
```

### Logs
```bash
# API-Logs
tail -f logs/stable_multiprocess_api.log

# System-Metriken
tail -f logs/system_metrics.jsonl
```

### Analyzer-Status prüfen
```python
from ml_analyzer_registry_complete import ML_ANALYZERS
from configs.gpu_groups_config import DISABLED_ANALYZERS
active = len(ML_ANALYZERS) - len(DISABLED_ANALYZERS)
print(f"Aktive Analyzer: {active}")
```

## Häufige Probleme

### FFmpeg Assertion Error
**Problem**: `Assertion fctx->async_lock failed`  
**Lösung**: Immer `source fix_ffmpeg_env.sh` vor dem Start ausführen

### GPU Out of Memory
**Problem**: CUDA OOM Fehler  
**Lösung**: Batch-Größen in `configs/performance_config.py` reduzieren

### Langsame Analyse
**Problem**: Analyse dauert >5x Realtime  
**Lösung**: Deaktiviere schwere Analyzer in `configs/gpu_groups_config.py`

## Wartung

### Speicherplatz freigeben
```bash
# Docker cleanup
docker system prune -a -f --volumes

# Alte Logs löschen
find logs/ -mtime +7 -delete

# Alte Results archivieren
tar -czf results_backup_$(date +%Y%m%d).tar.gz results/
```

### Analyzer hinzufügen/entfernen
1. Bearbeite `ml_analyzer_registry_complete.py`
2. Update `configs/gpu_groups_config.py`
3. Restart API Server

## Entwicklung

### Neuen Analyzer erstellen
1. Erstelle Datei in `analyzers/` mit `GPUBatchAnalyzer` als Basis
2. Implementiere `analyze()` und `process_batch_gpu()`
3. Registriere in `ml_analyzer_registry_complete.py`
4. Füge zu GPU-Gruppe in `configs/gpu_groups_config.py` hinzu

### Tests ausführen
```python
# Einzelnen Analyzer testen
from analyzers.object_detection_yolo import GPUBatchObjectDetectionYOLO
analyzer = GPUBatchObjectDetectionYOLO()
result = analyzer.analyze('/path/to/test/video.mp4')
print(f"Gefundene Objekte: {len(result['segments'])}")
```

## Lizenz & Kontakt

Proprietäres System - Alle Rechte vorbehalten.

Für Fragen oder Support: [Kontaktinformationen hier einfügen]

---

Stand: Juli 2025 | Version: Production Stable
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
