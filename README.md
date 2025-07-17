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