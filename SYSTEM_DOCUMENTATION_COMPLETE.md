# TikTok Video Analyzer System - Vollständige Dokumentation
## Stand: 17. Juli 2025 - 100% Success Rate erreicht!

### 🎯 SYSTEM ÜBERSICHT

#### Erreichte Ziele:
- **Success Rate**: 100% (alle 24 Analyzer funktionieren)
- **Performance**: 11s für Qwen2-VL (vorher 128s) 
- **GPU-Nutzung**: 15.5GB effizient genutzt
- **Stabilität**: Keine Crashes mehr

#### System-Architektur:
- **Server**: Ubuntu 22.04.5 LTS
- **GPU**: NVIDIA Quadro RTX 8000 (46GB VRAM)
- **Python**: 3.10.12
- **PyTorch**: 2.4.0 mit CUDA 12.1
- **API**: Port 8003 (stable_production_api_multiprocess.py)

### 📁 WICHTIGE DATEIEN UND IHRE FUNKTIONEN

#### 1. API und Hauptdateien:
```
/home/user/tiktok_production/
├── api/stable_production_api_multiprocess.py  # Haupt-API mit 3 GPU-Workern
├── single_workflow.py                         # Einzelner Workflow für Tests
├── fix_ffmpeg_env.sh                         # MUSS vor jedem Start ausgeführt werden!
├── start.sh                                  # API-Starter Script
└── start_clean_server.sh                     # Alternative Start-Methode
```

#### 2. Analyzer (24 aktive):
```
analyzers/
├── qwen2_vl_temporal_analyzer.py          # OPTIMIERT: Batch Processing, 11s statt 128s
├── audio_analysis_ultimate.py             # FIXED: Top-level librosa import
├── audio_environment_enhanced.py          # FIXED: ProcessPool entfernt
├── speech_transcription_ultimate.py       # Whisper Transcription
├── gpu_batch_speech_emotion.py           # Speech Emotion Recognition
├── gpu_batch_speech_rate_enhanced.py     # Speech Rate Analysis
├── gpu_batch_speech_flow.py              # Speech Flow Analysis
├── cross_analyzer_intelligence_safe.py   # FIXED: Type-safe wrapper
├── gpu_batch_object_detection_yolo.py    # YOLOv8 Object Detection
├── background_segmentation_light.py       # SegFormer Segmentation
├── text_overlay_tiktok_fixed.py          # EasyOCR Text Detection
├── visual_effects_light_fixed.py         # Visual Effects Detection
├── gpu_batch_product_detection_light.py  # Product Detection
├── face_emotion_deepface.py              # Face Emotion Recognition
├── body_pose_yolov8.py                   # Body Pose Estimation
├── gpu_batch_eye_tracking.py             # Eye Tracking
├── age_gender_insightface.py             # Age/Gender Estimation
├── camera_analysis_fixed.py              # Camera Movement Analysis
├── gpu_batch_color_analysis.py           # Color Analysis
├── gpu_batch_content_quality_fixed.py    # Content Quality Assessment
├── cut_analysis_fixed.py                 # Cut Detection
├── scene_segmentation_fixed.py           # Scene Segmentation
├── composition_analysis_light.py         # Composition Analysis
└── narrative_analysis_wrapper.py         # Temporal Flow Analysis
```

#### 3. Konfigurationen:
```
configs/
├── gpu_groups_config.py                  # 4-Stage GPU Execution
├── ml_analyzer_registry_complete.py      # Registry mit allen Analyzern
├── performance_config.py                 # Frame sampling und batch sizes
└── system_config.py                      # System-weite Einstellungen
```

#### 4. Utils:
```
utils/
├── staged_gpu_executor.py                # KRITISCH: Audio-Fix implementiert
├── simple_multiprocess_executor.py       # Multiprocess GPU execution
├── gpu_cleanup.py                        # GPU memory management
└── output_normalizer.py                  # Output standardization
```

### 🔧 KRITISCHE FIXES UND OPTIMIERUNGEN

#### 1. Qwen2-VL Optimierung (91% schneller):
- **Datei**: `analyzers/qwen2_vl_temporal_analyzer.py`
- **Was wurde gemacht**:
  - Global model loading (nur EINMAL laden)
  - Batch processing aller Segmente
  - Mixed precision mit torch.cuda.amp.autocast()
  - Reduzierte Auflösung auf 512x384
  - device_map="cuda:0" statt "auto"
- **Kritisch**: Model MUSS global geladen werden!

#### 2. Audio-Analyzer ProcessPool Fix:
- **Problem**: ProcessPoolExecutor crasht mit librosa
- **Lösung in** `utils/staged_gpu_executor.py`:
```python
# Stage 4 - Audio Analyzer direkt ausführen
if analyzer_name in ['audio_analysis', 'audio_environment', 'speech_emotion', 
                     'speech_transcription', 'speech_flow', 'speech_rate']:
    # Direct execution - no ProcessPool
    from configs.ml_analyzer_registry_complete import ML_ANALYZERS
    analyzer_class = ML_ANALYZERS.get(analyzer_name)
    analyzer = analyzer_class()
    result = analyzer.analyze(video_path)
```

#### 3. GPU-Optimierungen:
- **Datei**: `utils/staged_gpu_executor.py` (oben)
```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
```

#### 4. Cross-Analyzer Fix:
- **Problem**: 'str' object has no attribute 'get'
- **Lösung**: Safe wrapper in `cross_analyzer_intelligence_safe.py`
- **Wichtig**: Methode heißt `analyze()` nicht `analyze_with_context()`

### 🚀 SYSTEM STARTEN

#### Schritt-für-Schritt Anleitung:
```bash
# 1. Ins Verzeichnis wechseln
cd /home/user/tiktok_production

# 2. FFmpeg Environment setzen (KRITISCH!)
source fix_ffmpeg_env.sh

# 3. GPU-Optimierungen setzen
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# 4. API starten
python3 api/stable_production_api_multiprocess.py

# 5. Warten bis Modelle geladen sind (ca. 30s)
# Check: curl http://localhost:8003/health

# 6. Video analysieren
curl -X POST "http://localhost:8003/analyze" \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/home/user/tiktok_production/test_videos/test1.mp4"}'
```

### 📊 PERFORMANCE METRIKEN

#### Erwartete Zeiten für 10s Video:
- **Erste Analyse**: ~250s (Model Loading)
- **Weitere Analysen**: ~110s (11x realtime)
- **Qwen2-VL direkt**: 11s (1.1x realtime)

#### GPU-Auslastung:
- **Beim Start**: ~400MB
- **Während Analyse**: 15-20GB
- **Nach Analyse**: ~16GB (Modelle im Speicher)

### ⚠️ HÄUFIGE PROBLEME UND LÖSUNGEN

#### 1. "Process pool terminated abruptly":
- **Ursache**: Audio-Analyzer mit ProcessPool
- **Lösung**: Sicherstellen dass Stage 4 Fix aktiv ist

#### 2. Qwen2-VL zu langsam (>100s):
- **Ursache**: Alte Version ohne Batch Processing
- **Lösung**: Prüfen ob qwen2_vl_temporal_analyzer.py die optimierte Version ist

#### 3. GPU Memory Error:
- **Ursache**: Zu viele Modelle gleichzeitig
- **Lösung**: torch.cuda.empty_cache() zwischen Stages

#### 4. Cross-Analyzer Fehler:
- **Ursache**: Falsche Methode aufgerufen
- **Lösung**: analyze() statt analyze_with_context()

### 🔄 SYSTEM WIEDERHERSTELLEN

Falls etwas kaputt geht:

1. **Backup wiederherstellen**:
```bash
# Alte funktionierende Version aus Archiv
cp /home/user/WORKING_BACKUP_*/analyzers/*.py /home/user/tiktok_production/analyzers/
cp /home/user/WORKING_BACKUP_*/utils/*.py /home/user/tiktok_production/utils/
cp /home/user/WORKING_BACKUP_*/configs/*.py /home/user/tiktok_production/configs/
```

2. **Kritische Fixes erneut anwenden**:
- Qwen2-VL Batch Processing
- Audio-Analyzer ProcessPool Fix
- GPU-Optimierungen

3. **Test durchführen**:
```bash
python3 test_final_100_percent.py
```

### 📋 ANALYZER LISTE (24 aktiv)

1. **qwen2_vl_temporal** - Video Understanding (HAUPTANALYZER)
2. **object_detection** - YOLOv8 Objekterkennung
3. **text_overlay** - EasyOCR Texterkennung
4. **background_segmentation** - SegFormer
5. **body_pose** - Körperhaltung
6. **age_estimation** - Alter/Geschlecht
7. **content_quality** - CLIP Qualität
8. **color_analysis** - Farbanalyse
9. **camera_analysis** - Kamerabewegung
10. **cut_analysis** - Schnitterkennung
11. **scene_segmentation** - Szenenwechsel
12. **temporal_flow** - Narrative Analyse
13. **eye_tracking** - Blickverfolgung
14. **audio_analysis** - Audio-Features
15. **audio_environment** - Umgebungsgeräusche
16. **speech_emotion** - Sprach-Emotionen
17. **speech_flow** - Sprachfluss
18. **speech_transcription** - Whisper
19. **speech_rate** - Sprechgeschwindigkeit
20. **face_emotion** - Gesichtserkennung & Emotionen
21. **visual_effects** - Visuelle Effekte
22. **product_detection** - Produkterkennung
23. **composition_analysis** - Bildkomposition
24. **cross_analyzer_intelligence** - Korrelation

### 💾 WICHTIGE ENVIRONMENT VARIABLEN

```bash
# In fix_ffmpeg_env.sh:
export LD_LIBRARY_PATH=/home/user/ffmpeg-install/lib:$LD_LIBRARY_PATH
export PATH=/home/user/ffmpeg-install/bin:$PATH
export OPENCV_FFMPEG_CAPTURE_OPTIONS="protocol_whitelist=file,http,https,tcp,tls"
export OPENCV_VIDEOIO_PRIORITY_BACKEND=4  # cv2.CAP_GSTREAMER
export OPENCV_FFMPEG_MULTITHREADED=0
export OPENCV_FFMPEG_DEBUG=1

# Für GPU-Optimierung:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TOKENIZERS_PARALLELISM=false
```

### 🎯 ERFOLGS-KRITERIEN

Das System funktioniert korrekt wenn:
- Health Check zeigt 24 aktive Analyzer
- Alle 24 aktiven Analyzer produzieren Segmente
- Qwen2-VL braucht <30s (nicht 128s)
- Keine "Process pool terminated" Fehler
- GPU nutzt >10GB während Analyse
- Results JSON ist 2-3MB groß

### 📈 SYSTEM MONITORING

#### GPU Monitoring:
```bash
# Realtime GPU-Nutzung
watch -n 1 nvidia-smi

# Detaillierte GPU-Metriken
nvidia-smi dmon -i 0 -s pucm -d 1
```

#### Log Monitoring:
```bash
# API Logs
tail -f /home/user/tiktok_production/logs/stable_multiprocess_api.log

# System Logs
tail -f /home/user/tiktok_production/logs/stable_api.log
```

#### Performance Test:
```bash
# Vollständiger Test aller Analyzer
python3 test_final_100_percent.py

# Schneller Test einzelner Analyzer
python3 quick_analyzer_test.py

# Direkter Qwen2-VL Test
python3 test_qwen2_direct.py
```

### 🔐 SICHERHEIT UND STABILITÄT

#### Automatische Bereinigung:
- GPU Memory wird nach jeder Stage bereinigt
- Modelle bleiben für Performance im Speicher
- Automatische Garbage Collection aktiviert

#### Error Handling:
- Jeder Analyzer hat Try-Catch Blöcke
- Fehler werden geloggt aber stoppen nicht die Pipeline
- Timeout-Protection für lange Analysen

#### Multiprocess Isolation:
- 3 separate GPU Worker Prozesse
- Worker 0: Reserviert für Qwen2-VL
- Worker 1-2: Andere Analyzer
- Keine Interferenz zwischen Prozessen

### 📝 WARTUNG UND UPDATES

#### Tägliche Checks:
1. API Health: `curl http://localhost:8003/health`
2. GPU Status: `nvidia-smi`
3. Log-Größe: `du -sh logs/`

#### Wöchentliche Wartung:
1. Alte Logs archivieren
2. Results-Ordner aufräumen
3. GPU-Speicher vollständig bereinigen (Neustart)

#### Updates:
- Neue Analyzer in `ml_analyzer_registry_complete.py` registrieren
- GPU-Stage in `gpu_groups_config.py` zuordnen
- Frame-Sampling in `performance_config.py` konfigurieren

---
Dokumentation erstellt: 17. Juli 2025
System-Status: 100% funktionsfähig
Entwicklungszeit: 6 Monate