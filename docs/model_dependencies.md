# ML Model Dependencies

Dieses Dokument listet alle ML-Modelle auf, die von den Analyzern verwendet werden, sowie deren Speicherbedarf und Status.

## Aktive Analyzer und ihre Modelle

### 1. qwen2_vl_temporal & qwen2_vl_optimized
- **Model**: `Qwen/Qwen2-VL-7B-Instruct`
- **Größe**: ~14GB
- **Typ**: Vision-Language Model
- **Verwendung**: Temporal video understanding, scene descriptions

### 2. object_detection
- **Model**: `yolov8x.pt`
- **Größe**: ~140MB
- **Typ**: Object Detection
- **Verwendung**: Objekterkennung in Videos

### 3. product_detection
- **Model**: `yolov8s.pt`
- **Größe**: ~22MB
- **Typ**: Object Detection (lightweight)
- **Verwendung**: Produkterkennung

### 4. text_overlay
- **Model**: EasyOCR (integriert)
- **Größe**: ~64MB pro Sprache
- **Typ**: OCR
- **Verwendung**: Texterkennung in Videos

### 5. speech_transcription
- **Model**: `openai/whisper-base`
- **Größe**: ~140MB
- **Typ**: Speech-to-Text
- **Verwendung**: Transkription von Sprache

### 6. background_segmentation
- **Model**: `nvidia/segformer-b0-finetuned-ade-512-512`
- **Größe**: ~15MB
- **Typ**: Semantic Segmentation
- **Verwendung**: Hintergrund/Vordergrund-Trennung

### 7. content_quality
- **Model**: CLIP (openai/clip-vit-base-patch32)
- **Größe**: ~340MB
- **Typ**: Vision-Language Embedding
- **Verwendung**: Qualitätsbewertung

### 8. speech_emotion
- **Model**: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
- **Größe**: ~1.2GB
- **Typ**: Audio Emotion Recognition
- **Verwendung**: Emotionserkennung in Sprache

### 9. eye_tracking
- **Model**: MediaPipe Face Mesh (integriert)
- **Größe**: ~10MB
- **Typ**: Face Landmark Detection
- **Verwendung**: Blickverfolgung

### 10. Andere Analyzer ohne explizite Modelle
Die folgenden Analyzer verwenden entweder:
- Algorithmus-basierte Ansätze (keine ML-Modelle)
- Integrierte kleine Modelle
- Librosa für Audio-Analyse

- age_estimation (DeepFace integriert)
- audio_analysis (Librosa)
- audio_environment (Librosa + Custom)
- camera_analysis (OpenCV Optical Flow)
- color_analysis (Scikit-learn Clustering)
- comment_cta_detection (Regex + Heuristiken)
- cut_analysis (Frame Differencing)
- scene_segmentation (Histogram-basiert)
- sound_effects (Audio Feature Extraction)
- speech_flow (Prosody Analysis)
- speech_rate (VAD + Timing)
- temporal_flow (Narrative Patterns)
- visual_effects (Computer Vision)

## Deaktivierte Analyzer und ihre Modelle (können gelöscht werden)

### Große Modelle (>1GB)
1. **video_llava**
   - Model: `LanguageBind/Video-LLaVA-7B`
   - Größe: ~14GB
   - **KANN GELÖSCHT WERDEN**

2. **blip2_video_analyzer**
   - Model: `Salesforce/blip2-opt-2.7b`
   - Größe: ~5.5GB
   - **KANN GELÖSCHT WERDEN**

3. **streaming_dense_captioning**
   - Model: `Salesforce/blip-image-captioning-base`
   - Größe: ~450MB
   - **KANN GELÖSCHT WERDEN**

### Mittlere Modelle (100MB-1GB)
4. **body_pose**
   - Model: `yolov8x-pose.pt`
   - Größe: ~140MB
   - **KANN GELÖSCHT WERDEN**

5. **composition_analysis**
   - Model: `openai/clip-vit-base-patch32`
   - Größe: ~340MB
   - **ACHTUNG**: Wird auch von content_quality verwendet!

### Kleine Modelle (<100MB)
Die folgenden deaktivierten Analyzer verwenden kleine oder keine expliziten Modelle:
- face_detection (MTCNN ~5MB)
- emotion_detection (FER ~300KB)
- facial_details (MediaPipe)
- hand_gesture (MediaPipe)
- gesture_recognition (MediaPipe)
- body_language (Pose Analysis)
- depth_estimation (MiDaS)
- temporal_consistency (Algorithmus)
- audio_visual_sync (Algorithmus)
- trend_analysis (Statistik)
- scene_description (Text Generation)
- auroracap_analyzer (Custom)
- tarsier_video_description (Custom)

## Speicherplatz-Freigabe-Potenzial

### Sicher zu löschende Modelle (Huggingface Cache)
```bash
# Video-LLaVA (14GB)
~/.cache/huggingface/hub/models--LanguageBind--Video-LLaVA-7B*

# BLIP2 (5.5GB)
~/.cache/huggingface/hub/models--Salesforce--blip2-opt-2.7b*

# BLIP Captioning (450MB)
~/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-base*

# YOLOv8 Pose (140MB)
~/.cache/torch/hub/checkpoints/yolov8x-pose.pt
```

**Gesamtes Freigabe-Potenzial: ~20GB**

### NICHT löschen (von aktiven Analyzern verwendet)
```bash
# Qwen2-VL
~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct*

# CLIP (von content_quality verwendet)
~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32*

# Whisper
~/.cache/huggingface/hub/models--openai--whisper-base*

# SegFormer
~/.cache/huggingface/hub/models--nvidia--segformer-b0-finetuned-ade-512-512*

# Wav2Vec2
~/.cache/huggingface/hub/models--ehcalabres--wav2vec2-lg-xlsr-en-speech-emotion-recognition*
```

## Empfohlene Cleanup-Strategie

1. **Zuerst**: Backup der Model-Liste erstellen
   ```bash
   ls -la ~/.cache/huggingface/hub/ > ~/model_backup_list.txt
   du -sh ~/.cache/huggingface/hub/models--* | sort -hr > ~/model_sizes.txt
   ```

2. **Dann**: Große ungenutzte Modelle löschen
   ```bash
   # Nach Bestätigung
   rm -rf ~/.cache/huggingface/hub/models--LanguageBind--Video-LLaVA-7B*
   rm -rf ~/.cache/huggingface/hub/models--Salesforce--blip2-opt-2.7b*
   ```

3. **Optional**: Weitere ungenutzte Modelle identifizieren
   ```bash
   # Alle Modelle im Cache auflisten
   find ~/.cache/huggingface/hub -name "config.json" -type f | sort
   ```

## Wartungshinweise

- **Vor Model-Löschung**: Immer prüfen ob Analyzer wirklich deaktiviert ist
- **Nach Updates**: Diese Liste aktualisieren wenn Analyzer hinzugefügt/entfernt werden
- **Cache-Management**: Huggingface Cache wächst mit der Zeit, regelmäßig prüfen