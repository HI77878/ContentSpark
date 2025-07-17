# GPU Groups Configuration Documentation

Dieses Dokument beschreibt die GPU-Workload-Verteilung und Performance-Optimierungen für das TikTok Analyzer System.

## Übersicht

Das System nutzt eine gestaffelte GPU-Gruppierung, um die Quadro RTX 8000 (44.5GB VRAM) optimal auszulasten. Analyzer werden basierend auf ihrem Speicherbedarf und ihrer Ausführungszeit in Gruppen eingeteilt.

## GPU-Gruppen

### Stage 1: GPU Heavy (2 concurrent)
Die speicherintensivsten Modelle mit höchster Priorität.

| Analyzer | Zeit (s) | Beschreibung |
|----------|----------|--------------|
| qwen2_vl_temporal | 110.0 | Temporal multi-frame analysis mit Qwen2-VL-7B |
| product_detection | 50.4 | YOLOv8s Produkterkennung |
| object_detection | 50.3 | YOLOv8x Objekterkennung |
| visual_effects | 22.5 | Visuelle Effekte Erkennung |

**Gesamtzeit**: ~233s bei sequenzieller Ausführung
**GPU-Memory**: ~15-20GB peak

### Stage 2: GPU Medium (3 concurrent)
Mittelschwere Modelle mit moderatem Speicherbedarf.

| Analyzer | Zeit (s) | Beschreibung |
|----------|----------|--------------|
| streaming_dense_captioning | 15.0 | Optimierte Video-Beschreibung |
| camera_analysis | 36.1 | Kamerabewegung & Techniken |
| text_overlay | 37.1 | EasyOCR Texterkennung |
| background_segmentation | 41.2 | SegFormer Segmentierung |
| speech_rate | 14.1 | Sprechgeschwindigkeit |

**Gesamtzeit**: ~143s bei sequenzieller Ausführung
**GPU-Memory**: ~5-10GB peak

### Stage 3: GPU Light (4 concurrent)
Leichte Modelle die parallel ausgeführt werden können.

| Analyzer | Zeit (s) | Beschreibung |
|----------|----------|--------------|
| composition_analysis | 13.6 | CLIP-basierte Komposition |
| color_analysis | 16.4 | Farbanalyse |
| content_quality | 11.7 | CLIP Qualitätsbewertung |
| eye_tracking | 10.4 | MediaPipe Blickverfolgung |
| scene_segmentation | 10.6 | Szenenübergänge |

**Gesamtzeit**: ~63s bei sequenzieller Ausführung
**GPU-Memory**: ~2-5GB peak

### Stage 4: GPU Fast (4 concurrent)
Sehr schnelle, leichte Analyzer.

| Analyzer | Zeit (s) | Beschreibung |
|----------|----------|--------------|
| cut_analysis | 4.1 | Schnitterkennung |
| age_estimation | 1.1 | Altersschätzung |

**Gesamtzeit**: ~5s
**GPU-Memory**: <1GB

### CPU Parallel (8 concurrent)
Audio und Metadaten-Analyzer die keine GPU benötigen.

| Analyzer | Zeit (s) | Beschreibung |
|----------|----------|--------------|
| speech_transcription | 4.5 | Whisper Transkription |
| sound_effects | 5.9 | Sound-Effekt Erkennung |
| speech_emotion | 1.6 | Emotionserkennung |
| speech_flow | - | Sprachfluss |
| comment_cta_detection | - | CTA Erkennung |
| audio_environment | 0.5 | Umgebungsgeräusche |
| temporal_flow | 2.1 | Narrative Analyse |
| audio_analysis | 0.2 | Audio Features |

**Gesamtzeit**: ~15s (parallel)
**CPU-Cores**: 8

## Performance-Metriken

### Aktuelle Performance (30s Video)
- **Stage 1**: ~60s (2 concurrent)
- **Stage 2**: ~48s (3 concurrent)  
- **Stage 3**: ~16s (4 concurrent)
- **Stage 4**: ~2s (4 concurrent)
- **CPU**: ~15s (8 concurrent)
- **Gesamt**: ~90-120s (<3x Realtime)

### GPU-Auslastung
- **Peak Memory**: ~25GB/44.5GB (56%)
- **Durchschnitt**: 85-95% Compute
- **Bottleneck**: Stage 1 (qwen2_vl_temporal)

## Optimierungspotential

### Identifizierte Probleme
1. **qwen2_vl_temporal** (110s) - Hauptbottleneck
2. **text_overlay** (37s) - Könnte optimiert werden
3. **object_detection** (50s) - TensorRT möglich
4. **speech_rate** (14s) - Sollte CPU sein

### Empfohlene Optimierungen

#### 1. Qwen2-VL Temporal (110s → <50s)
- Flash Attention 2 aktivieren
- INT8 Quantization
- Batch Processing verbessern
- CUDA Graphs nutzen

#### 2. Text Overlay (37s → <15s)
- Batch OCR implementieren
- Frame Deduplication
- GPU-optimierte Alternative prüfen

#### 3. Object Detection (50s → <20s)
- TensorRT Export
- FP16 Inference
- Optimale Batch-Größe

#### 4. Load Balancing
- Stage 1 ist überlastet (233s total)
- Stage 4 ist unterlastet (5s total)
- Umverteilung würde Performance verbessern

## Memory Management

### Cleanup-Strategie
```python
GPU_MEMORY_CONFIG = {
    'cleanup_after_stage': True,  # GPU nach jeder Stage leeren
    'max_concurrent': {
        'stage1': 2,  # Max 2 Heavy Models gleichzeitig
        'stage2': 3,  # Max 3 Medium Models
        'stage3': 4,  # Max 4 Light Models
        'stage4': 4   # Max 4 Fast Models
    }
}
```

### Memory Thresholds
- **Warnung**: 35GB (78%)
- **Kritisch**: 40GB (90%)
- **Auto-Cleanup**: Nach jeder Stage

## Deaktivierte Analyzer (19)

Diese Analyzer sind deaktiviert und ihre GPU-Zeiten sind nicht relevant:

- **Video Understanding**: video_llava, blip2_video_analyzer, vid2seq, streaming_dense_captioning
- **Face/Body**: face_detection, emotion_detection, body_pose, body_language, facial_details
- **Gestures**: hand_gesture, gesture_recognition
- **Technical**: depth_estimation, temporal_consistency, audio_visual_sync
- **Other**: scene_description, composition_analysis, trend_analysis, auroracap_analyzer, tarsier_video_description

## Wartung und Monitoring

### GPU Monitoring
```bash
# Live GPU Stats
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv'

# Analyzer Timing
tail -f logs/stable_multiprocess_api.log | grep "completed in"
```

### Performance Tuning
1. **Batch Sizes**: In `configs/performance_config.py` anpassen
2. **Concurrency**: `max_concurrent` Werte ändern
3. **Stage Assignment**: Analyzer zwischen Stages verschieben

### Hinzufügen neuer Analyzer
1. Timing mit Einzeltest messen
2. Basierend auf Zeit und Memory zur passenden Stage hinzufügen
3. `max_concurrent` ggf. anpassen

## Zusammenfassung

Das aktuelle System erreicht <3x Realtime Performance durch:
- Intelligente GPU-Gruppierung
- Parallele CPU-Verarbeitung
- Memory Management zwischen Stages
- Optimierte Model Loading

Hauptverbesserungspotential liegt bei den Stage 1 Analyzern, insbesondere qwen2_vl_temporal.