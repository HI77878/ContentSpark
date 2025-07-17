# Analyzer Documentation - TikTok Video Analysis System

## Overview

The TikTok Video Analysis System employs 19 active ML analyzers distributed across GPU and CPU workers to provide comprehensive video analysis. Each analyzer is optimized for specific aspects of video content and designed for maximum GPU utilization and model caching efficiency.

## Analyzer Architecture

### Base Classes

#### GPUBatchAnalyzer
All GPU-based analyzers inherit from this base class:

```python
class GPUBatchAnalyzer:
    def __init__(self, batch_size=8):
        self.batch_size = batch_size
        self.analyzer_name = ""
        self.model = None
        
    def _load_model_impl(self):
        # Lazy model loading - implemented by subclasses
        pass
        
    def analyze(self, video_path):
        # Main entry point - standardized across all analyzers
        if self.model is None:
            self._load_model_impl()
        return self._process_video(video_path)
        
    def process_batch_gpu(self, frames, frame_times):
        # GPU batch processing - implemented by subclasses
        pass
```

## GPU Worker 0 - Video Understanding (Dedicated)

### qwen2_vl_temporal - Temporal Video Analysis
- **Model**: Qwen2-VL-7B-Instruct (Alibaba)
- **File**: `analyzers/qwen2_vl_video_analyzer.py`
- **GPU Memory**: 16GB (exclusive)
- **Purpose**: Comprehensive temporal video understanding with scene descriptions
- **Input**: 16 frames per 2-second segment
- **Processing Time**: 60s (optimized from 110s)
- **Batch Size**: 1 (memory intensive)

#### Technical Details
```python
class Qwen2VLVideoAnalyzer:
    def __init__(self):
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
        self.processor = Qwen2VLProcessor.from_pretrained(self.model_name)
        self.device = "cuda"
        
    def _load_model_impl(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
```

#### Output Structure
```json
{
  "segment_id": "qwen_temporal_0",
  "start_time": 0.0,
  "end_time": 2.0,
  "description": "A person energetically shows a product to the camera while standing in a modern room with warm lighting.",
  "objects_detected": ["person", "product", "indoor_space"],
  "scene_context": "product_demonstration",
  "confidence": 0.95,
  "temporal_continuity": "high"
}
```

## GPU Worker 1 - Visual Analysis (8-10GB VRAM)

### object_detection - Object Detection and Tracking
- **Model**: YOLOv8x (Ultralytics)
- **File**: `analyzers/gpu_batch_object_detection_yolo.py`
- **GPU Memory**: 3-4GB
- **Purpose**: Detects 80 COCO object classes + TikTok-specific objects
- **Processing Time**: 15s (optimized from 25s)
- **Frame Sampling**: Every 10th frame
- **Batch Size**: 64

#### Technical Implementation
```python
class GPUBatchObjectDetectionYOLO:
    def _load_model_impl(self):
        self.model = YOLO('yolov8x.pt')
        self.model.to('cuda')
        
    def process_batch_gpu(self, frames, frame_times):
        results = self.model(frames, verbose=False)
        detections = []
        for i, result in enumerate(results):
            for box in result.boxes:
                detections.append({
                    "timestamp": frame_times[i],
                    "object_class": result.names[int(box.cls)],
                    "confidence_score": float(box.conf),
                    "bounding_box": {
                        "x": int(box.xyxy[0][0]),
                        "y": int(box.xyxy[0][1]),
                        "width": int(box.xyxy[0][2] - box.xyxy[0][0]),
                        "height": int(box.xyxy[0][3] - box.xyxy[0][1])
                    }
                })
        return detections
```

### text_overlay - Text Recognition and Subtitles
- **Model**: EasyOCR (optimized for TikTok)
- **File**: `analyzers/text_overlay_tiktok_fixed.py`
- **GPU Memory**: 2-3GB
- **Purpose**: Recognizes subtitles, overlays, and embedded text
- **Processing Time**: 25s (optimized from 37s)
- **Frame Sampling**: Every 30th frame + motion-based
- **Batch Size**: 16

#### Specialized Features
```python
class TikTokTextOverlayAnalyzer:
    def __init__(self):
        self.reader = easyocr.Reader(['en', 'de', 'es', 'fr'], gpu=True)
        self.text_regions = []  # TikTok-specific regions
        
    def detect_tiktok_text_regions(self, frame):
        # Optimized for TikTok subtitle placement
        height, width = frame.shape[:2]
        regions = [
            frame[int(height*0.7):height, :],  # Bottom subtitles
            frame[:int(height*0.2), :],        # Top overlays
            frame[:, int(width*0.8):width]     # Side text
        ]
        return regions
```

### background_segmentation - Semantic Segmentation
- **Model**: SegFormer (NVIDIA)
- **File**: `analyzers/background_segmentation_light.py`
- **GPU Memory**: 2-3GB
- **Purpose**: Separates foreground/background and scene classification
- **Processing Time**: 18s (optimized from 41s)
- **Frame Sampling**: Every 15th frame
- **Batch Size**: 8

### camera_analysis - Camera Movement Analysis
- **Model**: Optical Flow + Custom CV
- **File**: `analyzers/camera_analysis_fixed.py`
- **GPU Memory**: 1-2GB
- **Purpose**: Detects zoom, pan, tilt, and stability
- **Processing Time**: 18s (optimized from 36s)
- **Frame Sampling**: Every 5th frame (dense)
- **Batch Size**: 16

#### Movement Detection
```python
class GPUBatchCameraAnalysisFixed:
    def analyze_movement(self, frame1, frame2):
        flow = cv2.calcOpticalFlowPyrLK(frame1, frame2, ...)
        
        # Movement classification
        movements = {
            "zoom": self.detect_zoom(flow),
            "pan": self.detect_pan(flow),
            "tilt": self.detect_tilt(flow),
            "shake": self.detect_shake(flow)
        }
        return movements
```

## GPU Worker 2 - Detail Analysis (5-7GB VRAM)

### scene_segmentation - Scene Boundary Detection
- **Model**: Custom similarity detection
- **File**: `analyzers/scene_segmentation_fixed.py`
- **Purpose**: Identifies scene cuts and transitions
- **Processing Time**: 10.6s
- **Output**: Scene boundaries with transition types

### color_analysis - Color Palette and Mood
- **Model**: K-Means + Color Space Analysis
- **File**: `analyzers/gpu_batch_color_analysis.py`
- **Purpose**: Extracts color palettes and color temperature
- **Processing Time**: 16.4s
- **Features**: Dominant colors, color harmony, mood analysis

### body_pose - Human Pose Estimation
- **Model**: YOLOv8x-Pose
- **File**: `analyzers/body_pose_yolov8.py`
- **Purpose**: Skeletal keypoint detection and gesture recognition
- **Processing Time**: 20s
- **Output**: 17 keypoints per person, gesture classifications

### age_estimation - Demographics Analysis
- **Model**: InsightFace
- **File**: `analyzers/age_gender_insightface.py`
- **Purpose**: Age and gender estimation for visible persons
- **Processing Time**: 8s
- **Privacy**: Local processing only, no data retention

### content_quality - Technical Quality Assessment
- **Model**: CLIP + Custom Metrics
- **File**: `analyzers/gpu_batch_content_quality_fixed.py`
- **Purpose**: Technical and aesthetic quality scoring
- **Metrics**: Sharpness, exposure, composition, noise levels

### eye_tracking - Gaze Analysis
- **Model**: MediaPipe Iris
- **File**: `analyzers/gpu_batch_eye_tracking.py`
- **Purpose**: Eye gaze direction and attention tracking
- **Output**: Gaze vectors, attention maps, eye contact detection

### cut_analysis - Edit Point Detection
- **Model**: Frame Difference + ML
- **File**: `analyzers/cut_analysis_fixed.py`
- **Purpose**: Identifies cuts, transitions, and edit points
- **Processing Time**: 4.1s
- **Types**: Hard cuts, fades, wipes, dissolves

## CPU Workers - Audio and Metadata (Parallel)

### speech_transcription - Speech-to-Text
- **Model**: Whisper Large V3
- **File**: `analyzers/speech_transcription_ultimate.py`
- **Purpose**: High-accuracy transcription with language detection
- **Processing Time**: 4.5s
- **Features**: Timestamps, speaker diarization, confidence scores

#### Enhanced Features
```python
class UltimateSpeechTranscription:
    def __init__(self):
        self.model = whisper.load_model("large-v3")
        self.vad_model = webrtcvad.Vad(3)  # Voice activity detection
        
    def transcribe_with_metadata(self, audio):
        # Enhanced transcription with metadata
        result = self.model.transcribe(
            audio,
            language='auto',
            task='transcribe',
            word_timestamps=True
        )
        return self.enhance_with_prosody(result)
```

### audio_analysis - Audio Feature Extraction
- **Model**: Librosa + Custom Features
- **File**: `analyzers/audio_analysis_ultimate.py`
- **Purpose**: Tempo, pitch, spectral features, music classification
- **Processing Time**: 0.2s
- **Features**: MFCC, spectral centroid, rhythm analysis

### audio_environment - Environmental Sound Classification
- **Model**: YAMNet + Custom
- **File**: `analyzers/audio_environment_enhanced.py`
- **Purpose**: Classifies background sounds and environment type
- **Processing Time**: 0.5s
- **Categories**: Indoor/outdoor, urban/nature, music/speech

### speech_emotion - Vocal Emotion Recognition
- **Model**: Wav2Vec2 + Emotion Classifier
- **File**: `analyzers/gpu_batch_speech_emotion.py`
- **Purpose**: Emotion detection from speech prosody
- **Processing Time**: 1.6s
- **Emotions**: Happy, sad, excited, angry, neutral, surprised

### temporal_flow - Narrative Structure Analysis
- **Model**: Custom NLP + Sequence Analysis
- **File**: `analyzers/narrative_analysis_wrapper.py`
- **Purpose**: Story structure and narrative flow analysis
- **Processing Time**: 2.1s
- **Output**: Narrative segments, story arc, pacing analysis

### speech_flow - Prosody and Rhythm Analysis
- **Model**: VAD + Prosody Analysis
- **File**: `analyzers/gpu_batch_speech_flow.py`
- **Purpose**: Speech patterns, emphasis, pauses, rhythm
- **Processing Time**: 1.6s
- **Features**: Pause detection, stress patterns, speaking rate variations

## Analyzer Performance Optimization

### Model Caching Strategy
```python
# Persistent model management across analyses
class PersistentModelManager:
    def __init__(self):
        self.cached_models = {}
        self.memory_monitor = GPUMemoryMonitor()
        
    def get_analyzer(self, name, analyzer_class):
        if name not in self.cached_models:
            analyzer = analyzer_class()
            analyzer._load_model_impl()
            self.cached_models[name] = analyzer
        return self.cached_models[name]
```

### Frame Sampling Optimization
```python
# Intelligent frame sampling per analyzer
FRAME_INTERVALS = {
    'qwen2_vl_temporal': 30,    # 1 FPS for detailed analysis
    'object_detection': 10,     # 3 FPS for object tracking
    'text_overlay': 30,         # 1 FPS for text detection
    'camera_analysis': 5,       # 6 FPS for movement detection
    'eye_tracking': 15,         # 2 FPS for gaze tracking
}
```

### Batch Processing Configuration
```python
# Optimized batch sizes based on model complexity
BATCH_SIZES = {
    'object_detection': 64,      # YOLOv8 handles large batches well
    'text_overlay': 16,          # EasyOCR moderate batches
    'background_segmentation': 8, # SegFormer memory-intensive
    'body_pose': 16,             # YOLOv8-pose moderate batches
    'eye_tracking': 16,          # MediaPipe efficient
    'color_analysis': 32,        # Lightweight processing
}
```

## Quality Assurance

### Output Validation
Each analyzer implements standardized output validation:

```python
def validate_output(self, result):
    required_fields = ['timestamp', 'confidence', 'data']
    for field in required_fields:
        if field not in result:
            raise ValidationError(f"Missing required field: {field}")
    
    if result['confidence'] < 0.0 or result['confidence'] > 1.0:
        raise ValidationError("Invalid confidence score")
        
    return True
```

### Error Recovery
```python
class AnalyzerErrorHandler:
    def handle_gpu_oom(self, analyzer_name):
        # Reduce batch size and retry
        new_batch_size = self.batch_sizes[analyzer_name] // 2
        self.batch_sizes[analyzer_name] = max(1, new_batch_size)
        torch.cuda.empty_cache()
        
    def handle_model_error(self, analyzer_name):
        # Clear cached model and reload
        if analyzer_name in self.cached_models:
            del self.cached_models[analyzer_name]
        torch.cuda.empty_cache()
```

## Performance Metrics

### Processing Time Optimizations
| Analyzer | Baseline | Optimized | Improvement |
|----------|----------|-----------|-------------|
| qwen2_vl_temporal | 110s | 60s | 45% |
| object_detection | 25s | 15s | 40% |
| text_overlay | 37s | 25s | 32% |
| background_segmentation | 41s | 18s | 56% |
| camera_analysis | 36s | 18s | 50% |

### GPU Memory Utilization
- **Worker 0**: 16GB (Qwen2-VL exclusive)
- **Worker 1**: 8-10GB (4 visual analyzers)
- **Worker 2**: 5-7GB (7 detail analyzers)
- **Total**: 29-33GB of 45GB available (65-73% utilization)

### Cache Hit Rates
- **First Analysis**: 0% (models load from disk)
- **Second Analysis**: 85%+ (models reused from cache)
- **Memory Overhead**: ~2GB for cached models
- **Performance Gain**: 80-90% faster processing

This analyzer architecture enables comprehensive video analysis while maintaining high performance through intelligent resource management and optimization strategies.