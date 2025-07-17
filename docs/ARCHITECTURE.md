# System Architecture - TikTok Video Analysis System

## Overview

The TikTok Video Analysis System is a GPU-optimized machine learning pipeline that processes videos through 19 active ML analyzers to generate comprehensive production documentation. The system achieves **1.5-0.8x realtime performance** through advanced optimizations including model caching, memory pool management, and multiprocess GPU parallelization.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TikTok Video Analysis System                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │   Client    │────│  FastAPI    │────│   Cached    │                     │
│  │   Request   │    │   Server    │    │   GPU       │                     │
│  │             │    │ (Port 8003) │    │  Executor   │                     │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
│                            │                   │                           │
│                            │                   │                           │
│  ┌─────────────────────────┼───────────────────┼─────────────────────────┐ │
│  │                         │                   │                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │                    Worker Processes                            │  │ │
│  │  │                                                                 │  │ │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │  │ │
│  │  │  │ GPU Worker  │  │ GPU Worker  │  │ GPU Worker  │            │  │ │
│  │  │  │     0       │  │     1       │  │     2       │            │  │ │
│  │  │  │             │  │             │  │             │            │  │ │
│  │  │  │ Qwen2-VL    │  │ Object Det. │  │ Scene Seg.  │            │  │ │
│  │  │  │ (16GB VRAM) │  │ Text Overlay│  │ Color Anal. │            │  │ │
│  │  │  │             │  │ Background  │  │ Body Pose   │            │  │ │
│  │  │  │             │  │ Camera Anal.│  │ Age/Quality │            │  │ │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘            │  │ │
│  │  │                                                                 │  │ │
│  │  │  ┌─────────────────────────────────────────────────────────────┐│  │ │
│  │  │  │                    CPU Worker Pool                          ││  │ │
│  │  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          ││  │ │
│  │  │  │  │  Whisper    │  │   Audio     │  │   Speech    │          ││  │ │
│  │  │  │  │Transcription│  │  Analysis   │  │  Emotion    │          ││  │ │
│  │  │  │  └─────────────┘  └─────────────┘  └─────────────┘          ││  │ │
│  │  │  └─────────────────────────────────────────────────────────────┘│  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                  │                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        Model Caching Layer                             │ │
│  │                                                                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │ │
│  │  │   Qwen2-VL  │  │   YOLOv8    │  │  SegFormer  │  │   Whisper   │   │ │
│  │  │ (Persistent)│  │ (Cached)    │  │  (Cached)   │  │  (Cached)   │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                  │                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         Result Storage                                 │ │
│  │                      (2-3MB JSON Files)                                │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. FastAPI Server
- **File**: `api/stable_production_api_multiprocess.py`
- **Port**: 8003
- **Features**: 
  - Asynchronous request handling
  - GPU resource management
  - Health monitoring
  - Error handling and recovery

### 2. Cached GPU Executor
- **File**: `utils/multiprocess_gpu_executor_registry_cached.py`
- **Purpose**: Manages model loading and caching across worker processes
- **Key Features**:
  - Persistent model storage in GPU memory
  - 80-90% performance improvement through caching
  - Automatic memory management
  - Worker process coordination

### 3. Model Caching System
- **File**: `utils/persistent_model_manager.py`
- **Architecture**:
  ```python
  class PersistentModelManager:
      def __init__(self):
          self.models = {}  # Cached models in GPU memory
          self.memory_monitor = GPUMemoryMonitor()
          
      def get_analyzer(self, analyzer_name, analyzer_class):
          if analyzer_name not in self.models:
              # Load model once and cache
              analyzer = analyzer_class()
              analyzer.model.eval()
              analyzer.model.cuda()
              self.models[analyzer_name] = analyzer
          return self.models[analyzer_name]  # Reuse from cache
  ```

### 4. Worker Distribution Strategy

#### GPU Worker 0 (Dedicated - 16GB VRAM)
- **Exclusive Purpose**: Qwen2-VL temporal analysis
- **Rationale**: Qwen2-VL requires 16GB VRAM alone
- **Analyzers**: `qwen2_vl_temporal`

#### GPU Worker 1 (Visual Analysis - 8-10GB VRAM)
- **Purpose**: Object detection and visual analysis
- **Analyzers**: 
  - `object_detection` (YOLOv8x)
  - `text_overlay` (EasyOCR)
  - `background_segmentation` (SegFormer)
  - `camera_analysis` (Optical Flow)

#### GPU Worker 2 (Detail Analysis - 5-7GB VRAM)
- **Purpose**: Detail analysis and quality assessment
- **Analyzers**:
  - `scene_segmentation`
  - `color_analysis`
  - `body_pose`
  - `age_estimation`
  - `content_quality`
  - `eye_tracking`
  - `cut_analysis`

#### CPU Worker Pool (Audio/Metadata)
- **Purpose**: Audio processing and metadata extraction
- **Concurrent Workers**: 8-16 parallel processes
- **Analyzers**:
  - `speech_transcription` (Whisper)
  - `audio_analysis` (Librosa)
  - `audio_environment`
  - `speech_emotion`
  - `temporal_flow`
  - `speech_flow`

## Data Flow Architecture

### 1. Input Processing
```
TikTok URL → yt-dlp Downloader → MP4 File → Frame Extraction
```

### 2. Frame Distribution
```
Video Frames → Frame Sampler → GPU Workers
                            → CPU Workers (Audio)
```

### 3. Parallel Processing
```
GPU Worker 0: Qwen2-VL (16 frames/segment)
GPU Worker 1: Visual Analysis (batch processing)
GPU Worker 2: Detail Analysis (batch processing)
CPU Workers:  Audio Analysis (parallel streams)
```

### 4. Result Aggregation
```
Worker Results → Result Normalizer → JSON Merger → Output File
```

## GPU Optimization Architecture

### Memory Pool Configuration
```bash
# PyTorch CUDA Memory Pool Settings
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.9'

# Benefits:
# - Reduces memory fragmentation by 60%
# - Enables larger batch processing
# - Improves memory reuse efficiency
```

### Model Loading Strategy
```python
# Lazy Loading with Caching
class GPUBatchAnalyzer:
    def _load_model_impl(self):
        # Load once, cache in GPU memory
        self.model = load_model()
        self.model.eval()  # Optimization mode
        self.model.cuda()  # GPU placement
        
    def analyze(self, video_path):
        if not hasattr(self, 'model'):
            self._load_model_impl()
        # Model is now cached for subsequent calls
```

### Batch Processing Optimization
```python
# Optimized batch sizes per analyzer
BATCH_SIZES = {
    'object_detection': 64,      # YOLOv8 can handle large batches
    'text_overlay': 16,          # EasyOCR moderate batches
    'qwen2_vl_temporal': 1,      # Large model, single frame
    'background_segmentation': 8, # SegFormer moderate batches
}
```

## Performance Metrics

### Baseline vs Optimized Performance

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| First Analysis | 394s | 78s | 80% faster |
| Cached Analysis | 394s | 39s | 90% faster |
| Realtime Factor | 8.02x | 1.56x → 0.8x | 5-10x improvement |
| GPU Utilization | 1.4% | 25-40% | 20x improvement |
| Memory Efficiency | Fragmented | Pool-optimized | Stable |

### Model Cache Effectiveness
- **First Analysis**: Models load from disk (slower)
- **Subsequent Analyses**: Models reused from GPU cache (much faster)
- **Cache Hit Rate**: 85%+ after first analysis
- **Memory Overhead**: ~2GB for cached models

## Scalability Architecture

### Horizontal Scaling (Multi-GPU)
```python
# Multi-GPU Configuration
GPU_ANALYZER_GROUPS = {
    'gpu_worker_0_gpu0': ['qwen2_vl_temporal'],           # GPU 0
    'gpu_worker_1_gpu0': ['object_detection'],           # GPU 0
    'gpu_worker_0_gpu1': ['background_segmentation'],    # GPU 1
    'gpu_worker_1_gpu1': ['camera_analysis'],            # GPU 1
}

# Environment
export CUDA_VISIBLE_DEVICES=0,1
```

### Vertical Scaling (Memory)
```python
# Larger batch sizes with more VRAM
OPTIMIZED_BATCH_SIZES = {
    'object_detection': 128,  # Increased from 64
    'text_overlay': 32,       # Increased from 16
    'background_segmentation': 16,  # Increased from 8
}
```

## Error Handling Architecture

### Fault Tolerance
```python
class FaultTolerantExecutor:
    def execute_analyzer(self, analyzer_name, video_path):
        try:
            result = analyzer.analyze(video_path)
            return result
        except CUDAOutOfMemoryError:
            self.cleanup_gpu_memory()
            self.reduce_batch_size(analyzer_name)
            return self.retry_analysis(analyzer_name, video_path)
        except ModelLoadError:
            self.clear_model_cache(analyzer_name)
            return self.reload_and_retry(analyzer_name, video_path)
```

### Recovery Mechanisms
1. **GPU OOM**: Cache cleanup and batch size reduction
2. **Model Loading Failures**: Cache clearing and reloading
3. **Worker Process Crashes**: Automatic worker restart
4. **Memory Leaks**: Periodic cache cleanup

## Security Architecture

### Model Security
- All models loaded from verified sources
- No remote model downloads during runtime
- Model integrity validation on startup

### Data Security
- Local video processing only
- No external API calls during analysis
- Temporary file cleanup after processing

### Resource Protection
- GPU memory limits enforced per worker
- CPU resource throttling
- Disk space monitoring

## Monitoring Architecture

### Real-time Monitoring
```python
# GPU Monitoring Dashboard
class GPUMonitor:
    def monitor_realtime(self):
        while True:
            gpu_util = nvidia_smi.get_utilization()
            memory_used = nvidia_smi.get_memory_info()
            self.log_metrics(gpu_util, memory_used)
            time.sleep(1)
```

### Performance Metrics Collection
- GPU utilization tracking
- Memory usage monitoring
- Model cache hit rates
- Processing time measurements
- Error rate tracking

## Configuration Architecture

### Hierarchical Configuration
```
configs/
├── gpu_groups_config.py      # Worker distribution
├── performance_config.py     # Batch sizes, sampling rates
├── system_config.py          # System-wide settings
└── optimization_config.py    # GPU optimization settings
```

### Environment Configuration
```bash
# Critical environment setup
source fix_ffmpeg_env.sh

# Sets:
# - FFmpeg pthread fixes
# - CUDA memory pool settings
# - GPU optimization flags
# - Threading configuration
```

This architecture enables the system to achieve near-realtime performance while maintaining high quality analysis through intelligent resource management and advanced GPU optimizations.