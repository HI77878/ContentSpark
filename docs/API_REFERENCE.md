# API Reference - TikTok Video Analysis System

## Overview

The TikTok Video Analysis System provides a RESTful API built with FastAPI that processes videos through 17 ML analyzers. The API is optimized for GPU utilization and supports both TikTok URL downloads and local file analysis.

## Base Information

- **Host**: `localhost`
- **Port**: `8003`
- **Protocol**: HTTP
- **Framework**: FastAPI 0.100.0+
- **Documentation**: `http://localhost:8003/docs` (Swagger UI)
- **OpenAPI**: `http://localhost:8003/openapi.json`

## Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible on localhost.

## Global Response Format

All API responses follow this structure:

```json
{
  "status": "success" | "error",
  "timestamp": "2025-07-13T07:00:00Z",
  "data": {}, // Endpoint-specific data
  "error": null | "Error message",
  "processing_info": {} // Optional processing metadata
}
```

## Endpoints

### Health Check

#### `GET /health`

Returns system health status and GPU information.

**Request:**
```bash
curl -X GET "http://localhost:8003/health"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-13T07:00:00Z",
  "gpu": {
    "gpu_available": true,
    "gpu_name": "Quadro RTX 8000",
    "gpu_memory": {
      "used_mb": 2048,
      "total_mb": 45541,
      "utilization": "15.0%"
    }
  },
  "active_analyzers": 17,
  "parallelization": "multiprocess",
  "model_cache": {
    "cached_models": 5,
    "cache_hit_rate": "85%",
    "memory_usage_mb": 18432
  },
  "system": {
    "cpu_usage": "25%",
    "memory_usage": "65%",
    "disk_space_free": "450GB"
  }
}
```

**Response Fields:**
- `gpu_available`: Boolean indicating CUDA availability
- `gpu_name`: GPU model name
- `gpu_memory`: Current GPU memory usage statistics
- `active_analyzers`: Number of working analyzers
- `parallelization`: Execution mode (multiprocess/sequential)
- `model_cache`: Model caching performance metrics
- `system`: System resource utilization

**Error Responses:**
```json
{
  "status": "error",
  "error": "GPU not available",
  "timestamp": "2025-07-13T07:00:00Z"
}
```

### Video Analysis

#### `POST /analyze`

Analyzes a video using all active ML analyzers.

**Request:**
```bash
# TikTok URL Analysis
curl -X POST "http://localhost:8003/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "tiktok_url": "https://www.tiktok.com/@username/video/123456789",
    "priority": 5,
    "analyzers": ["object_detection", "qwen2_vl_temporal"],
    "options": {
      "save_frames": false,
      "quality": "high"
    }
  }'

# Local File Analysis
curl -X POST "http://localhost:8003/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/local/video.mp4",
    "priority": 8
  }'
```

**Request Body Schema:**
```json
{
  "tiktok_url": "string (optional)",        // TikTok video URL for download
  "video_path": "string (optional)",       // Local video file path
  "analyzers": ["string"] (optional),      // Specific analyzers to run
  "priority": 1-10 (optional, default: 5), // Processing priority
  "options": {
    "save_frames": false,                   // Save extracted frames
    "quality": "high|medium|fast",          // Analysis quality mode
    "max_duration": 300,                    // Max video duration in seconds
    "skip_audio": false                     // Skip audio analysis
  }
}
```

**Required Fields:**
- Either `tiktok_url` OR `video_path` must be provided
- If both are provided, `video_path` takes precedence

**Success Response:**
```json
{
  "status": "success",
  "timestamp": "2025-07-13T07:00:00Z",
  "data": {
    "video_path": "/home/user/tiktok_videos/videos/123456789.mp4",
    "video_info": {
      "duration": 49.2,
      "fps": 30,
      "resolution": "1080x1920",
      "size_mb": 15.3,
      "format": "mp4"
    },
    "processing_time": 78.5,
    "realtime_factor": 1.56,
    "successful_analyzers": 17,
    "failed_analyzers": 0,
    "total_analyzers": 17,
    "results_file": "/home/user/tiktok_production/results/123456789_analysis.json",
    "analysis_id": "analysis_20250713_070000",
    "gpu_usage": {
      "peak_utilization": "42%",
      "average_utilization": "35%",
      "peak_memory_mb": 22016
    },
    "cache_performance": {
      "cache_hits": 12,
      "cache_misses": 5,
      "models_loaded": 5,
      "models_reused": 12
    }
  }
}
```

**Error Responses:**

*Invalid URL:*
```json
{
  "status": "error",
  "error": "Invalid TikTok URL format",
  "timestamp": "2025-07-13T07:00:00Z",
  "details": "URL must match pattern: https://www.tiktok.com/@username/video/id"
}
```

*File Not Found:*
```json
{
  "status": "error",
  "error": "Video file not found",
  "timestamp": "2025-07-13T07:00:00Z",
  "details": "File path does not exist: /path/to/video.mp4"
}
```

*GPU Out of Memory:*
```json
{
  "status": "error",
  "error": "GPU out of memory",
  "timestamp": "2025-07-13T07:00:00Z",
  "details": "CUDA error: out of memory on GPU 0",
  "retry_possible": true,
  "suggested_action": "Wait for current analysis to complete or restart API"
}
```

*Processing Timeout:*
```json
{
  "status": "error",
  "error": "Analysis timeout",
  "timestamp": "2025-07-13T07:00:00Z",
  "details": "Processing exceeded 10 minutes",
  "partial_results": "/home/user/tiktok_production/results/partial_123.json"
}
```

### Analysis Status

#### `GET /status/{analysis_id}`

Check the status of a running analysis.

**Request:**
```bash
curl -X GET "http://localhost:8003/status/analysis_20250713_070000"
```

**Response:**
```json
{
  "status": "success",
  "timestamp": "2025-07-13T07:00:00Z",
  "data": {
    "analysis_id": "analysis_20250713_070000",
    "status": "running", // "queued", "running", "completed", "failed"
    "progress": {
      "completed_analyzers": 12,
      "total_analyzers": 17,
      "percentage": 70.6,
      "current_analyzer": "qwen2_vl_temporal",
      "estimated_remaining": 25.3
    },
    "start_time": "2025-07-13T07:00:00Z",
    "processing_time": 52.7,
    "gpu_usage": {
      "current_utilization": "38%",
      "memory_usage_mb": 18432
    }
  }
}
```

### Results Retrieval

#### `GET /results/{analysis_id}`

Retrieve analysis results by ID.

**Request:**
```bash
curl -X GET "http://localhost:8003/results/analysis_20250713_070000"
```

**Response:**
```json
{
  "status": "success",
  "timestamp": "2025-07-13T07:00:00Z",
  "data": {
    "analysis_id": "analysis_20250713_070000",
    "video_path": "/home/user/tiktok_videos/videos/123.mp4",
    "processing_info": {
      "total_time": 78.5,
      "successful_analyzers": 17,
      "failed_analyzers": 0
    },
    "results": {
      "qwen2_vl_temporal": [...],
      "object_detection": [...],
      "text_overlay": [...],
      // ... all analyzer results
    }
  }
}
```

### Download Results

#### `GET /download/{analysis_id}`

Download analysis results as JSON file.

**Request:**
```bash
curl -X GET "http://localhost:8003/download/analysis_20250713_070000" \
  -H "Accept: application/json" \
  -o analysis_results.json
```

**Response:**
- **Content-Type**: `application/json`
- **Content-Disposition**: `attachment; filename="analysis_20250713_070000.json"`
- **Body**: Complete analysis results JSON (2-3MB)

### Analyzer Information

#### `GET /analyzers`

Get information about all available analyzers.

**Request:**
```bash
curl -X GET "http://localhost:8003/analyzers"
```

**Response:**
```json
{
  "status": "success",
  "timestamp": "2025-07-13T07:00:00Z",
  "data": {
    "total_analyzers": 30,
    "active_analyzers": 17,
    "disabled_analyzers": 13,
    "analyzers": {
      "qwen2_vl_temporal": {
        "status": "active",
        "gpu_worker": "gpu_worker_0",
        "model": "Qwen2-VL-7B-Instruct",
        "vram_required": "16GB",
        "avg_processing_time": 60.0,
        "description": "Temporal video understanding with scene descriptions"
      },
      "object_detection": {
        "status": "active",
        "gpu_worker": "gpu_worker_1",
        "model": "YOLOv8x",
        "vram_required": "3-4GB",
        "avg_processing_time": 15.0,
        "description": "Object detection and tracking"
      }
      // ... all analyzers
    }
  }
}
```

### Performance Metrics

#### `GET /metrics`

Get system performance metrics.

**Request:**
```bash
curl -X GET "http://localhost:8003/metrics"
```

**Response:**
```json
{
  "status": "success",
  "timestamp": "2025-07-13T07:00:00Z",
  "data": {
    "performance": {
      "average_processing_time": 45.2,
      "realtime_factor": 0.9,
      "throughput_videos_per_hour": 72
    },
    "gpu_metrics": {
      "average_utilization": "32%",
      "peak_utilization": "85%",
      "memory_efficiency": "78%"
    },
    "cache_metrics": {
      "hit_rate": "87%",
      "memory_usage": "35GB",
      "models_cached": 15
    },
    "analyzer_performance": {
      "qwen2_vl_temporal": {
        "avg_time": 60.0,
        "success_rate": "98%",
        "cache_hits": 145
      }
      // ... per-analyzer metrics
    }
  }
}
```

## Error Handling

### HTTP Status Codes

- **200**: Success
- **400**: Bad Request (invalid parameters)
- **404**: Not Found (analysis ID not found)
- **422**: Unprocessable Entity (validation error)
- **500**: Internal Server Error
- **503**: Service Unavailable (GPU not available)

### Error Response Format

All errors follow this format:
```json
{
  "status": "error",
  "error": "Error description",
  "details": "Detailed error information",
  "timestamp": "2025-07-13T07:00:00Z",
  "error_code": "GPU_OOM",
  "retry_possible": true,
  "suggested_action": "Specific action to resolve"
}
```

### Common Error Codes

- `GPU_OOM`: GPU out of memory
- `INVALID_URL`: Malformed TikTok URL
- `FILE_NOT_FOUND`: Video file doesn't exist
- `DOWNLOAD_FAILED`: TikTok download failed
- `ANALYZER_FAILED`: Specific analyzer crashed
- `TIMEOUT`: Processing timeout exceeded
- `INVALID_FORMAT`: Unsupported video format

## Rate Limiting

Currently, no rate limiting is implemented. The API processes requests sequentially to optimize GPU utilization.

## WebSocket Support (Future)

Real-time progress updates via WebSocket:

```javascript
// Future WebSocket implementation
const ws = new WebSocket('ws://localhost:8003/ws/analysis/analysis_id');
ws.onmessage = (event) => {
  const progress = JSON.parse(event.data);
  console.log(`Progress: ${progress.percentage}%`);
};
```

## SDKs and Client Libraries

### Python Client Example
```python
import requests
import json

class TikTokAnalysisClient:
    def __init__(self, base_url="http://localhost:8003"):
        self.base_url = base_url
        
    def analyze_video(self, tiktok_url=None, video_path=None, analyzers=None):
        payload = {}
        if tiktok_url:
            payload['tiktok_url'] = tiktok_url
        if video_path:
            payload['video_path'] = video_path
        if analyzers:
            payload['analyzers'] = analyzers
            
        response = requests.post(f"{self.base_url}/analyze", json=payload)
        return response.json()
        
    def get_health(self):
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Usage
client = TikTokAnalysisClient()
result = client.analyze_video(tiktok_url="https://www.tiktok.com/@user/video/123")
print(f"Analysis completed in {result['data']['processing_time']}s")
```

### JavaScript/Node.js Client Example
```javascript
class TikTokAnalysisClient {
  constructor(baseUrl = 'http://localhost:8003') {
    this.baseUrl = baseUrl;
  }
  
  async analyzeVideo(options) {
    const response = await fetch(`${this.baseUrl}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(options)
    });
    return await response.json();
  }
  
  async getHealth() {
    const response = await fetch(`${this.baseUrl}/health`);
    return await response.json();
  }
}

// Usage
const client = new TikTokAnalysisClient();
const result = await client.analyzeVideo({
  tiktok_url: 'https://www.tiktok.com/@user/video/123'
});
console.log(`Processing time: ${result.data.processing_time}s`);
```

## Performance Considerations

### Request Optimization
- **Batch Processing**: For multiple videos, send requests sequentially
- **Analyzer Selection**: Specify only needed analyzers to reduce processing time
- **Quality Mode**: Use "fast" mode for quick previews, "high" for final analysis

### Response Caching
- Analysis results are cached automatically
- Subsequent requests for the same video return cached results
- Cache invalidation happens after 24 hours

### Resource Management
- Only one analysis runs at a time to optimize GPU usage
- Queue system manages multiple concurrent requests
- Automatic cleanup of temporary files after processing

This API provides comprehensive access to the TikTok Video Analysis System with optimized performance and detailed error handling.