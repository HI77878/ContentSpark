# TikTok Mass Processing System

A distributed system for downloading and analyzing thousands of TikTok videos with 90% GPU utilization.

## Features

- **Automatic Download Pipeline**: yt-dlp integration with rate limiting
- **Distributed Queue System**: Redis + Celery for scalable processing  
- **GPU Optimization**: Maintains 90% GPU utilization with Pipeline V2
- **Real-time Monitoring**: Web dashboard for queue and worker status
- **Fault Tolerance**: Automatic retries and worker recovery
- **Priority Support**: Process urgent videos first

## Quick Start

### 1. Install Requirements
```bash
cd /home/user/tiktok_production/mass_processing
pip install -r requirements.txt
```

### 2. Start Workers
```bash
chmod +x start_workers.sh
./start_workers.sh
```

### 3. Add Videos
```bash
# Single URL
python bulk_processor.py add -u "https://www.tiktok.com/@user/video/123"

# From file
python bulk_processor.py add -f urls.txt -p 5

# High priority
python bulk_processor.py process-now "https://www.tiktok.com/@user/video/123"
```

### 4. Monitor Progress
- Dashboard: http://localhost:5000
- Flower: http://localhost:5555

## Architecture

### Components

1. **Download Workers**: Fetch videos from TikTok
2. **GPU Workers**: Run 30 ML analyzers per video
3. **CPU Workers**: Fallback for lightweight analysis
4. **Queue Manager**: Redis-based priority queues
5. **Monitor**: Real-time web dashboard

### Queue Flow

```
TikTok URL → Download Queue → Download Worker → Video File
                                                     ↓
Dashboard ← Supabase ← GPU Worker ← Processing Queue
```

### Worker Distribution

- **Download Workers**: 2 concurrent (rate limited)
- **GPU Workers**: 1 per GPU (90% utilization)
- **CPU Workers**: 4 workers (fallback)
- **Monitor Worker**: 1 (system tasks)

## Commands

### Queue Management
```bash
# Check status
python bulk_processor.py status

# Requeue failed
python bulk_processor.py requeue

# Clean old tasks
python bulk_processor.py cleanup --days 7

# Export stats
python bulk_processor.py export -o stats.json
```

### Monitoring
```bash
# Open dashboard
python bulk_processor.py monitor

# Check workers
celery -A mass_processing.celery_config inspect active

# View logs
tail -f logs/mass_processing/*.log
```

## Configuration

### Priority Levels
- 1-3: Low priority (background processing)
- 4-6: Normal priority (default)
- 7-9: High priority (faster processing)
- 10: Urgent (immediate processing)

### Rate Limiting
- Downloads: 30/minute (configurable)
- GPU tasks: 10/minute per worker
- Minimum 2s delay between downloads

### Retry Policy
- Download failures: 3 retries with exponential backoff
- Processing failures: 3 retries, then CPU fallback
- Permanent failures: No retry (private/deleted videos)

## Database Schema

### Main Tables
- `video_analysis`: Complete analysis results
- `processing_queue`: Queue status tracking
- `worker_status`: Worker health monitoring
- `performance_metrics`: System metrics

### Queue States
- `pending`: Waiting to be processed
- `downloading`: Currently downloading
- `processing`: Being analyzed
- `completed`: Successfully processed
- `failed`: Processing failed

## Performance Optimization

### GPU Utilization
- Multi-stream CUDA processing
- Frame prefetching
- Batch optimization per analyzer
- Memory-aware scheduling

### Throughput
- Target: 10-15 videos/hour per GPU
- Actual: Depends on video length
- Monitoring: Real-time dashboard

## Troubleshooting

### Common Issues

1. **Low GPU Utilization**
   - Check worker count
   - Verify queue has videos
   - Review batch sizes

2. **High Failure Rate**
   - Check TikTok rate limits
   - Verify GPU memory
   - Review error logs

3. **Workers Offline**
   - Check Redis connection
   - Review worker logs
   - Restart with script

### Logs Location
```
/home/user/tiktok_production/logs/mass_processing/
├── download_worker_*.log
├── gpu_worker_*.log
├── cpu_worker_*.log
├── celery_beat.log
├── flower.log
└── monitoring_dashboard.log
```

## Scaling

### Vertical (Same Machine)
- Add more GPUs
- Increase worker count
- Optimize batch sizes

### Horizontal (Multiple Machines)
- Use Redis cluster
- Distributed Celery
- Shared storage (NFS/S3)

## API Endpoints

### Monitoring API
- `GET /api/metrics` - System metrics
- `POST /api/queue/add` - Add URLs
- `POST /api/queue/requeue` - Requeue failed
- `GET /api/export/json` - Export stats

## Best Practices

1. **Batch Processing**: Add URLs in batches for efficiency
2. **Priority Usage**: Reserve high priority for urgent videos
3. **Regular Cleanup**: Run cleanup weekly to free space
4. **Monitor Actively**: Check dashboard for issues
5. **Log Rotation**: Configure log rotation to prevent disk fill

## Example Workflow

```bash
# 1. Prepare URL list
echo "https://www.tiktok.com/@user1/video/123" > urls.txt
echo "https://www.tiktok.com/@user2/video/456" >> urls.txt

# 2. Start system
./start_workers.sh

# 3. Add videos
python bulk_processor.py add -f urls.txt -p 5 -m

# 4. Monitor progress
# Check http://localhost:5000

# 5. Export results when done
python bulk_processor.py export -o results.json
```

## Support

For issues or questions:
1. Check worker logs
2. Review queue status
3. Verify GPU availability
4. Check Supabase connection