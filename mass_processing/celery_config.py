#!/usr/bin/env python3
"""
Celery configuration for distributed TikTok video processing
"""

from celery import Celery
from kombu import Queue, Exchange
import os
from datetime import timedelta

# Redis configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Create Celery app
app = Celery('tiktok_processor')

# Celery configuration
app.conf.update(
    # Broker settings
    broker_url=REDIS_URL,
    result_backend=REDIS_URL,
    broker_connection_retry_on_startup=True,
    
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task execution settings
    task_acks_late=True,  # Tasks acknowledged after completion
    task_reject_on_worker_lost=True,
    task_track_started=True,
    task_time_limit=1800,  # 30 minutes hard limit
    task_soft_time_limit=1500,  # 25 minutes soft limit
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    
    # Queue configuration
    task_default_queue='default',
    task_default_exchange='default',
    task_default_exchange_type='direct',
    task_default_routing_key='default',
    
    task_queues=(
        # Download queue - for downloading videos
        Queue('download', Exchange('download'), routing_key='download',
              queue_arguments={'x-max-priority': 10}),
              
        # GPU processing queue - for ML analysis
        Queue('gpu_processing', Exchange('gpu'), routing_key='gpu',
              queue_arguments={'x-max-priority': 10}),
              
        # CPU processing queue - for lightweight tasks
        Queue('cpu_processing', Exchange('cpu'), routing_key='cpu'),
        
        # Priority queue - for urgent tasks
        Queue('priority', Exchange('priority'), routing_key='priority',
              queue_arguments={'x-max-priority': 10}),
              
        # Monitoring queue - for system tasks
        Queue('monitoring', Exchange('monitoring'), routing_key='monitoring'),
    ),
    
    # Task routing
    task_routes={
        'tasks.download_tiktok_task': {
            'queue': 'download',
            'routing_key': 'download',
        },
        'tasks.process_video_gpu_task': {
            'queue': 'gpu_processing',
            'routing_key': 'gpu',
        },
        'tasks.process_video_cpu_task': {
            'queue': 'cpu_processing',
            'routing_key': 'cpu',
        },
        'tasks.priority_process_task': {
            'queue': 'priority',
            'routing_key': 'priority',
        },
        'tasks.cleanup_old_files': {
            'queue': 'monitoring',
            'routing_key': 'monitoring',
        },
        'tasks.update_metrics': {
            'queue': 'monitoring',
            'routing_key': 'monitoring',
        },
        'tasks.worker_health_check': {
            'queue': 'monitoring',
            'routing_key': 'monitoring',
        },
    },
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Only fetch one task at a time for GPU tasks
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks
    worker_disable_rate_limits=False,
    worker_send_task_events=True,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'cleanup-old-files': {
            'task': 'tasks.cleanup_old_files',
            'schedule': timedelta(hours=1),
            'options': {'queue': 'monitoring'}
        },
        'update-metrics': {
            'task': 'tasks.update_metrics',
            'schedule': timedelta(minutes=1),
            'options': {'queue': 'monitoring'}
        },
        'worker-health-check': {
            'task': 'tasks.worker_health_check',
            'schedule': timedelta(seconds=30),
            'options': {'queue': 'monitoring'}
        },
        'requeue-failed': {
            'task': 'tasks.requeue_failed_tasks',
            'schedule': timedelta(minutes=15),
            'options': {'queue': 'monitoring'}
        },
        'optimize-gpu-scheduling': {
            'task': 'tasks.optimize_gpu_scheduling',
            'schedule': timedelta(minutes=5),
            'options': {'queue': 'monitoring'}
        },
    },
    
    # Task annotations for specific settings
    task_annotations={
        'tasks.process_video_gpu_task': {
            'rate_limit': '10/m',  # Max 10 GPU tasks per minute
            'time_limit': 1800,    # 30 minutes
        },
        'tasks.download_tiktok_task': {
            'rate_limit': '30/m',  # Max 30 downloads per minute
            'time_limit': 300,     # 5 minutes
        },
    },
    
    # Control settings
    worker_pool='threads',  # Use threads for I/O bound tasks
    worker_concurrency=4,   # Number of concurrent workers
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Configure Celery to use JSON for serialization
app.conf.task_serializer = 'json'
app.conf.result_serializer = 'json'
app.conf.accept_content = ['json']

# Import tasks
app.autodiscover_tasks(['mass_processing.tasks'])