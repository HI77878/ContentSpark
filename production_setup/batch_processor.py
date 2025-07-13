#!/usr/bin/env python3
"""
Batch Processing System for TikTok Video Analyzer
Handles queue management, priority processing, and error recovery
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import httpx
from queue import PriorityQueue

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/user/tiktok_production/logs/batch_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class JobStatus(Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class JobPriority(Enum):
    URGENT = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

@dataclass
class VideoJob:
    job_id: str
    tiktok_url: str
    creator_username: Optional[str]
    priority: JobPriority
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    video_duration: Optional[float] = None
    processing_time: Optional[float] = None
    result_file: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """For priority queue comparison"""
        return self.priority.value < other.priority.value

class BatchProcessor:
    def __init__(self, max_concurrent_jobs: int = 3, api_url: str = "http://localhost:8003"):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.api_url = api_url
        self.db_path = "/home/user/tiktok_production/batch_jobs.db"
        self.queue = PriorityQueue()
        self.active_jobs: Dict[str, VideoJob] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self._init_db()
        self._load_pending_jobs()
        
    def _init_db(self):
        """Initialize SQLite database for job persistence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_jobs (
                job_id TEXT PRIMARY KEY,
                tiktok_url TEXT NOT NULL,
                creator_username TEXT,
                priority INTEGER,
                status TEXT,
                created_at TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                video_duration REAL,
                processing_time REAL,
                result_file TEXT,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_status ON video_jobs(status)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_priority ON video_jobs(priority)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_created ON video_jobs(created_at)
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_pending_jobs(self):
        """Load incomplete jobs from database on startup"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM video_jobs 
            WHERE status IN ('pending', 'downloading', 'processing', 'retrying')
            ORDER BY priority, created_at
        ''')
        
        for row in cursor.fetchall():
            job = self._row_to_job(row)
            if job.status in [JobStatus.DOWNLOADING, JobStatus.PROCESSING]:
                # Reset interrupted jobs to pending
                job.status = JobStatus.PENDING
                self._update_job_in_db(job)
            
            self.queue.put(job)
            logger.info(f"Loaded pending job: {job.job_id}")
        
        conn.close()
    
    def _row_to_job(self, row) -> VideoJob:
        """Convert database row to VideoJob object"""
        return VideoJob(
            job_id=row[0],
            tiktok_url=row[1],
            creator_username=row[2],
            priority=JobPriority(row[3]),
            status=JobStatus(row[4]),
            created_at=datetime.fromisoformat(row[5]),
            started_at=datetime.fromisoformat(row[6]) if row[6] else None,
            completed_at=datetime.fromisoformat(row[7]) if row[7] else None,
            video_duration=row[8],
            processing_time=row[9],
            result_file=row[10],
            error_message=row[11],
            retry_count=row[12]
        )
    
    def _save_job_to_db(self, job: VideoJob):
        """Save job to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO video_jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job.job_id,
            job.tiktok_url,
            job.creator_username,
            job.priority.value,
            job.status.value,
            job.created_at.isoformat(),
            job.started_at.isoformat() if job.started_at else None,
            job.completed_at.isoformat() if job.completed_at else None,
            job.video_duration,
            job.processing_time,
            job.result_file,
            job.error_message,
            job.retry_count
        ))
        
        conn.commit()
        conn.close()
    
    def _update_job_in_db(self, job: VideoJob):
        """Update existing job in database"""
        self._save_job_to_db(job)
    
    def add_job(self, tiktok_url: str, priority: JobPriority = JobPriority.NORMAL,
                creator_username: Optional[str] = None) -> str:
        """Add a new job to the queue"""
        job_id = f"job_{int(time.time())}_{hash(tiktok_url) % 10000}"
        
        # Check if URL already in queue or processing
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT job_id FROM video_jobs 
            WHERE tiktok_url = ? AND status NOT IN ('completed', 'failed')
        ''', (tiktok_url,))
        
        existing = cursor.fetchone()
        conn.close()
        
        if existing:
            logger.warning(f"Video already in queue: {tiktok_url}")
            return existing[0]
        
        job = VideoJob(
            job_id=job_id,
            tiktok_url=tiktok_url,
            creator_username=creator_username,
            priority=priority,
            status=JobStatus.PENDING,
            created_at=datetime.now()
        )
        
        self._save_job_to_db(job)
        self.queue.put(job)
        
        logger.info(f"Added job {job_id} for {tiktok_url}")
        return job_id
    
    def add_batch(self, urls: List[str], priority: JobPriority = JobPriority.NORMAL) -> List[str]:
        """Add multiple jobs at once"""
        job_ids = []
        
        # Sort by video length estimation (shorter videos first)
        for url in urls:
            job_id = self.add_job(url, priority)
            job_ids.append(job_id)
        
        return job_ids
    
    async def process_job(self, job: VideoJob):
        """Process a single video job"""
        logger.info(f"Starting job {job.job_id}: {job.tiktok_url}")
        
        try:
            # Update status
            job.status = JobStatus.DOWNLOADING
            job.started_at = datetime.now()
            self._update_job_in_db(job)
            
            # Download video
            from mass_processing.tiktok_downloader import TikTokDownloader
            downloader = TikTokDownloader()
            download_result = downloader.download_video(job.tiktok_url)
            
            if not download_result['success']:
                raise Exception(f"Download failed: {download_result.get('error', 'Unknown error')}")
            
            video_path = download_result['video_path']
            metadata = download_result.get('metadata', {})
            job.video_duration = metadata.get('duration', 0)
            job.creator_username = metadata.get('username', job.creator_username)
            
            # Update status
            job.status = JobStatus.PROCESSING
            self._update_job_in_db(job)
            
            # Analyze video
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{self.api_url}/analyze",
                    json={
                        "video_path": video_path,
                        "tiktok_url": job.tiktok_url,
                        "creator_username": job.creator_username
                    }
                )
                
                if response.status_code != 200:
                    raise Exception(f"Analysis failed: {response.text}")
                
                result = response.json()
                
                if result['status'] == 'success':
                    job.status = JobStatus.COMPLETED
                    job.result_file = result.get('results_file')
                    job.completed_at = datetime.now()
                    job.processing_time = (job.completed_at - job.started_at).total_seconds()
                    
                    logger.info(f"Job {job.job_id} completed in {job.processing_time:.1f}s")
                else:
                    raise Exception(result.get('error', 'Unknown error'))
            
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {str(e)}")
            job.error_message = str(e)
            job.retry_count += 1
            
            if job.retry_count < job.max_retries:
                job.status = JobStatus.RETRYING
                # Re-add to queue with delay
                await asyncio.sleep(30 * job.retry_count)  # Exponential backoff
                self.queue.put(job)
            else:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
        
        finally:
            self._update_job_in_db(job)
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
    
    async def worker(self):
        """Worker coroutine to process jobs from queue"""
        while True:
            try:
                # Check if we can process more jobs
                if len(self.active_jobs) >= self.max_concurrent_jobs:
                    await asyncio.sleep(5)
                    continue
                
                # Get next job from queue
                if not self.queue.empty():
                    job = self.queue.get()
                    self.active_jobs[job.job_id] = job
                    
                    # Process job in background
                    asyncio.create_task(self.process_job(job))
                else:
                    await asyncio.sleep(10)  # Wait if queue is empty
                    
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
                await asyncio.sleep(5)
    
    async def start(self, num_workers: int = None):
        """Start the batch processor with multiple workers"""
        if num_workers is None:
            num_workers = self.max_concurrent_jobs
        
        logger.info(f"Starting batch processor with {num_workers} workers")
        
        # Start workers
        workers = [asyncio.create_task(self.worker()) for _ in range(num_workers)]
        
        # Start status reporter
        asyncio.create_task(self.status_reporter())
        
        # Wait for workers
        await asyncio.gather(*workers)
    
    async def status_reporter(self):
        """Periodically report processing status"""
        while True:
            await asyncio.sleep(30)  # Report every 30 seconds
            
            # Get statistics
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT status, COUNT(*) FROM video_jobs 
                GROUP BY status
            ''')
            
            status_counts = dict(cursor.fetchall())
            
            cursor.execute('''
                SELECT AVG(processing_time) FROM video_jobs 
                WHERE status = 'completed' AND processing_time IS NOT NULL
            ''')
            
            avg_time = cursor.fetchone()[0] or 0
            
            conn.close()
            
            logger.info(f"Queue Status: {status_counts}")
            logger.info(f"Active Jobs: {len(self.active_jobs)}")
            logger.info(f"Average Processing Time: {avg_time:.1f}s")
            logger.info(f"Queue Size: {self.queue.qsize()}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a specific job"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM video_jobs WHERE job_id = ?', (job_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            job = self._row_to_job(row)
            return asdict(job)
        
        return None
    
    def get_queue_status(self) -> Dict:
        """Get overall queue status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Status counts
        cursor.execute('''
            SELECT status, COUNT(*) FROM video_jobs 
            GROUP BY status
        ''')
        status_counts = dict(cursor.fetchall())
        
        # Recent completions
        cursor.execute('''
            SELECT COUNT(*) FROM video_jobs 
            WHERE status = 'completed' 
            AND completed_at > datetime('now', '-1 hour')
        ''')
        recent_completions = cursor.fetchone()[0]
        
        # Average times
        cursor.execute('''
            SELECT 
                AVG(processing_time) as avg_processing,
                AVG(video_duration) as avg_duration,
                AVG(processing_time / NULLIF(video_duration, 0)) as avg_realtime_factor
            FROM video_jobs 
            WHERE status = 'completed' 
            AND processing_time IS NOT NULL 
            AND video_duration IS NOT NULL
        ''')
        
        timing_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            'status_counts': status_counts,
            'active_jobs': len(self.active_jobs),
            'queue_size': self.queue.qsize(),
            'recent_completions_per_hour': recent_completions,
            'average_processing_time': timing_stats[0] or 0,
            'average_video_duration': timing_stats[1] or 0,
            'average_realtime_factor': timing_stats[2] or 0
        }

# CLI Interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Video Processor")
    parser.add_argument('command', choices=['start', 'add', 'status', 'queue'])
    parser.add_argument('-u', '--url', help='TikTok URL to process')
    parser.add_argument('-f', '--file', help='File with URLs (one per line)')
    parser.add_argument('-p', '--priority', choices=['urgent', 'high', 'normal', 'low'], 
                       default='normal')
    parser.add_argument('-j', '--job-id', help='Job ID for status check')
    parser.add_argument('-w', '--workers', type=int, default=3, 
                       help='Number of concurrent workers')
    
    args = parser.parse_args()
    
    if args.command == 'start':
        # Start the batch processor
        processor = BatchProcessor(max_concurrent_jobs=args.workers)
        asyncio.run(processor.start())
        
    elif args.command == 'add':
        processor = BatchProcessor()
        
        if args.url:
            # Add single URL
            priority = JobPriority[args.priority.upper()]
            job_id = processor.add_job(args.url, priority)
            print(f"Added job: {job_id}")
            
        elif args.file:
            # Add URLs from file
            with open(args.file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            priority = JobPriority[args.priority.upper()]
            job_ids = processor.add_batch(urls, priority)
            print(f"Added {len(job_ids)} jobs")
            
    elif args.command == 'status':
        processor = BatchProcessor()
        
        if args.job_id:
            # Get specific job status
            status = processor.get_job_status(args.job_id)
            if status:
                print(json.dumps(status, indent=2, default=str))
            else:
                print(f"Job not found: {args.job_id}")
        else:
            print("Please provide a job ID with -j")
            
    elif args.command == 'queue':
        # Get queue status
        processor = BatchProcessor()
        status = processor.get_queue_status()
        print(json.dumps(status, indent=2))