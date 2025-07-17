#!/usr/bin/env python3
"""
Batch Processing System for TikTok Videos
Optimized for processing 1000+ videos daily
"""
import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import threading

sys.path.append('/home/user/tiktok_production')
from production_pipeline import ProductionPipeline
TikTokAnalysisPipeline = ProductionPipeline
OptimizedPipeline = ProductionPipeline  # Alias

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """High-throughput batch processing system"""
    
    def __init__(self, max_workers: int = 3, use_optimized: bool = True):
        self.max_workers = max_workers
        self.pipeline_class = OptimizedPipeline if use_optimized else TikTokAnalysisPipeline
        self.results_queue = queue.Queue()
        self.stats = {
            "total": 0,
            "completed": 0,
            "failed": 0,
            "in_progress": 0,
            "start_time": None,
            "end_time": None
        }
        
    def process_urls_file(self, file_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Process URLs from a file with progress tracking
        
        Args:
            file_path: Path to file with URLs (one per line)
            output_dir: Directory for results (default: results/batch_TIMESTAMP)
            
        Returns:
            Batch processing summary
        """
        # Setup output directory
        if not output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"/home/user/tiktok_production/results/batch_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Read URLs
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and line.startswith('http')]
        
        logger.info(f"📊 Starting batch processing of {len(urls)} videos")
        logger.info(f"💾 Results will be saved to: {output_dir}")
        
        self.stats["total"] = len(urls)
        self.stats["start_time"] = datetime.now()
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i, url in enumerate(urls):
                future = executor.submit(self._process_single_url, url, i, output_dir)
                futures.append((i, url, future))
            
            # Monitor progress
            for i, url, future in futures:
                try:
                    result = future.result(timeout=900)  # 15 min timeout
                    self.stats["completed"] += 1
                    logger.info(f"✅ Completed {self.stats['completed']}/{len(urls)}: {url}")
                except Exception as e:
                    self.stats["failed"] += 1
                    logger.error(f"❌ Failed {i+1}/{len(urls)}: {url} - {str(e)}")
                
                # Update progress
                self._print_progress()
        
        self.stats["end_time"] = datetime.now()
        
        # Generate final report
        report = self._generate_batch_report(output_dir)
        report_path = f"{output_dir}/batch_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self._print_final_summary()
        
        return report
    
    def _process_single_url(self, url: str, index: int, output_dir: str) -> Dict[str, Any]:
        """Process a single URL"""
        self.stats["in_progress"] += 1
        
        pipeline = self.pipeline_class()
        result = pipeline.process_tiktok_url(url)
        
        # Save individual result
        result_path = f"{output_dir}/video_{index:04d}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        self.stats["in_progress"] -= 1
        return result
    
    def _print_progress(self):
        """Print progress bar"""
        completed = self.stats["completed"]
        failed = self.stats["failed"]
        total = self.stats["total"]
        progress = (completed + failed) / total * 100
        
        bar_length = 50
        filled = int(bar_length * progress / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        elapsed = (datetime.now() - self.stats["start_time"]).total_seconds()
        rate = (completed + failed) / elapsed if elapsed > 0 else 0
        eta = (total - completed - failed) / rate if rate > 0 else 0
        
        print(f"\r[{bar}] {progress:.1f}% | "
              f"✅ {completed} | ❌ {failed} | "
              f"⏱️ {elapsed:.0f}s | "
              f"📊 {rate:.1f} videos/min | "
              f"⏳ ETA: {eta/60:.1f} min", end='', flush=True)
    
    def _print_final_summary(self):
        """Print final summary"""
        print("\n\n" + "="*60)
        print("🎉 BATCH PROCESSING COMPLETE")
        print("="*60)
        print(f"Total Videos: {self.stats['total']}")
        print(f"✅ Successful: {self.stats['completed']}")
        print(f"❌ Failed: {self.stats['failed']}")
        
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        print(f"\n⏱️ Total Time: {duration/60:.1f} minutes")
        print(f"📊 Average: {duration/self.stats['total']:.1f} seconds per video")
        print(f"🚀 Throughput: {self.stats['total']/duration*60:.1f} videos per minute")
    
    def _generate_batch_report(self, output_dir: str) -> Dict[str, Any]:
        """Generate comprehensive batch report"""
        # Collect all results
        all_results = []
        for file_path in sorted(os.listdir(output_dir)):
            if file_path.startswith('video_') and file_path.endswith('.json'):
                with open(f"{output_dir}/{file_path}", 'r') as f:
                    all_results.append(json.load(f))
        
        # Analyzer statistics
        analyzer_stats = {}
        total_processing_time = 0
        
        for result in all_results:
            if result.get('status') == 'success':
                total_processing_time += result.get('processing_time', 0)
                
                if 'analysis' in result and 'results' in result['analysis']:
                    for analyzer, data in result['analysis']['results'].items():
                        if analyzer not in analyzer_stats:
                            analyzer_stats[analyzer] = {
                                "total_runs": 0,
                                "successful": 0,
                                "failed": 0,
                                "avg_results": 0,
                                "total_results": 0
                            }
                        
                        analyzer_stats[analyzer]["total_runs"] += 1
                        
                        if 'error' not in data:
                            analyzer_stats[analyzer]["successful"] += 1
                            # Count results (e.g., segments, detections)
                            if isinstance(data, dict) and 'segments' in data:
                                count = len(data['segments'])
                                analyzer_stats[analyzer]["total_results"] += count
                        else:
                            analyzer_stats[analyzer]["failed"] += 1
        
        # Calculate averages
        for analyzer, stats in analyzer_stats.items():
            if stats["successful"] > 0:
                stats["avg_results"] = stats["total_results"] / stats["successful"]
        
        return {
            "batch_summary": {
                "total_videos": self.stats["total"],
                "successful": self.stats["completed"],
                "failed": self.stats["failed"],
                "total_duration": (self.stats["end_time"] - self.stats["start_time"]).total_seconds(),
                "avg_processing_time": total_processing_time / self.stats["completed"] if self.stats["completed"] > 0 else 0,
                "throughput_per_minute": self.stats["total"] / ((self.stats["end_time"] - self.stats["start_time"]).total_seconds() / 60)
            },
            "analyzer_performance": analyzer_stats,
            "timestamp": datetime.now().isoformat(),
            "output_directory": output_dir
        }


class QueuedBatchProcessor(BatchProcessor):
    """Advanced batch processor with queue management and priority handling"""
    
    def __init__(self, max_workers: int = 3):
        super().__init__(max_workers, use_optimized=True)
        self.priority_queue = queue.PriorityQueue()
        self.processing = True
        
    def add_url(self, url: str, priority: int = 5):
        """Add URL to processing queue with priority (1=highest, 10=lowest)"""
        self.priority_queue.put((priority, datetime.now(), url))
        
    def start_processing(self):
        """Start processing queue in background"""
        self.processing = True
        
        # Start worker threads
        workers = []
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker, args=(i,))
            worker.daemon = True
            worker.start()
            workers.append(worker)
        
        logger.info(f"🚀 Started {self.max_workers} worker threads")
        return workers
    
    def _worker(self, worker_id: int):
        """Worker thread for processing queue"""
        pipeline = self.pipeline_class()
        
        while self.processing:
            try:
                # Get next item from queue
                priority, timestamp, url = self.priority_queue.get(timeout=5)
                
                logger.info(f"Worker {worker_id}: Processing {url} (priority: {priority})")
                
                # Process URL
                result = pipeline.process_tiktok_url(url)
                
                # Mark as done
                self.priority_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
    
    def stop_processing(self):
        """Stop processing queue"""
        self.processing = False
        logger.info("Stopping queue processing...")


def create_test_batch_file():
    """Create a test batch file with sample URLs"""
    test_urls = [
        "https://www.tiktok.com/@cristiano/video/7294871462286863649",
        "https://www.tiktok.com/@khaby.lame/video/7293456789012345678",
        "https://www.tiktok.com/@charlidamelio/video/7292345678901234567",
        # Add more test URLs here
    ]
    
    with open("/home/user/tiktok_production/test_batch.txt", 'w') as f:
        for url in test_urls:
            f.write(url + '\n')
    
    print(f"Created test batch file with {len(test_urls)} URLs")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process TikTok videos")
    parser.add_argument("--file", help="Path to file with URLs")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel workers")
    parser.add_argument("--create-test", action="store_true", help="Create test batch file")
    
    args = parser.parse_args()
    
    if args.create_test:
        create_test_batch_file()
    elif args.file:
        processor = BatchProcessor(max_workers=args.workers)
        processor.process_urls_file(args.file)
    else:
        print("Usage: python batch_processor.py --file urls.txt [--workers 3]")
        print("       python batch_processor.py --create-test")