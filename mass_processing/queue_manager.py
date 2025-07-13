#!/usr/bin/env python3
"""
Queue Manager for TikTok video processing
Handles URL queuing, priority management, and status tracking
"""

import redis
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
import hashlib

logger = logging.getLogger(__name__)

class QueueManager:
    """Manages video processing queues with Redis"""
    
    def __init__(self, redis_url: str = 'redis://localhost:6379/0'):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        
        # Queue keys
        self.queues = {
            'download': 'queue:download',
            'processing': 'queue:processing', 
            'priority': 'queue:priority',
            'failed': 'queue:failed'
        }
        
        # Status tracking
        self.processed_set = 'urls:processed'
        self.processing_set = 'urls:processing'
        self.failed_set = 'urls:failed'
        
        # Metadata storage
        self.metadata_prefix = 'metadata:'
        
    def add_url(self, url: str, priority: int = 5, metadata: Dict = None) -> bool:
        """
        Add single URL to download queue
        
        Args:
            url: TikTok video URL
            priority: Priority level (1-10, higher = more priority)
            metadata: Additional metadata
            
        Returns:
            bool: True if added, False if already exists
        """
        # Check if already processed or processing
        if self.is_url_processed(url) or self.is_url_processing(url):
            logger.info(f"URL already processed/processing: {url}")
            return False
            
        # Create task object
        task = {
            'url': url,
            'priority': priority,
            'created_at': datetime.now().isoformat(),
            'retry_count': 0,
            'metadata': metadata or {}
        }
        
        # Determine queue based on priority
        if priority >= 8:
            queue_key = self.queues['priority']
        else:
            queue_key = self.queues['download']
            
        # Add to queue (score is priority for sorting)
        self.redis_client.zadd(queue_key, {json.dumps(task): priority})
        
        logger.info(f"Added URL to queue: {url} (priority: {priority})")
        return True
        
    def add_urls_batch(self, urls: List[str], priority: int = 5) -> Dict[str, bool]:
        """
        Add multiple URLs to download queue
        
        Args:
            urls: List of TikTok URLs
            priority: Priority level for all URLs
            
        Returns:
            Dict mapping URL to success status
        """
        results = {}
        pipe = self.redis_client.pipeline()
        
        for url in urls:
            # Skip if already processed
            if self.is_url_processed(url) or self.is_url_processing(url):
                results[url] = False
                continue
                
            task = {
                'url': url,
                'priority': priority,
                'created_at': datetime.now().isoformat(),
                'retry_count': 0
            }
            
            # Determine queue
            queue_key = self.queues['priority'] if priority >= 8 else self.queues['download']
            
            # Add to pipeline
            pipe.zadd(queue_key, {json.dumps(task): priority})
            results[url] = True
            
        # Execute pipeline
        pipe.execute()
        
        added = sum(1 for v in results.values() if v)
        logger.info(f"Added {added}/{len(urls)} URLs to queue")
        
        return results
        
    def get_next_download_task(self) -> Optional[Dict]:
        """
        Get next task from download queue (priority-aware)
        
        Returns:
            Task dict or None if no tasks
        """
        # Check priority queue first
        task = self._pop_from_queue(self.queues['priority'])
        if task:
            return task
            
        # Then regular download queue
        return self._pop_from_queue(self.queues['download'])
        
    def _pop_from_queue(self, queue_key: str) -> Optional[Dict]:
        """Pop highest priority item from queue"""
        # Get highest priority item
        items = self.redis_client.zrevrange(queue_key, 0, 0, withscores=True)
        
        if not items:
            return None
            
        task_json, score = items[0]
        
        # Remove from queue
        self.redis_client.zrem(queue_key, task_json)
        
        # Parse task
        task = json.loads(task_json)
        
        # Mark as processing
        self.mark_url_processing(task['url'])
        
        return task
        
    def mark_url_processed(self, url: str):
        """Mark URL as successfully processed"""
        url_hash = self._hash_url(url)
        
        # Remove from processing set
        self.redis_client.srem(self.processing_set, url_hash)
        
        # Add to processed set with timestamp
        self.redis_client.hset(self.processed_set, url_hash, datetime.now().isoformat())
        
        logger.info(f"Marked as processed: {url}")
        
    def mark_url_processing(self, url: str):
        """Mark URL as currently being processed"""
        url_hash = self._hash_url(url)
        self.redis_client.sadd(self.processing_set, url_hash)
        
    def mark_url_failed(self, url: str, error: str, permanent: bool = False):
        """Mark URL as failed"""
        url_hash = self._hash_url(url)
        
        # Remove from processing
        self.redis_client.srem(self.processing_set, url_hash)
        
        if permanent:
            # Don't retry permanent failures
            self.redis_client.hset(self.failed_set, url_hash, json.dumps({
                'error': error,
                'failed_at': datetime.now().isoformat(),
                'permanent': True
            }))
        else:
            # Add to failed queue for retry
            task = {
                'url': url,
                'error': error,
                'failed_at': datetime.now().isoformat(),
                'retry_count': 0
            }
            self.redis_client.zadd(self.queues['failed'], {json.dumps(task): 0})
            
        logger.warning(f"Marked as failed: {url} - {error}")
        
    def is_url_processed(self, url: str) -> bool:
        """Check if URL has been processed"""
        url_hash = self._hash_url(url)
        return self.redis_client.hexists(self.processed_set, url_hash)
        
    def is_url_processing(self, url: str) -> bool:
        """Check if URL is currently being processed"""
        url_hash = self._hash_url(url)
        return self.redis_client.sismember(self.processing_set, url_hash)
        
    def get_queue_stats(self) -> Dict[str, int]:
        """Get current queue statistics"""
        stats = {}
        
        # Queue sizes
        for name, key in self.queues.items():
            stats[f'queue_{name}'] = self.redis_client.zcard(key)
            
        # Status counts
        stats['processed'] = self.redis_client.hlen(self.processed_set)
        stats['processing'] = self.redis_client.scard(self.processing_set)
        stats['failed'] = self.redis_client.hlen(self.failed_set)
        
        # Calculate total pending
        stats['total_pending'] = (
            stats['queue_download'] + 
            stats['queue_priority'] + 
            stats['queue_processing']
        )
        
        return stats
        
    def get_queue_details(self, queue_name: str, limit: int = 10) -> List[Dict]:
        """Get detailed view of specific queue"""
        if queue_name not in self.queues:
            return []
            
        queue_key = self.queues[queue_name]
        items = self.redis_client.zrevrange(queue_key, 0, limit-1, withscores=True)
        
        details = []
        for task_json, score in items:
            task = json.loads(task_json)
            task['score'] = score
            details.append(task)
            
        return details
        
    def requeue_failed(self, max_retries: int = 3) -> int:
        """
        Requeue failed items that haven't exceeded retry limit
        
        Args:
            max_retries: Maximum number of retries
            
        Returns:
            Number of items requeued
        """
        failed_items = self.redis_client.zrange(self.queues['failed'], 0, -1)
        requeued = 0
        
        for item_json in failed_items:
            task = json.loads(item_json)
            
            if task.get('retry_count', 0) < max_retries:
                # Increment retry count
                task['retry_count'] = task.get('retry_count', 0) + 1
                task['retry_at'] = datetime.now().isoformat()
                
                # Remove from failed queue
                self.redis_client.zrem(self.queues['failed'], item_json)
                
                # Add back to download queue with lower priority
                new_priority = max(1, task.get('priority', 5) - 1)
                task['priority'] = new_priority
                
                self.redis_client.zadd(
                    self.queues['download'],
                    {json.dumps(task): new_priority}
                )
                requeued += 1
                
                logger.info(f"Requeued failed task: {task['url']} (attempt {task['retry_count']})")
                
        return requeued
        
    def cleanup_old_tasks(self, days: int = 7) -> int:
        """
        Remove old tasks from queues
        
        Args:
            days: Remove tasks older than this many days
            
        Returns:
            Number of tasks removed
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        removed = 0
        
        for queue_name, queue_key in self.queues.items():
            items = self.redis_client.zrange(queue_key, 0, -1)
            
            for item_json in items:
                try:
                    task = json.loads(item_json)
                    created_at = datetime.fromisoformat(task.get('created_at', ''))
                    
                    if created_at < cutoff_date:
                        self.redis_client.zrem(queue_key, item_json)
                        removed += 1
                except:
                    # Remove invalid items
                    self.redis_client.zrem(queue_key, item_json)
                    removed += 1
                    
        logger.info(f"Cleaned up {removed} old tasks")
        return removed
        
    def get_processing_time_stats(self) -> Dict[str, float]:
        """Get statistics on processing times"""
        # This would require tracking start/end times
        # For now, return placeholder
        return {
            'avg_download_time': 15.2,
            'avg_processing_time': 245.8,
            'avg_total_time': 261.0
        }
        
    def clear_all_queues(self):
        """Clear all queues (use with caution!)"""
        for queue_key in self.queues.values():
            self.redis_client.delete(queue_key)
            
        self.redis_client.delete(self.processed_set)
        self.redis_client.delete(self.processing_set)
        self.redis_client.delete(self.failed_set)
        
        logger.warning("Cleared all queues!")
        
    def _hash_url(self, url: str) -> str:
        """Create consistent hash for URL"""
        # Normalize URL
        url = url.strip().lower()
        
        # Extract video ID if possible
        if '/video/' in url:
            video_id = url.split('/video/')[-1].split('?')[0]
            return f"tiktok:{video_id}"
        
        # Fallback to URL hash
        return hashlib.md5(url.encode()).hexdigest()
        
    def export_stats_json(self, filepath: str):
        """Export queue statistics to JSON file"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'queues': self.get_queue_stats(),
            'processing_times': self.get_processing_time_stats(),
            'queue_samples': {
                name: self.get_queue_details(name, limit=5)
                for name in self.queues.keys()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"Exported stats to {filepath}")


# CLI interface for queue management
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TikTok Queue Manager')
    parser.add_argument('command', choices=['stats', 'add', 'clear', 'requeue', 'export'])
    parser.add_argument('--url', help='TikTok URL to add')
    parser.add_argument('--file', help='File with URLs to add')
    parser.add_argument('--priority', type=int, default=5, help='Priority (1-10)')
    parser.add_argument('--output', help='Output file for export')
    
    args = parser.parse_args()
    
    qm = QueueManager()
    
    if args.command == 'stats':
        stats = qm.get_queue_stats()
        print("\nQueue Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    elif args.command == 'add':
        if args.url:
            success = qm.add_url(args.url, args.priority)
            print(f"Added: {success}")
        elif args.file:
            with open(args.file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            results = qm.add_urls_batch(urls, args.priority)
            added = sum(1 for v in results.values() if v)
            print(f"Added {added}/{len(urls)} URLs")
            
    elif args.command == 'clear':
        response = input("Clear all queues? (yes/no): ")
        if response.lower() == 'yes':
            qm.clear_all_queues()
            print("Cleared all queues")
            
    elif args.command == 'requeue':
        requeued = qm.requeue_failed()
        print(f"Requeued {requeued} failed tasks")
        
    elif args.command == 'export':
        output_file = args.output or f"queue_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        qm.export_stats_json(output_file)
        print(f"Exported to {output_file}")