#!/usr/bin/env python3
"""
TikTok Video Downloader with yt-dlp integration
Handles rate limiting, retries, and metadata extraction
"""

import yt_dlp
import logging
from pathlib import Path
import json
import time
from typing import Dict, Optional, List
import hashlib
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class TikTokDownloader:
    """Download TikTok videos with metadata and error handling"""
    
    def __init__(self, download_dir: str = "/home/user/tiktok_videos"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organization
        (self.download_dir / "videos").mkdir(exist_ok=True)
        (self.download_dir / "metadata").mkdir(exist_ok=True)
        (self.download_dir / "thumbnails").mkdir(exist_ok=True)
        
        # Rate limiting
        self.last_download_time = 0
        self.min_delay = 2  # Minimum 2 seconds between downloads
        
        # yt-dlp configuration
        self.ydl_opts = {
            'outtmpl': str(self.download_dir / 'videos' / '%(id)s.%(ext)s'),
            'format': 'best[ext=mp4]/best',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'socket_timeout': 30,
            'retries': 3,
            'fragment_retries': 3,
            'concurrent_fragment_downloads': 4,
            'writedescription': True,
            'writeinfojson': True,
            'writethumbnail': True,
            'cookiefile': 'cookies.txt',  # Optional: Use cookies for better access
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'extractor_args': {
                'tiktok': {
                    'api_hostname': 'api16-normal-c-useast1a.tiktokv.com'
                }
            },
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
        }
        
    def _rate_limit(self):
        """Enforce rate limiting between downloads"""
        current_time = time.time()
        time_since_last = current_time - self.last_download_time
        
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            logger.info(f"Rate limiting: sleeping for {sleep_time:.1f}s")
            time.sleep(sleep_time)
            
        self.last_download_time = time.time()
        
    def _extract_tiktok_id(self, url: str) -> str:
        """Extract TikTok video ID from URL"""
        # Handle different URL formats
        if '/video/' in url:
            return url.split('/video/')[-1].split('?')[0]
        elif '@' in url and '/' in url:
            parts = url.split('/')
            for i, part in enumerate(parts):
                if part.startswith('@') and i + 1 < len(parts):
                    return parts[i + 1].split('?')[0]
        
        # Fallback: use URL hash
        return hashlib.md5(url.encode()).hexdigest()[:16]
        
    def download_video(self, url: str, priority: int = 5) -> Dict[str, any]:
        """
        Download TikTok video with metadata
        
        Args:
            url: TikTok video URL
            priority: Download priority (1-10)
            
        Returns:
            Dict with download results and metadata
        """
        # Rate limiting
        self._rate_limit()
        
        try:
            start_time = time.time()
            logger.info(f"Downloading: {url}")
            
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # Extract info
                info = ydl.extract_info(url, download=True)
                
                # Get file paths
                video_id = info.get('id', self._extract_tiktok_id(url))
                video_path = self.download_dir / 'videos' / f"{video_id}.mp4"
                
                # Ensure video file exists
                if not video_path.exists():
                    # Try alternative extensions
                    for ext in ['webm', 'mov', 'avi']:
                        alt_path = self.download_dir / 'videos' / f"{video_id}.{ext}"
                        if alt_path.exists():
                            video_path = alt_path
                            break
                
                # Extract comprehensive metadata
                metadata = {
                    'tiktok_id': video_id,
                    'tiktok_url': url,
                    'username': info.get('uploader', '') or info.get('creator', ''),
                    'user_id': info.get('uploader_id', ''),
                    'description': info.get('description', ''),
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0),
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'comment_count': info.get('comment_count', 0),
                    'share_count': info.get('repost_count', 0),
                    'download_path': str(video_path),
                    'thumbnail': info.get('thumbnail', ''),
                    'upload_date': info.get('upload_date', ''),
                    'timestamp': info.get('timestamp', 0),
                    'hashtags': self._extract_hashtags(info.get('description', '')),
                    'music': {
                        'title': info.get('track', ''),
                        'author': info.get('artist', ''),
                    },
                    'download_time': time.time() - start_time,
                    'downloaded_at': datetime.now().isoformat(),
                    'priority': priority,
                }
                
                # Save metadata separately
                metadata_path = self.download_dir / 'metadata' / f"{video_id}.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Downloaded successfully: {video_id} ({metadata['download_time']:.1f}s)")
                
                return {
                    'success': True,
                    'video_path': str(video_path),
                    'metadata': metadata,
                    'video_id': video_id
                }
                
        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e)
            logger.error(f"Download error for {url}: {error_msg}")
            
            # Check for specific errors
            if 'private' in error_msg.lower():
                return {'success': False, 'error': 'Video is private', 'url': url}
            elif 'removed' in error_msg.lower():
                return {'success': False, 'error': 'Video has been removed', 'url': url}
            elif '404' in error_msg:
                return {'success': False, 'error': 'Video not found', 'url': url}
            else:
                return {'success': False, 'error': error_msg, 'url': url}
                
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            return {'success': False, 'error': str(e), 'url': url}
            
    def _extract_hashtags(self, description: str) -> List[str]:
        """Extract hashtags from description"""
        import re
        hashtags = re.findall(r'#\w+', description)
        return [tag.lower() for tag in hashtags]
        
    def download_batch(self, urls: List[str], max_workers: int = 3) -> Dict[str, any]:
        """
        Download multiple videos with concurrent workers
        
        Args:
            urls: List of TikTok URLs
            max_workers: Maximum concurrent downloads
            
        Returns:
            Dict with results for each URL
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all downloads
            future_to_url = {
                executor.submit(self.download_video, url): url 
                for url in urls
            }
            
            # Process completed downloads
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results[url] = result
                except Exception as e:
                    logger.error(f"Failed to download {url}: {e}")
                    results[url] = {'success': False, 'error': str(e)}
                    
        return results
        
    def cleanup_old_videos(self, days: int = 7):
        """Remove videos older than specified days"""
        import shutil
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned = 0
        
        for video_file in (self.download_dir / 'videos').glob('*.mp4'):
            # Check metadata for download date
            metadata_file = self.download_dir / 'metadata' / f"{video_file.stem}.json"
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                downloaded_at = datetime.fromisoformat(metadata.get('downloaded_at', ''))
                if downloaded_at < cutoff_date:
                    video_file.unlink()
                    metadata_file.unlink()
                    cleaned += 1
                    
        logger.info(f"Cleaned up {cleaned} old videos")
        return cleaned


if __name__ == "__main__":
    # Test downloader
    downloader = TikTokDownloader()
    
    # Test single download
    test_url = "https://www.tiktok.com/@user/video/1234567890"
    result = downloader.download_video(test_url)
    print(json.dumps(result, indent=2))