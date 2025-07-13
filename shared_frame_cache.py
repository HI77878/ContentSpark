"""
Shared Frame Cache
Simple frame caching for analyzers
"""
import time
from typing import Dict, List, Tuple, Optional
import numpy as np

class SharedFrameCache:
    """Simple frame cache to avoid duplicate frame extraction"""
    
    def __init__(self):
        self._cache = {}
        self._access_times = {}
        self._max_cache_size = 1000  # Max number of frames to cache
        
    def get_frames(self, video_path: str, frame_indices: List[int]) -> Optional[Tuple[List[np.ndarray], List[float]]]:
        """Get frames from cache if available"""
        cache_key = f"{video_path}:{','.join(map(str, sorted(frame_indices)))}"
        
        if cache_key in self._cache:
            self._access_times[cache_key] = time.time()
            return self._cache[cache_key]
        
        return None
    
    def set_frames(self, video_path: str, frame_indices: List[int], 
                   frames: List[np.ndarray], timestamps: List[float]):
        """Store frames in cache"""
        cache_key = f"{video_path}:{','.join(map(str, sorted(frame_indices)))}"
        
        # Evict old entries if cache is full
        if len(self._cache) >= self._max_cache_size:
            # Remove least recently used
            oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
        
        self._cache[cache_key] = (frames, timestamps)
        self._access_times[cache_key] = time.time()
    
    def clear(self):
        """Clear the cache"""
        self._cache.clear()
        self._access_times.clear()

# Global cache instance
frame_cache = SharedFrameCache()
FRAME_CACHE = frame_cache  # Alias for compatibility