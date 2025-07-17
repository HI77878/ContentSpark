from abc import ABC, abstractmethod
# FFmpeg pthread fix
import os
os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"

# cuDNN optimization für ConvNets
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import cv2
import numpy as np
import gc
import time
import sys
import logging
sys.path.append('/home/user/tiktok_production')
from gpu_force_config import GPUForce, DEVICE, force_gpu
from utils.gpu_memory_optimizer import GPUMemoryOptimizer

logger = logging.getLogger(__name__)

class GPUBatchAnalyzer(ABC):
    def __init__(self, batch_size=64):  # Increased from 32 to 64 for maximum GPU utilization
        self.batch_size = batch_size
        # Force GPU usage with optimizations
        self.device = DEVICE  # Use global GPU device
        if self.device.type != 'cuda':
            print(f"WARNING: {self.__class__.__name__} running on CPU!")
        
        # Apply GPU optimizations
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.model = None  # NICHT beim Init laden!
        self.model_loaded = False
        # GPU Memory Optimizer
        self.gpu_optimizer = GPUMemoryOptimizer()
        # Model caching support
        self.use_model_cache = True
        self._model_cache = {}
        
        # Performance tracking
        self.total_gpu_time = 0.0
        self.total_frames_processed = 0
    
    @abstractmethod
    @torch.inference_mode()
    def process_batch_gpu(self, frames, frame_times):
        pass
    
    @staticmethod
    def is_valid_frame(frame):
        """Check if frame is valid (handles numpy array boolean ambiguity)"""
        if frame is None:
            return False
        if isinstance(frame, np.ndarray):
            return frame.size > 0
        return bool(frame)
    
    @staticmethod
    def is_empty_frames(frames):
        """Check if frames list is empty (handles numpy array boolean ambiguity)"""
        if frames is None:
            return True
        if isinstance(frames, np.ndarray):
            return frames.size == 0
        return len(frames) == 0
    
    def extract_frames(self, video_path, sample_rate=None, max_frames=1000):
        """Extract frames from video with memory optimization"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate optimal sample rate from config
        if sample_rate is None:
            # Import config here to avoid circular imports
            from configs.performance_config import FRAME_EXTRACTION_INTERVALS
            from configs.gpu_groups_config import get_frame_interval
            
            # Get analyzer name from class
            analyzer_name = getattr(self, 'analyzer_name', self.__class__.__name__.lower())
            
            # Try multiple sources for frame interval
            if hasattr(self, 'frame_interval'):
                # If analyzer has explicit frame_interval
                sample_rate = self.frame_interval
            elif analyzer_name in FRAME_EXTRACTION_INTERVALS:
                # Use performance config
                sample_rate = FRAME_EXTRACTION_INTERVALS[analyzer_name]
            else:
                # Try GPU groups config
                sample_rate = get_frame_interval(analyzer_name)
            
            print(f"[{self.__class__.__name__}] Using sample_rate={sample_rate} from config (fps={fps:.1f})")
        
        frames = []
        timestamps = []
        frame_count = 0
        extracted_count = 0
        
        # Pre-allocate arrays for better performance
        if max_frames:
            frames = []
            timestamps = []
        
        while cap.isOpened() and (max_frames is None or extracted_count < max_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_rate == 0:
                frames.append(frame)
                timestamps.append(frame_count / fps)
                extracted_count += 1
                
            frame_count += 1
        
        cap.release()
        
        # Convert to pinned memory for faster GPU transfer
        if torch.cuda.is_available() and frames:
            pinned_frames = self.gpu_optimizer.pin_memory_batch(frames)
            print(f"[{self.__class__.__name__}] Extracted {len(frames)} frames with pinned memory")
        else:
            pinned_frames = frames
            print(f"[{self.__class__.__name__}] Extracted {len(frames)} frames (sampled from {total_frames} total)")
        
        return frames, timestamps
    
    def load_model(self):
        """Lädt Model nur wenn gebraucht"""
        if not self.model_loaded:
            # Force GPU
            force_gpu()
            print(f"[{self.__class__.__name__}] Loading model...")
            self._load_model_impl()  # Jeder Analyzer implementiert das
            self.model_loaded = True
            if torch.cuda.is_available():
                print(f"[{self.__class__.__name__}] GPU Memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    def unload_model(self):
        """Entlädt Model und gibt GPU Memory frei"""
        if self.model_loaded:
            print(f"[{self.__class__.__name__}] Unloading model...")
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            if hasattr(self, 'detector') and self.detector is not None:
                del self.detector
                self.detector = None
            if hasattr(self, 'reader') and self.reader is not None:
                del self.reader
                self.reader = None
            if hasattr(self, 'whisper_model') and self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None
            
            # Minimal cleanup - only garbage collection
            gc.collect()
            # GPU cleanup removed for better performance
            # torch.cuda.empty_cache() - not needed
            # torch.cuda.synchronize() - not needed
            # time.sleep(0.1) - removed for speed
            
            self.model_loaded = False
    
    def _load_model_impl(self):
        """To be implemented by each analyzer - default does nothing"""
        # Default implementation for backward compatibility
        pass
    
    @torch.inference_mode()
    def analyze(self, video_path):
        """Lädt Model automatisch wenn nötig - optimiert mit inference_mode"""
        # Check if we're being called from _analyze_impl to avoid recursion
        if hasattr(self, '_in_analyze') and self._in_analyze:
            # This is the old analyze method being called from _analyze_impl
            # So don't do anything special, just let it run
            raise NotImplementedError("Old analyze method called")
            
        if not self.model_loaded:
            self.load_model()
        return self._analyze_impl(video_path)
    
    def _analyze_impl(self, video_path):
        """To be implemented by each analyzer - default calls old analyze"""
        # Backward compatibility - call old analyze method if it exists
        if hasattr(self, 'analyze') and self.analyze != GPUBatchAnalyzer.analyze:
            # Temporarily set flag to avoid infinite recursion
            self._in_analyze = True
            result = self.analyze.__func__(self, video_path)
            self._in_analyze = False
            return result
        else:
            raise NotImplementedError("Analyzer must implement _analyze_impl or old analyze method")