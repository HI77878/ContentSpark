#!/usr/bin/env python3
"""
Standardized Base Analyzer - Einheitliches Interface für alle Analyzer
Behebt Interface-Mismatches und Rekursionsprobleme
"""

import torch
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import gc
import time

logger = logging.getLogger(__name__)


class StandardizedBaseAnalyzer(ABC):
    """
    Standardisierte Basisklasse für alle Analyzer
    - Einheitliches Interface: analyze(video_path)
    - Keine _analyze_impl Verwirrung mehr
    - Klare Vererbungshierarchie
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False
        self.analyzer_name = self.__class__.__name__
        
    @abstractmethod
    def load_model(self):
        """
        Load the model for this analyzer.
        Must be implemented by each analyzer.
        """
        pass
        
    @abstractmethod
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process the video and return analysis results.
        Must be implemented by each analyzer.
        """
        pass
        
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """
        Main entry point for all analyzers.
        Handles model loading, processing, and cleanup.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Analysis results dictionary
        """
        start_time = time.time()
        
        try:
            # Lazy load model
            if not self.model_loaded:
                logger.info(f"[{self.analyzer_name}] Loading model...")
                self.load_model()
                self.model_loaded = True
                
            # Process video
            logger.info(f"[{self.analyzer_name}] Analyzing {video_path}")
            results = self.process_video(video_path)
            
            # Add metadata
            elapsed = time.time() - start_time
            results['metadata'] = {
                'analyzer': self.analyzer_name,
                'processing_time': elapsed,
                'device': str(self.device)
            }
            
            logger.info(f"[{self.analyzer_name}] Completed in {elapsed:.1f}s")
            return results
            
        except Exception as e:
            logger.error(f"[{self.analyzer_name}] Analysis failed: {e}")
            return {
                'error': str(e),
                'analyzer': self.analyzer_name,
                'metadata': {
                    'processing_time': time.time() - start_time,
                    'device': str(self.device)
                }
            }
        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()


class GPUBatchAnalyzerStandardized(StandardizedBaseAnalyzer):
    """
    Standardisierte GPU Batch Analyzer Basisklasse
    Für Analyzer die Frames in Batches verarbeiten
    """
    
    def __init__(self, batch_size: int = 8, frame_interval: int = 30):
        super().__init__()
        self.batch_size = batch_size
        self.frame_interval = frame_interval
        
    @abstractmethod
    def process_batch(self, frames: torch.Tensor, frame_times: list) -> Dict[str, Any]:
        """
        Process a batch of frames.
        Must be implemented by each batch analyzer.
        """
        pass
        
    def extract_frames(self, video_path: str) -> tuple:
        """Extract frames from video at specified interval"""
        import cv2
        import numpy as np
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        frame_times = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % self.frame_interval == 0:
                frames.append(frame)
                frame_times.append(frame_count / fps)
                
            frame_count += 1
            
        cap.release()
        
        logger.info(f"[{self.analyzer_name}] Extracted {len(frames)} frames")
        return frames, frame_times
        
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process video by extracting frames and processing in batches"""
        # Extract frames
        frames, frame_times = self.extract_frames(video_path)
        
        if not frames:
            return {'segments': [], 'error': 'No frames extracted'}
            
        # Process in batches
        all_results = []
        
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i + self.batch_size]
            batch_times = frame_times[i:i + self.batch_size]
            
            # Convert to tensor
            import numpy as np
            frames_array = np.stack(batch_frames)
            frames_tensor = torch.from_numpy(frames_array).to(self.device)
            frames_tensor = frames_tensor.permute(0, 3, 1, 2).float() / 255.0
            
            # Process batch
            batch_results = self.process_batch(frames_tensor, batch_times)
            all_results.extend(batch_results.get('segments', []))
            
        return {'segments': all_results}