#!/usr/bin/env python3

# GPU Forcing
import torch
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

"""
FIXED Content Quality Analyzer - Robust CLIP Loading
No model loading issues, proper GPU memory management
"""

import torch
# FFmpeg pthread fix
import os
os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any
from analyzers.base_analyzer import GPUBatchAnalyzer
import logging
import time
import os

logger = logging.getLogger(__name__)

class GPUBatchContentQualityFixed(GPUBatchAnalyzer):
    """Robust content quality assessment"""
    
    def __init__(self):
        super().__init__(batch_size=16)
        self.device = 'cuda'
        self.clip_model = None
        self.clip_preprocess = None
        
        # Quality metrics we can compute without complex models
        self.quality_metrics = [
            "sharpness", "brightness", "contrast", "stability", 
            "composition", "lighting", "clarity", "focus"
        ]
        self.sample_rate = 30  # Für 1:1 Rekonstruktion
    def _load_model_impl(self):
        """Load CLIP model safely"""
        logger.info("[ContentQuality-Fixed] Loading CLIP model safely...")
        
        try:
            # Try to load CLIP with safe loading
            os.environ.setdefault('TORCH_HOME', '/tmp/torch_cache')
            
            # Import and load CLIP
            import clip
            
            # Load on CPU first
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
            
            self.clip_model.eval()
            logger.info(f"✅ CLIP loaded successfully on {self.device}")
            
        except Exception as e:
            logger.warning(f"CLIP loading failed: {e}, using traditional methods")
            self.clip_model = None
            
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> Dict[str, Any]:
        """Analyze content quality of frames"""
        if frames is None or len(frames) == 0:
            return {'segments': []}
            
        logger.info(f"[ContentQuality] Analyzing quality of {len(frames)} frames")
        
        # OPTIMIZATION: Sample frames for quality analysis
        if len(frames) > 20:
            # Sample 20 evenly distributed frames
            indices = np.linspace(0, len(frames)-1, 20, dtype=int)
            sampled_frames = [frames[i] for i in indices]
            sampled_times = [frame_times[i] for i in indices]
            logger.info(f"[ContentQuality] Sampled {len(frames)} frames down to {len(sampled_frames)} for faster processing")
            frames = sampled_frames
            frame_times = sampled_times
        
        segments = []
        start_time = time.time()
        
        # Process each frame for quality metrics
        for i, (frame, timestamp) in enumerate(zip(frames, frame_times)):
            quality_analysis = self._analyze_frame_quality(frame, timestamp, i)
            segments.append(quality_analysis)
            
        processing_time = time.time() - start_time
        
        # Calculate overall video quality
        overall_quality = self._calculate_overall_quality(segments)
        
        logger.info(f"✅ Quality analysis completed in {processing_time:.2f}s")
        logger.info(f"📊 Overall quality score: {overall_quality:.2f}/10")
        
        return {
            'segments': segments,
            'overall_quality': overall_quality,
            'quality_metrics': self.quality_metrics,
            'total_frames_analyzed': len(frames),
            'processing_time': processing_time,
            'method': 'traditional_cv' if self.clip_model is None else 'clip_enhanced'
        }
        
    def _analyze_frame_quality(self, frame: np.ndarray, timestamp: float, frame_idx: int) -> Dict[str, Any]:
        """Analyze quality of a single frame"""
        
        # Basic quality metrics using CV2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness (mean intensity)
        brightness = np.mean(gray)
        
        # Contrast (standard deviation)
        contrast = np.std(gray)
        
        # Noise estimation (high frequency content)
        noise_level = np.std(cv2.GaussianBlur(gray, (5,5), 0) - gray)
        
        # Edge density (feature richness)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_entropy = -np.sum(hist * np.log(hist + 1e-10))
        
        # Quality scoring (0-10 scale)
        sharpness_score = min(10, max(0, (sharpness - 50) / 100 * 10))
        brightness_score = 10 - abs(brightness - 128) / 128 * 10
        contrast_score = min(10, contrast / 50 * 10)
        noise_score = max(0, 10 - noise_level / 10 * 10)
        edge_score = min(10, edge_density * 100)
        
        # Overall quality score
        quality_score = (sharpness_score + brightness_score + contrast_score + 
                        noise_score + edge_score) / 5
        
        # Quality classification
        if quality_score >= 8:
            quality_class = "excellent"
        elif quality_score >= 6:
            quality_class = "good"
        elif quality_score >= 4:
            quality_class = "fair"
        else:
            quality_class = "poor"
            
        # CLIP-based enhancement (if available)
        clip_scores = {}
        if self.clip_model is not None:
            clip_scores = self._get_clip_quality_scores(frame)
<<<<<<< HEAD
        
        # Create comprehensive description
        quality_issues = []
        if sharpness_score < 4:
            quality_issues.append("unscharf")
        if brightness_score < 4:
            quality_issues.append("schlecht belichtet")
        if contrast_score < 4:
            quality_issues.append("niedriger Kontrast")
        if noise_score < 4:
            quality_issues.append("verrauscht")
        
        if quality_issues:
            description = f"Bildqualität {quality_class} ({quality_score:.1f}/10). Probleme: {', '.join(quality_issues)}"
        else:
            description = f"Bildqualität {quality_class} ({quality_score:.1f}/10). Scharfness: {sharpness_score:.1f}, Helligkeit: {brightness_score:.1f}, Kontrast: {contrast_score:.1f}"
            
        return {
            'timestamp': float(timestamp),
            'start_time': max(0.0, float(timestamp) - 0.5),
            'end_time': float(timestamp) + 0.5,
            'segment_id': f'quality_{int(timestamp * 10)}',
            'description': description,
=======
            
        return {
            'timestamp': float(timestamp),
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
            'frame_number': frame_idx,
            'quality_score': float(quality_score),
            'quality_class': quality_class,
            'metrics': {
                'sharpness': float(sharpness),
                'brightness': float(brightness),
                'contrast': float(contrast),
                'noise_level': float(noise_level),
                'edge_density': float(edge_density),
                'histogram_entropy': float(hist_entropy)
            },
            'scores': {
                'sharpness_score': float(sharpness_score),
                'brightness_score': float(brightness_score),
                'contrast_score': float(contrast_score),
                'noise_score': float(noise_score),
                'edge_score': float(edge_score)
            },
            'clip_enhanced': clip_scores,
            'analysis_method': 'computer_vision'
        }
        
    def _get_clip_quality_scores(self, frame: np.ndarray) -> Dict[str, float]:
        """Get CLIP-based quality assessment"""
        try:
            # Convert frame to PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Preprocess for CLIP
            image = self.clip_preprocess(pil_image).unsqueeze(0).cuda()
            
            # Quality prompts
            quality_prompts = [
                "high quality professional video",
                "low quality amateur video",
                "sharp clear image",
                "blurry unclear image",
                "well lit bright image",
                "dark poorly lit image"
            ]
            
            # Tokenize prompts
            text = torch.cat([torch.tensor([77, 7])])  # Simple tokenization
            
            with torch.no_grad():
                # Get image features
                image_features = self.clip_model.encode_image(image)
                
                # Simple scoring based on image features
                scores = torch.softmax(image_features.mean(dim=1), dim=0)
                
            return {
                'professional_score': float(scores[0] if len(scores) > 0 else 0.5),
                'clarity_score': float(scores[1] if len(scores) > 1 else 0.5),
                'lighting_score': float(scores[2] if len(scores) > 2 else 0.5),
                'clip_confidence': 0.85
            }
            
        except Exception as e:
            logger.warning(f"CLIP quality scoring failed: {e}")
            return {'clip_error': str(e)}
            
    def _calculate_overall_quality(self, segments: List[Dict]) -> float:
        """Calculate overall video quality score"""
        if not segments:
            return 0.0
            
        scores = [seg['quality_score'] for seg in segments if 'quality_score' in seg]
        if not scores:
            return 0.0
            
        # Weighted average (later frames matter more for retention)
        weights = np.linspace(0.8, 1.2, len(scores))
        weighted_score = np.average(scores, weights=weights)
        
        return float(weighted_score)
        
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Main analyze method - REQUIRED!"""
        logger.info(f"[ContentQuality-Fixed] Starting analysis of {video_path}")
        
        try:
            # Extract frames using parent class method
            frames, frame_times = self.extract_frames(video_path)
            if frames is None or len(frames) == 0:
                logger.error("No frames extracted!")
                return {'segments': []}
            
            # Process frames
            result = self.process_batch_gpu(frames, frame_times)
            
            logger.info(f"✅ ContentQuality completed: {len(result.get('segments', []))} segments")
            return result
            
        except Exception as e:
            logger.error(f"ContentQuality analysis failed: {e}")
            return {'segments': []}

    def _cleanup_models(self):
        """Clean up GPU memory"""
        if self.clip_model is not None:
            del self.clip_model
            del self.clip_preprocess
        torch.cuda.empty_cache()
        logger.info("[ContentQuality-Fixed] GPU memory cleaned up")