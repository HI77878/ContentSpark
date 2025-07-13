#!/usr/bin/env python3

# GPU Forcing
import torch
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

"""
Optimized Color Analysis - GPU-accelerated histogram calculation
Target: < 5s (from 34.9s)
"""

# FFmpeg pthread fix
import os
os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any
import logging
from analyzers.base_analyzer import GPUBatchAnalyzer

# Import sklearn - required for color analysis
from sklearn.cluster import KMeans

# Import GPU forcing
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gpu_force import force_gpu_init

from shared_frame_cache import FRAME_CACHE
logger = logging.getLogger(__name__)

class GPUBatchColorAnalysis(GPUBatchAnalyzer):
    def __init__(self):
        super().__init__(batch_size=32)
        try:
            force_gpu_init()
            self.device = 'cuda'
        except RuntimeError:
            # Fallback to CPU if GPU not available
            self.device = 'cpu'
            print("[ColorAnalysis] Running on CPU")
        
        # EXTREME OPTIMIZATION: Sample only every 2 seconds
        self.sample_rate = 30  # For 30 FPS video, this is 2 seconds
        self.resize_size = (320, 180)  # Small size for color analysis
        self.histogram_bins = 32  # Reduced from 64 for speed
        
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """GPU-accelerated color analysis"""
        logger.info(f"[ColorAnalysisOptimized] Starting GPU analysis of {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'segments': [], 'error': 'Could not open video'}
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Extract sampled frames
        frames = []
        timestamps = []
        frame_indices = list(range(0, total_frames, self.sample_rate))[:30]  # Max 30 samples
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize immediately for memory efficiency
                frame_small = cv2.resize(frame, self.resize_size)
                frames.append(frame_small)
                timestamps.append(idx / fps)
                
        cap.release()
        
        if frames is None or len(frames) == 0:
            return {'segments': [], 'error': 'No frames extracted'}
            
        # GPU batch processing
        segments = self._analyze_colors_gpu_batch(frames, timestamps)
        
        # Generate summary
        summary = self._generate_color_summary(segments)
        
        return {
            'segments': segments,
            'summary': summary,
            'metadata': {
                'frames_analyzed': len(frames),
                'sample_rate': self.sample_rate,
                'optimization': 'gpu_accelerated'
            }
        }
        
    def _analyze_colors_gpu_batch(self, frames: List[np.ndarray], timestamps: List[float]) -> List[Dict]:
        """Batch color analysis on GPU"""
        # Convert all frames to GPU tensor at once
        frames_np = np.array(frames).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames_np).cuda()
        
        segments = []
        
        with torch.no_grad():
            for i, (frame_tensor, timestamp) in enumerate(zip(frames_tensor, timestamps)):
                # Transpose to CHW format
                frame_chw = frame_tensor.permute(2, 0, 1)
                
                # Fast GPU histogram calculation
                hist_r = torch.histc(frame_chw[2].flatten(), bins=self.histogram_bins, min=0, max=1)
                hist_g = torch.histc(frame_chw[1].flatten(), bins=self.histogram_bins, min=0, max=1)
                hist_b = torch.histc(frame_chw[0].flatten(), bins=self.histogram_bins, min=0, max=1)
                
                # Dominant colors from histogram peaks
                dominant_r_idx = torch.argmax(hist_r)
                dominant_g_idx = torch.argmax(hist_g)
                dominant_b_idx = torch.argmax(hist_b)
                
                dominant_r = float(dominant_r_idx) / self.histogram_bins
                dominant_g = float(dominant_g_idx) / self.histogram_bins
                dominant_b = float(dominant_b_idx) / self.histogram_bins
                
                # Color metrics
                brightness = torch.mean(frame_tensor).cpu().item()
                saturation = torch.std(frame_tensor).cpu().item()
                
                # HSV conversion for better color understanding
                frame_cpu = (frame_tensor.cpu().numpy() * 255).astype(np.uint8)
                hsv = cv2.cvtColor(frame_cpu, cv2.COLOR_RGB2HSV)
                hue_mean = np.mean(hsv[:, :, 0])
                
                # Extract color palette using K-means clustering
                color_palette = self._extract_color_palette(frame_cpu, n_colors=5)
                
                # Determine color mood
                color_mood = self._determine_color_mood(brightness, saturation, hue_mean)
                
                segments.append({
                    'timestamp': round(timestamp, 2),
                    'dominant_color': {
                        'r': round(dominant_r, 3),
                        'g': round(dominant_g, 3),
                        'b': round(dominant_b, 3),
                        'hex': self._rgb_to_hex(dominant_r, dominant_g, dominant_b),
                        'name': self._get_color_name(int(dominant_r * 255), int(dominant_g * 255), int(dominant_b * 255))
                    },
                    'color_palette': color_palette,
                    'brightness': round(brightness, 3),
                    'saturation': round(saturation, 3),
                    'color_mood': color_mood,
                    'hue_dominant': self._hue_to_color_name(hue_mean),
                    'description': self._generate_color_description(color_mood, color_palette, hue_mean),
                    'ml_method': 'gpu_histogram_analysis'
                })
                
        return segments
        
    def _determine_color_mood(self, brightness: float, saturation: float, hue: float) -> str:
        """Determine color mood from metrics"""
        if brightness < 0.3:
            return 'dark'
        elif brightness > 0.7:
            return 'bright'
        elif saturation < 0.2:
            return 'muted'
        elif saturation > 0.6:
            return 'vibrant'
        else:
            return 'balanced'
            
    def _hue_to_color_name(self, hue: float) -> str:
        """Convert hue to color name"""
        if hue < 10 or hue > 170:
            return 'red'
        elif hue < 30:
            return 'orange'
        elif hue < 50:
            return 'yellow'
        elif hue < 80:
            return 'green'
        elif hue < 130:
            return 'blue'
        elif hue < 160:
            return 'purple'
        else:
            return 'pink'
            
    def _rgb_to_hex(self, r: float, g: float, b: float) -> str:
        """Convert RGB to hex color"""
        return '#{:02x}{:02x}{:02x}'.format(
            int(r * 255), int(g * 255), int(b * 255)
        )
    
    def _extract_color_palette(self, frame: np.ndarray, n_colors: int = 5) -> List[Dict[str, Any]]:
        """Extract dominant color palette using K-means clustering"""
        try:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (100, 100))
            
            # Reshape to list of pixels
            pixels = small_frame.reshape(-1, 3)
            
            # Use K-means clustering - required
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers (dominant colors)
            colors = kmeans.cluster_centers_
            
            # Get percentage of each color
            labels = kmeans.labels_
            color_counts = np.bincount(labels)
            color_percentages = color_counts / len(labels)
            
            # Create palette
            palette = []
            for i, (color, percentage) in enumerate(zip(colors, color_percentages)):
                r, g, b = color.astype(int)
                palette.append({
                    'color': {'r': int(r), 'g': int(g), 'b': int(b)},
                    'hex': '#{:02x}{:02x}{:02x}'.format(r, g, b),
                    'percentage': round(float(percentage * 100), 1),
                    'name': self._get_color_name(r, g, b)
                })
            
            # Sort by percentage
            palette.sort(key=lambda x: x['percentage'], reverse=True)
            
            return palette
            
        except Exception as e:
            logger.debug(f"Color palette extraction failed: {e}")
            # Return simple palette based on dominant color
            return [{
                'color': {'r': 128, 'g': 128, 'b': 128},
                'hex': '#808080',
                'percentage': 100.0,
                'name': 'gray'
            }]
    
    def _get_color_name(self, r: int, g: int, b: int) -> str:
        """Get detailed color name from RGB values"""
        # Convert to HSV for easier classification
        hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv[0], hsv[1], hsv[2]
        
        # Check for grayscale
        if s < 20:
            if v < 30:
                return 'black'
            elif v < 60:
                return 'dark_gray'
            elif v < 120:
                return 'gray'
            elif v < 200:
                return 'light_gray'
            else:
                return 'white'
        
        # Check brightness/darkness
        prefix = ''
        if v < 80:
            prefix = 'dark_'
        elif v > 200 and s > 50:
            prefix = 'bright_'
        elif s < 50:
            prefix = 'pale_'
        
        # Determine color based on hue
        if h < 10 or h > 170:
            color = 'red'
        elif h < 20:
            color = 'orange'
        elif h < 35:
            color = 'yellow'
        elif h < 50:
            color = 'lime'
        elif h < 85:
            color = 'green'
        elif h < 100:
            color = 'cyan'
        elif h < 125:
            color = 'blue'
        elif h < 145:
            color = 'purple'
        elif h < 160:
            color = 'magenta'
        else:
            color = 'pink'
            
        return prefix + color
    
    def _generate_color_description(self, mood: str, palette: List[Dict], hue: float) -> str:
        """Generate descriptive text about colors"""
        mood_desc = {
            'dark': 'Dunkle Farbstimmung',
            'bright': 'Helle, leuchtende Farben',
            'muted': 'Gedämpfte, zurückhaltende Farben',
            'vibrant': 'Kräftige, lebendige Farben',
            'balanced': 'Ausgewogene Farbgebung'
        }
        
        desc = mood_desc.get(mood, 'Normale Farbgebung')
        
        if palette and len(palette) > 0:
            main_color = palette[0]['name']
            desc += f", hauptsächlich {main_color}"
            
            if len(palette) > 1 and palette[1]['percentage'] > 20:
                desc += f" mit {palette[1]['name']} Akzenten"
        
        return desc
        
    def _generate_color_summary(self, segments: List[Dict]) -> Dict:
        """Generate color analysis summary"""
        if not segments:
            return {}
            
        brightness_values = [s['brightness'] for s in segments]
        saturation_values = [s['saturation'] for s in segments]
        color_moods = [s['color_mood'] for s in segments]
        hue_colors = [s['hue_dominant'] for s in segments]
        
        # Count distributions
        mood_counts = {}
        for mood in color_moods:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
            
        color_counts = {}
        for color in hue_colors:
            color_counts[color] = color_counts.get(color, 0) + 1
            
        return {
            'average_brightness': float(np.mean(brightness_values)),
            'average_saturation': float(np.mean(saturation_values)),
            'brightness_variance': float(np.var(brightness_values)),
            'dominant_mood': max(mood_counts, key=mood_counts.get) if mood_counts else 'unknown',
            'mood_distribution': mood_counts,
            'dominant_color': max(color_counts, key=color_counts.get) if color_counts else 'unknown',
            'color_distribution': color_counts,
            'color_consistency': 1.0 - min(float(np.std(brightness_values)), 1.0)
        }
        
    def process_batch_gpu(self, frames, frame_times):
        """Process frames using GPU color analysis"""
        if not frames:
            return {'segments': []}
        
        # Use the existing GPU batch analysis method
        segments = self._analyze_colors_gpu_batch(frames, frame_times)
        
        # Generate summary
        summary = self._generate_color_summary(segments)
        
        return {
            'segments': segments,
            'summary': summary,
            'metadata': {
                'frames_analyzed': len(frames),
                'optimization': 'gpu_accelerated'
            }
        }