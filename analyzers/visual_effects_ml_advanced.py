#!/usr/bin/env python3
"""
Advanced Visual Effects Detection using ML Models
Detects real video effects: transitions, filters, speed changes, color grading, etc.
Uses a combination of ML models and advanced CV algorithms
"""

# GPU Forcing
import torch
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from analyzers.base_analyzer import GPUBatchAnalyzer
import logging
from collections import deque
import torchvision.transforms as transforms
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class VisualEffectsMLAdvanced(GPUBatchAnalyzer):
    """Advanced ML-based visual effects detection"""
    
    def __init__(self):
        super().__init__(batch_size=8)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models_loaded = False
        
        # Effect detection thresholds
        self.transition_threshold = 0.7
        self.filter_threshold = 0.6
        
        # Frame sampling for temporal effects
        self.temporal_window = 5  # frames
        self.sample_rate = 15  # Sample every 0.5 seconds at 30fps
        
        # History tracking
        self.frame_history = deque(maxlen=self.temporal_window)
        self.effect_history = deque(maxlen=30)  # Track effects over 10 seconds
        
        # Models
        self.style_classifier = None
        self.aesthetic_scorer = None
        self.transform = None
        
        logger.info("[VisualEffectsML] Initialized with advanced ML detection")
    
    def _load_model_impl(self):
        """Load ML models for effect detection"""
        if self.models_loaded:
            return
            
        try:
            logger.info("[VisualEffectsML] Loading ML models...")
            
            # Load style transfer detection model
            self.style_classifier = pipeline(
                "image-classification",
                model="cafeai/cafe_style",  # Detects artistic styles
                device=0 if self.device == 'cuda' else -1
            )
            
            # Load aesthetic scoring model
            self.aesthetic_processor = AutoImageProcessor.from_pretrained("cafeai/cafe_aesthetic")
            self.aesthetic_model = AutoModelForImageClassification.from_pretrained(
                "cafeai/cafe_aesthetic"
            ).to(self.device)
            self.aesthetic_model.eval()
            
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.models_loaded = True
            logger.info("[VisualEffectsML] ✅ Models loaded successfully")
            
        except Exception as e:
            logger.error(f"[VisualEffectsML] Failed to load models: {e}")
            # Fallback to CV-only mode
            self.models_loaded = False
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Analyze video for visual effects"""
        logger.info(f"[VisualEffectsML] Starting analysis of {video_path}")
        
        # Load models if needed
        if not self.models_loaded:
            self._load_model_impl()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'segments': [], 'error': 'Could not open video'}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        segments = []
        frame_count = 0
        last_transition_time = -2.0  # Prevent duplicate transitions
        
        # Process video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.sample_rate == 0:
                timestamp = frame_count / fps
                
                # Add to frame history
                self.frame_history.append({
                    'frame': frame.copy(),
                    'timestamp': timestamp,
                    'frame_idx': frame_count
                })
                
                # Analyze effects when we have enough history
                if len(self.frame_history) >= 3:
                    effects = self._analyze_temporal_effects()
                    
                    if effects and (not effects.get('transition') or 
                                  timestamp - last_transition_time > 1.0):
                        
                        if effects.get('transition'):
                            last_transition_time = timestamp
                        
                        segment = {
                            'timestamp': round(timestamp, 2),
                            'start_time': round(timestamp - 0.5, 2),
                            'end_time': round(timestamp + 0.5, 2),
                            'effects': effects,
                            'confidence': self._calculate_confidence(effects),
                            'description': self._generate_description(effects)
                        }
                        segments.append(segment)
                        
                        # Track effect continuity
                        self.effect_history.append(effects)
            
            frame_count += 1
        
        cap.release()
        
        # Analyze overall video style
        overall_analysis = self._analyze_overall_style()
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        result = {
            'segments': segments,
            'summary': overall_analysis,
            'metadata': {
                'total_effects': len(segments),
                'duration': round(duration, 2),
                'analysis_method': 'ml_based' if self.models_loaded else 'cv_based'
            }
        }
        
        logger.info(f"[VisualEffectsML] Found {len(segments)} effect segments")
        return result
    
    def _analyze_temporal_effects(self) -> Dict[str, Any]:
        """Analyze effects using temporal context"""
        effects = {}
        
        if len(self.frame_history) < 2:
            return effects
        
        curr_data = self.frame_history[-1]
        prev_data = self.frame_history[-2]
        
        curr_frame = curr_data['frame']
        prev_frame = prev_data['frame']
        timestamp = curr_data['timestamp']
        
        # 1. Transition Detection
        transition = self._detect_transition_ml(prev_frame, curr_frame)
        if transition:
            effects['transition'] = transition
        
        # 2. Speed Effects (need more frames)
        if len(self.frame_history) >= self.temporal_window:
            speed_effect = self._detect_speed_effects()
            if speed_effect:
                effects['speed'] = speed_effect
        
        # 3. Filter/Style Detection
        filter_style = self._detect_filter_style(curr_frame)
        if filter_style:
            effects['filter'] = filter_style
        
        # 4. Color Grading
        color_grading = self._detect_color_grading(curr_frame)
        if color_grading:
            effects['color_grading'] = color_grading
        
        # 5. Visual Effects (blur, glow, etc.)
        visual_fx = self._detect_visual_effects(curr_frame, prev_frame)
        if visual_fx:
            effects.update(visual_fx)
        
        # 6. Camera Effects
        camera_fx = self._detect_camera_effects(curr_frame, prev_frame)
        if camera_fx:
            effects['camera_effect'] = camera_fx
        
        return effects
    
    def _detect_transition_ml(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Optional[Dict]:
        """Detect transitions using ML and CV"""
        # Basic frame difference
        diff = cv2.absdiff(prev_frame, curr_frame)
        diff_score = np.mean(diff) / 255.0
        
        # Histogram comparison
        hist_score = self._compare_histograms(prev_frame, curr_frame)
        
        # Optical flow magnitude
        flow_mag = self._calculate_optical_flow_magnitude(prev_frame, curr_frame)
        
        # ML-based style change detection
        style_change = 0.0
        if self.models_loaded:
            style_change = self._detect_style_change(prev_frame, curr_frame)
        
        # Combine scores
        transition_score = (diff_score * 0.3 + hist_score * 0.2 + 
                          flow_mag * 0.2 + style_change * 0.3)
        
        if transition_score > self.transition_threshold:
            transition_type = self._classify_transition(
                prev_frame, curr_frame, diff_score, flow_mag
            )
            return {
                'type': transition_type,
                'confidence': float(min(transition_score, 1.0)),
                'style_change': style_change > 0.5
            }
        
        return None
    
    def _detect_filter_style(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect artistic filters and styles"""
        if not self.models_loaded:
            return self._detect_filter_cv_only(frame)
        
        try:
            # Use style classifier
            results = self.style_classifier(frame)
            
            # Filter results
            high_conf_styles = [r for r in results if r['score'] > self.filter_threshold]
            
            if high_conf_styles:
                top_style = high_conf_styles[0]
                
                # Map model labels to user-friendly names
                style_map = {
                    'vintage': 'Vintage/Retro',
                    'cinematic': 'Cinematic',
                    'anime': 'Anime/Cartoon',
                    'aesthetic': 'Aesthetic',
                    'dark': 'Dark/Moody',
                    'bright': 'Bright/Vibrant',
                    'minimal': 'Minimalist',
                    'film': 'Film Look'
                }
                
                style_name = style_map.get(top_style['label'].lower(), top_style['label'])
                
                # Get aesthetic score
                aesthetic_score = self._get_aesthetic_score(frame)
                
                return {
                    'style': style_name,
                    'confidence': float(top_style['score']),
                    'aesthetic_score': aesthetic_score,
                    'characteristics': self._analyze_style_characteristics(frame)
                }
        
        except Exception as e:
            logger.debug(f"Style detection failed: {e}")
        
        return self._detect_filter_cv_only(frame)
    
    def _detect_filter_cv_only(self, frame: np.ndarray) -> Optional[Dict]:
        """Fallback filter detection using only CV"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Analyze color characteristics
        saturation = np.mean(hsv[:, :, 1])
        value = np.mean(hsv[:, :, 2])
        
        # Edge detection for sharpness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        filter_type = None
        confidence = 0.0
        
        if saturation < 30:
            filter_type = "Black & White"
            confidence = 0.9
        elif saturation < 60:
            filter_type = "Desaturated/Faded"
            confidence = 0.8
        elif saturation > 180:
            filter_type = "Hypersaturated"
            confidence = 0.85
        elif edge_density < 0.02:
            filter_type = "Beauty/Smooth"
            confidence = 0.7
        elif value < 60:
            filter_type = "Dark/Moody"
            confidence = 0.75
        elif value > 200:
            filter_type = "Bright/Overexposed"
            confidence = 0.8
        
        if filter_type:
            return {
                'style': filter_type,
                'confidence': confidence,
                'characteristics': {
                    'saturation': float(saturation),
                    'brightness': float(value),
                    'sharpness': float(edge_density)
                }
            }
        
        return None
    
    def _detect_color_grading(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect color grading effects"""
        # Analyze color channels
        b, g, r = cv2.split(frame)
        
        # Channel means
        b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
        
        # Color cast detection
        color_cast = None
        strength = 0.0
        
        if r_mean > g_mean * 1.2 and r_mean > b_mean * 1.2:
            color_cast = "Warm/Orange"
            strength = (r_mean - max(g_mean, b_mean)) / 255.0
        elif b_mean > r_mean * 1.2 and b_mean > g_mean * 1.2:
            color_cast = "Cool/Blue"
            strength = (b_mean - max(r_mean, g_mean)) / 255.0
        elif g_mean > r_mean * 1.1 and g_mean > b_mean * 1.1:
            color_cast = "Green/Teal"
            strength = (g_mean - max(r_mean, b_mean)) / 255.0
        
        # Contrast analysis
        contrast = np.std(frame) / 255.0
        
        # Shadows/Highlights
        dark_pixels = np.sum(frame < 50) / frame.size
        bright_pixels = np.sum(frame > 200) / frame.size
        
        if color_cast or contrast > 0.3 or dark_pixels > 0.3 or bright_pixels > 0.3:
            return {
                'color_cast': color_cast,
                'strength': float(strength),
                'contrast': 'high' if contrast > 0.3 else 'normal',
                'shadows': 'crushed' if dark_pixels > 0.3 else 'normal',
                'highlights': 'blown' if bright_pixels > 0.3 else 'normal'
            }
        
        return None
    
    def _detect_speed_effects(self) -> Optional[Dict]:
        """Detect speed ramping, slow motion, time lapse"""
        if len(self.frame_history) < self.temporal_window:
            return None
        
        # Calculate motion between consecutive frames
        motions = []
        for i in range(1, len(self.frame_history)):
            prev = self.frame_history[i-1]['frame']
            curr = self.frame_history[i]['frame']
            
            # Optical flow
            flow_mag = self._calculate_optical_flow_magnitude(prev, curr)
            motions.append(flow_mag)
        
        # Analyze motion pattern
        avg_motion = np.mean(motions)
        motion_variance = np.var(motions)
        
        # Detect patterns
        if avg_motion < 0.02 and len(set(motions)) == 1:
            return {'type': 'freeze_frame', 'confidence': 0.95}
        elif avg_motion < 0.05:
            return {'type': 'slow_motion', 'confidence': 0.8}
        elif avg_motion > 0.3:
            return {'type': 'time_lapse', 'confidence': 0.85}
        elif motion_variance > 0.1:
            # Check for ramping
            if motions[-1] > motions[0] * 2:
                return {'type': 'speed_ramp_up', 'confidence': 0.7}
            elif motions[-1] < motions[0] * 0.5:
                return {'type': 'speed_ramp_down', 'confidence': 0.7}
        
        return None
    
    def _detect_visual_effects(self, curr_frame: np.ndarray, 
                             prev_frame: np.ndarray) -> Dict[str, Any]:
        """Detect various visual effects"""
        effects = {}
        
        # Motion blur
        blur_score = self._detect_motion_blur(curr_frame)
        if blur_score > 0.7:
            effects['motion_blur'] = {
                'intensity': 'heavy' if blur_score > 0.85 else 'medium',
                'score': float(blur_score)
            }
        
        # Lens flare / Light leaks
        flare = self._detect_lens_flare(curr_frame)
        if flare:
            effects['lens_flare'] = flare
        
        # Glitch effects
        glitch = self._detect_glitch_effect(curr_frame, prev_frame)
        if glitch:
            effects['glitch'] = glitch
        
        # Vignette
        vignette = self._detect_vignette(curr_frame)
        if vignette:
            effects['vignette'] = vignette
        
        # Chromatic aberration
        chromatic = self._detect_chromatic_aberration(curr_frame)
        if chromatic:
            effects['chromatic_aberration'] = chromatic
        
        return effects
    
    def _detect_camera_effects(self, curr_frame: np.ndarray, 
                             prev_frame: np.ndarray) -> Optional[str]:
        """Detect camera movement effects"""
        # Calculate homography
        try:
            # Feature matching
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(
                cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), None
            )
            kp2, des2 = orb.detectAndCompute(
                cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY), None
            )
            
            if des1 is not None and des2 is not None and len(kp1) > 10 and len(kp2) > 10:
                # Match features
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                
                if len(matches) > 10:
                    # Calculate transformation
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    
                    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if M is not None:
                        # Analyze transformation
                        scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
                        rotation = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
                        translation = np.sqrt(M[0, 2]**2 + M[1, 2]**2)
                        
                        if scale > 1.1:
                            return "zoom_in"
                        elif scale < 0.9:
                            return "zoom_out"
                        elif abs(rotation) > 5:
                            return "rotation"
                        elif translation > 20:
                            return "pan" if abs(M[0, 2]) > abs(M[1, 2]) else "tilt"
                        
        except Exception as e:
            logger.debug(f"Camera effect detection failed: {e}")
        
        return None
    
    def _calculate_optical_flow_magnitude(self, prev_frame: np.ndarray, 
                                        curr_frame: np.ndarray) -> float:
        """Calculate average optical flow magnitude"""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Resize for faster computation
        scale = 0.5
        prev_small = cv2.resize(prev_gray, None, fx=scale, fy=scale)
        curr_small = cv2.resize(curr_gray, None, fx=scale, fy=scale)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_small, curr_small, None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return float(np.mean(magnitude))
    
    def _compare_histograms(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compare color histograms between frames"""
        hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
        
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        return 1.0 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def _detect_style_change(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Detect style change using ML model"""
        try:
            # Get style classifications for both frames
            prev_styles = self.style_classifier(prev_frame)
            curr_styles = self.style_classifier(curr_frame)
            
            # Compare top styles
            if prev_styles and curr_styles:
                prev_top = prev_styles[0]['label']
                curr_top = curr_styles[0]['label']
                
                if prev_top != curr_top:
                    return 1.0
                else:
                    # Compare confidence changes
                    conf_diff = abs(prev_styles[0]['score'] - curr_styles[0]['score'])
                    return conf_diff
                    
        except Exception as e:
            logger.debug(f"Style change detection failed: {e}")
        
        return 0.0
    
    def _classify_transition(self, prev_frame: np.ndarray, curr_frame: np.ndarray,
                           diff_score: float, flow_mag: float) -> str:
        """Classify the type of transition"""
        # Brightness analysis
        prev_brightness = np.mean(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
        curr_brightness = np.mean(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY))
        brightness_change = curr_brightness - prev_brightness
        
        if diff_score > 0.8:
            return "hard_cut"
        elif abs(brightness_change) > 100:
            if brightness_change > 0:
                return "fade_in"
            else:
                return "fade_out"
        elif flow_mag > 0.3:
            # Directional analysis
            return "swipe"
        elif diff_score > 0.5:
            return "dissolve"
        else:
            return "smooth_transition"
    
    def _get_aesthetic_score(self, frame: np.ndarray) -> float:
        """Get aesthetic quality score"""
        try:
            # Preprocess for aesthetic model
            inputs = self.aesthetic_processor(images=frame, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.aesthetic_model(**inputs)
                logits = outputs.logits
                
                # Convert to probability
                probs = torch.nn.functional.softmax(logits, dim=-1)
                # Aesthetic score (assuming binary classification)
                aesthetic_score = float(probs[0, 1].cpu().item())
                
            return aesthetic_score
            
        except Exception as e:
            logger.debug(f"Aesthetic scoring failed: {e}")
            return 0.5
    
    def _analyze_style_characteristics(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze detailed style characteristics"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        return {
            'saturation': float(np.mean(hsv[:, :, 1])),
            'brightness': float(np.mean(hsv[:, :, 2])),
            'contrast': float(np.std(frame)),
            'sharpness': float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        }
    
    def _detect_motion_blur(self, frame: np.ndarray) -> float:
        """Detect motion blur intensity"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range (inverse - low variance = high blur)
        blur_score = 1.0 - min(laplacian_var / 500.0, 1.0)
        return blur_score
    
    def _detect_lens_flare(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect lens flare or light leak effects"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Look for bright, low-saturation areas (typical of lens flares)
        bright_mask = hsv[:, :, 2] > 240
        low_sat_mask = hsv[:, :, 1] < 30
        flare_mask = bright_mask & low_sat_mask
        
        flare_pixels = np.sum(flare_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        flare_ratio = flare_pixels / total_pixels
        
        if flare_ratio > 0.05:  # More than 5% of image
            # Find flare location
            y_coords, x_coords = np.where(flare_mask)
            if len(x_coords) > 0:
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                
                return {
                    'intensity': 'strong' if flare_ratio > 0.15 else 'subtle',
                    'coverage': float(flare_ratio),
                    'position': f"{center_x},{center_y}"
                }
        
        return None
    
    def _detect_glitch_effect(self, curr_frame: np.ndarray, 
                            prev_frame: np.ndarray) -> Optional[Dict]:
        """Detect digital glitch effects"""
        # Look for horizontal bands (common in glitch effects)
        diff = cv2.absdiff(curr_frame, prev_frame)
        
        # Sum differences along horizontal lines
        horizontal_diffs = np.mean(diff, axis=(1, 2))
        
        # Look for sharp spikes in horizontal differences
        max_diff = np.max(horizontal_diffs)
        mean_diff = np.mean(horizontal_diffs)
        
        if max_diff > mean_diff * 5:  # Sharp spike
            # Count discontinuous lines
            glitch_lines = np.sum(horizontal_diffs > mean_diff * 3)
            
            if glitch_lines > 10:
                return {
                    'type': 'digital_glitch',
                    'intensity': 'strong' if glitch_lines > 50 else 'subtle',
                    'lines_affected': int(glitch_lines)
                }
        
        # Check for color channel shifts
        b, g, r = cv2.split(curr_frame)
        shift_threshold = 10
        
        # Calculate channel misalignment
        bg_shift = np.mean(np.abs(b.astype(float) - g.astype(float)))
        gr_shift = np.mean(np.abs(g.astype(float) - r.astype(float)))
        
        if bg_shift > shift_threshold or gr_shift > shift_threshold:
            return {
                'type': 'rgb_shift',
                'intensity': float(max(bg_shift, gr_shift) / 255.0)
            }
        
        return None
    
    def _detect_vignette(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect vignette effect"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Sample center and corners
        center_region = gray[h//3:2*h//3, w//3:2*w//3]
        corner_regions = [
            gray[:h//4, :w//4],           # top-left
            gray[:h//4, -w//4:],          # top-right
            gray[-h//4:, :w//4],          # bottom-left
            gray[-h//4:, -w//4:]          # bottom-right
        ]
        
        center_brightness = np.mean(center_region)
        corner_brightness = np.mean([np.mean(corner) for corner in corner_regions])
        
        brightness_ratio = corner_brightness / (center_brightness + 1e-6)
        
        if brightness_ratio < 0.7:  # Corners significantly darker
            return {
                'strength': 'strong' if brightness_ratio < 0.5 else 'subtle',
                'darkness_ratio': float(1.0 - brightness_ratio)
            }
        
        return None
    
    def _detect_chromatic_aberration(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect chromatic aberration (color fringing)"""
        # Look at edges for color separation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to get edge regions
        kernel = np.ones((3, 3), np.uint8)
        edge_regions = cv2.dilate(edges, kernel, iterations=2)
        
        # Check color variance at edges
        b, g, r = cv2.split(frame)
        
        edge_pixels = edge_regions > 0
        if np.sum(edge_pixels) > 100:
            # Calculate color channel differences at edges
            bg_diff = np.mean(np.abs(b[edge_pixels].astype(float) - g[edge_pixels].astype(float)))
            gr_diff = np.mean(np.abs(g[edge_pixels].astype(float) - r[edge_pixels].astype(float)))
            
            max_diff = max(bg_diff, gr_diff)
            
            if max_diff > 15:  # Significant color separation
                return {
                    'intensity': 'strong' if max_diff > 25 else 'subtle',
                    'color_shift': float(max_diff / 255.0)
                }
        
        return None
    
    def _calculate_confidence(self, effects: Dict[str, Any]) -> float:
        """Calculate overall confidence for detected effects"""
        if not effects:
            return 0.0
        
        # Different weights for different effect types
        weights = {
            'transition': 0.3,
            'filter': 0.25,
            'speed': 0.2,
            'color_grading': 0.15,
            'motion_blur': 0.1,
            'camera_effect': 0.15,
            'lens_flare': 0.1,
            'glitch': 0.1,
            'vignette': 0.05,
            'chromatic_aberration': 0.05
        }
        
        total_confidence = 0.0
        total_weight = 0.0
        
        for effect_type, weight in weights.items():
            if effect_type in effects:
                effect_data = effects[effect_type]
                
                # Get confidence from effect data
                if isinstance(effect_data, dict) and 'confidence' in effect_data:
                    conf = effect_data['confidence']
                elif isinstance(effect_data, dict) and 'score' in effect_data:
                    conf = effect_data['score']
                else:
                    conf = 0.8  # Default confidence
                
                total_confidence += conf * weight
                total_weight += weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.5
    
    def _generate_description(self, effects: Dict[str, Any]) -> str:
        """Generate human-readable description of effects"""
        descriptions = []
        
        if 'transition' in effects:
            trans = effects['transition']
            descriptions.append(f"{trans['type'].replace('_', ' ').title()} transition")
        
        if 'filter' in effects:
            filt = effects['filter']
            descriptions.append(f"{filt['style']} filter applied")
        
        if 'speed' in effects:
            speed = effects['speed']
            descriptions.append(f"{speed['type'].replace('_', ' ').title()} effect")
        
        if 'color_grading' in effects:
            grading = effects['color_grading']
            if grading.get('color_cast'):
                descriptions.append(f"{grading['color_cast']} color grading")
        
        if 'camera_effect' in effects:
            descriptions.append(f"Camera {effects['camera_effect']}")
        
        if 'motion_blur' in effects:
            blur = effects['motion_blur']
            descriptions.append(f"{blur['intensity'].title()} motion blur")
        
        if 'lens_flare' in effects:
            descriptions.append("Lens flare effect")
        
        if 'glitch' in effects:
            descriptions.append("Digital glitch effect")
        
        if 'vignette' in effects:
            vig = effects['vignette']
            descriptions.append(f"{vig['strength'].title()} vignette")
        
        if descriptions:
            return ", ".join(descriptions)
        else:
            return "No significant effects detected"
    
    def _analyze_overall_style(self) -> Dict[str, Any]:
        """Analyze overall video style from effect history"""
        if not self.effect_history:
            return {
                'primary_style': 'natural',
                'effect_density': 0.0,
                'consistency': 1.0
            }
        
        # Count effect types
        effect_counts = {}
        filter_styles = []
        transitions = []
        
        for effects in self.effect_history:
            for effect_type, effect_data in effects.items():
                effect_counts[effect_type] = effect_counts.get(effect_type, 0) + 1
                
                if effect_type == 'filter' and isinstance(effect_data, dict):
                    filter_styles.append(effect_data.get('style', 'unknown'))
                elif effect_type == 'transition' and isinstance(effect_data, dict):
                    transitions.append(effect_data.get('type', 'unknown'))
        
        # Determine primary style
        if filter_styles:
            from collections import Counter
            style_counter = Counter(filter_styles)
            primary_style = style_counter.most_common(1)[0][0]
        else:
            primary_style = 'natural'
        
        # Calculate metrics
        total_segments = len(self.effect_history)
        segments_with_effects = sum(1 for e in self.effect_history if e)
        effect_density = segments_with_effects / total_segments if total_segments > 0 else 0
        
        # Style consistency (how often the same style appears)
        if filter_styles and len(filter_styles) > 1:
            style_counts = Counter(filter_styles)
            most_common_count = style_counts.most_common(1)[0][1]
            consistency = most_common_count / len(filter_styles)
        else:
            consistency = 1.0
        
        return {
            'primary_style': primary_style,
            'effect_density': float(effect_density),
            'consistency': float(consistency),
            'total_effects': sum(effect_counts.values()),
            'effect_types': list(effect_counts.keys()),
            'transition_count': len(transitions),
            'unique_transitions': list(set(transitions)) if transitions else [],
            'editing_pace': 'fast' if len(transitions) > 10 else 'moderate' if len(transitions) > 5 else 'slow'
        }
    
    def process_batch_gpu(self, frames: List[np.ndarray], 
                         frame_times: List[float]) -> Dict[str, Any]:
        """Process batch of frames on GPU"""
        # This analyzer needs temporal context, so we process sequentially
        # but we can still use GPU for individual operations
        segments = []
        
        for i, (frame, timestamp) in enumerate(zip(frames, frame_times)):
            self.frame_history.append({
                'frame': frame,
                'timestamp': timestamp,
                'frame_idx': i
            })
            
            if len(self.frame_history) >= 3:
                effects = self._analyze_temporal_effects()
                
                if effects:
                    segment = {
                        'timestamp': round(timestamp, 2),
                        'effects': effects,
                        'confidence': self._calculate_confidence(effects),
                        'description': self._generate_description(effects)
                    }
                    segments.append(segment)
                    self.effect_history.append(effects)
        
        return {'segments': segments}