#!/usr/bin/env python3
"""
Fixed Lightweight Visual Effects Detection
Uses standardized interface with analyze() method
"""

import cv2
import numpy as np
from typing import List, Dict, Any
from analyzers.standardized_base_analyzer import StandardizedBaseAnalyzer
import logging
from scipy import ndimage

logger = logging.getLogger(__name__)

class VisualEffectsLight(StandardizedBaseAnalyzer):
    """Fixed lightweight visual effects analyzer with standardized interface"""
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 5  # Every 0.17 seconds for dense effect detection
        self.prev_frame = None
        self.prev_hsv = None
        self.effect_history = []  # Track effects over time
        
    def load_model(self):
        """No models needed - only CV algorithms"""
        logger.info("[VisualEffects-Light] Ready - using CV algorithms only")
        
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process video and detect visual effects"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        segments = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % self.sample_rate == 0:
                timestamp = frame_count / fps
                effects = self._analyze_frame_effects(frame, timestamp)
                
                if effects:
                    segments.append({
                        'timestamp': timestamp,
                        'effects': effects,
                        'confidence': 0.8
                    })
                    
            frame_count += 1
            
        cap.release()
        
        logger.info(f"[VisualEffects-Light] Found {len(segments)} effect segments")
        return {'segments': segments}
    
    def _analyze_frame_effects(self, frame: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """Analyze visual effects in a single frame with DETAILED detection"""
        effects = {}
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Motion blur detection
        blur_level = self._detect_motion_blur(frame)
        if blur_level:
            effects['motion_blur'] = blur_level
            
        # Advanced color analysis
        color_effects = self._analyze_color_effects(frame, hsv)
        if color_effects:
            effects.update(color_effects)
            
        # Filter detection
        filter_type = self._detect_filters(frame, hsv)
        if filter_type:
            effects['filter'] = filter_type
            
        # Glitch effects
        glitch = self._detect_glitch_effects(frame)
        if glitch:
            effects['glitch'] = glitch
            
        # Lens effects
        lens_effect = self._detect_lens_effects(frame)
        if lens_effect:
            effects['lens'] = lens_effect
            
        # Transition detection
        if self.prev_frame is not None:
            transition = self._detect_transition(self.prev_frame, frame)
            if transition:
                effects['transition'] = transition
                
            # Speed ramping detection
            speed_effect = self._detect_speed_ramping(self.prev_frame, frame)
            if speed_effect:
                effects['speed_effect'] = speed_effect
                
            # Shake/stabilization effects
            shake = self._detect_camera_shake(self.prev_frame, frame)
            if shake:
                effects['camera_shake'] = shake
        
        # Text/graphic overlay detection (enhanced)
        overlay = self._detect_overlay_elements(frame)
        if overlay:
            effects['overlay'] = overlay
            
        # Zoom/scale effects
        if self.prev_frame is not None:
            zoom = self._detect_zoom_effect(self.prev_frame, frame)
            if zoom:
                effects['zoom'] = zoom
                
        # Split screen / multi-frame detection
        split = self._detect_split_screen(frame)
        if split:
            effects['split_screen'] = split
            
        # Green screen / chroma key detection
        chroma = self._detect_chroma_key(frame, hsv)
        if chroma:
            effects['chroma_key'] = chroma
                
        self.prev_frame = frame.copy()
        self.prev_hsv = hsv.copy()
        
        # Track effect continuity
        self.effect_history.append(effects)
        if len(self.effect_history) > 10:
            self.effect_history.pop(0)
        
        # Add timestamp for precise tracking
        if effects:
            effects['timestamp'] = timestamp
        
        return effects
    
    def _detect_motion_blur(self, frame: np.ndarray) -> str:
        """Detect motion blur using Laplacian variance"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian < 50:
            return 'heavy'
        elif laplacian < 100:
            return 'medium'
        elif laplacian < 200:
            return 'light'
        return None
    
    def _analyze_saturation(self, frame: np.ndarray) -> str:
        """Analyze color saturation"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].mean()
        
        if saturation < 30:
            return 'desaturated'
        elif saturation > 180:
            return 'oversaturated'
        return 'normal'
    
    def _analyze_brightness(self, frame: np.ndarray) -> str:
        """Analyze brightness levels"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        
        if brightness < 50:
            return 'very_dark'
        elif brightness < 100:
            return 'dark'
        elif brightness > 200:
            return 'bright'
        elif brightness > 240:
            return 'overexposed'
        return 'normal'
    
    def _detect_transition(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> str:
        """Detect transitions between frames"""
        # Calculate frame difference
        diff = cv2.absdiff(prev_frame, curr_frame)
        diff_mean = diff.mean()
        
        if diff_mean > 100:
            return 'hard_cut'
        elif diff_mean > 50:
            # Check for fade
            prev_bright = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).mean()
            curr_bright = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY).mean()
            
            if abs(prev_bright - curr_bright) > 50:
                if curr_bright < prev_bright:
                    return 'fade_out'
                else:
                    return 'fade_in'
            return 'dissolve'
            
        return None
    
    def _analyze_color_effects(self, frame: np.ndarray, hsv: np.ndarray) -> Dict[str, Any]:
        """Analyze advanced color effects"""
        effects = {}
        
        # Saturation analysis
        saturation = hsv[:, :, 1].mean()
        if saturation < 30:
            effects['saturation'] = 'black_and_white'
        elif saturation < 60:
            effects['saturation'] = 'desaturated'
        elif saturation > 180:
            effects['saturation'] = 'hypersaturated'
            
        # Brightness/Value analysis
        value = hsv[:, :, 2].mean()
        if value < 50:
            effects['brightness'] = 'very_dark'
        elif value > 200:
            effects['brightness'] = 'overexposed'
            
        # Color tone analysis
        hue = hsv[:, :, 0]
        hue_hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hue_hist)
        
        # Detect color grading
        if 15 < dominant_hue < 25:  # Orange tones
            effects['color_grading'] = 'warm_orange'
        elif 100 < dominant_hue < 130:  # Blue tones
            effects['color_grading'] = 'cool_blue'
        elif 160 < dominant_hue < 180:  # Pink/purple tones
            effects['color_grading'] = 'pink_aesthetic'
            
        return effects
    
    def _detect_filters(self, frame: np.ndarray, hsv: np.ndarray) -> str:
        """Detect common TikTok filters"""
        # Check for high contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Analyze histogram shape
        low_vals = hist[:50].sum()
        high_vals = hist[-50:].sum()
        mid_vals = hist[50:-50].sum()
        
        total = hist.sum()
        if (low_vals + high_vals) / total > 0.6:
            return 'high_contrast'
        
        # Check for vintage filter characteristics
        b, g, r = cv2.split(frame)
        if r.mean() > g.mean() * 1.2 and r.mean() > b.mean() * 1.3:
            return 'vintage_warm'
            
        # Check for beauty filter (skin smoothing)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (edges.shape[0] * edges.shape[1] * 255)
        if edge_density < 0.02:  # Very few edges = smoothed
            return 'beauty_filter'
            
        return None
    
    def _detect_speed_ramping(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> str:
        """Detect speed ramping effects"""
        # Calculate optical flow
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        avg_motion = magnitude.mean()
        
        # Track motion history
        if not hasattr(self, 'motion_history'):
            self.motion_history = []
        
        self.motion_history.append(avg_motion)
        if len(self.motion_history) > 5:
            self.motion_history.pop(0)
            
        # Detect sudden changes in motion
        if len(self.motion_history) >= 3:
            recent_avg = np.mean(self.motion_history[-3:])
            if recent_avg > 20:
                return 'slow_motion'
            elif recent_avg < 2 and max(self.motion_history) > 10:
                return 'freeze_frame'
                
        return None
    
    def _detect_glitch_effects(self, frame: np.ndarray) -> str:
        """Detect digital glitch effects"""
        # Check for RGB channel shifts
        b, g, r = cv2.split(frame)
        
        # Calculate channel offsets
        shift_threshold = 10
        h, w = frame.shape[:2]
        
        # Check horizontal shifts
        if np.mean(np.abs(r[:-shift_threshold] - r[shift_threshold:])) > 50:
            return 'rgb_shift'
            
        # Check for digital artifacts (sharp color blocks)
        edges = cv2.Canny(frame, 100, 200)
        edge_density = edges.sum() / (h * w * 255)
        
        if edge_density > 0.3:  # Lots of hard edges
            return 'digital_glitch'
            
        return None
    
    def _detect_lens_effects(self, frame: np.ndarray) -> str:
        """Detect lens-based effects"""
        h, w = frame.shape[:2]
        center = (w//2, h//2)
        
        # Check for vignette (darker edges)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        center_brightness = gray[h//2-50:h//2+50, w//2-50:w//2+50].mean()
        edge_brightness = np.mean([
            gray[:50, :].mean(),
            gray[-50:, :].mean(),
            gray[:, :50].mean(),
            gray[:, -50:].mean()
        ])
        
        if center_brightness > edge_brightness * 1.5:
            return 'vignette'
            
        # Check for fisheye/barrel distortion (simplified)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None and len(lines) < 5:  # Few straight lines = possible distortion
            return 'fisheye'
            
        return None
    
    def _detect_camera_shake(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> str:
        """Detect camera shake or stabilization"""
        # Use feature matching for shake detection
        orb = cv2.ORB_create(nfeatures=100)
        kp1, des1 = orb.detectAndCompute(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = orb.detectAndCompute(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY), None)
        
        if des1 is None or des2 is None:
            return None
            
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if len(matches) < 10:
            return None
            
        # Calculate average displacement
        displacements = []
        for match in matches[:20]:  # Use top 20 matches
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            displacement = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            displacements.append(displacement)
            
        avg_displacement = np.mean(displacements)
        
        if avg_displacement > 20:
            return 'heavy_shake'
        elif avg_displacement > 10:
            return 'moderate_shake'
        elif avg_displacement > 5:
            return 'slight_shake'
            
        return None
    
    def _detect_split_screen(self, frame: np.ndarray) -> str:
        """Detect split screen effects"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect vertical lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=frame.shape[0]*0.8, maxLineGap=10)
        
        if lines is not None:
            # Check for vertical lines near center
            h, w = frame.shape[:2]
            center_x = w // 2
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < 5:  # Vertical line
                    if abs(x1 - center_x) < 50:  # Near center
                        return 'vertical_split'
                        
        # Check for horizontal split
        lines_h = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=frame.shape[1]*0.8, maxLineGap=10)
        if lines_h is not None:
            center_y = frame.shape[0] // 2
            for line in lines_h:
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) < 5:  # Horizontal line
                    if abs(y1 - center_y) < 50:  # Near center
                        return 'horizontal_split'
                        
        return None
    
    def _detect_chroma_key(self, frame: np.ndarray, hsv: np.ndarray) -> str:
        """Detect green screen / chroma key usage"""
        # Check for dominant green or blue regions
        h, s, v = cv2.split(hsv)
        
        # Green screen detection (Hue ~60-80)
        green_mask = cv2.inRange(h, 50, 80)
        green_ratio = cv2.countNonZero(green_mask) / (h.shape[0] * h.shape[1])
        
        if green_ratio > 0.3:  # 30% of frame is green
            # Check if it's uniform (likely green screen)
            green_regions = hsv[green_mask > 0]
            if len(green_regions) > 0:
                sat_std = np.std(green_regions[:, 1])
                if sat_std < 30:  # Uniform saturation
                    return 'green_screen'
                    
        # Blue screen detection (Hue ~100-120)
        blue_mask = cv2.inRange(h, 100, 130)
        blue_ratio = cv2.countNonZero(blue_mask) / (h.shape[0] * h.shape[1])
        
        if blue_ratio > 0.3:
            blue_regions = hsv[blue_mask > 0]
            if len(blue_regions) > 0:
                sat_std = np.std(blue_regions[:, 1])
                if sat_std < 30:
                    return 'blue_screen'
                    
        return None
    
    def _detect_overlay_elements(self, frame: np.ndarray) -> str:
        """Detect graphic overlay elements with detailed classification"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect high contrast regions (typical for text/graphics)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(binary)
        total_pixels = frame.shape[0] * frame.shape[1]
        
        # Check different regions for specific overlay types
        h, w = frame.shape[:2]
        
        # Top region (titles, watermarks)
        top_region = binary[0:int(h*0.2), :]
        top_whites = cv2.countNonZero(top_region)
        
        # Bottom region (captions, credits)
        bottom_region = binary[int(h*0.8):h, :]
        bottom_whites = cv2.countNonZero(bottom_region)
        
        # Center region (main text)
        center_region = binary[int(h*0.3):int(h*0.7), int(w*0.2):int(w*0.8)]
        center_whites = cv2.countNonZero(center_region)
        
        # Corners (UI elements, logos)
        corners = [
            binary[0:100, 0:100],      # Top-left
            binary[0:100, w-100:w],    # Top-right
            binary[h-100:h, 0:100],    # Bottom-left
            binary[h-100:h, w-100:w]   # Bottom-right
        ]
        corner_whites = sum(cv2.countNonZero(corner) for corner in corners)
        
        # Classify overlay type
        if top_whites > 10000:
            return 'title_text'
        elif bottom_whites > 10000:
            return 'caption_text'
        elif center_whites > 20000:
            return 'center_text'
        elif corner_whites > 5000:
            return 'ui_elements'
        elif white_pixels / total_pixels > 0.1:  # 10% of frame
            return 'graphic_overlay'
            
        return None
    
    def _detect_zoom_effect(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> str:
        """Detect zoom in/out effects"""
        # Simplified zoom detection using template matching
        h, w = prev_frame.shape[:2]
        
        # Get center region of previous frame
        center_size = 0.6
        y1, y2 = int(h * (1-center_size)/2), int(h * (1+center_size)/2)
        x1, x2 = int(w * (1-center_size)/2), int(w * (1+center_size)/2)
        
        prev_center = prev_frame[y1:y2, x1:x2]
        
        # Resize to match current frame for zoom detection
        prev_center_large = cv2.resize(prev_center, (w, h))
        prev_center_small = cv2.resize(prev_frame, (x2-x1, y2-y1))
        
        # Check similarity
        diff_zoom_in = cv2.absdiff(prev_center_large, curr_frame).mean()
        diff_zoom_out = cv2.absdiff(prev_center_small, curr_frame[y1:y2, x1:x2]).mean()
        
        if diff_zoom_in < 30:
            return 'zoom_in'
        elif diff_zoom_out < 30:
            return 'zoom_out'
            
        return None