#!/usr/bin/env python3
"""
Camera Analysis Fixed - Complete Implementation
Detects camera movements, shot types, stability, and zoom levels
Essential for video reconstruction
"""

# FFmpeg pthread fix
import os
os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"
import cv2
import numpy as np
import torch
# Force GPU usage
if torch.cuda.is_available():
    torch.set_default_device('cuda')
    torch.backends.cudnn.benchmark = True
from typing import List, Dict, Any, Tuple, Optional
from analyzers.base_analyzer import GPUBatchAnalyzer
import logging

logger = logging.getLogger(__name__)

class GPUBatchCameraAnalysisFixed(GPUBatchAnalyzer):
    def __init__(self):
        super().__init__(batch_size=64)  # Increased batch size for faster processing
        self.device = torch.device('cuda:0')  # Force GPU
        self.optical_flow = None
        self.previous_frame = None
        self.movement_threshold = 0.3  # More sensitive to detect subtle movements
        self.zoom_threshold = 0.01  # More sensitive zoom detection
        self.handheld_threshold = 5.0  # Threshold for handheld detection
        print(f"[CameraAnalysis-Fixed] Initializing with enhanced movement detection on {self.device}")
        self.sample_rate = 45  # Optimized: every 1.5 seconds instead of every second
    def _load_model_impl(self):
        """Initialize optical flow calculator"""
        try:
            # Use Farneback optical flow (CPU-based but fast)
            self.optical_flow_params = {
                'pyr_scale': 0.5,
                'levels': 3,
                'winsize': 15,
                'iterations': 3,
                'poly_n': 5,
                'poly_sigma': 1.2,
                'flags': 0
            }
            
            # Initialize face cascade once
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            print("✅ Optical flow and face detection initialized for camera movement detection")
        except Exception as e:
            logger.error(f"Failed to initialize optical flow: {e}")
            raise
    
    def calculate_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        """Calculate sparse optical flow using Lucas-Kanade (much faster)"""
        # Downsample for faster processing
        scale = 0.25  # Even smaller for speed
        prev_small = cv2.resize(prev_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(curr_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        # Detect features in the first frame
        feature_params = dict(
            maxCorners=50,  # Fewer features for speed
            qualityLevel=0.01,  # Lower quality threshold to detect more movement
            minDistance=10,
            blockSize=7
        )
        
        # Lucas-Kanade parameters
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Find features
        p0 = cv2.goodFeaturesToTrack(prev_small, mask=None, **feature_params)
        
        if p0 is None or len(p0) < 10:
            # Return zero flow if not enough features
            return np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_small, curr_small, p0, None, **lk_params)
        
        # Select good points
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            # Convert sparse flow to dense representation
            flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)
            
            if len(good_new) > 5:
                # Calculate average flow
                avg_flow = np.mean(good_new - good_old, axis=0) / scale
                flow[:, :, 0] = avg_flow[0]
                flow[:, :, 1] = avg_flow[1]
            
            return flow
        else:
            return np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)
    
    def analyze_camera_movement(self, flow: np.ndarray) -> Dict[str, Any]:
        """Analyze optical flow to detect camera movement"""
        # Calculate average flow vectors
        avg_flow_x = np.mean(flow[:, :, 0])
        avg_flow_y = np.mean(flow[:, :, 1])
        
        # Calculate flow magnitude
        magnitude = np.sqrt(avg_flow_x**2 + avg_flow_y**2)
        
        # Calculate flow variance for handheld detection
        flow_variance = np.var(flow[:, :, 0]) + np.var(flow[:, :, 1])
        
        # Determine movement type
        movement = {
            'type': 'static',
            'direction': None,
            'speed': float(magnitude),
            'confidence': 0.0,
            'is_handheld': False
        }
        
        # Check for handheld camera shake
        if flow_variance > self.handheld_threshold:
            movement['is_handheld'] = True
        
        if magnitude > self.movement_threshold:
            # Determine primary movement direction
            angle = np.arctan2(avg_flow_y, avg_flow_x) * 180 / np.pi
            
            # Horizontal movement (pan)
            if -45 <= angle <= 45:
                movement['type'] = 'pan_right'
                movement['direction'] = 'horizontal'
            elif angle >= 135 or angle <= -135:
                movement['type'] = 'pan_left'
                movement['direction'] = 'horizontal'
            # Vertical movement (tilt)
            elif 45 < angle < 135:
                movement['type'] = 'tilt_down'
                movement['direction'] = 'vertical'
            else:  # -135 < angle < -45
                movement['type'] = 'tilt_up'
                movement['direction'] = 'vertical'
            
            # Add handheld modifier if detected
            if movement['is_handheld']:
                movement['type'] = f"{movement['type']}_handheld"
            
            movement['confidence'] = min(1.0, magnitude / 5.0)  # More sensitive confidence
        
        return movement
    
    def detect_zoom(self, flow: np.ndarray, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Detect zoom in/out based on radial flow pattern"""
        h, w = frame_shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Downsample flow for faster processing
        scale = 0.25
        flow_small = cv2.resize(flow, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h_small, w_small = flow_small.shape[:2]
        
        # Create coordinate grids for small size
        y, x = np.mgrid[0:h_small, 0:w_small]
        center_x_small, center_y_small = w_small // 2, h_small // 2
        
        # Calculate radial vectors from center
        dx = x - center_x_small
        dy = y - center_y_small
        
        # Normalize radial vectors
        radial_magnitude = np.sqrt(dx**2 + dy**2) + 1e-6
        radial_x = dx / radial_magnitude
        radial_y = dy / radial_magnitude
        
        # Calculate dot product with flow
        radial_flow = flow_small[:, :, 0] * radial_x + flow_small[:, :, 1] * radial_y
        
        # Average radial flow (positive = zoom out, negative = zoom in)
        avg_radial_flow = np.mean(radial_flow)
        
        zoom_info = {
            'type': 'none',
            'magnitude': float(abs(avg_radial_flow)),
            'confidence': 0.0
        }
        
        if abs(avg_radial_flow) > self.zoom_threshold:
            if avg_radial_flow > 0:
                zoom_info['type'] = 'zoom_out'
            else:
                zoom_info['type'] = 'zoom_in'
            zoom_info['confidence'] = min(1.0, abs(avg_radial_flow) / 0.1)
        
        return zoom_info
    
    def detect_shot_type(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect shot type based on face/body detection"""
        # Downsample for faster processing
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        # Simple heuristic based on face detection
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 3)  # Faster parameters
        
        h, w = small_frame.shape[:2]
        shot_info = {
            'type': 'wide_shot',
            'face_ratio': 0.0,
            'confidence': 0.8
        }
        
        if len(faces) > 0:
            # Calculate largest face area ratio (adjust for scale)
            largest_face_area = max([face[2] * face[3] for face in faces]) / (scale * scale)
            frame_area = (h * w) / (scale * scale)
            face_ratio = largest_face_area / frame_area
            
            shot_info['face_ratio'] = float(face_ratio)
            
            if face_ratio > 0.3:
                shot_info['type'] = 'close_up'
                shot_info['confidence'] = 0.9
            elif face_ratio > 0.1:
                shot_info['type'] = 'medium_shot'
                shot_info['confidence'] = 0.85
            else:
                shot_info['type'] = 'wide_shot'
                shot_info['confidence'] = 0.8
        
        return shot_info
    
    def calculate_stability_score(self, flow: np.ndarray) -> float:
        """Calculate camera stability score (0-1, higher is more stable)"""
        # Calculate flow variance
        flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        variance = np.var(flow_magnitude)
        
        # Convert to stability score (inverse of variance)
        # Use exponential decay for smooth transition
        stability = np.exp(-variance / 10.0)
        
        return float(stability)
    
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> Dict[str, Any]:
        """Process frames to detect camera movements"""
        segments = []
        
        # Pre-convert all frames to grayscale for efficiency
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        
        for i in range(len(frames)):
            frame_gray = gray_frames[i]
            
            if i > 0:
                prev_gray = gray_frames[i-1]
                
                # Calculate optical flow
                flow = self.calculate_optical_flow(prev_gray, frame_gray)
                
                # Analyze movement
                movement = self.analyze_camera_movement(flow)
                zoom = self.detect_zoom(flow, frame_gray.shape)
                stability = self.calculate_stability_score(flow)
                
            else:
                # First frame - no movement
                movement = {
                    'type': 'static',
                    'direction': None,
                    'speed': 0.0,
                    'confidence': 1.0,
                    'is_handheld': False
                }
                zoom = {
                    'type': 'none',
                    'magnitude': 0.0,
                    'confidence': 1.0
                }
                stability = 1.0
            
            # Detect shot type
            shot_info = self.detect_shot_type(frames[i])
            
            # Create comprehensive segment data
            segment = {
                'timestamp': float(frame_times[i]),
                'start_time': float(frame_times[i]),
                'end_time': float(frame_times[i] + (frame_times[1] - frame_times[0]) if i < len(frame_times) - 1 else 0.1),
                'movement': movement['type'],  # Primary field for movement type
                'movement_type': movement['type'],  # Compatibility field
                'movement_speed': movement['speed'],
                'movement_confidence': movement['confidence'],
                'movement_direction': movement['direction'],
                'is_handheld': movement['is_handheld'],
                'description': self._generate_description(movement, zoom, shot_info, stability),
                'camera_movement': movement,  # Full movement details
                'zoom': zoom,
                'shot_type': shot_info,
                'stability_score': stability,
                'frame_index': i
            }
            
            segments.append(segment)
        
        return {'segments': segments}
    
    def _generate_description(self, movement: Dict, zoom: Dict, shot_info: Dict, stability: float = 1.0) -> str:
        """Generate descriptive text for the camera work"""
        parts = []
        
        # Shot type
        parts.append(shot_info['type'].replace('_', ' '))
        
        # Movement
        if movement['type'] != 'static':
            move_desc = movement['type'].replace('_', ' ')
            if movement['is_handheld'] and 'handheld' not in move_desc:
                move_desc += ' (handheld)'
            parts.append(f"with {move_desc}")
        
        # Zoom
        if zoom['type'] != 'none':
            parts.append(f"and {zoom['type'].replace('_', ' ')}")
        
        # Stability
        if stability < 0.3:
            parts.append("(very shaky)")
        elif stability < 0.7:
            parts.append("(slightly unstable)")
        
        return ' '.join(parts)
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Main analysis function"""
        logger.info(f"[CameraAnalysis-Fixed] Analyzing {video_path}")
        
        # Load model if needed
        if not self.model_loaded:
            self._load_model_impl()
            self.model_loaded = True
        
        # Extract frames for GPU analysis (optimized for production)
        # Use config values if available
        from configs.performance_config import OPTIMIZED_FRAME_INTERVALS
        
        analyzer_name = 'camera_analysis'
        sample_rate = self.sample_rate  # Use optimized sample rate (45)
        max_frames = 40  # Balance between speed and accuracy
        
        frames, timestamps = self.extract_frames(video_path, sample_rate=sample_rate, max_frames=max_frames)
        
        if len(frames) == 0:
            return {'segments': [], 'error': 'No frames extracted'}
        
        # Process frames
        result = self.process_batch_gpu(frames, timestamps)
        
        # Add comprehensive summary statistics
        movements = [s['movement'] for s in result['segments']]  # Use 'movement' field directly
        movement_types = [m.replace('_handheld', '') for m in movements]  # Base movement types
        
        # Count all movement types
        from collections import Counter
        movement_counts = Counter(movements)
        base_movement_counts = Counter(movement_types)
        
        result['summary'] = {
            'total_segments': len(result['segments']),
            'movement_distribution': dict(movement_counts),
            'base_movement_distribution': {
                'static': base_movement_counts.get('static', 0),
                'pan_left': base_movement_counts.get('pan_left', 0),
                'pan_right': base_movement_counts.get('pan_right', 0),
                'tilt_up': base_movement_counts.get('tilt_up', 0),
                'tilt_down': base_movement_counts.get('tilt_down', 0)
            },
            'handheld_segments': sum(1 for s in result['segments'] if s.get('is_handheld', False)),
            'average_stability': float(np.mean([s['stability_score'] for s in result['segments']])),
            'has_zoom': any(s['zoom']['type'] != 'none' for s in result['segments']),
            'zoom_segments': sum(1 for s in result['segments'] if s['zoom']['type'] != 'none'),
            'dominant_shot_type': max(
                set([s['shot_type']['type'] for s in result['segments']]),
                key=[s['shot_type']['type'] for s in result['segments']].count
            ),
            'average_movement_speed': float(np.mean([s['movement_speed'] for s in result['segments']]))
        }
        
        logger.info(f"[CameraAnalysis-Fixed] Found {len(result['segments'])} segments")
        
        return result