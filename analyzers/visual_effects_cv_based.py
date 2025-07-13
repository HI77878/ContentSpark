#!/usr/bin/env python3
"""
Computer Vision-based Visual Effects Detection
Detects real video effects using CV algorithms instead of unreliable ML models
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from analyzers.base_analyzer import GPUBatchAnalyzer
import logging
from collections import deque
from scipy import signal

logger = logging.getLogger(__name__)

class VisualEffectsCVBased(GPUBatchAnalyzer):
    """CV-based visual effects detection for real results"""
    
    def __init__(self):
        super().__init__(batch_size=16)
        self.sample_rate = 15  # Every 0.5 seconds
        
        # History tracking
        self.frame_history = deque(maxlen=5)
        self.histogram_history = deque(maxlen=10)
        self.motion_history = deque(maxlen=5)
        
        # Thresholds
        self.motion_blur_threshold = 100  # Laplacian variance
        self.transition_threshold = 0.4   # Frame difference
        self.color_shift_threshold = 30  # Histogram difference
        
        logger.info("[VisualEffectsCV] Initialized with CV-based detection")
    
    def _detect_motion_blur(self, frame):
        """Detect motion blur using Laplacian variance"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return {
            'has_motion_blur': laplacian_var < self.motion_blur_threshold,
            'blur_score': float(laplacian_var),
            'blur_level': 'high' if laplacian_var < 50 else 'medium' if laplacian_var < 100 else 'low'
        }
    
    def _detect_color_grading(self, frame):
        """Detect color grading and filters"""
        # Analyze color distribution
        hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
        
        # Check for color shifts
        mean_b = np.mean(frame[:,:,0])
        mean_g = np.mean(frame[:,:,1])
        mean_r = np.mean(frame[:,:,2])
        
        # Detect common filters
        filters = []
        
        # Warm filter (orange/yellow tint)
        if mean_r > mean_b + 20 and mean_g > mean_b + 10:
            filters.append('warm_filter')
        
        # Cool filter (blue tint)
        elif mean_b > mean_r + 20:
            filters.append('cool_filter')
        
        # Vintage filter (reduced contrast, slight yellow)
        contrast = np.std(frame)
        if contrast < 40 and mean_g > mean_b:
            filters.append('vintage_filter')
        
        # High contrast
        if contrast > 80:
            filters.append('high_contrast')
        
        # Desaturated/B&W
        color_variance = np.std([mean_b, mean_g, mean_r])
        if color_variance < 5:
            filters.append('desaturated' if color_variance > 1 else 'black_white')
        
        return {
            'filters_detected': filters,
            'color_balance': {
                'red': float(mean_r),
                'green': float(mean_g),
                'blue': float(mean_b)
            },
            'contrast': float(contrast),
            'has_color_grading': len(filters) > 0
        }
    
    def _detect_transitions(self, frame, prev_frame):
        """Detect video transitions"""
        if prev_frame is None:
            return None
        
        # Frame difference
        diff = cv2.absdiff(frame, prev_frame)
        diff_score = np.mean(diff) / 255.0
        
        transition_type = None
        if diff_score > self.transition_threshold:
            # Analyze transition type
            
            # Fade detection (uniform change)
            if np.std(diff) < 30:
                if np.mean(frame) > np.mean(prev_frame):
                    transition_type = 'fade_in'
                else:
                    transition_type = 'fade_out'
            
            # Cut detection (sharp change)
            elif diff_score > 0.6:
                transition_type = 'hard_cut'
            
            # Wipe/slide detection (directional change)
            else:
                # Check horizontal gradient
                h_gradient = np.mean(np.diff(diff.mean(axis=0)))
                v_gradient = np.mean(np.diff(diff.mean(axis=1)))
                
                if abs(h_gradient) > abs(v_gradient):
                    transition_type = 'horizontal_wipe'
                else:
                    transition_type = 'vertical_wipe'
        
        return {
            'has_transition': transition_type is not None,
            'transition_type': transition_type,
            'transition_score': float(diff_score)
        }
    
    def _detect_speed_effects(self, timestamps):
        """Detect slow motion or time lapse"""
        if len(timestamps) < 3:
            return None
        
        # Calculate frame intervals
        intervals = np.diff(timestamps)
        mean_interval = np.mean(intervals)
        
        # Expected interval for 30fps
        expected_interval = 1.0 / 30.0
        
        speed_factor = expected_interval / mean_interval if mean_interval > 0 else 1.0
        
        effect = None
        if speed_factor < 0.5:
            effect = 'slow_motion'
        elif speed_factor > 2.0:
            effect = 'time_lapse'
        
        return {
            'has_speed_effect': effect is not None,
            'speed_effect': effect,
            'speed_factor': float(speed_factor)
        }
    
    def _detect_camera_effects(self, frame, prev_frames):
        """Detect zoom, pan, tilt effects"""
        if len(prev_frames) < 2:
            return None
        
        # Simple motion detection using optical flow
        prev_gray = cv2.cvtColor(prev_frames[-1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Analyze flow patterns
        h_flow = np.mean(flow[:,:,0])
        v_flow = np.mean(flow[:,:,1])
        flow_magnitude = np.sqrt(h_flow**2 + v_flow**2)
        
        effect = None
        if flow_magnitude > 2:
            if abs(h_flow) > abs(v_flow):
                effect = 'pan_left' if h_flow < 0 else 'pan_right'
            else:
                effect = 'tilt_up' if v_flow < 0 else 'tilt_down'
        
        # Check for zoom (radial flow pattern)
        center = (frame.shape[1]//2, frame.shape[0]//2)
        radial_flow = 0
        for y in range(0, frame.shape[0], 10):
            for x in range(0, frame.shape[1], 10):
                dx = x - center[0]
                dy = y - center[1]
                if dx != 0 or dy != 0:
                    radial = (flow[y,x,0]*dx + flow[y,x,1]*dy) / np.sqrt(dx**2 + dy**2)
                    radial_flow += radial
        
        radial_flow /= (frame.shape[0] * frame.shape[1] / 100)
        
        if abs(radial_flow) > 0.5:
            effect = 'zoom_in' if radial_flow > 0 else 'zoom_out'
        
        return {
            'has_camera_effect': effect is not None,
            'camera_effect': effect,
            'motion_magnitude': float(flow_magnitude)
        }
    
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> Dict[str, Any]:
        """Process frames for effect detection"""
        segments = []
        
        for i, (frame, timestamp) in enumerate(zip(frames, frame_times)):
            effects = {
                'timestamp': float(timestamp),
                'effects': []
            }
            
            # Motion blur detection
            blur_result = self._detect_motion_blur(frame)
            if blur_result['has_motion_blur']:
                effects['effects'].append({
                    'type': 'motion_blur',
                    'details': blur_result
                })
            
            # Color grading/filter detection
            color_result = self._detect_color_grading(frame)
            if color_result['has_color_grading']:
                effects['effects'].append({
                    'type': 'color_grading',
                    'details': color_result
                })
            
            # Transition detection
            if len(self.frame_history) > 0:
                transition_result = self._detect_transitions(frame, self.frame_history[-1])
                if transition_result and transition_result['has_transition']:
                    effects['effects'].append({
                        'type': 'transition',
                        'details': transition_result
                    })
            
            # Camera effects
            if len(self.frame_history) >= 2:
                camera_result = self._detect_camera_effects(frame, list(self.frame_history))
                if camera_result and camera_result['has_camera_effect']:
                    effects['effects'].append({
                        'type': 'camera_movement',
                        'details': camera_result
                    })
            
            # Update history
            self.frame_history.append(frame)
            
            # Create description
            if effects['effects']:
                effect_names = [e['type'] for e in effects['effects']]
                effects['description'] = f"Effects detected: {', '.join(effect_names)}"
            else:
                effects['description'] = "No special effects"
            
            segments.append(effects)
        
        # Detect speed effects across batch
        if len(frame_times) > 2:
            speed_result = self._detect_speed_effects(frame_times)
            if speed_result and speed_result['has_speed_effect']:
                # Add to all segments in batch
                for segment in segments:
                    segment['effects'].append({
                        'type': 'speed_change',
                        'details': speed_result
                    })
        
        return {'segments': segments}
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Main analysis function"""
        logger.info(f"[VisualEffectsCV] Starting analysis of {video_path.split('/')[-1]}")
        
        try:
            # Extract frames
            frames, frame_times = self.extract_frames(video_path, sample_rate=self.sample_rate)
            
            if not frames:
                return {'segments': [], 'error': 'No frames extracted'}
            
            logger.info(f"[VisualEffectsCV] Processing {len(frames)} frames")
            
            # Process in batches
            all_segments = []
            for i in range(0, len(frames), self.batch_size):
                batch_frames = frames[i:i+self.batch_size]
                batch_times = frame_times[i:i+self.batch_size]
                
                result = self.process_batch_gpu(batch_frames, batch_times)
                all_segments.extend(result['segments'])
            
            # Generate summary
            total_effects = sum(len(s['effects']) for s in all_segments)
            effect_types = {}
            for segment in all_segments:
                for effect in segment['effects']:
                    effect_type = effect['type']
                    effect_types[effect_type] = effect_types.get(effect_type, 0) + 1
            
            return {
                'segments': all_segments,
                'summary': {
                    'total_segments': len(all_segments),
                    'total_effects': total_effects,
                    'effect_distribution': effect_types,
                    'most_common_effect': max(effect_types.items(), key=lambda x: x[1])[0] if effect_types else None
                }
            }
            
        except Exception as e:
            logger.error(f"[VisualEffectsCV] Analysis failed: {e}")
            return {'segments': [], 'error': str(e)}
        
        finally:
            # Cleanup
            self.frame_history.clear()
            self.histogram_history.clear()
            self.motion_history.clear()