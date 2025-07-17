#!/usr/bin/env python3
"""
Temporal Flow Analysis for understanding video narrative structure and time-based patterns
"""
# FFmpeg pthread fix
import os
os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from analyzers.base_analyzer import GPUBatchAnalyzer
from analyzers.description_helpers import DescriptionHelpers

logger = logging.getLogger(__name__)

class CPUBatchTemporalFlow(GPUBatchAnalyzer):
    def __init__(self):
        super().__init__(batch_size=8)
        self.device = 'cpu'
        self.flow_analyzer = TemporalFlowAnalyzer()
        self.narrative_analyzer = NarrativeStructureAnalyzer()
        self.pacing_analyzer = PacingAnalyzer()
        print("[TemporalFlow] Initializing temporal flow and narrative analysis")
        
        # Initialize flow tracking
        self.previous_frame = None
        self.flow_history = []
        self.scene_transitions = []
        self.activity_levels = []
    
    def analyze_optical_flow(self, current_frame: np.ndarray, 
                           previous_frame: np.ndarray) -> Dict[str, Any]:
        """Analyze optical flow between consecutive frames"""
        
        try:
            # Convert to grayscale
            gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow using Lucas-Kanade method
            flow = cv2.calcOpticalFlowPyrLK(
                gray_previous, gray_current, None, None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Alternative: Dense optical flow
            flow_dense = cv2.calcOpticalFlowFarneback(
                gray_previous, gray_current, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Analyze flow magnitude and direction
            magnitude, angle = cv2.cartToPolar(flow_dense[..., 0], flow_dense[..., 1])
            
            # Calculate flow statistics
            avg_magnitude = np.mean(magnitude)
            max_magnitude = np.max(magnitude)
            flow_density = np.sum(magnitude > 1.0) / magnitude.size
            
            # Determine movement characteristics
            movement_type = self.classify_movement_pattern(flow_dense, magnitude, angle)
            
            # Calculate directional flow
            directional_flow = self.analyze_directional_flow(flow_dense, magnitude)
            
            return {
                'average_magnitude': float(avg_magnitude),
                'max_magnitude': float(max_magnitude),
                'flow_density': float(flow_density),
                'movement_type': movement_type,
                'directional_flow': directional_flow,
                'flow_complexity': self.calculate_flow_complexity(flow_dense),
                'motion_coherence': self.calculate_motion_coherence(flow_dense, magnitude)
            }
            
        except Exception as e:
            logger.debug(f"Optical flow analysis error: {e}")
            return {
                'average_magnitude': 0.0,
                'movement_type': 'static',
                'flow_density': 0.0
            }
    
    def classify_movement_pattern(self, flow: np.ndarray, magnitude: np.ndarray, 
                                angle: np.ndarray) -> str:
        """Classify the type of movement in the video"""
        
        avg_magnitude = np.mean(magnitude)
        
        # Static scene
        if avg_magnitude < 0.5:
            return 'static'
        
        # Analyze flow direction patterns
        # Convert angles to degrees for easier analysis
        angles_deg = angle * 180 / np.pi
        
        # Calculate dominant direction
        hist, _ = np.histogram(angles_deg[magnitude > 1.0], bins=8, range=(0, 360))
        dominant_direction_idx = np.argmax(hist)
        
        # Check for specific patterns
        # Zoom in/out detection
        center_y, center_x = flow.shape[0] // 2, flow.shape[1] // 2
        center_region = flow[center_y-20:center_y+20, center_x-20:center_x+20]
        
        if center_region.size > 0:
            center_flow_magnitude = np.mean(np.sqrt(center_region[..., 0]**2 + center_region[..., 1]**2))
            
            # Radial flow patterns (zoom)
            radial_flow = self.detect_radial_flow(flow, magnitude)
            if radial_flow['is_radial']:
                if radial_flow['direction'] == 'outward':
                    return 'zoom_out'
                else:
                    return 'zoom_in'
        
        # Pan detection
        horizontal_flow = np.mean(flow[..., 0])
        vertical_flow = np.mean(flow[..., 1])
        
        if abs(horizontal_flow) > abs(vertical_flow) * 2:
            if horizontal_flow > 0.5:
                return 'pan_right'
            elif horizontal_flow < -0.5:
                return 'pan_left'
        elif abs(vertical_flow) > abs(horizontal_flow) * 2:
            if vertical_flow > 0.5:
                return 'tilt_down'
            elif vertical_flow < -0.5:
                return 'tilt_up'
        
        # Object movement vs camera movement
        if avg_magnitude > 2.0:
            if np.std(magnitude) > np.mean(magnitude):
                return 'object_movement'
            else:
                return 'camera_movement'
        
        return 'general_movement'
    
    def detect_radial_flow(self, flow: np.ndarray, magnitude: np.ndarray) -> Dict[str, Any]:
        """Detect radial flow patterns (zoom in/out)"""
        
        height, width = flow.shape[:2]
        center_y, center_x = height // 2, width // 2
        
        # Create coordinate grids
        y, x = np.mgrid[0:height, 0:width]
        
        # Calculate vectors from center
        dx_from_center = x - center_x
        dy_from_center = y - center_y
        
        # Normalize
        distances = np.sqrt(dx_from_center**2 + dy_from_center**2)
        distances[distances == 0] = 1  # Avoid division by zero
        
        expected_x = dx_from_center / distances
        expected_y = dy_from_center / distances
        
        # Compare with actual flow
        actual_flow_x = flow[..., 0]
        actual_flow_y = flow[..., 1]
        
        # Calculate correlation with radial pattern
        mask = magnitude > 0.5  # Only consider significant flow
        
        if np.sum(mask) == 0:
            return {'is_radial': False, 'direction': 'none'}
        
        correlation_x = np.corrcoef(
            expected_x[mask].flatten(), 
            actual_flow_x[mask].flatten()
        )[0, 1] if len(expected_x[mask].flatten()) > 1 else 0
        
        correlation_y = np.corrcoef(
            expected_y[mask].flatten(), 
            actual_flow_y[mask].flatten()
        )[0, 1] if len(expected_y[mask].flatten()) > 1 else 0
        
        avg_correlation = (correlation_x + correlation_y) / 2
        
        # Check if flow is radial
        if abs(avg_correlation) > 0.3:
            direction = 'outward' if avg_correlation > 0 else 'inward'
            return {'is_radial': True, 'direction': direction, 'strength': abs(avg_correlation)}
        
        return {'is_radial': False, 'direction': 'none'}
    
    def analyze_directional_flow(self, flow: np.ndarray, magnitude: np.ndarray) -> Dict[str, Any]:
        """Analyze directional characteristics of flow"""
        
        # Calculate flow in different regions
        height, width = flow.shape[:2]
        
        regions = {
            'top': flow[:height//3, :],
            'middle': flow[height//3:2*height//3, :],
            'bottom': flow[2*height//3:, :],
            'left': flow[:, :width//3],
            'center': flow[:, width//3:2*width//3],
            'right': flow[:, 2*width//3:]
        }
        
        regional_flow = {}
        
        for region_name, region_flow in regions.items():
            if region_flow.size > 0:
                avg_flow_x = np.mean(region_flow[..., 0])
                avg_flow_y = np.mean(region_flow[..., 1])
                magnitude = np.sqrt(avg_flow_x**2 + avg_flow_y**2)
                
                direction = 'static'
                if magnitude > 0.5:
                    angle = np.arctan2(avg_flow_y, avg_flow_x) * 180 / np.pi
                    if -22.5 <= angle < 22.5:
                        direction = 'right'
                    elif 22.5 <= angle < 67.5:
                        direction = 'down_right'
                    elif 67.5 <= angle < 112.5:
                        direction = 'down'
                    elif 112.5 <= angle < 157.5:
                        direction = 'down_left'
                    elif abs(angle) >= 157.5:
                        direction = 'left'
                    elif -157.5 <= angle < -112.5:
                        direction = 'up_left'
                    elif -112.5 <= angle < -67.5:
                        direction = 'up'
                    elif -67.5 <= angle < -22.5:
                        direction = 'up_right'
                
                regional_flow[region_name] = {
                    'direction': direction,
                    'magnitude': float(magnitude),
                    'flow_x': float(avg_flow_x),
                    'flow_y': float(avg_flow_y)
                }
        
        return regional_flow
    
    def calculate_flow_complexity(self, flow: np.ndarray) -> str:
        """Calculate complexity of optical flow"""
        
        # Calculate flow magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Variance in flow direction
        angles = np.arctan2(flow[..., 1], flow[..., 0])
        angle_variance = np.var(angles[magnitude > 0.5])
        
        # Spatial coherence
        spatial_variance = np.var(magnitude)
        
        # Classify complexity
        if angle_variance > 2.0 or spatial_variance > 5.0:
            return 'very_complex'
        elif angle_variance > 1.0 or spatial_variance > 2.0:
            return 'complex'
        elif angle_variance > 0.5 or spatial_variance > 1.0:
            return 'moderate'
        else:
            return 'simple'
    
    def calculate_motion_coherence(self, flow: np.ndarray, magnitude: np.ndarray) -> float:
        """Calculate how coherent the motion is across the frame"""
        
        # Only consider significant motion
        mask = magnitude > 0.5
        if np.sum(mask) == 0:
            return 1.0  # No motion = perfect coherence
        
        # Calculate dominant flow direction
        flow_x = flow[..., 0][mask]
        flow_y = flow[..., 1][mask]
        
        # Average flow direction
        avg_flow_x = np.mean(flow_x)
        avg_flow_y = np.mean(flow_y)
        
        # Calculate how much each flow vector deviates from average
        deviations = np.sqrt((flow_x - avg_flow_x)**2 + (flow_y - avg_flow_y)**2)
        avg_deviation = np.mean(deviations)
        avg_magnitude = np.mean(magnitude[mask])
        
        # Coherence score (1.0 = perfect coherence, 0.0 = random motion)
        coherence = max(0.0, 1.0 - (avg_deviation / (avg_magnitude + 1e-6)))
        
        return float(coherence)
    
    def detect_scene_transitions(self, frames: List[np.ndarray], timestamps: List[float]) -> List[Dict[str, Any]]:
        """Detect scene transitions and cuts"""
        
        transitions = []
        
        for i in range(1, len(frames)):
            if frames[i] is None or frames[i-1] is None:
                continue
            
            try:
                # Calculate frame difference
                prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                
                # Histogram comparison
                hist_prev = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
                hist_curr = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
                
                # Calculate correlation
                correlation = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
                
                # Calculate structural similarity
                mean_diff = np.mean(np.abs(prev_gray.astype(float) - curr_gray.astype(float)))
                
                # Detect cut if significant change
                if correlation < 0.7 or mean_diff > 30:
                    transition_type = self.classify_transition_type(
                        frames[i-1], frames[i], correlation, mean_diff
                    )
                    
                    transitions.append({
                        'timestamp': timestamps[i],
                        'type': transition_type,
                        'correlation': float(correlation),
                        'mean_difference': float(mean_diff),
                        'transition_strength': self.calculate_transition_strength(correlation, mean_diff)
                    })
                    
            except Exception as e:
                logger.debug(f"Scene transition detection error at {timestamps[i]}: {e}")
                continue
        
        return transitions
    
    def classify_transition_type(self, frame1: np.ndarray, frame2: np.ndarray, 
                               correlation: float, mean_diff: float) -> str:
        """Classify the type of scene transition"""
        
        # Hard cut (sudden complete change)
        if correlation < 0.3 and mean_diff > 50:
            return 'hard_cut'
        
        # Soft cut (gradual change)
        elif correlation < 0.7 and mean_diff > 20:
            return 'soft_cut'
        
        # Fade transition
        elif correlation > 0.7 and mean_diff > 30:
            # Check for fade by looking at brightness changes
            bright1 = np.mean(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY))
            bright2 = np.mean(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY))
            
            if abs(bright1 - bright2) > 50:
                if bright2 < bright1:
                    return 'fade_to_black'
                else:
                    return 'fade_from_black'
            else:
                return 'cross_fade'
        
        # Scene change (same location, different content)
        elif correlation > 0.5:
            return 'scene_change'
        
        return 'unknown_transition'
    
    def calculate_transition_strength(self, correlation: float, mean_diff: float) -> str:
        """Calculate the strength of a transition"""
        
        # Combine correlation and mean difference for strength assessment
        strength_score = (1 - correlation) + (mean_diff / 100)
        
        if strength_score > 1.5:
            return 'sehr stark'
        elif strength_score > 1.0:
            return 'stark'
        elif strength_score > 0.5:
            return 'moderat'
        else:
            return 'schwach'
    
    def analyze_pacing_rhythm(self, flow_history: List[Dict[str, Any]], 
                            transitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the pacing and rhythm of the video"""
        
        if not flow_history:
            return {'pacing': 'unknown', 'rhythm': 'unknown'}
        
        # Extract activity levels over time
        activity_levels = [flow.get('average_magnitude', 0) for flow in flow_history]
        
        # Calculate pacing metrics
        avg_activity = np.mean(activity_levels)
        activity_variance = np.var(activity_levels)
        
        # Transition frequency
        transition_frequency = len(transitions) / len(flow_history) if flow_history else 0
        
        # Determine pacing
        if avg_activity > 2.0 and transition_frequency > 0.3:
            pacing = 'sehr schnell'
        elif avg_activity > 1.5 and transition_frequency > 0.2:
            pacing = 'schnell'
        elif avg_activity > 0.8 and transition_frequency > 0.1:
            pacing = 'moderat'
        elif avg_activity > 0.3:
            pacing = 'langsam'
        else:
            pacing = 'sehr langsam'
        
        # Determine rhythm
        if activity_variance > 2.0:
            rhythm = 'unregelmäßig'
        elif activity_variance > 1.0:
            rhythm = 'variabel'
        else:
            rhythm = 'gleichmäßig'
        
        # Detect rhythm patterns
        rhythm_pattern = self.detect_rhythm_patterns(activity_levels)
        
        return {
            'pacing': pacing,
            'rhythm': rhythm,
            'rhythm_pattern': rhythm_pattern,
            'average_activity': float(avg_activity),
            'activity_variance': float(activity_variance),
            'transition_frequency': float(transition_frequency),
            'total_transitions': len(transitions)
        }
    
    def detect_rhythm_patterns(self, activity_levels: List[float]) -> str:
        """Detect rhythmic patterns in activity"""
        
        if len(activity_levels) < 4:
            return 'insufficient_data'
        
        # Look for repeating patterns
        # Simple approach: check for alternating high/low activity
        high_threshold = np.mean(activity_levels) + np.std(activity_levels)
        low_threshold = np.mean(activity_levels) - np.std(activity_levels)
        
        pattern = []
        for level in activity_levels:
            if level > high_threshold:
                pattern.append('high')
            elif level < low_threshold:
                pattern.append('low')
            else:
                pattern.append('medium')
        
        # Check for alternating pattern
        alternating_count = 0
        for i in range(1, len(pattern)):
            if pattern[i] != pattern[i-1]:
                alternating_count += 1
        
        alternating_ratio = alternating_count / (len(pattern) - 1)
        
        if alternating_ratio > 0.8:
            return 'alternating'
        elif alternating_ratio > 0.6:
            return 'variable'
        elif pattern.count('high') > len(pattern) * 0.7:
            return 'consistently_active'
        elif pattern.count('low') > len(pattern) * 0.7:
            return 'consistently_calm'
        else:
            return 'mixed'
    
    def analyze_narrative_structure(self, flow_data: List[Dict[str, Any]], 
                                  transitions: List[Dict[str, Any]], 
                                  duration: float) -> Dict[str, Any]:
        """Analyze narrative structure and story progression"""
        
        if not flow_data or duration <= 0:
            return {'structure': 'unknown', 'phases': []}
        
        # Divide video into narrative phases
        phase_duration = duration / 3  # Simple 3-act structure
        phases = []
        
        # Analyze each phase
        for phase_idx in range(3):
            phase_start = phase_idx * phase_duration
            phase_end = (phase_idx + 1) * phase_duration
            
            # Get data for this phase
            phase_flow = [f for f in flow_data 
                         if phase_start <= f.get('timestamp', 0) < phase_end]
            phase_transitions = [t for t in transitions 
                               if phase_start <= t.get('timestamp', 0) < phase_end]
            
            if phase_flow:
                phase_activity = np.mean([f.get('average_magnitude', 0) for f in phase_flow])
                phase_complexity = len(phase_transitions)
                
                # Classify phase
                if phase_idx == 0:  # Beginning
                    if phase_activity > 1.0:
                        phase_type = 'dynamic_opening'
                    else:
                        phase_type = 'calm_introduction'
                elif phase_idx == 1:  # Middle
                    if phase_complexity > 2:
                        phase_type = 'complex_development'
                    elif phase_activity > 1.5:
                        phase_type = 'active_development'
                    else:
                        phase_type = 'steady_development'
                else:  # End
                    if phase_activity > phase_flow[0].get('average_magnitude', 0) if phase_flow else False:
                        phase_type = 'climactic_ending'
                    else:
                        phase_type = 'calm_conclusion'
                
                phases.append({
                    'phase': phase_idx + 1,
                    'type': phase_type,
                    'start_time': phase_start,
                    'end_time': phase_end,
                    'activity_level': float(phase_activity),
                    'transitions': len(phase_transitions),
                    'description': self.describe_phase(phase_type, phase_activity, len(phase_transitions))
                })
        
        # Overall structure classification
        if len(phases) >= 3:
            structure = self.classify_overall_structure(phases)
        else:
            structure = 'simple'
        
        return {
            'structure': structure,
            'phases': phases,
            'total_duration': duration,
            'narrative_complexity': len(transitions) + sum(p.get('transitions', 0) for p in phases)
        }
    
    def describe_phase(self, phase_type: str, activity: float, transitions: int) -> str:
        """Create German description for narrative phase"""
        
        descriptions = {
            'dynamic_opening': 'Dynamischer Einstieg mit viel Bewegung',
            'calm_introduction': 'Ruhige Einführung',
            'complex_development': 'Komplexe Entwicklung mit vielen Wechseln',
            'active_development': 'Aktive Entwicklung',
            'steady_development': 'Stetige Entwicklung',
            'climactic_ending': 'Spannungsreicher Abschluss',
            'calm_conclusion': 'Ruhiger Abschluss'
        }
        
        base_desc = descriptions.get(phase_type, 'Narrative Phase')
        
        if transitions > 3:
            base_desc += ' mit häufigen Szenenwechseln'
        elif activity > 2.0:
            base_desc += ' mit hoher Aktivität'
        
        return base_desc
    
    def classify_overall_structure(self, phases: List[Dict[str, Any]]) -> str:
        """Classify overall narrative structure"""
        
        if len(phases) < 3:
            return 'simple'
        
        # Analyze activity progression
        activities = [p.get('activity_level', 0) for p in phases]
        
        # Classic rising action structure
        if activities[0] < activities[1] and activities[1] > activities[2]:
            return 'classic_arc'
        
        # Consistent energy
        elif max(activities) - min(activities) < 0.5:
            return 'consistent_energy'
        
        # Building intensity
        elif activities[0] < activities[1] < activities[2]:
            return 'building_intensity'
        
        # Declining intensity
        elif activities[0] > activities[1] > activities[2]:
            return 'declining_intensity'
        
        # Complex/irregular
        else:
            return 'complex_structure'
    
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> Dict[str, Any]:
        """Process frames for temporal flow analysis"""
        if frames is None or len(frames) == 0:
            return {'segments': []}
        
        segments = []
        flow_data = []
        
        # Calculate video duration
        duration = frame_times[-1] - frame_times[0] if len(frame_times) > 1 else 0
        
        # Process optical flow between consecutive frames
        for i in range(1, len(frames)):
            if frames[i] is None or frames[i-1] is None:
                continue
                
            try:
                timestamp = frame_times[i]
                
                # Calculate optical flow
                flow_analysis = self.analyze_optical_flow(frames[i], frames[i-1])
                flow_analysis['timestamp'] = timestamp
                flow_data.append(flow_analysis)
                
<<<<<<< HEAD
                # Create flow segment with temporal window
                segment_start = max(0, timestamp - 0.5)  # 0.5s window before
                segment_end = min(duration + frame_times[0], timestamp + 0.5)  # 0.5s window after
                
                segment = {
                    'start_time': float(segment_start),
                    'end_time': float(segment_end),
                    'timestamp': float(timestamp),
                    'type': 'temporal_flow',
                    'segment_id': f'temporal_flow_{int(timestamp * 10)}',
                    # Enhanced descriptions
                    'description': self.create_movement_description(flow_analysis),
=======
                # Create flow segment
                segment = {
                    'timestamp': float(timestamp),
                    'type': 'temporal_flow',
                    # Enhanced descriptions
                    'movement_description': self.create_movement_description(flow_analysis),
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
                    'movement_type': flow_analysis.get('movement_type', 'unknown'),
                    'flow_intensity': self.classify_flow_intensity(flow_analysis.get('average_magnitude', 0)),
                    'motion_coherence': flow_analysis.get('motion_coherence', 0.5),
                    'flow_complexity': flow_analysis.get('flow_complexity', 'unknown'),
                    # Technical data
                    'flow_magnitude': flow_analysis.get('average_magnitude', 0),
                    'flow_density': flow_analysis.get('flow_density', 0),
                    'directional_flow': flow_analysis.get('directional_flow', {}),
                    'analysis_method': 'Optical Flow'
                }
                
                segments.append(segment)
                
            except Exception as e:
                logger.error(f"Temporal flow error at {frame_times[i]}s: {e}")
                continue
        
        # Detect scene transitions
        transitions = self.detect_scene_transitions(frames, frame_times)
        
        # Analyze pacing and rhythm
        pacing_analysis = self.analyze_pacing_rhythm(flow_data, transitions)
        
        # Analyze narrative structure
        narrative_analysis = self.analyze_narrative_structure(flow_data, transitions, duration)
        
<<<<<<< HEAD
        # Convert transitions into temporal segments
        for transition in transitions:
            transition_timestamp = transition.get('timestamp', 0)
            segments.append({
                'start_time': max(0, transition_timestamp - 0.5),
                'end_time': min(duration + frame_times[0], transition_timestamp + 0.5),
                'timestamp': transition_timestamp,
                'type': 'scene_transition',
                'segment_id': f'transition_{int(transition_timestamp * 10)}',
                'description': f"Szenenwechsel mit Stärke {transition.get('mean_difference', 0):.2f}",
                'transition_strength': transition.get('mean_difference', 0),
                'frame_difference': transition.get('frame_difference', 0)
            })
        
        # Add pacing segments distributed across video
        if duration > 0:
            pacing_segments = 5  # 5 pacing analysis points
            for i in range(pacing_segments):
                segment_time = (duration * i) / (pacing_segments - 1) if pacing_segments > 1 else 0
                segments.append({
                    'start_time': max(0, segment_time - 2.0),
                    'end_time': min(duration + frame_times[0], segment_time + 2.0),
                    'timestamp': segment_time,
                    'type': 'pacing_analysis',
                    'segment_id': f'pacing_{i}',
                    'description': self.create_pacing_description(pacing_analysis),
                    'pacing': pacing_analysis.get('pacing', 'unknown'),
                    'rhythm': pacing_analysis.get('rhythm', 'unknown')
                })
=======
        # Add summary segments
        segments.append({
            'type': 'pacing_analysis',
            'pacing': pacing_analysis.get('pacing', 'unknown'),
            'rhythm': pacing_analysis.get('rhythm', 'unknown'),
            'rhythm_pattern': pacing_analysis.get('rhythm_pattern', 'unknown'),
            'pacing_description': self.create_pacing_description(pacing_analysis),
            'technical_metrics': pacing_analysis
        })
        
        segments.append({
            'type': 'narrative_structure',
            'structure': narrative_analysis.get('structure', 'unknown'),
            'phases': narrative_analysis.get('phases', []),
            'structure_description': self.create_structure_description(narrative_analysis),
            'narrative_complexity': narrative_analysis.get('narrative_complexity', 0)
        })
        
        # Add transition summary
        if transitions:
            segments.append({
                'type': 'scene_transitions',
                'transitions': transitions,
                'total_transitions': len(transitions),
                'transition_summary': f"{len(transitions)} Szenenwechsel erkannt",
                'avg_transition_strength': np.mean([
                    t.get('mean_difference', 0) for t in transitions
                ]) if transitions else 0
            })
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
        
        return {'segments': segments}
    
    def create_movement_description(self, flow_analysis: Dict[str, Any]) -> str:
        """Create German description for movement type"""
        
        movement_type = flow_analysis.get('movement_type', 'unknown')
        magnitude = flow_analysis.get('average_magnitude', 0)
        coherence = flow_analysis.get('motion_coherence', 0.5)
        
        # Base movement descriptions
        movement_descriptions = {
            'static': 'Statische Szene ohne Bewegung',
            'pan_left': 'Kameraschwenk nach links',
            'pan_right': 'Kameraschwenk nach rechts',
            'tilt_up': 'Kamera neigt sich nach oben',
            'tilt_down': 'Kamera neigt sich nach unten',
            'zoom_in': 'Heranzoomen auf das Motiv',
            'zoom_out': 'Herauszoomen für Überblick',
            'object_movement': 'Objektbewegung im Bild',
            'camera_movement': 'Kamerabewegung',
            'general_movement': 'Allgemeine Bewegung'
        }
        
        base_desc = movement_descriptions.get(movement_type, 'Bewegung erkannt')
        
        # Add intensity qualifier
        if magnitude > 3.0:
            intensity = 'sehr dynamisch'
        elif magnitude > 2.0:
            intensity = 'dynamisch'
        elif magnitude > 1.0:
            intensity = 'moderat'
        elif magnitude > 0.5:
            intensity = 'langsam'
        else:
            intensity = 'minimal'
        
        # Add coherence qualifier
        if coherence > 0.8:
            coherence_desc = 'sehr flüssig'
        elif coherence > 0.6:
            coherence_desc = 'flüssig'
        elif coherence > 0.4:
            coherence_desc = 'teilweise ruckartig'
        else:
            coherence_desc = 'unruhig'
        
        return f"{base_desc}, {intensity}, {coherence_desc}"
    
    def classify_flow_intensity(self, magnitude: float) -> str:
        """Classify flow intensity in German"""
        
        if magnitude > 3.0:
            return 'sehr hoch'
        elif magnitude > 2.0:
            return 'hoch'
        elif magnitude > 1.0:
            return 'moderat'
        elif magnitude > 0.5:
            return 'niedrig'
        else:
            return 'minimal'
    
    def create_pacing_description(self, pacing_analysis: Dict[str, Any]) -> str:
        """Create German description for pacing"""
        
        pacing = pacing_analysis.get('pacing', 'unknown')
        rhythm = pacing_analysis.get('rhythm', 'unknown')
        pattern = pacing_analysis.get('rhythm_pattern', 'unknown')
        
        desc_parts = [f"Tempo: {pacing}"]
        
        if rhythm != 'unknown':
            desc_parts.append(f"Rhythmus: {rhythm}")
        
        if pattern not in ['unknown', 'insufficient_data']:
            pattern_descriptions = {
                'alternating': 'mit wechselnden Intensitäten',
                'variable': 'mit variablen Intensitäten',
                'consistently_active': 'durchgehend aktiv',
                'consistently_calm': 'durchgehend ruhig',
                'mixed': 'mit gemischten Intensitäten'
            }
            pattern_desc = pattern_descriptions.get(pattern, pattern)
            desc_parts.append(pattern_desc)
        
        return ", ".join(desc_parts)
    
    def create_structure_description(self, narrative_analysis: Dict[str, Any]) -> str:
        """Create German description for narrative structure"""
        
        structure = narrative_analysis.get('structure', 'unknown')
        phases = narrative_analysis.get('phases', [])
        
        structure_descriptions = {
            'classic_arc': 'Klassischer Spannungsbogen mit Aufbau und Auflösung',
            'consistent_energy': 'Gleichmäßige Energie über gesamte Dauer',
            'building_intensity': 'Steigende Intensität zum Ende hin',
            'declining_intensity': 'Abnehmende Intensität',
            'complex_structure': 'Komplexe narrative Struktur',
            'simple': 'Einfache lineare Struktur'
        }
        
        base_desc = structure_descriptions.get(structure, 'Narrative Struktur erkannt')
        
        if phases:
            phase_count = len(phases)
            base_desc += f" in {phase_count} Phasen"
        
        return base_desc
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Analyze video for temporal flow and narrative structure"""
<<<<<<< HEAD
        logger.info(f"[TemporalFlow] Analyzing {video_path}")
        
        # Extract frames with higher frequency for flow analysis (1 frame per second for temporal coverage)
        frames, timestamps = self.extract_frames(video_path, max_frames=600, frame_interval=30)
        
        if frames is None or len(frames) == 0:
            logger.warning(f"[TemporalFlow] No frames extracted from {video_path}")
            return {'segments': []}
        
        logger.info(f"[TemporalFlow] Extracted {len(frames)} frames for analysis")
        
        # Process frames
        result = self.process_batch_gpu(frames, timestamps)
        
        logger.info(f"[TemporalFlow] Generated {len(result['segments'])} temporal segments")
=======
        print(f"[TemporalFlow] Analyzing {video_path}")
        
        # Extract frames with higher frequency for flow analysis
        frames, timestamps = self.extract_frames(video_path, max_frames=20)
        
        if frames is None or len(frames) == 0:
            return {'segments': []}
        
        # Process frames
        result = self.process_batch_gpu(frames, timestamps)
        
        print(f"[TemporalFlow] Analyzed temporal flow with {len(result['segments'])} segments")
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
        return result


class TemporalFlowAnalyzer:
    """Core temporal flow analysis functionality"""
    
    def __init__(self):
        self.flow_buffer = []
        self.transition_threshold = 0.3
    
    def track_flow_changes(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track changes in flow over time"""
        self.flow_buffer.append(flow_data)
        
        # Keep only recent flow data
        if len(self.flow_buffer) > 10:
            self.flow_buffer = self.flow_buffer[-10:]
        
        # Analyze flow trends
        if len(self.flow_buffer) > 2:
            recent_magnitudes = [f.get('average_magnitude', 0) for f in self.flow_buffer[-3:]]
            trend = self.calculate_trend(recent_magnitudes)
            return {'trend': trend, 'stability': self.calculate_stability()}
        
        return {'trend': 'unknown', 'stability': 'unknown'}
    
    def calculate_trend(self, values: List[float]) -> str:
        """Calculate trend in flow values"""
        if len(values) < 2:
            return 'unknown'
        
        if values[-1] > values[0] * 1.2:
            return 'increasing'
        elif values[-1] < values[0] * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def calculate_stability(self) -> str:
        """Calculate flow stability"""
        if len(self.flow_buffer) < 3:
            return 'unknown'
        
        magnitudes = [f.get('average_magnitude', 0) for f in self.flow_buffer]
        variance = np.var(magnitudes)
        
        if variance > 2.0:
            return 'unstable'
        elif variance > 1.0:
            return 'variable'
        else:
            return 'stable'


class NarrativeStructureAnalyzer:
    """Analyze narrative structure and story elements"""
    
    def __init__(self):
        self.narrative_elements = []
    
    def detect_narrative_beats(self, flow_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect narrative beats and story points"""
        beats = []
        
        # This would implement more sophisticated narrative analysis
        # For now, basic implementation
        
        return beats


class PacingAnalyzer:
    """Analyze video pacing and rhythm"""
    
    def __init__(self):
        self.pacing_history = []
    
    def analyze_pacing_patterns(self, activity_levels: List[float]) -> Dict[str, Any]:
        """Analyze patterns in video pacing"""
        
        if len(activity_levels) < 3:
            return {'pattern': 'insufficient_data'}
        
        # Detect acceleration/deceleration
        first_third = np.mean(activity_levels[:len(activity_levels)//3])
        last_third = np.mean(activity_levels[-len(activity_levels)//3:])
        
        if last_third > first_third * 1.5:
            return {'pattern': 'accelerating', 'change': last_third - first_third}
        elif last_third < first_third * 0.67:
            return {'pattern': 'decelerating', 'change': first_third - last_third}
        else:
            return {'pattern': 'consistent', 'change': abs(last_third - first_third)}