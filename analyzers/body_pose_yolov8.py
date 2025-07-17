#!/usr/bin/env python3
"""
Body Pose and Gesture Analysis using YOLOv8-Pose
Advanced pose estimation with body language interpretation
"""

# GPU Forcing
import torch
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from analyzers.base_analyzer import GPUBatchAnalyzer
import logging
import warnings
warnings.filterwarnings('ignore')
from collections import deque

# YOLOv8 imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics YOLO not available. Install with: pip install ultralytics")

logger = logging.getLogger(__name__)

class BodyPoseYOLOv8(GPUBatchAnalyzer):
    """Body pose detection and gesture analysis using YOLOv8-Pose"""
    
    def __init__(self):
        super().__init__(batch_size=16)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.models_loaded = False
        
        # Configuration
        self.sample_rate = 10  # Every ~0.33 seconds at 30fps
        self.conf_threshold = 0.5
        self.model_size = 'yolov8x-pose.pt'  # Best accuracy
        
        # Pose keypoints mapping (COCO format)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Body part connections for skeleton
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # Head
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],  # Upper body
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],  # Arms
            [2, 4], [3, 5], [4, 6], [5, 7]  # Lower body
        ]
        
        # Gesture recognition patterns
        self.gesture_patterns = {
            'hands_up': self._check_hands_up,
            'pointing': self._check_pointing,
            'crossed_arms': self._check_crossed_arms,
            'hands_on_hips': self._check_hands_on_hips,
            'waving': self._check_waving,
            'thumbs_up': self._check_thumbs_up,
            'clapping': self._check_clapping
        }
        
        # Body language patterns
        self.body_language_patterns = {
            'open': self._check_open_posture,
            'closed': self._check_closed_posture,
            'confident': self._check_confident_posture,
            'relaxed': self._check_relaxed_posture,
            'tense': self._check_tense_posture,
            'leaning': self._check_leaning
        }
        
        # Movement tracking
        self.pose_history = deque(maxlen=10)  # Track last 10 poses
        self.person_tracks = {}
        
        logger.info("[BodyPoseYOLOv8] Initialized with YOLOv8x-pose")
    
    def _load_model_impl(self):
        """Load YOLOv8-Pose model"""
        if self.models_loaded:
            return
        
        if not YOLO_AVAILABLE:
            raise RuntimeError("Ultralytics YOLO not installed. Cannot load model.")
        
        try:
            logger.info(f"[BodyPoseYOLOv8] Loading {self.model_size}...")
            
            # Load YOLOv8-Pose model
            self.model = YOLO(self.model_size)
            
            # Move to GPU if available
            if self.device == 'cuda':
                self.model.to('cuda')
            
            self.models_loaded = True
            logger.info("[BodyPoseYOLOv8] ✅ YOLOv8-Pose loaded successfully")
            
        except Exception as e:
            logger.error(f"[BodyPoseYOLOv8] Failed to load model: {e}")
            raise RuntimeError(f"Failed to load YOLOv8-Pose: {e}")
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Analyze video for body poses and gestures"""
        logger.info(f"[BodyPoseYOLOv8] Starting analysis of {video_path}")
        
        # Load model if needed
        if not self.models_loaded:
            self._load_model_impl()
        
        # Extract frames
        frames, frame_times = self.extract_frames(video_path, self.sample_rate)
        
        if not frames:
            return {
                'segments': [],
                'summary': {'error': 'No frames extracted'}
            }
        
        # Process frames
        result = self.process_batch_gpu(frames, frame_times)
        
        # Add movement analysis
        result['movement_analysis'] = self._analyze_movement_patterns()
        
        # Add summary
        result['summary'] = self._generate_summary(result['segments'])
        
        logger.info(f"[BodyPoseYOLOv8] Completed with {len(result['segments'])} segments")
        return result
    
    def process_batch_gpu(self, frames: List[np.ndarray], 
                         frame_times: List[float]) -> Dict[str, Any]:
        """Process frames for pose detection"""
        segments = []
        
        # Process frames in batches
        batch_size = self.batch_size
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_times = frame_times[i:i+batch_size]
            
            try:
                # Run pose detection on batch
                results = self.model(batch_frames, conf=self.conf_threshold, verbose=False)
                
                # Process each result
                for j, (result, timestamp) in enumerate(zip(results, batch_times)):
                    frame_poses = []
                    
                    if result.keypoints is not None and len(result.keypoints.data) > 0:
                        # Extract poses from result
                        keypoints_data = result.keypoints.data.cpu().numpy()
                        boxes_data = result.boxes.data.cpu().numpy() if result.boxes is not None else None
                        
                        for person_idx, keypoints in enumerate(keypoints_data):
                            # keypoints shape: (17, 3) - x, y, confidence
                            
                            # Get bounding box if available
                            if boxes_data is not None and person_idx < len(boxes_data):
                                box = boxes_data[person_idx]
                                bbox = {
                                    'x': int(box[0]),
                                    'y': int(box[1]),
                                    'width': int(box[2] - box[0]),
                                    'height': int(box[3] - box[1]),
                                    'confidence': float(box[4])
                                }
                            else:
                                # Calculate bbox from keypoints
                                valid_points = keypoints[keypoints[:, 2] > 0.3]
                                if len(valid_points) > 0:
                                    x_min, y_min = valid_points[:, :2].min(axis=0)
                                    x_max, y_max = valid_points[:, :2].max(axis=0)
                                    bbox = {
                                        'x': int(x_min),
                                        'y': int(y_min),
                                        'width': int(x_max - x_min),
                                        'height': int(y_max - y_min),
                                        'confidence': 0.9
                                    }
                                else:
                                    continue
                            
                            # Convert keypoints to dictionary
                            pose_keypoints = {}
                            for k, name in enumerate(self.keypoint_names):
                                if k < len(keypoints):
                                    pose_keypoints[name] = {
                                        'x': float(keypoints[k, 0]),
                                        'y': float(keypoints[k, 1]),
                                        'confidence': float(keypoints[k, 2])
                                    }
                            
                            # Analyze pose
                            pose_analysis = self._analyze_pose(pose_keypoints)
                            
                            # Detect gestures
                            detected_gestures = self._detect_gestures(pose_keypoints)
                            
                            # Analyze body language
                            body_language = self._analyze_body_language(pose_keypoints)
                            
                            # Track person
                            person_id = self._track_person(bbox, pose_keypoints, timestamp)
                            
                            pose_data = {
                                'person_id': person_id,
                                'bbox': bbox,
                                'keypoints': pose_keypoints,
                                'pose_analysis': pose_analysis,
                                'gestures': detected_gestures,
                                'body_language': body_language,
                                'movement_state': self._analyze_movement_state(person_id)
                            }
                            
                            frame_poses.append(pose_data)
                    
                    # Create segment
                    segment = {
                        'timestamp': round(timestamp, 2),
                        'people_detected': len(frame_poses),
                        'poses': frame_poses,
                        'scene_analysis': self._analyze_scene_poses(frame_poses)
                    }
                    
                    segments.append(segment)
                    
                    # Update pose history
                    if frame_poses:
                        self.pose_history.append({
                            'timestamp': timestamp,
                            'poses': frame_poses
                        })
                    
            except Exception as e:
                logger.error(f"[BodyPoseYOLOv8] Error processing batch: {e}")
                # Add empty segments for failed batch
                for timestamp in batch_times:
                    segments.append({
                        'timestamp': round(timestamp, 2),
                        'people_detected': 0,
                        'error': str(e)
                    })
        
        return {
            'segments': segments,
            'metadata': {
                'model': self.model_size,
                'total_people': len(self.person_tracks)
            }
        }
    
    def _analyze_pose(self, keypoints: Dict) -> Dict[str, Any]:
        """Analyze overall pose characteristics"""
        analysis = {}
        
        # Check if pose is complete
        valid_keypoints = sum(1 for kp in keypoints.values() if kp['confidence'] > 0.3)
        analysis['completeness'] = valid_keypoints / len(self.keypoint_names)
        
        # Analyze body orientation
        if all(k in keypoints for k in ['left_shoulder', 'right_shoulder']):
            left_shoulder = keypoints['left_shoulder']
            right_shoulder = keypoints['right_shoulder']
            
            if left_shoulder['confidence'] > 0.3 and right_shoulder['confidence'] > 0.3:
                shoulder_width = abs(right_shoulder['x'] - left_shoulder['x'])
                
                # Estimate orientation
                if shoulder_width < 30:  # Threshold depends on image size
                    analysis['orientation'] = 'side_view'
                elif left_shoulder['x'] > right_shoulder['x']:
                    analysis['orientation'] = 'back_view'
                else:
                    analysis['orientation'] = 'front_view'
            else:
                analysis['orientation'] = 'unknown'
        
        # Analyze pose symmetry
        analysis['symmetry'] = self._calculate_pose_symmetry(keypoints)
        
        # Analyze pose openness
        analysis['openness'] = self._calculate_pose_openness(keypoints)
        
        # Activity level
        analysis['activity_level'] = self._estimate_activity_level(keypoints)
        
        return analysis
    
    def _detect_gestures(self, keypoints: Dict) -> List[Dict]:
        """Detect specific gestures"""
        detected = []
        
        for gesture_name, check_func in self.gesture_patterns.items():
            result = check_func(keypoints)
            if result['detected']:
                detected.append({
                    'gesture': gesture_name,
                    'confidence': result['confidence'],
                    'description': result.get('description', gesture_name)
                })
        
        return detected
    
    def _check_hands_up(self, keypoints: Dict) -> Dict:
        """Check if hands are raised"""
        try:
            left_wrist = keypoints.get('left_wrist', {})
            right_wrist = keypoints.get('right_wrist', {})
            left_shoulder = keypoints.get('left_shoulder', {})
            right_shoulder = keypoints.get('right_shoulder', {})
            
            if (left_wrist.get('confidence', 0) > 0.3 and 
                left_shoulder.get('confidence', 0) > 0.3 and
                right_wrist.get('confidence', 0) > 0.3 and 
                right_shoulder.get('confidence', 0) > 0.3):
                
                # Check if wrists are above shoulders
                left_up = left_wrist['y'] < left_shoulder['y']
                right_up = right_wrist['y'] < right_shoulder['y']
                
                if left_up and right_up:
                    return {'detected': True, 'confidence': 0.9, 'description': 'Both hands raised'}
                elif left_up or right_up:
                    return {'detected': True, 'confidence': 0.7, 'description': 'One hand raised'}
        except:
            pass
        
        return {'detected': False, 'confidence': 0}
    
    def _check_pointing(self, keypoints: Dict) -> Dict:
        """Check if person is pointing"""
        try:
            for side in ['left', 'right']:
                wrist = keypoints.get(f'{side}_wrist', {})
                elbow = keypoints.get(f'{side}_elbow', {})
                shoulder = keypoints.get(f'{side}_shoulder', {})
                
                if (wrist.get('confidence', 0) > 0.3 and 
                    elbow.get('confidence', 0) > 0.3 and
                    shoulder.get('confidence', 0) > 0.3):
                    
                    # Check if arm is extended
                    arm_vector = np.array([wrist['x'] - shoulder['x'], wrist['y'] - shoulder['y']])
                    arm_length = np.linalg.norm(arm_vector)
                    
                    elbow_vector = np.array([elbow['x'] - shoulder['x'], elbow['y'] - shoulder['y']])
                    elbow_length = np.linalg.norm(elbow_vector)
                    
                    if arm_length > elbow_length * 1.5:  # Arm is extended
                        return {
                            'detected': True, 
                            'confidence': 0.8,
                            'description': f'Pointing with {side} hand'
                        }
        except:
            pass
        
        return {'detected': False, 'confidence': 0}
    
    def _check_crossed_arms(self, keypoints: Dict) -> Dict:
        """Check if arms are crossed"""
        try:
            left_wrist = keypoints.get('left_wrist', {})
            right_wrist = keypoints.get('right_wrist', {})
            left_elbow = keypoints.get('left_elbow', {})
            right_elbow = keypoints.get('right_elbow', {})
            
            if all(kp.get('confidence', 0) > 0.3 for kp in [left_wrist, right_wrist, left_elbow, right_elbow]):
                # Check if wrists are on opposite sides of body
                left_crossed = left_wrist['x'] > right_elbow['x']
                right_crossed = right_wrist['x'] < left_elbow['x']
                
                if left_crossed and right_crossed:
                    return {'detected': True, 'confidence': 0.85, 'description': 'Arms crossed'}
        except:
            pass
        
        return {'detected': False, 'confidence': 0}
    
    def _check_hands_on_hips(self, keypoints: Dict) -> Dict:
        """Check if hands are on hips"""
        try:
            left_wrist = keypoints.get('left_wrist', {})
            right_wrist = keypoints.get('right_wrist', {})
            left_hip = keypoints.get('left_hip', {})
            right_hip = keypoints.get('right_hip', {})
            
            if all(kp.get('confidence', 0) > 0.3 for kp in [left_wrist, right_wrist, left_hip, right_hip]):
                # Check proximity of wrists to hips
                left_distance = np.sqrt((left_wrist['x'] - left_hip['x'])**2 + 
                                      (left_wrist['y'] - left_hip['y'])**2)
                right_distance = np.sqrt((right_wrist['x'] - right_hip['x'])**2 + 
                                       (right_wrist['y'] - right_hip['y'])**2)
                
                threshold = 50  # pixels
                if left_distance < threshold and right_distance < threshold:
                    return {'detected': True, 'confidence': 0.8, 'description': 'Hands on hips'}
        except:
            pass
        
        return {'detected': False, 'confidence': 0}
    
    def _check_waving(self, keypoints: Dict) -> Dict:
        """Check if person is waving (requires history)"""
        # This would need temporal analysis
        # Simplified version - check if hand is raised and moving
        hands_up = self._check_hands_up(keypoints)
        if hands_up['detected']:
            # Would need to track hand movement over time
            return {'detected': True, 'confidence': 0.6, 'description': 'Possible waving gesture'}
        
        return {'detected': False, 'confidence': 0}
    
    def _check_thumbs_up(self, keypoints: Dict) -> Dict:
        """Check for thumbs up gesture"""
        # YOLOv8-Pose doesn't include finger keypoints
        # Would need additional hand detection model
        return {'detected': False, 'confidence': 0}
    
    def _check_clapping(self, keypoints: Dict) -> Dict:
        """Check if person is clapping"""
        try:
            left_wrist = keypoints.get('left_wrist', {})
            right_wrist = keypoints.get('right_wrist', {})
            
            if left_wrist.get('confidence', 0) > 0.3 and right_wrist.get('confidence', 0) > 0.3:
                # Check if hands are close together in front of body
                distance = np.sqrt((left_wrist['x'] - right_wrist['x'])**2 + 
                                 (left_wrist['y'] - right_wrist['y'])**2)
                
                if distance < 50:  # threshold
                    # Check if hands are in front of body
                    nose = keypoints.get('nose', {})
                    if nose.get('confidence', 0) > 0.3:
                        hands_y = (left_wrist['y'] + right_wrist['y']) / 2
                        if nose['y'] < hands_y < nose['y'] + 200:  # hands at chest level
                            return {'detected': True, 'confidence': 0.7, 'description': 'Clapping'}
        except:
            pass
        
        return {'detected': False, 'confidence': 0}
    
    def _analyze_body_language(self, keypoints: Dict) -> Dict:
        """Analyze body language patterns"""
        patterns = {}
        
        for pattern_name, check_func in self.body_language_patterns.items():
            result = check_func(keypoints)
            if result['detected']:
                patterns[pattern_name] = {
                    'confidence': result['confidence'],
                    'description': result.get('description', '')
                }
        
        # Determine dominant body language
        if patterns:
            dominant = max(patterns.items(), key=lambda x: x[1]['confidence'])
            return {
                'dominant': dominant[0],
                'patterns': patterns,
                'description': self._describe_body_language(dominant[0], patterns)
            }
        else:
            return {
                'dominant': 'neutral',
                'patterns': {},
                'description': 'Neutral body posture'
            }
    
    def _check_open_posture(self, keypoints: Dict) -> Dict:
        """Check for open body posture"""
        try:
            # Check if arms are not crossed and shoulders are back
            arms_crossed = self._check_crossed_arms(keypoints)
            if not arms_crossed['detected']:
                # Check shoulder alignment
                left_shoulder = keypoints.get('left_shoulder', {})
                right_shoulder = keypoints.get('right_shoulder', {})
                
                if left_shoulder.get('confidence', 0) > 0.3 and right_shoulder.get('confidence', 0) > 0.3:
                    shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
                    if shoulder_width > 50:  # Shoulders are apart
                        return {'detected': True, 'confidence': 0.8, 'description': 'Open and approachable'}
        except:
            pass
        
        return {'detected': False, 'confidence': 0}
    
    def _check_closed_posture(self, keypoints: Dict) -> Dict:
        """Check for closed body posture"""
        arms_crossed = self._check_crossed_arms(keypoints)
        if arms_crossed['detected']:
            return {'detected': True, 'confidence': 0.85, 'description': 'Closed/defensive posture'}
        
        return {'detected': False, 'confidence': 0}
    
    def _check_confident_posture(self, keypoints: Dict) -> Dict:
        """Check for confident posture"""
        hands_on_hips = self._check_hands_on_hips(keypoints)
        if hands_on_hips['detected']:
            return {'detected': True, 'confidence': 0.8, 'description': 'Confident stance'}
        
        # Check for straight posture
        try:
            nose = keypoints.get('nose', {})
            left_hip = keypoints.get('left_hip', {})
            right_hip = keypoints.get('right_hip', {})
            
            if all(kp.get('confidence', 0) > 0.3 for kp in [nose, left_hip, right_hip]):
                hip_center_x = (left_hip['x'] + right_hip['x']) / 2
                alignment = abs(nose['x'] - hip_center_x)
                if alignment < 30:  # Good alignment
                    return {'detected': True, 'confidence': 0.7, 'description': 'Upright confident posture'}
        except:
            pass
        
        return {'detected': False, 'confidence': 0}
    
    def _check_relaxed_posture(self, keypoints: Dict) -> Dict:
        """Check for relaxed posture"""
        # Simplified check - arms at sides
        try:
            left_wrist = keypoints.get('left_wrist', {})
            right_wrist = keypoints.get('right_wrist', {})
            left_hip = keypoints.get('left_hip', {})
            right_hip = keypoints.get('right_hip', {})
            
            if all(kp.get('confidence', 0) > 0.3 for kp in [left_wrist, right_wrist, left_hip, right_hip]):
                # Check if arms are hanging naturally
                left_natural = abs(left_wrist['x'] - left_hip['x']) < 100
                right_natural = abs(right_wrist['x'] - right_hip['x']) < 100
                
                if left_natural and right_natural:
                    return {'detected': True, 'confidence': 0.7, 'description': 'Relaxed posture'}
        except:
            pass
        
        return {'detected': False, 'confidence': 0}
    
    def _check_tense_posture(self, keypoints: Dict) -> Dict:
        """Check for tense posture"""
        # Check for raised shoulders
        try:
            left_shoulder = keypoints.get('left_shoulder', {})
            right_shoulder = keypoints.get('right_shoulder', {})
            left_ear = keypoints.get('left_ear', {})
            right_ear = keypoints.get('right_ear', {})
            
            if all(kp.get('confidence', 0) > 0.3 for kp in [left_shoulder, right_shoulder, left_ear, right_ear]):
                # Check if shoulders are raised (close to ears)
                left_distance = abs(left_shoulder['y'] - left_ear['y'])
                right_distance = abs(right_shoulder['y'] - right_ear['y'])
                
                if left_distance < 50 and right_distance < 50:
                    return {'detected': True, 'confidence': 0.7, 'description': 'Tense/stressed posture'}
        except:
            pass
        
        return {'detected': False, 'confidence': 0}
    
    def _check_leaning(self, keypoints: Dict) -> Dict:
        """Check if person is leaning"""
        try:
            nose = keypoints.get('nose', {})
            left_hip = keypoints.get('left_hip', {})
            right_hip = keypoints.get('right_hip', {})
            
            if all(kp.get('confidence', 0) > 0.3 for kp in [nose, left_hip, right_hip]):
                hip_center_x = (left_hip['x'] + right_hip['x']) / 2
                lean_distance = nose['x'] - hip_center_x
                
                if abs(lean_distance) > 50:
                    direction = 'forward' if lean_distance > 0 else 'backward'
                    return {
                        'detected': True, 
                        'confidence': 0.75,
                        'description': f'Leaning {direction}'
                    }
        except:
            pass
        
        return {'detected': False, 'confidence': 0}
    
    def _calculate_pose_symmetry(self, keypoints: Dict) -> float:
        """Calculate how symmetric the pose is"""
        symmetry_pairs = [
            ('left_shoulder', 'right_shoulder'),
            ('left_elbow', 'right_elbow'),
            ('left_wrist', 'right_wrist'),
            ('left_hip', 'right_hip'),
            ('left_knee', 'right_knee'),
            ('left_ankle', 'right_ankle')
        ]
        
        symmetry_scores = []
        
        for left, right in symmetry_pairs:
            if left in keypoints and right in keypoints:
                left_kp = keypoints[left]
                right_kp = keypoints[right]
                
                if left_kp['confidence'] > 0.3 and right_kp['confidence'] > 0.3:
                    # Calculate vertical alignment difference
                    y_diff = abs(left_kp['y'] - right_kp['y'])
                    symmetry_score = max(0, 1 - y_diff / 100)  # Normalize
                    symmetry_scores.append(symmetry_score)
        
        return np.mean(symmetry_scores) if symmetry_scores else 0.5
    
    def _calculate_pose_openness(self, keypoints: Dict) -> float:
        """Calculate how open/expansive the pose is"""
        # Calculate span of pose
        x_coords = []
        y_coords = []
        
        for kp in keypoints.values():
            if kp['confidence'] > 0.3:
                x_coords.append(kp['x'])
                y_coords.append(kp['y'])
        
        if len(x_coords) > 5:
            x_span = max(x_coords) - min(x_coords)
            y_span = max(y_coords) - min(y_coords)
            
            # Normalize (these values depend on image size)
            openness = min(1.0, (x_span * y_span) / (640 * 480))
            return openness
        
        return 0.5
    
    def _estimate_activity_level(self, keypoints: Dict) -> str:
        """Estimate activity level from pose"""
        # Check if person is likely sitting, standing, or active
        
        # Simple heuristic based on vertical span
        y_coords = [kp['y'] for kp in keypoints.values() if kp['confidence'] > 0.3]
        
        if len(y_coords) > 5:
            y_span = max(y_coords) - min(y_coords)
            
            if y_span < 200:  # Compressed pose
                return 'sitting'
            elif y_span < 400:
                return 'standing'
            else:
                return 'active'
        
        return 'unknown'
    
    def _track_person(self, bbox: Dict, keypoints: Dict, timestamp: float) -> str:
        """Track person identity across frames"""
        # Simple centroid tracking
        center_x = bbox['x'] + bbox['width'] / 2
        center_y = bbox['y'] + bbox['height'] / 2
        
        min_distance = float('inf')
        matched_id = None
        distance_threshold = 100
        
        for person_id, track in self.person_tracks.items():
            if timestamp - track['last_seen'] < 2.0:  # 2 second timeout
                prev_center = track['center']
                distance = np.sqrt((center_x - prev_center[0])**2 + 
                                 (center_y - prev_center[1])**2)
                
                if distance < min_distance and distance < distance_threshold:
                    min_distance = distance
                    matched_id = person_id
        
        if matched_id:
            # Update existing track
            self.person_tracks[matched_id].update({
                'center': (center_x, center_y),
                'last_seen': timestamp,
                'bbox': bbox,
                'appearances': self.person_tracks[matched_id]['appearances'] + 1
            })
            return matched_id
        else:
            # Create new track
            new_id = f"person_{len(self.person_tracks)}"
            self.person_tracks[new_id] = {
                'center': (center_x, center_y),
                'bbox': bbox,
                'first_seen': timestamp,
                'last_seen': timestamp,
                'appearances': 1
            }
            return new_id
    
    def _analyze_movement_state(self, person_id: str) -> str:
        """Analyze movement state of a person"""
        if person_id not in self.person_tracks:
            return 'unknown'
        
        track = self.person_tracks[person_id]
        
        # Would need velocity calculation from history
        # Simplified version
        if track['appearances'] > 5:
            return 'stationary'
        else:
            return 'moving'
    
    def _analyze_scene_poses(self, poses: List[Dict]) -> Dict[str, Any]:
        """Analyze overall scene from all poses"""
        if not poses:
            return {
                'activity': 'empty',
                'interaction': 'none',
                'description': 'No people detected'
            }
        
        # Count gestures and body language
        all_gestures = []
        all_body_language = []
        
        for pose in poses:
            all_gestures.extend(pose.get('gestures', []))
            if 'body_language' in pose:
                all_body_language.append(pose['body_language']['dominant'])
        
        # Determine scene activity
        if len(poses) == 1:
            activity = 'individual'
        elif len(poses) == 2:
            activity = 'pair'
        else:
            activity = 'group'
        
        # Check for interactions
        interaction = 'none'
        if len(poses) > 1:
            # Check if people are facing each other
            # Simplified - would need more sophisticated analysis
            interaction = 'possible_interaction'
        
        # Generate description
        gesture_summary = f"{len(all_gestures)} gestures detected" if all_gestures else "No specific gestures"
        
        from collections import Counter
        if all_body_language:
            body_language_counts = Counter(all_body_language)
            dominant_mood = body_language_counts.most_common(1)[0][0]
            mood_desc = f"Overall {dominant_mood} body language"
        else:
            mood_desc = "Neutral body language"
        
        return {
            'activity': activity,
            'people_count': len(poses),
            'interaction': interaction,
            'gesture_summary': gesture_summary,
            'mood': mood_desc,
            'description': f"{len(poses)} people, {mood_desc.lower()}"
        }
    
    def _describe_body_language(self, dominant: str, patterns: Dict) -> str:
        """Create description of body language"""
        descriptions = {
            'open': "Open and approachable body language",
            'closed': "Closed or defensive posture",
            'confident': "Confident and assertive stance",
            'relaxed': "Relaxed and comfortable posture",
            'tense': "Tense or stressed body language",
            'leaning': "Engaged, leaning posture"
        }
        
        base = descriptions.get(dominant, "Neutral posture")
        
        # Add additional patterns
        if len(patterns) > 1:
            other_patterns = [p for p in patterns.keys() if p != dominant]
            base += f" with hints of {', '.join(other_patterns)}"
        
        return base
    
    def _analyze_movement_patterns(self) -> Dict[str, Any]:
        """Analyze movement patterns from pose history"""
        if len(self.pose_history) < 2:
            return {'movement_type': 'insufficient_data'}
        
        # Analyze changes in poses over time
        movement_events = []
        
        for i in range(1, len(self.pose_history)):
            prev = self.pose_history[i-1]
            curr = self.pose_history[i]
            
            # Check for significant gesture changes
            prev_gestures = set()
            curr_gestures = set()
            
            for pose in prev['poses']:
                for gesture in pose.get('gestures', []):
                    prev_gestures.add(gesture['gesture'])
            
            for pose in curr['poses']:
                for gesture in pose.get('gestures', []):
                    curr_gestures.add(gesture['gesture'])
            
            new_gestures = curr_gestures - prev_gestures
            if new_gestures:
                movement_events.append({
                    'timestamp': curr['timestamp'],
                    'type': 'new_gesture',
                    'gestures': list(new_gestures)
                })
        
        return {
            'movement_events': movement_events,
            'activity_level': self._calculate_overall_activity(),
            'gesture_frequency': len(movement_events) / len(self.pose_history) if self.pose_history else 0
        }
    
    def _calculate_overall_activity(self) -> str:
        """Calculate overall activity level"""
        if not self.pose_history:
            return 'none'
        
        # Count total movements and gestures
        total_gestures = 0
        for entry in self.pose_history:
            for pose in entry['poses']:
                total_gestures += len(pose.get('gestures', []))
        
        avg_gestures = total_gestures / len(self.pose_history)
        
        if avg_gestures > 2:
            return 'high'
        elif avg_gestures > 0.5:
            return 'moderate'
        elif avg_gestures > 0:
            return 'low'
        else:
            return 'minimal'
    
    def _generate_summary(self, segments: List[Dict]) -> Dict[str, Any]:
        """Generate analysis summary"""
        if not segments:
            return {'status': 'no_data'}
        
        # Collect statistics
        total_people = len(self.person_tracks)
        total_poses = sum(s['people_detected'] for s in segments)
        
        # Gesture statistics
        all_gestures = []
        for segment in segments:
            for pose in segment.get('poses', []):
                all_gestures.extend(pose.get('gestures', []))
        
        from collections import Counter
        gesture_counts = Counter(g['gesture'] for g in all_gestures)
        
        # Body language statistics
        body_language_counts = Counter()
        for segment in segments:
            for pose in segment.get('poses', []):
                if 'body_language' in pose:
                    body_language_counts[pose['body_language']['dominant']] += 1
        
        # Activity analysis
        activity_levels = []
        for segment in segments:
            for pose in segment.get('poses', []):
                if 'pose_analysis' in pose:
                    activity_levels.append(pose['pose_analysis'].get('activity_level', 'unknown'))
        
        activity_counts = Counter(activity_levels)
        
        summary = {
            'status': 'success',
            'total_segments': len(segments),
            'unique_people': total_people,
            'total_pose_detections': total_poses,
            'gesture_statistics': {
                'total': len(all_gestures),
                'unique': len(gesture_counts),
                'most_common': dict(gesture_counts.most_common(5))
            },
            'body_language_distribution': dict(body_language_counts),
            'activity_distribution': dict(activity_counts),
            'dominant_body_language': body_language_counts.most_common(1)[0][0] if body_language_counts else 'none',
            'gesture_variety': len(gesture_counts),
            'interaction_detected': any(s.get('scene_analysis', {}).get('interaction') != 'none' for s in segments)
        }
        
        return summary