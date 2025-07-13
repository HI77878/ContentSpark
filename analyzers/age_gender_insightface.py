#!/usr/bin/env python3
"""
Advanced Age and Gender Detection using InsightFace
High accuracy face analysis with Buffalo_L model
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
from collections import Counter

# InsightFace imports
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available. Install with: pip install insightface")

logger = logging.getLogger(__name__)

class AgeGenderInsightFace(GPUBatchAnalyzer):
    """Advanced age and gender detection using InsightFace"""
    
    def __init__(self):
        super().__init__(batch_size=8)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.app = None
        self.models_loaded = False
        
        # Sampling configuration - MORE AGGRESSIVE
        self.sample_rate = 10  # Every 0.33 seconds for better coverage
        
        # Age groups with more granularity
        self.age_groups = {
            'infant': (0, 2),
            'toddler': (3, 5),
            'child': (6, 12),
            'teenager': (13, 17),
            'young_adult': (18, 25),
            'adult': (26, 35),
            'middle_aged': (36, 50),
            'mature': (51, 65),
            'senior': (66, 100)
        }
        
        # Gender mapping
        self.gender_map = {
            0: 'female',
            1: 'male'
        }
        
        logger.info("[AgeGenderInsightFace] Initialized with InsightFace backend")
    
    def _load_model_impl(self):
        """Load InsightFace models"""
        if self.models_loaded:
            return
        
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("InsightFace not installed. Cannot load models.")
        
        try:
            logger.info("[AgeGenderInsightFace] Loading InsightFace Buffalo_L model...")
            
            # Initialize FaceAnalysis with Buffalo_L model - force CUDA
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            logger.info(f"[AgeGenderInsightFace] Available ONNX providers: {available_providers}")
            
            # Use CUDA if available
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                logger.info("[AgeGenderInsightFace] Using CUDA for inference")
            else:
                providers = ['CPUExecutionProvider']
                logger.warning("[AgeGenderInsightFace] CUDA not available, using CPU")
            
            self.app = FaceAnalysis(
                name='buffalo_l',
                providers=providers
            )
            
            # Prepare the model with specific input size and LOWER detection threshold
            # Default is 0.5, we use 0.3 for better face detection
            self.app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640), det_thresh=0.3)
            
            self.models_loaded = True
            logger.info("[AgeGenderInsightFace] ✅ InsightFace Buffalo_L loaded successfully")
            
        except Exception as e:
            logger.error(f"[AgeGenderInsightFace] Failed to load InsightFace: {e}")
            raise RuntimeError(f"Failed to load InsightFace models: {e}")
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Analyze video for age and gender"""
        logger.info(f"[AgeGenderInsightFace] Starting analysis of {video_path}")
        
        # Load models if needed
        if not self.models_loaded:
            self._load_model_impl()
        
        # Extract frames - with max frames limit for better coverage
        from configs.performance_config import MAX_FRAMES_PER_ANALYZER
        max_frames = MAX_FRAMES_PER_ANALYZER.get('age_estimation', 150)
        
        frames, frame_times = self.extract_frames(
            video_path, 
            sample_rate=self.sample_rate,
            max_frames=max_frames
        )
        
        logger.info(f"[AgeGenderInsightFace] Extracted {len(frames)} frames with sample_rate={self.sample_rate}, max_frames={max_frames}")
        
        if not frames:
            return {
                'segments': [],
                'summary': {'error': 'No frames extracted'}
            }
        
        # Process frames
        result = self.process_batch_gpu(frames, frame_times)
        
        # Add summary statistics
        result['summary'] = self._generate_summary(result['segments'])
        
        logger.info(f"[AgeGenderInsightFace] Completed analysis with {len(result['segments'])} segments")
        return result
    
    def process_batch_gpu(self, frames: List[np.ndarray], 
                         frame_times: List[float]) -> Dict[str, Any]:
        """Process frames for face analysis"""
        segments = []
        face_tracks = {}  # Track faces across frames
        
        for frame_idx, (frame, timestamp) in enumerate(zip(frames, frame_times)):
            try:
                # Detect and analyze faces
                faces = self.app.get(frame)
                
                frame_faces = []
                
                for face in faces:
                    # Extract face attributes
                    bbox = face.bbox.astype(int)
                    
                    # Age and gender from InsightFace
                    age = int(face.age) if hasattr(face, 'age') else None
                    gender = self.gender_map.get(face.gender, 'unknown') if hasattr(face, 'gender') else 'unknown'
                    
                    # Face quality and pose
                    face_quality = self._assess_face_quality(face)
                    face_pose = self._analyze_face_pose(face)
                    
                    # Additional attributes
                    face_embedding = face.embedding if hasattr(face, 'embedding') else None
                    
                    # Track face identity
                    face_id = self._match_face_identity(face_embedding, face_tracks) if face_embedding is not None else f"face_{len(frame_faces)}"
                    
                    face_data = {
                        'face_id': face_id,
                        'bbox': {
                            'x': int(bbox[0]),
                            'y': int(bbox[1]),
                            'width': int(bbox[2] - bbox[0]),
                            'height': int(bbox[3] - bbox[1])
                        },
                        'age': age,
                        'age_group': self._get_age_group(age) if age else 'unknown',
                        'gender': gender,
                        'confidence': {
                            'detection': float(face.det_score) if hasattr(face, 'det_score') else 0.99,
                            'age': self._estimate_age_confidence(age, face),
                            'gender': 0.95 if gender != 'unknown' else 0.0
                        },
                        'quality': face_quality,
                        'pose': face_pose,
                        'appearance': self._analyze_appearance(face, frame, bbox)
                    }
                    
                    frame_faces.append(face_data)
                    
                    # Update face tracks
                    if face_embedding is not None:
                        face_tracks[face_id] = {
                            'embedding': face_embedding,
                            'last_seen': timestamp,
                            'appearances': face_tracks.get(face_id, {}).get('appearances', 0) + 1
                        }
                
                # Create segment
                segment = {
                    'timestamp': round(timestamp, 2),
                    'faces_detected': len(frame_faces),
                    'faces': frame_faces,
                    'frame_analysis': self._analyze_frame_demographics(frame_faces)
                }
                
                segments.append(segment)
                
            except Exception as e:
                logger.error(f"[AgeGenderInsightFace] Error processing frame at {timestamp}s: {e}")
                segments.append({
                    'timestamp': round(timestamp, 2),
                    'faces_detected': 0,
                    'error': str(e)
                })
        
        # Clean up face tracks
        unique_identities = len(face_tracks)
        
        return {
            'segments': segments,
            'metadata': {
                'unique_identities': unique_identities,
                'total_detections': sum(s['faces_detected'] for s in segments),
                'analysis_method': 'insightface_buffalo_l'
            }
        }
    
    def _assess_face_quality(self, face) -> Dict[str, Any]:
        """Assess face detection quality"""
        quality = {
            'detection_score': float(face.det_score) if hasattr(face, 'det_score') else 0.99,
            'clarity': 'high',  # Would need blur detection
            'occlusion': 'none',  # Would need occlusion model
            'lighting': 'good'  # Would need lighting analysis
        }
        
        # Classify overall quality
        if quality['detection_score'] > 0.9:
            quality['overall'] = 'excellent'
        elif quality['detection_score'] > 0.7:
            quality['overall'] = 'good'
        elif quality['detection_score'] > 0.5:
            quality['overall'] = 'fair'
        else:
            quality['overall'] = 'poor'
        
        return quality
    
    def _analyze_face_pose(self, face) -> Dict[str, Any]:
        """Analyze face pose/orientation"""
        # InsightFace provides landmarks
        if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
            landmarks = face.landmark_3d_68
            
            # Simple pose estimation from landmarks
            # (In production, use proper pose estimation)
            pose = {
                'yaw': 0.0,  # Left-right rotation
                'pitch': 0.0,  # Up-down rotation
                'roll': 0.0,  # Tilt
                'facing': 'forward'
            }
            
            # Estimate facing direction from eye positions
            if len(landmarks) >= 68:
                left_eye_center = np.mean(landmarks[36:42], axis=0)
                right_eye_center = np.mean(landmarks[42:48], axis=0)
                eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
                
                # Very rough estimation
                if eye_distance < 30:  # Threshold depends on image size
                    pose['facing'] = 'side'
                else:
                    pose['facing'] = 'forward'
        else:
            pose = {
                'yaw': 0.0,
                'pitch': 0.0,
                'roll': 0.0,
                'facing': 'unknown'
            }
        
        return pose
    
    def _analyze_appearance(self, face, frame: np.ndarray, bbox: List[int]) -> Dict[str, Any]:
        """Analyze facial appearance characteristics"""
        appearance = {}
        
        # Extract face region
        x1, y1, x2, y2 = bbox
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size > 0:
            # Skin tone analysis (simplified)
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            
            # Average color in face region
            avg_color = cv2.mean(face_roi)[:3]
            avg_hsv = cv2.mean(hsv)[:3]
            
            # Classify skin tone (very simplified - use proper model in production)
            brightness = avg_hsv[2]
            if brightness < 80:
                skin_tone = 'dark'
            elif brightness < 120:
                skin_tone = 'medium'
            elif brightness < 180:
                skin_tone = 'light'
            else:
                skin_tone = 'very_light'
            
            appearance['skin_tone'] = skin_tone
            
            # Hair detection (check top portion of face bbox)
            hair_region = frame[max(0, y1-30):y1+20, x1:x2]
            if hair_region.size > 0:
                hair_color = self._detect_hair_color(hair_region)
                appearance['hair_color'] = hair_color
            
            # Facial hair detection (for males)
            # This would require a specialized model
            appearance['facial_hair'] = 'unknown'
            
            # Accessories (glasses, etc.)
            # This would require additional detection
            appearance['glasses'] = 'unknown'
        
        return appearance
    
    def _detect_hair_color(self, hair_region: np.ndarray) -> str:
        """Simple hair color detection"""
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
        
        # Get dominant color
        avg_hue = cv2.mean(hsv)[0]
        avg_saturation = cv2.mean(hsv)[1]
        avg_value = cv2.mean(hsv)[2]
        
        # Classify hair color (simplified)
        if avg_value < 30:
            return 'black'
        elif avg_saturation < 30:
            if avg_value > 200:
                return 'white/gray'
            else:
                return 'gray'
        elif 10 < avg_hue < 20:
            return 'brown'
        elif 20 < avg_hue < 30:
            return 'blonde'
        elif 0 <= avg_hue < 10 or avg_hue > 170:
            return 'red/auburn'
        else:
            return 'colored/other'
    
    def _match_face_identity(self, embedding: np.ndarray, 
                           face_tracks: Dict) -> str:
        """Match face to existing identity or create new one"""
        if not face_tracks:
            return "person_0"
        
        # Compare with existing embeddings
        min_distance = float('inf')
        matched_id = None
        threshold = 0.6  # Similarity threshold
        
        for face_id, track_data in face_tracks.items():
            if 'embedding' in track_data:
                # Cosine similarity
                similarity = np.dot(embedding, track_data['embedding']) / (
                    np.linalg.norm(embedding) * np.linalg.norm(track_data['embedding'])
                )
                distance = 1 - similarity
                
                if distance < min_distance and distance < threshold:
                    min_distance = distance
                    matched_id = face_id
        
        if matched_id:
            return matched_id
        else:
            # Create new identity
            return f"person_{len(face_tracks)}"
    
    def _estimate_age_confidence(self, age: Optional[int], face) -> float:
        """Estimate confidence in age prediction"""
        if age is None:
            return 0.0
        
        # Base confidence on detection score
        base_confidence = float(face.det_score) if hasattr(face, 'det_score') else 0.8
        
        # Adjust based on face quality
        # (In production, use face quality metrics)
        
        return min(base_confidence * 0.9, 0.95)
    
    def _get_age_group(self, age: int) -> str:
        """Get age group from age value"""
        for group, (min_age, max_age) in self.age_groups.items():
            if min_age <= age <= max_age:
                return group
        return 'unknown'
    
    def _analyze_frame_demographics(self, faces: List[Dict]) -> Dict[str, Any]:
        """Analyze demographics of all faces in frame"""
        if not faces:
            return {
                'primary_demographic': 'none',
                'age_distribution': {},
                'gender_distribution': {}
            }
        
        # Age distribution
        age_groups = [f['age_group'] for f in faces if f.get('age_group') != 'unknown']
        age_distribution = {}
        for group in age_groups:
            age_distribution[group] = age_distribution.get(group, 0) + 1
        
        # Gender distribution
        genders = [f['gender'] for f in faces if f.get('gender') != 'unknown']
        gender_distribution = {}
        for gender in genders:
            gender_distribution[gender] = gender_distribution.get(gender, 0) + 1
        
        # Determine primary demographic
        if age_groups:
            primary_age = max(age_distribution, key=age_distribution.get)
            primary_gender = max(gender_distribution, key=gender_distribution.get) if gender_distribution else 'unknown'
            primary_demographic = f"{primary_age}_{primary_gender}"
        else:
            primary_demographic = 'unknown'
        
        return {
            'primary_demographic': primary_demographic,
            'age_distribution': age_distribution,
            'gender_distribution': gender_distribution,
            'average_age': np.mean([f['age'] for f in faces if f.get('age')]) if any(f.get('age') for f in faces) else None
        }
    
    def _generate_summary(self, segments: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not segments:
            return {'status': 'no_data'}
        
        # Collect all face data
        all_faces = []
        for segment in segments:
            if 'faces' in segment:
                all_faces.extend(segment['faces'])
        
        if not all_faces:
            return {
                'status': 'no_faces_detected',
                'total_segments': len(segments)
            }
        
        # Age statistics
        ages = [f['age'] for f in all_faces if f.get('age')]
        age_groups = [f['age_group'] for f in all_faces if f.get('age_group') != 'unknown']
        
        # Gender statistics
        genders = [f['gender'] for f in all_faces if f.get('gender') != 'unknown']
        
        # Count unique identities
        unique_ids = set(f['face_id'] for f in all_faces if 'face_id' in f)
        
        # Age group distribution
        from collections import Counter
        age_group_counts = Counter(age_groups)
        gender_counts = Counter(genders)
        
        summary = {
            'status': 'success',
            'total_segments': len(segments),
            'total_faces_detected': len(all_faces),
            'unique_individuals': len(unique_ids),
            'age_statistics': {
                'mean': float(np.mean(ages)) if ages else None,
                'median': float(np.median(ages)) if ages else None,
                'min': int(np.min(ages)) if ages else None,
                'max': int(np.max(ages)) if ages else None,
                'std': float(np.std(ages)) if ages else None
            },
            'age_distribution': dict(age_group_counts),
            'gender_distribution': dict(gender_counts),
            'demographics': {
                'primary_age_group': age_group_counts.most_common(1)[0][0] if age_group_counts else 'unknown',
                'primary_gender': gender_counts.most_common(1)[0][0] if gender_counts else 'unknown',
                'diversity_score': self._calculate_diversity_score(age_group_counts, gender_counts)
            }
        }
        
        # Add temporal analysis
        if len(segments) > 1:
            summary['temporal_analysis'] = self._analyze_temporal_patterns(segments)
        
        return summary
    
    def _calculate_diversity_score(self, age_groups: Counter, genders: Counter) -> float:
        """Calculate demographic diversity score"""
        # Simple diversity measure (0-1)
        age_diversity = len(age_groups) / len(self.age_groups) if age_groups else 0
        gender_diversity = len(genders) / 2 if genders else 0  # Max 2 genders
        
        return round((age_diversity + gender_diversity) / 2, 3)
    
    def _analyze_temporal_patterns(self, segments: List[Dict]) -> Dict[str, Any]:
        """Analyze how demographics change over time"""
        # Track changes in number of faces
        face_counts = [s['faces_detected'] for s in segments]
        
        # Identify patterns
        patterns = {
            'face_count_trend': 'stable',
            'max_faces': max(face_counts),
            'min_faces': min(face_counts),
            'avg_faces': round(np.mean(face_counts), 2)
        }
        
        # Determine trend
        if len(face_counts) > 3:
            first_half = np.mean(face_counts[:len(face_counts)//2])
            second_half = np.mean(face_counts[len(face_counts)//2:])
            
            if second_half > first_half * 1.3:
                patterns['face_count_trend'] = 'increasing'
            elif second_half < first_half * 0.7:
                patterns['face_count_trend'] = 'decreasing'
        
        return patterns