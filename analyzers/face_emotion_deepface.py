#!/usr/bin/env python3
"""
Face Detection and Emotion Recognition using DeepFace
Combines face detection with emotion analysis for comprehensive facial understanding
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

# DeepFace imports
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logging.warning("DeepFace not available. Install with: pip install deepface")

logger = logging.getLogger(__name__)

class FaceEmotionDeepFace(GPUBatchAnalyzer):
    """Face detection and emotion recognition using DeepFace"""
    
    def __init__(self):
        super().__init__(batch_size=4)  # DeepFace is memory intensive
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models_loaded = False
        
        # Configuration
        self.sample_rate = 15  # Every 0.5 seconds at 30fps
        self.detector_backend = 'retinaface'  # Most accurate
        self.emotion_model = 'Emotion'
        self.analyze_age_gender = True  # Also get age/gender
        
        # Emotion mapping
        self.emotion_map = {
            'angry': 'wütend',
            'disgust': 'angewidert', 
            'fear': 'ängstlich',
            'happy': 'glücklich',
            'sad': 'traurig',
            'surprise': 'überrascht',
            'neutral': 'neutral'
        }
        
        # Face tracking
        self.face_tracks = {}
        self.next_face_id = 0
        
        logger.info("[FaceEmotionDeepFace] Initialized with DeepFace backend")
    
    def _load_model_impl(self):
        """Load DeepFace models"""
        if self.models_loaded:
            return
        
        if not DEEPFACE_AVAILABLE:
            raise RuntimeError("DeepFace not installed. Cannot load models.")
        
        try:
            logger.info("[FaceEmotionDeepFace] Loading DeepFace models...")
            
            # Pre-load models by running a dummy analysis
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            dummy_img[30:70, 30:70] = 255  # White square for face
            
            # This will download and cache models on first run
            try:
                _ = DeepFace.analyze(
                    dummy_img,
                    actions=['emotion'],
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    silent=True
                )
            except:
                pass  # Expected to fail, just loading models
            
            self.models_loaded = True
            logger.info(f"[FaceEmotionDeepFace] ✅ DeepFace models loaded with {self.detector_backend}")
            
        except Exception as e:
            logger.error(f"[FaceEmotionDeepFace] Failed to load DeepFace: {e}")
            raise RuntimeError(f"Failed to load DeepFace models: {e}")
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Analyze video for faces and emotions"""
        logger.info(f"[FaceEmotionDeepFace] Starting analysis of {video_path}")
        
        # Load models if needed
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
        
        # Add emotional journey analysis
        result['emotional_journey'] = self._analyze_emotional_journey(result['segments'])
        
        # Add summary
        result['summary'] = self._generate_summary(result['segments'])
        
        logger.info(f"[FaceEmotionDeepFace] Completed with {len(result['segments'])} segments")
        return result
    
    def process_batch_gpu(self, frames: List[np.ndarray], 
                         frame_times: List[float]) -> Dict[str, Any]:
        """Process frames for face and emotion detection"""
        segments = []
        
        for frame_idx, (frame, timestamp) in enumerate(zip(frames, frame_times)):
            try:
                # Analyze frame with DeepFace
                analyses = DeepFace.analyze(
                    frame,
                    actions=['emotion', 'age', 'gender'] if self.analyze_age_gender else ['emotion'],
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    silent=True
                )
                
                # DeepFace returns list if multiple faces
                if not isinstance(analyses, list):
                    analyses = [analyses]
                
                frame_faces = []
                
                for face_data in analyses:
                    # Skip if no face detected
                    if 'region' not in face_data:
                        continue
                    
                    # Extract face region
                    region = face_data['region']
                    bbox = {
                        'x': region['x'],
                        'y': region['y'],
                        'width': region['w'],
                        'height': region['h']
                    }
                    
                    # Extract emotions
                    emotions = face_data.get('emotion', {})
                    dominant_emotion = face_data.get('dominant_emotion', 'neutral')
                    
                    # Extract additional attributes
                    age = face_data.get('age', None)
                    gender = face_data.get('gender', {})
                    dominant_gender = face_data.get('dominant_gender', 'unknown')
                    
                    # Track face identity
                    face_center = (
                        bbox['x'] + bbox['width'] // 2,
                        bbox['y'] + bbox['height'] // 2
                    )
                    face_id = self._track_face(face_center, bbox, timestamp)
                    
                    # Analyze emotion intensity and authenticity
                    emotion_analysis = self._analyze_emotion_depth(emotions, dominant_emotion)
                    
                    # Facial expression analysis
                    expression_features = self._analyze_facial_expression(emotions)
                    
                    face_info = {
                        'face_id': face_id,
                        'bbox': bbox,
                        'emotions': {k: round(v, 3) for k, v in emotions.items()},
                        'dominant_emotion': dominant_emotion,
                        'dominant_emotion_de': self.emotion_map.get(dominant_emotion, dominant_emotion),
                        'emotion_confidence': round(emotions.get(dominant_emotion, 0), 3),
                        'emotion_intensity': emotion_analysis['intensity'],
                        'emotion_authenticity': emotion_analysis['authenticity'],
                        'emotion_complexity': emotion_analysis['complexity'],
                        'expression_features': expression_features,
                        'age': int(age) if age else None,
                        'gender': dominant_gender,
                        'gender_confidence': round(gender.get(dominant_gender, 0), 3) if isinstance(gender, dict) else None,
                        'face_quality': self._assess_face_quality(frame, bbox)
                    }
                    
                    frame_faces.append(face_info)
                
                # Create segment
                segment = {
                    'timestamp': round(timestamp, 2),
                    'faces_detected': len(frame_faces),
                    'faces': frame_faces,
                    'scene_emotion': self._analyze_scene_emotion(frame_faces),
                    'interaction_analysis': self._analyze_face_interactions(frame_faces) if len(frame_faces) > 1 else None
                }
                
                segments.append(segment)
                
            except Exception as e:
                logger.error(f"[FaceEmotionDeepFace] Error at {timestamp}s: {e}")
                segments.append({
                    'timestamp': round(timestamp, 2),
                    'faces_detected': 0,
                    'error': str(e)
                })
        
        # Clean old face tracks
        self._clean_face_tracks(frame_times[-1] if frame_times else 0)
        
        return {
            'segments': segments,
            'metadata': {
                'unique_faces': len(self.face_tracks),
                'detector': self.detector_backend,
                'emotion_model': self.emotion_model
            }
        }
    
    def _track_face(self, center: Tuple[int, int], bbox: Dict, 
                    timestamp: float) -> str:
        """Track face identity across frames"""
        # Simple distance-based tracking
        min_distance = float('inf')
        matched_id = None
        distance_threshold = 100  # pixels
        
        current_time = timestamp
        
        for face_id, track in self.face_tracks.items():
            # Skip if face hasn't been seen recently
            if current_time - track['last_seen'] > 2.0:  # 2 seconds
                continue
            
            # Calculate distance
            prev_center = track['center']
            distance = np.sqrt(
                (center[0] - prev_center[0])**2 + 
                (center[1] - prev_center[1])**2
            )
            
            if distance < min_distance and distance < distance_threshold:
                min_distance = distance
                matched_id = face_id
        
        if matched_id:
            # Update existing track
            self.face_tracks[matched_id]['center'] = center
            self.face_tracks[matched_id]['last_seen'] = timestamp
            self.face_tracks[matched_id]['bbox'] = bbox
            return matched_id
        else:
            # Create new track
            new_id = f"face_{self.next_face_id}"
            self.next_face_id += 1
            self.face_tracks[new_id] = {
                'center': center,
                'bbox': bbox,
                'last_seen': timestamp,
                'first_seen': timestamp
            }
            return new_id
    
    def _clean_face_tracks(self, current_time: float):
        """Remove old face tracks"""
        timeout = 5.0  # seconds
        tracks_to_remove = []
        
        for face_id, track in self.face_tracks.items():
            if current_time - track['last_seen'] > timeout:
                tracks_to_remove.append(face_id)
        
        for face_id in tracks_to_remove:
            del self.face_tracks[face_id]
    
    def _analyze_emotion_depth(self, emotions: Dict[str, float], 
                              dominant: str) -> Dict[str, Any]:
        """Analyze emotion intensity and authenticity"""
        # Calculate intensity (strength of dominant emotion)
        intensity = emotions.get(dominant, 0)
        
        # Calculate authenticity (how much dominant emotion stands out)
        emotion_values = list(emotions.values())
        if len(emotion_values) > 1:
            sorted_values = sorted(emotion_values, reverse=True)
            authenticity = sorted_values[0] - sorted_values[1]
        else:
            authenticity = intensity
        
        # Calculate complexity (how many emotions are present)
        significant_emotions = sum(1 for v in emotion_values if v > 10)
        
        # Classify intensity
        if intensity > 80:
            intensity_level = 'very_high'
        elif intensity > 60:
            intensity_level = 'high'
        elif intensity > 40:
            intensity_level = 'moderate'
        elif intensity > 20:
            intensity_level = 'low'
        else:
            intensity_level = 'very_low'
        
        # Classify authenticity
        if authenticity > 50:
            authenticity_level = 'genuine'
        elif authenticity > 30:
            authenticity_level = 'clear'
        elif authenticity > 15:
            authenticity_level = 'mixed'
        else:
            authenticity_level = 'ambiguous'
        
        # Classify complexity
        if significant_emotions >= 4:
            complexity = 'very_complex'
        elif significant_emotions >= 3:
            complexity = 'complex'
        elif significant_emotions >= 2:
            complexity = 'mixed'
        else:
            complexity = 'simple'
        
        return {
            'intensity': intensity_level,
            'intensity_score': round(intensity, 2),
            'authenticity': authenticity_level,
            'authenticity_score': round(authenticity, 2),
            'complexity': complexity,
            'significant_emotions': significant_emotions
        }
    
    def _analyze_facial_expression(self, emotions: Dict[str, float]) -> Dict[str, Any]:
        """Analyze facial expression characteristics"""
        features = {}
        
        # Positive vs negative emotion balance
        positive_emotions = ['happy', 'surprise']
        negative_emotions = ['angry', 'disgust', 'fear', 'sad']
        
        positive_score = sum(emotions.get(e, 0) for e in positive_emotions)
        negative_score = sum(emotions.get(e, 0) for e in negative_emotions)
        neutral_score = emotions.get('neutral', 0)
        
        # Determine valence
        if positive_score > negative_score + 20:
            features['valence'] = 'positive'
        elif negative_score > positive_score + 20:
            features['valence'] = 'negative'
        else:
            features['valence'] = 'neutral'
        
        # Arousal level (emotional activation)
        high_arousal = ['angry', 'fear', 'surprise', 'happy']
        low_arousal = ['sad', 'neutral']
        
        arousal_score = sum(emotions.get(e, 0) for e in high_arousal)
        if arousal_score > 60:
            features['arousal'] = 'high'
        elif arousal_score > 30:
            features['arousal'] = 'moderate'
        else:
            features['arousal'] = 'low'
        
        # Expression strength
        max_emotion = max(emotions.values()) if emotions else 0
        features['expression_strength'] = 'strong' if max_emotion > 70 else 'moderate' if max_emotion > 40 else 'subtle'
        
        # Micro-expression indicators
        emotion_variance = np.var(list(emotions.values())) if len(emotions) > 1 else 0
        features['micro_expression_likely'] = emotion_variance < 100 and max_emotion < 50
        
        return features
    
    def _assess_face_quality(self, frame: np.ndarray, bbox: Dict) -> Dict[str, Any]:
        """Assess face detection quality"""
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        
        quality = {}
        
        # Size quality (larger faces = better)
        face_area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        size_ratio = face_area / frame_area
        
        if size_ratio > 0.2:
            quality['size'] = 'large'
        elif size_ratio > 0.1:
            quality['size'] = 'medium'
        elif size_ratio > 0.05:
            quality['size'] = 'small'
        else:
            quality['size'] = 'tiny'
        
        # Position quality (centered = better)
        center_x = x + w/2
        center_y = y + h/2
        frame_center_x = frame.shape[1] / 2
        frame_center_y = frame.shape[0] / 2
        
        distance_from_center = np.sqrt(
            (center_x - frame_center_x)**2 + 
            (center_y - frame_center_y)**2
        )
        
        if distance_from_center < frame.shape[1] * 0.2:
            quality['position'] = 'centered'
        elif distance_from_center < frame.shape[1] * 0.4:
            quality['position'] = 'off_center'
        else:
            quality['position'] = 'peripheral'
        
        # Blur detection
        if face_roi.size > 0:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var > 500:
                quality['sharpness'] = 'sharp'
            elif laplacian_var > 100:
                quality['sharpness'] = 'normal'
            else:
                quality['sharpness'] = 'blurry'
        else:
            quality['sharpness'] = 'unknown'
        
        # Overall quality score
        quality_score = 0
        if quality['size'] in ['large', 'medium']:
            quality_score += 0.4
        if quality['position'] == 'centered':
            quality_score += 0.3
        if quality['sharpness'] in ['sharp', 'normal']:
            quality_score += 0.3
        
        quality['overall'] = 'excellent' if quality_score > 0.8 else 'good' if quality_score > 0.5 else 'fair'
        quality['score'] = round(quality_score, 2)
        
        return quality
    
    def _analyze_scene_emotion(self, faces: List[Dict]) -> Dict[str, Any]:
        """Analyze overall emotional tone of the scene"""
        if not faces:
            return {
                'dominant': 'none',
                'intensity': 'none',
                'description': 'No faces detected'
            }
        
        # Aggregate emotions across all faces
        emotion_totals = {}
        for face in faces:
            for emotion, value in face.get('emotions', {}).items():
                emotion_totals[emotion] = emotion_totals.get(emotion, 0) + value
        
        # Average emotions
        num_faces = len(faces)
        emotion_averages = {k: v/num_faces for k, v in emotion_totals.items()}
        
        # Find dominant scene emotion
        if emotion_averages:
            dominant = max(emotion_averages, key=emotion_averages.get)
            intensity = emotion_averages[dominant]
        else:
            dominant = 'neutral'
            intensity = 0
        
        # Classify scene mood
        if dominant == 'happy' and intensity > 50:
            mood = 'joyful'
        elif dominant in ['angry', 'disgust'] and intensity > 40:
            mood = 'tense'
        elif dominant in ['sad', 'fear'] and intensity > 40:
            mood = 'somber'
        elif dominant == 'surprise' and intensity > 50:
            mood = 'dramatic'
        elif dominant == 'neutral' or intensity < 30:
            mood = 'calm'
        else:
            mood = 'mixed'
        
        # Generate description
        if num_faces == 1:
            desc = f"Single person showing {self.emotion_map.get(dominant, dominant)}"
        else:
            desc = f"{num_faces} people, overall {mood} mood"
        
        return {
            'dominant': dominant,
            'dominant_de': self.emotion_map.get(dominant, dominant),
            'intensity': round(intensity, 2),
            'mood': mood,
            'emotion_diversity': len([e for e in emotion_averages.values() if e > 10]),
            'description': desc
        }
    
    def _analyze_face_interactions(self, faces: List[Dict]) -> Dict[str, Any]:
        """Analyze interactions between multiple faces"""
        if len(faces) < 2:
            return None
        
        # Analyze emotional synchrony
        emotions_list = [f.get('emotions', {}) for f in faces]
        
        # Calculate emotion similarity between faces
        similarities = []
        for i in range(len(faces)):
            for j in range(i+1, len(faces)):
                similarity = self._calculate_emotion_similarity(
                    emotions_list[i], emotions_list[j]
                )
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0
        
        # Determine interaction type
        dominant_emotions = [f.get('dominant_emotion', 'neutral') for f in faces]
        
        if avg_similarity > 0.7:
            if all(e == dominant_emotions[0] for e in dominant_emotions):
                interaction_type = 'synchronized'
            else:
                interaction_type = 'harmonious'
        elif avg_similarity > 0.4:
            interaction_type = 'connected'
        elif avg_similarity > 0.2:
            interaction_type = 'independent'
        else:
            interaction_type = 'contrasting'
        
        # Check for specific interaction patterns
        patterns = []
        
        # All happy
        if all(e == 'happy' for e in dominant_emotions):
            patterns.append('group_joy')
        
        # Mixed positive/negative
        positive = sum(1 for e in dominant_emotions if e in ['happy', 'surprise'])
        negative = sum(1 for e in dominant_emotions if e in ['angry', 'sad', 'fear', 'disgust'])
        if positive > 0 and negative > 0:
            patterns.append('emotional_contrast')
        
        # Leader-follower (one strong emotion, others neutral)
        emotion_intensities = [f.get('emotion_confidence', 0) for f in faces]
        if max(emotion_intensities) > 70 and min(emotion_intensities) < 30:
            patterns.append('leader_follower')
        
        return {
            'type': interaction_type,
            'emotional_similarity': round(avg_similarity, 3),
            'patterns': patterns,
            'face_count': len(faces),
            'description': f"{interaction_type.capitalize()} emotional dynamic between {len(faces)} people"
        }
    
    def _calculate_emotion_similarity(self, emotions1: Dict, emotions2: Dict) -> float:
        """Calculate similarity between two emotion sets"""
        # Get all emotion keys
        all_emotions = set(emotions1.keys()) | set(emotions2.keys())
        
        if not all_emotions:
            return 0.0
        
        # Calculate cosine similarity
        vec1 = [emotions1.get(e, 0) for e in all_emotions]
        vec2 = [emotions2.get(e, 0) for e in all_emotions]
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = np.sqrt(sum(a**2 for a in vec1))
        magnitude2 = np.sqrt(sum(b**2 for b in vec2))
        
        if magnitude1 * magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _analyze_emotional_journey(self, segments: List[Dict]) -> List[Dict]:
        """Analyze emotional progression through video"""
        journey = []
        
        # Group segments into time windows
        window_size = 3.0  # 3 second windows
        current_window = []
        window_start = 0
        
        for segment in segments:
            if segment['timestamp'] - window_start > window_size:
                if current_window:
                    journey.append(self._summarize_emotion_window(
                        current_window, window_start
                    ))
                window_start = segment['timestamp']
                current_window = [segment]
            else:
                current_window.append(segment)
        
        # Add final window
        if current_window:
            journey.append(self._summarize_emotion_window(
                current_window, window_start
            ))
        
        # Add transitions
        for i in range(1, len(journey)):
            prev = journey[i-1]
            curr = journey[i]
            
            if prev['dominant_emotion'] != curr['dominant_emotion']:
                transition = self._describe_emotion_transition(
                    prev['dominant_emotion'],
                    curr['dominant_emotion']
                )
                journey[i]['transition'] = transition
        
        return journey
    
    def _summarize_emotion_window(self, segments: List[Dict], 
                                 start_time: float) -> Dict:
        """Summarize emotions in time window"""
        all_emotions = {}
        total_faces = 0
        
        for segment in segments:
            for face in segment.get('faces', []):
                total_faces += 1
                for emotion, value in face.get('emotions', {}).items():
                    all_emotions[emotion] = all_emotions.get(emotion, 0) + value
        
        if total_faces > 0:
            # Average emotions
            avg_emotions = {k: v/total_faces for k, v in all_emotions.items()}
            dominant = max(avg_emotions, key=avg_emotions.get) if avg_emotions else 'neutral'
            
            # Determine stability
            emotion_values = list(avg_emotions.values())
            stability = 1.0 - (np.std(emotion_values) / 100) if len(emotion_values) > 1 else 1.0
        else:
            dominant = 'none'
            avg_emotions = {}
            stability = 0
        
        return {
            'timestamp': round(start_time, 2),
            'dominant_emotion': dominant,
            'dominant_emotion_de': self.emotion_map.get(dominant, dominant),
            'emotions': {k: round(v, 2) for k, v in avg_emotions.items()},
            'face_count': total_faces,
            'stability': round(stability, 3),
            'description': self._describe_emotion_state(dominant, total_faces)
        }
    
    def _describe_emotion_state(self, emotion: str, face_count: int) -> str:
        """Create description of emotional state"""
        if face_count == 0:
            return "No faces visible"
        
        emotion_de = self.emotion_map.get(emotion, emotion)
        
        if face_count == 1:
            return f"Person ist {emotion_de}"
        else:
            return f"{face_count} Personen, überwiegend {emotion_de}"
    
    def _describe_emotion_transition(self, from_emotion: str, to_emotion: str) -> str:
        """Describe emotional transition"""
        from_de = self.emotion_map.get(from_emotion, from_emotion)
        to_de = self.emotion_map.get(to_emotion, to_emotion)
        
        # Classify transition type
        positive = ['happy', 'surprise']
        negative = ['angry', 'sad', 'fear', 'disgust']
        
        if from_emotion in negative and to_emotion in positive:
            transition_type = "Stimmungsaufhellung"
        elif from_emotion in positive and to_emotion in negative:
            transition_type = "Stimmungsverschlechterung"
        elif from_emotion == 'neutral' and to_emotion != 'neutral':
            transition_type = "Emotionale Aktivierung"
        elif from_emotion != 'neutral' and to_emotion == 'neutral':
            transition_type = "Emotionale Beruhigung"
        else:
            transition_type = "Stimmungswechsel"
        
        return f"{transition_type}: {from_de} → {to_de}"
    
    def _generate_summary(self, segments: List[Dict]) -> Dict[str, Any]:
        """Generate analysis summary"""
        if not segments:
            return {'status': 'no_data'}
        
        # Collect all face data
        all_faces = []
        all_emotions = {}
        
        for segment in segments:
            for face in segment.get('faces', []):
                all_faces.append(face)
                for emotion, value in face.get('emotions', {}).items():
                    all_emotions[emotion] = all_emotions.get(emotion, 0) + value
        
        if not all_faces:
            return {
                'status': 'no_faces',
                'total_segments': len(segments)
            }
        
        # Calculate statistics
        num_faces = len(all_faces)
        unique_faces = len(set(f['face_id'] for f in all_faces if 'face_id' in f))
        
        # Average emotions
        avg_emotions = {k: v/num_faces for k, v in all_emotions.items()}
        dominant_overall = max(avg_emotions, key=avg_emotions.get) if avg_emotions else 'neutral'
        
        # Emotion distribution
        from collections import Counter
        dominant_emotions = [f.get('dominant_emotion', 'neutral') for f in all_faces]
        emotion_counts = Counter(dominant_emotions)
        
        # Calculate emotional diversity
        emotion_diversity = len([e for e, c in emotion_counts.items() if c > num_faces * 0.1])
        
        summary = {
            'status': 'success',
            'total_segments': len(segments),
            'total_faces': num_faces,
            'unique_individuals': unique_faces,
            'dominant_emotion': dominant_overall,
            'dominant_emotion_de': self.emotion_map.get(dominant_overall, dominant_overall),
            'emotion_distribution': dict(emotion_counts),
            'average_emotions': {k: round(v, 2) for k, v in avg_emotions.items()},
            'emotional_diversity': emotion_diversity,
            'emotional_consistency': round(
                emotion_counts.most_common(1)[0][1] / num_faces, 3
            ) if emotion_counts else 0,
            'face_quality': {
                'excellent': sum(1 for f in all_faces if f.get('face_quality', {}).get('overall') == 'excellent'),
                'good': sum(1 for f in all_faces if f.get('face_quality', {}).get('overall') == 'good'),
                'fair': sum(1 for f in all_faces if f.get('face_quality', {}).get('overall') == 'fair')
            }
        }
        
        # Add demographic summary if available
        ages = [f.get('age') for f in all_faces if f.get('age')]
        genders = [f.get('gender') for f in all_faces if f.get('gender') != 'unknown']
        
        if ages:
            summary['age_stats'] = {
                'mean': round(np.mean(ages), 1),
                'min': min(ages),
                'max': max(ages)
            }
        
        if genders:
            gender_counts = Counter(genders)
            summary['gender_distribution'] = dict(gender_counts)
        
        return summary