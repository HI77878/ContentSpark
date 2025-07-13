#!/usr/bin/env python3
"""
Face Detection and Emotion Recognition using MediaPipe + FER
Stable alternative to DeepFace with better TensorFlow compatibility
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any, Optional
from analyzers.base_analyzer import GPUBatchAnalyzer
import logging
import warnings
warnings.filterwarnings('ignore')

# Import FER for emotion detection
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    logging.warning("FER not available. Install with: pip install fer")

logger = logging.getLogger(__name__)

class FaceEmotionMediaPipe(GPUBatchAnalyzer):
    """Face detection and emotion recognition using MediaPipe + FER"""
    
    def __init__(self):
        super().__init__(batch_size=8)  # Can handle more frames than DeepFace
        self.models_loaded = False
        
        # Configuration
        self.sample_rate = 15  # Every 0.5 seconds at 30fps
        
        # Initialize detectors
        self.mp_face_detection = None
        self.emotion_detector = None
        
        # Face tracking
        self.face_tracks = {}
        self.next_face_id = 0
        
        logger.info("[FaceEmotionMediaPipe] Initialized")
    
    def _load_models(self):
        """Load MediaPipe and FER models"""
        if self.models_loaded:
            return
            
        try:
            # Store MediaPipe module reference
            self.mp_face_detection_module = mp.solutions.face_detection
            
            # Initialize FER emotion detector
            if FER_AVAILABLE:
                # Use MTCNN for better face detection
                self.emotion_detector = FER(mtcnn=True)
            else:
                logger.error("FER not available - emotion detection disabled")
                
            self.models_loaded = True
            logger.info("[FaceEmotionMediaPipe] ✅ Models loaded successfully")
            
        except Exception as e:
            logger.error(f"[FaceEmotionMediaPipe] Failed to load models: {e}")
            raise
    
    def _detect_faces_mediapipe(self, image):
        """Detect faces using MediaPipe"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use context manager for MediaPipe
        with self.mp_face_detection_module.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        ) as face_detection:
            # Process the image
            results = face_detection.process(image_rgb)
        
        faces = []
        if results.detections:
            h, w = image.shape[:2]
            
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure valid bbox
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                faces.append({
                    'bbox': {'x': x, 'y': y, 'width': width, 'height': height},
                    'confidence': detection.score[0] if detection.score else 0.0
                })
        
        return faces
    
    def _detect_emotions_fer(self, image, faces):
        """Detect emotions using FER"""
        if not FER_AVAILABLE or not self.emotion_detector:
            return faces
            
        # FER can detect faces and emotions in one go
        fer_results = self.emotion_detector.detect_emotions(image)
        
        # Match FER results with MediaPipe faces
        for i, face in enumerate(faces):
            if i < len(fer_results):
                # Add emotion data
                emotions = fer_results[i].get('emotions', {})
                
                # Find dominant emotion
                if emotions:
                    dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                    face['emotion'] = {
                        'dominant': dominant_emotion[0],
                        'confidence': dominant_emotion[1],
                        'all_emotions': emotions
                    }
                else:
                    face['emotion'] = {
                        'dominant': 'neutral',
                        'confidence': 0.0,
                        'all_emotions': {}
                    }
        
        return faces
    
    def _track_faces(self, faces, timestamp):
        """Simple face tracking across frames"""
        tracked_faces = []
        
        for face in faces:
            # Simple tracking based on position
            face_center = (
                face['bbox']['x'] + face['bbox']['width'] // 2,
                face['bbox']['y'] + face['bbox']['height'] // 2
            )
            
            # Find closest existing track
            min_dist = float('inf')
            matched_id = None
            
            for face_id, track in self.face_tracks.items():
                if timestamp - track['last_seen'] > 2.0:  # 2 second timeout
                    continue
                    
                dist = np.sqrt(
                    (face_center[0] - track['center'][0])**2 + 
                    (face_center[1] - track['center'][1])**2
                )
                
                if dist < min_dist and dist < 100:  # 100 pixel threshold
                    min_dist = dist
                    matched_id = face_id
            
            # Assign ID
            if matched_id is not None:
                face['person_id'] = f"person_{matched_id}"
                self.face_tracks[matched_id] = {
                    'center': face_center,
                    'last_seen': timestamp
                }
            else:
                face['person_id'] = f"person_{self.next_face_id}"
                self.face_tracks[self.next_face_id] = {
                    'center': face_center,
                    'last_seen': timestamp
                }
                self.next_face_id += 1
            
            tracked_faces.append(face)
        
        return tracked_faces
    
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> Dict[str, Any]:
        """Process batch of frames for face and emotion detection"""
        if not self.models_loaded:
            self._load_models()
        
        all_segments = []
        
        for frame, timestamp in zip(frames, frame_times):
            try:
                # Detect faces with MediaPipe
                faces = self._detect_faces_mediapipe(frame)
                
                # Detect emotions with FER
                faces = self._detect_emotions_fer(frame, faces)
                
                # Track faces
                faces = self._track_faces(faces, timestamp)
                
                # Create segment
                segment = {
                    'timestamp': float(timestamp),
                    'faces_detected': len(faces),
                    'faces': faces
                }
                
                all_segments.append(segment)
                
                # Log progress
                if len(faces) > 0:
                    emotions = [f['emotion']['dominant'] for f in faces if 'emotion' in f]
                    logger.info(f"   Timestamp {timestamp:.1f}s: {len(faces)} faces, emotions: {emotions}")
                
            except Exception as e:
                logger.error(f"[FaceEmotionMediaPipe] Error at {timestamp}s: {e}")
                segment = {
                    'timestamp': float(timestamp),
                    'faces_detected': 0,
                    'error': str(e)
                }
                all_segments.append(segment)
        
        return {'segments': all_segments}
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Main analysis function"""
        logger.info(f"[FaceEmotionMediaPipe] Starting analysis of {video_path.split('/')[-1]}")
        
        try:
            # Extract frames
            frames, frame_times = self.extract_frames(video_path, sample_rate=self.sample_rate)
            
            if not frames:
                return {'segments': [], 'error': 'No frames extracted'}
            
            logger.info(f"[FaceEmotionMediaPipe] Processing {len(frames)} frames")
            
            # Process in batches
            all_segments = []
            for i in range(0, len(frames), self.batch_size):
                batch_frames = frames[i:i+self.batch_size]
                batch_times = frame_times[i:i+self.batch_size]
                
                result = self.process_batch_gpu(batch_frames, batch_times)
                all_segments.extend(result['segments'])
            
            # Generate summary
            total_faces = sum(s['faces_detected'] for s in all_segments)
            segments_with_faces = sum(1 for s in all_segments if s['faces_detected'] > 0)
            
            # Emotion statistics
            emotion_counts = {}
            for segment in all_segments:
                for face in segment.get('faces', []):
                    if 'emotion' in face:
                        emotion = face['emotion']['dominant']
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            return {
                'segments': all_segments,
                'summary': {
                    'total_segments': len(all_segments),
                    'segments_with_faces': segments_with_faces,
                    'total_faces_detected': total_faces,
                    'emotion_distribution': emotion_counts,
                    'unique_people': self.next_face_id
                }
            }
            
        except Exception as e:
            logger.error(f"[FaceEmotionMediaPipe] Analysis failed: {e}")
            return {'segments': [], 'error': str(e)}
        
        finally:
            # Cleanup face tracking but NOT MediaPipe (handled by context manager)
            self.face_tracks = {}
            self.next_face_id = 0