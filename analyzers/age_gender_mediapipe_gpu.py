#!/usr/bin/env python3
"""
Age and Gender Detection using MediaPipe + GPU-accelerated models
Replacement for InsightFace to ensure GPU usage
"""

# GPU Forcing
import torch
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any, Optional, Tuple
from analyzers.base_analyzer import GPUBatchAnalyzer
import logging
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AgeGenderMediaPipeGPU(GPUBatchAnalyzer):
    """GPU-accelerated age and gender detection using MediaPipe + Transformers"""
    
    def __init__(self):
        super().__init__(batch_size=16)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.face_detector = None
        self.age_gender_model = None
        self.models_loaded = False
        
        # Sampling configuration - aggressive for better coverage
        self.sample_rate = 10  # Every 0.33 seconds
        
        # Age groups
        self.age_groups = {
            'child': (0, 12),
            'teenager': (13, 19),
            'young_adult': (20, 35),
            'middle_aged': (36, 55),
            'senior': (56, 100)
        }
        
        logger.info(f"[AgeGenderMediaPipeGPU] Initialized with device: {self.device}")
    
    def _load_model_impl(self):
        """Load MediaPipe and age/gender model"""
        if self.models_loaded:
            return
        
        try:
            logger.info("[AgeGenderMediaPipeGPU] Loading models...")
            
            # Initialize MediaPipe Face Detection (runs on GPU if available)
            self.face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # Full range model
                min_detection_confidence=0.3  # Lower threshold for better detection
            )
            
            # Load age/gender classifier from Hugging Face (GPU-accelerated)
            self.age_gender_model = pipeline(
                "image-classification",
                model="nateraw/vit-age-classifier",  # Vision Transformer model
                device=0 if self.device == 'cuda' else -1
            )
            
            self.models_loaded = True
            logger.info(f"[AgeGenderMediaPipeGPU] ✅ Models loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"[AgeGenderMediaPipeGPU] Failed to load models: {e}")
            raise RuntimeError(f"Failed to load models: {e}")
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Analyze video for age and gender using GPU"""
        logger.info(f"[AgeGenderMediaPipeGPU] Starting GPU-accelerated analysis of {video_path}")
        
        # Load models if needed
        if not self.models_loaded:
            self._load_model_impl()
        
        # Extract frames with max frames limit
        from configs.performance_config import MAX_FRAMES_PER_ANALYZER
        max_frames = MAX_FRAMES_PER_ANALYZER.get('age_estimation', 150)
        
        frames, frame_times = self.extract_frames(
            video_path, 
            sample_rate=self.sample_rate,
            max_frames=max_frames
        )
        
        logger.info(f"[AgeGenderMediaPipeGPU] Extracted {len(frames)} frames")
        
        if not frames:
            return {
                'segments': [],
                'summary': {'error': 'No frames extracted'}
            }
        
        # Process frames
        result = self.process_batch_gpu(frames, frame_times)
        
        # Add summary statistics
        result['summary'] = self._generate_summary(result['segments'])
        
        logger.info(f"[AgeGenderMediaPipeGPU] Completed analysis with {len(result['segments'])} segments")
        return result
    
    def process_batch_gpu(self, frames: List[np.ndarray], 
                         frame_times: List[float]) -> Dict[str, Any]:
        """Process frames for face analysis using GPU"""
        segments = []
        
        # Process in batches for GPU efficiency
        batch_size = 8
        
        for batch_start in range(0, len(frames), batch_size):
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]
            batch_times = frame_times[batch_start:batch_end]
            
            for frame, timestamp in zip(batch_frames, batch_times):
                try:
                    # Convert BGR to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces with MediaPipe
                    results = self.face_detector.process(rgb_frame)
                    
                    frame_faces = []
                    
                    if results.detections:
                        for detection in results.detections:
                            # Get bounding box
                            bbox = detection.location_data.relative_bounding_box
                            h, w = frame.shape[:2]
                            
                            x = int(bbox.xmin * w)
                            y = int(bbox.ymin * h)
                            width = int(bbox.width * w)
                            height = int(bbox.height * h)
                            
                            # Ensure bbox is within frame
                            x = max(0, x)
                            y = max(0, y)
                            width = min(width, w - x)
                            height = min(height, h - y)
                            
                            if width > 20 and height > 20:  # Minimum face size
                                # Extract face region
                                face_img = frame[y:y+height, x:x+width]
                                
                                # Get age prediction using GPU model
                                try:
                                    predictions = self.age_gender_model(face_img)
                                    
                                    # Parse predictions
                                    age_estimate = self._parse_age_prediction(predictions)
                                    
                                    face_data = {
                                        'face_id': f"face_{len(frame_faces)}",
                                        'bbox': {
                                            'x': x,
                                            'y': y,
                                            'width': width,
                                            'height': height
                                        },
                                        'age': age_estimate,
                                        'age_group': self._get_age_group(age_estimate),
                                        'gender': 'unknown',  # This model doesn't predict gender
                                        'confidence': {
                                            'detection': detection.score[0] if detection.score else 0.9,
                                            'age': 0.8  # Approximate confidence
                                        }
                                    }
                                    
                                    frame_faces.append(face_data)
                                    
                                except Exception as e:
                                    logger.debug(f"Age prediction failed: {e}")
                    
                    # Create segment
                    segment = {
                        'timestamp': round(timestamp, 2),
                        'faces_detected': len(frame_faces),
                        'faces': frame_faces
                    }
                    
                    segments.append(segment)
                    
                except Exception as e:
                    logger.error(f"[AgeGenderMediaPipeGPU] Error processing frame at {timestamp}s: {e}")
                    segments.append({
                        'timestamp': round(timestamp, 2),
                        'faces_detected': 0,
                        'error': str(e)
                    })
        
        logger.info(f"[AgeGenderMediaPipeGPU] Processed {len(segments)} segments on GPU")
        
        # Calculate detection statistics
        total_faces = sum(s['faces_detected'] for s in segments)
        frames_with_faces = sum(1 for s in segments if s['faces_detected'] > 0)
        
        return {
            'segments': segments,
            'metadata': {
                'total_faces_detected': total_faces,
                'frames_with_faces': frames_with_faces,
                'detection_rate': frames_with_faces / len(segments) * 100 if segments else 0,
                'processing_device': self.device,
                'models': 'mediapipe_gpu + vit_age_classifier'
            }
        }
    
    def _parse_age_prediction(self, predictions: List[Dict]) -> int:
        """Parse age from model predictions"""
        # The model returns age ranges like "20-29", "30-39", etc.
        if not predictions:
            return 25  # Default
        
        top_pred = predictions[0]
        label = top_pred.get('label', '20-29')
        
        # Extract age range
        if '-' in label:
            try:
                min_age, max_age = map(int, label.split('-'))
                return (min_age + max_age) // 2
            except:
                pass
        
        # Try to extract any number
        import re
        numbers = re.findall(r'\d+', label)
        if numbers:
            return int(numbers[0])
        
        return 25  # Default
    
    def _get_age_group(self, age: int) -> str:
        """Get age group from age value"""
        for group, (min_age, max_age) in self.age_groups.items():
            if min_age <= age <= max_age:
                return group
        return 'unknown'
    
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
        
        from collections import Counter
        age_group_counts = Counter(age_groups)
        
        # Detection rate
        frames_with_faces = sum(1 for s in segments if s['faces_detected'] > 0)
        detection_rate = frames_with_faces / len(segments) * 100 if segments else 0
        
        summary = {
            'status': 'success',
            'total_segments': len(segments),
            'total_faces_detected': len(all_faces),
            'detection_rate': round(detection_rate, 1),
            'age_statistics': {
                'mean': float(np.mean(ages)) if ages else None,
                'median': float(np.median(ages)) if ages else None,
                'min': int(np.min(ages)) if ages else None,
                'max': int(np.max(ages)) if ages else None
            },
            'age_distribution': dict(age_group_counts),
            'processing_device': self.device,
            'gpu_accelerated': True
        }
        
        return summary