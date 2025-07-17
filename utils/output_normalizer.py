#!/usr/bin/env python3
"""
Output Normalizer - Standardisiert Analyzer Outputs für konsistente Datenextraktion
Löst das Problem inkonsistenter Feldnamen zwischen verschiedenen Analyzern
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class AnalyzerOutputNormalizer:
    """
    Standardisiert alle Analyzer Outputs auf ein einheitliches Schema
    """
    
    # Field mappings für jeden Analyzer
    FIELD_MAPPINGS = {
        'eye_tracking': {
            # Map German and alternative field names to standard
            'gaze_direction_general': 'gaze_direction',
            'blickrichtung': 'gaze_direction',
            'augen_zustand': 'eye_state',
            'eye_state': 'eye_state',
            'gaze_confidence': 'confidence',
            'pupillary_distance': 'pupil_distance',
            'looking_at_camera': 'camera_contact',
            'eye_contact': 'camera_contact'
        },
        'speech_rate': {
            # Standardize pitch fields
            'average_pitch': 'pitch_hz',
            'pitch_range': 'pitch_range_hz',
            'pitch_std': 'pitch_std_hz',
            'words_per_minute': 'wpm',
            'speech_rate': 'wpm'
        },
        'object_detection': {
            # Normalize object class fields
            'object': 'object_class',
            'class': 'object_class',
            'label': 'object_class',
            'category': 'object_category',
            'confidence': 'confidence_score',
            'bbox': 'bounding_box',
            'bbox_exact': 'bounding_box'
        },
        'product_detection': {
            # Same structure as object detection
            'product': 'product_class',
            'class': 'product_class',
            'label': 'product_class',
            'category': 'product_category',
            'confidence': 'confidence_score',
            'bbox': 'bounding_box'
        },
        'face_detection': {
            'dominant_emotion': 'emotion',
            'emotion_confidence': 'emotion_score',
            'age': 'estimated_age',
            'gender': 'detected_gender'
        },
        'emotion_detection': {
            'dominant_emotion': 'emotion',
            'dominant_emotion_de': 'emotion_de',
            'emotions': 'emotion_scores'
        },
        'age_estimation': {
            'age': 'estimated_age',
            'age_range': 'age_range',
            'confidence': 'confidence_score'
        },
        'text_overlay': {
            'timestamp': 'time',
            'start_time': 'time',
            'texts': 'text_blocks',  # Changed to match expected output
            'text': 'text',  # Keep original text field
            'combined_text': 'text',  # Map combined_text to text
            'confidence': 'confidence_score'
        },
        'camera_analysis': {
            'movement': 'camera_movement',
            'movement_type': 'camera_movement',
            'shot_type': 'shot_info',
            'description': 'movement_description'
        },
        'visual_effects': {
            'effects': 'detected_effects',
            'effect_type': 'effect_category',
            'intensity': 'effect_intensity'
        }
    }
    
    # Expected output structure for validation
    EXPECTED_STRUCTURE = {
        'segments': list,  # All analyzers should have segments
        'metadata': dict   # Optional metadata
    }
    
    def __init__(self):
        self.normalization_stats = {
            'total_processed': 0,
            'fields_normalized': 0,
            'analyzers_processed': set()
        }
    
    def normalize(self, analyzer_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalisiert Analyzer Output auf einheitliches Schema
        
        Args:
            analyzer_name: Name des Analyzers
            data: Original Analyzer Output
            
        Returns:
            Normalisierter Output mit standardisierten Feldnamen
        """
        if not data:
            return data
        
        self.normalization_stats['total_processed'] += 1
        self.normalization_stats['analyzers_processed'].add(analyzer_name)
        
        # Normalize segments if present
        if 'segments' in data and isinstance(data['segments'], list):
            data['segments'] = self._normalize_segments(analyzer_name, data['segments'])
        
        # Normalize metadata if present
        if 'metadata' in data and isinstance(data['metadata'], dict):
            data['metadata'] = self._normalize_dict(analyzer_name, data['metadata'])
        
        # Special handling for analyzers that don't use segments structure
        if analyzer_name in ['audio_analysis', 'temporal_flow'] and 'segments' not in data:
            # These analyzers might have different structures
            data = self._normalize_dict(analyzer_name, data)
        
        return data
    
    def _normalize_segments(self, analyzer_name: str, segments: List[Dict]) -> List[Dict]:
        """Normalize each segment in the segments list"""
        normalized_segments = []
        
        for segment in segments:
            if isinstance(segment, dict):
                normalized_segment = self._normalize_dict(analyzer_name, segment)
                
                # Special handling for object detection - flatten if needed
                if analyzer_name == 'object_detection':
                    normalized_segment = self._ensure_standard_object_structure(normalized_segment)
                elif analyzer_name == 'product_detection':
                    normalized_segment = self._ensure_standard_product_structure(normalized_segment)
                
                normalized_segments.append(normalized_segment)
            else:
                normalized_segments.append(segment)
        
        return normalized_segments
    
    def _normalize_dict(self, analyzer_name: str, data_dict: Dict) -> Dict:
        """Normalize field names in a dictionary"""
        if analyzer_name not in self.FIELD_MAPPINGS:
            return data_dict
        
        mapping = self.FIELD_MAPPINGS[analyzer_name]
        normalized = {}
        
        for key, value in data_dict.items():
            # Check if this field needs mapping
            if key in mapping:
                new_key = mapping[key]
                normalized[new_key] = value
                self.normalization_stats['fields_normalized'] += 1
                logger.debug(f"[{analyzer_name}] Normalized field: {key} -> {new_key}")
            else:
                # Keep original field
                normalized[key] = value
            
            # Recursively normalize nested dictionaries
            if isinstance(value, dict):
                normalized[key] = self._normalize_dict(analyzer_name, value)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # List of dictionaries
                normalized[key] = [self._normalize_dict(analyzer_name, item) for item in value]
        
        return normalized
    
    def _ensure_standard_object_structure(self, segment: Dict) -> Dict:
        """
        Ensure object detection data follows standard structure
        Some analyzers put object data directly in segment instead of 'objects' array
        """
        if 'objects' not in segment and ('object_class' in segment or 'object' in segment):
            # Move object data into standard structure
            object_data = {
                'object_class': segment.pop('object_class', segment.pop('object', None)),
                'confidence_score': segment.pop('confidence_score', segment.pop('confidence', 0)),
                'bounding_box': segment.pop('bounding_box', segment.pop('bbox', [])),
                'object_category': segment.pop('object_category', segment.pop('category', None)),
                'object_id': segment.pop('object_id', None),
                'class': segment.pop('class', None),
                'label': segment.pop('label', None)
            }
            # Remove None values
            object_data = {k: v for k, v in object_data.items() if v is not None}
            
            # Add to objects array
            segment['objects'] = [object_data]
        
        return segment
    
    def _ensure_standard_product_structure(self, segment: Dict) -> Dict:
        """
        Ensure product detection data follows standard structure
        """
        if 'products' not in segment and ('product_class' in segment or 'product' in segment):
            # Move product data into standard structure
            product_data = {
                'product_class': segment.pop('product_class', segment.pop('product', None)),
                'confidence_score': segment.pop('confidence_score', segment.pop('confidence', 0)),
                'bounding_box': segment.pop('bounding_box', segment.pop('bbox', [])),
                'product_category': segment.pop('product_category', segment.pop('category', None)),
                'product_id': segment.pop('product_id', None),
                'class': segment.pop('class', None),
                'label': segment.pop('label', None)
            }
            # Remove None values
            product_data = {k: v for k, v in product_data.items() if v is not None}
            
            # Add to products array
            segment['products'] = [product_data]
        
        return segment
    
    def get_normalization_stats(self) -> Dict[str, Any]:
        """Get statistics about normalization process"""
        return {
            **self.normalization_stats,
            'analyzers_processed': list(self.normalization_stats['analyzers_processed'])
        }
    
    def validate_output(self, analyzer_name: str, data: Dict[str, Any]) -> List[str]:
        """
        Validate that output follows expected structure
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check for required fields
        if 'segments' not in data and analyzer_name not in ['audio_analysis', 'temporal_flow']:
            errors.append(f"{analyzer_name}: Missing 'segments' field")
        
        # Check segment structure
        if 'segments' in data:
            if not isinstance(data['segments'], list):
                errors.append(f"{analyzer_name}: 'segments' must be a list")
            elif data['segments']:
                # Check first segment
                first_segment = data['segments'][0]
                if not isinstance(first_segment, dict):
                    errors.append(f"{analyzer_name}: Segments must be dictionaries")
                elif 'timestamp' not in first_segment and 'start_time' not in first_segment:
                    errors.append(f"{analyzer_name}: Segments missing timestamp field")
        
        return errors


def create_unified_field_extractor():
    """
    Create helper functions for extracting commonly needed fields
    """
    
    def extract_gaze_direction(segment: Dict) -> str:
        """Extract gaze direction from various possible fields"""
        return segment.get('gaze_direction', 
               segment.get('gaze_direction_general',
               segment.get('blickrichtung', 'unknown')))
    
    def extract_pitch_data(segment: Dict) -> float:
        """Extract pitch data from various possible fields"""
        return segment.get('pitch_hz',
               segment.get('average_pitch',
               segment.get('pitch', 0.0)))
    
    def extract_emotion(segment: Dict) -> str:
        """Extract emotion from various possible fields"""
        return segment.get('emotion',
               segment.get('dominant_emotion',
               segment.get('detected_emotion', 'neutral')))
    
    def extract_confidence(segment: Dict) -> float:
        """Extract confidence score from various possible fields"""
        return segment.get('confidence_score',
               segment.get('confidence',
               segment.get('score', 0.0)))
    
    return {
        'gaze_direction': extract_gaze_direction,
        'pitch': extract_pitch_data,
        'emotion': extract_emotion,
        'confidence': extract_confidence
    }


# Example usage
if __name__ == "__main__":
    normalizer = AnalyzerOutputNormalizer()
    
    # Test eye tracking normalization
    eye_data = {
        'segments': [
            {
                'timestamp': 1.0,
                'gaze_direction_general': 'in_kamera',
                'eye_state': 'open',
                'gaze_confidence': 0.95
            }
        ]
    }
    
    normalized = normalizer.normalize('eye_tracking', eye_data)
    print("Eye tracking normalized:", normalized['segments'][0])
    
    # Test speech rate normalization  
    speech_data = {
        'segments': [
            {
                'timestamp': 1.0,
                'average_pitch': 150.5,
                'pitch_range': [100, 200],
                'words_per_minute': 120
            }
        ]
    }
    
    normalized = normalizer.normalize('speech_rate', speech_data)
    print("Speech rate normalized:", normalized['segments'][0])
    
    print("\nNormalization stats:", normalizer.get_normalization_stats())