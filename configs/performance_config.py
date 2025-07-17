"""
<<<<<<< HEAD
Clean Server Performance Configuration - MVP Version
Optimized for 1-second segment processing
"""

# Main performance configuration dictionary
PERFORMANCE_CONFIG = {
    'frame_intervals': {
        # Standard intervals for 30 FPS video
        'dense': 10,        # 3 frames per second (0.33s)
        'default': 15,      # 2 frames per second (0.5s)
        'medium': 20,       # 1.5 frames per second
        'sparse': 30,       # 1 frame per second
        
        # Specific for clean server analyzers
        'video_description': 30,    # 1 frame per second for video understanding
        'face_tracking': 15,        # 2 frames per second for face analysis
        'object_tracking': 15,      # 2 frames per second for object detection
        'text_detection': 15,       # 2 frames per second for text overlay
        'audio_sync': 15,           # 2 frames per second for audio alignment
        'body_tracking': 15,        # 2 frames per second for body pose
        'emotion_tracking': 15,     # 2 frames per second for emotions
    },
    
    'max_frames': {
        # Frame limits for processing efficiency
        'default': 200,
        'video_analysis': 150,
        'face_detection': 200,
        'object_detection': 300,
        'text_overlay': 150,
        'body_pose': 200,
    },
    
    'quality_settings': {
        'min_description_length': 100,   # 100 characters for descriptions
        'require_details': True,         # Force detailed outputs
        'include_position': True,        # Position of objects/persons
        'include_timestamps': True,      # Exact timestamps
=======
PERFORMANCE CONFIGURATION - OPTIMIERT FÜR 1:1 VIDEO REKONSTRUKTION
Hauptziel: JEDE Sekunde muss analysiert werden!
"""

# HAUPTZIEL: 1:1 VIDEO-REKONSTRUKTION
# Jede Sekunde muss analysiert werden!

# Main performance configuration dictionary
PERFORMANCE_CONFIG = {
    'frame_intervals': {
        # FÜR 30 FPS VIDEO:
        'dense': 15,        # 2 Frames pro Sekunde (alle 0.5s)
        'default': 30,      # 1 Frame pro Sekunde (MINIMUM!)
        'medium': 30,       # 1 Frame pro Sekunde  
        'sparse': 60,       # Alle 2 Sekunden (nur für Overview)
        
        # SPEZIFISCH FÜR REKONSTRUKTION:
        'video_description': 30,    # video_llava: JEDE Sekunde beschreiben
        'face_tracking': 15,        # Gesichter 2x pro Sekunde
        'object_tracking': 30,      # Objekte jede Sekunde
        'text_detection': 15,       # Text 2x pro Sekunde (kann schnell wechseln)
        'audio_sync': 30,           # Audio-Visual sync jede Sekunde
        'body_tracking': 15,        # Körperhaltung 2x pro Sekunde
        'emotion_tracking': 30,     # Emotionen jede Sekunde
    },
    
    'max_frames': {
        # Keine Limits - wir brauchen ALLE Daten!
        'default': None,
        'video_analysis': None,
        'face_detection': None,
        'object_detection': None,
        'text_overlay': None,
        'body_pose': None,
    },
    
    'quality_settings': {
        'min_description_length': 100,   # Mindestens 100 Zeichen pro Beschreibung
        'require_details': True,         # Erzwinge detaillierte Ausgaben
        'include_position': True,        # Position von Objekten/Personen
        'include_timestamps': True,      # Exakte Zeitstempel
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
        'include_confidence': True,      # Confidence scores
    },
    
    'batch_sizes': {
<<<<<<< HEAD
        # Optimized batches for clean server
        'heavy_models': 1,    # Single frame for heavy models
        'medium_models': 8,   # 8 frames for medium models
        'light_models': 16,   # 16 frames for light models
        'cpu_models': 1,      # Single frame for CPU
    }
}

# Clean Server Requirements
RECONSTRUCTION_REQUIREMENTS = {
    'temporal_coverage': 1.0,        # Every second must be covered
    'min_analyzers_per_second': 8,   # At least 8 analyzers per second
    'description_detail_level': 'medium',
    'position_accuracy': 'bbox',
    'color_depth': 'standard',
    'motion_tracking': True,
    'facial_expression_detail': 'medium',
    'audio_transcription_accuracy': 0.90,
}

# Analyzer-specific settings for clean server
ANALYZER_SETTINGS = {
    'qwen2_vl_temporal': {
        'sample_rate': 30,
        'max_frames': 150,
        'prompt': "Describe this video segment in detail"
    },
    'object_detection': {
        'sample_rate': 15,
        'confidence_threshold': 0.5,
        'include_all_objects': True
    },
    'face_emotion': {
=======
        # Kleinere Batches für mehr Detail
        'heavy_models': 4,
        'medium_models': 8, 
        'light_models': 16,
        'cpu_models': 1,
    }
}

# FÜR 1:1 REKONSTRUKTION KRITISCH:
RECONSTRUCTION_REQUIREMENTS = {
    'temporal_coverage': 1.0,        # JEDE Sekunde muss abgedeckt sein
    'min_analyzers_per_second': 10,  # Mindestens 10 Analyzer pro Sekunde
    'description_detail_level': 'high',
    'position_accuracy': 'pixel',
    'color_depth': 'full',
    'motion_tracking': True,
    'facial_expression_detail': 'high',
    'audio_transcription_accuracy': 0.95,
}

# Analyzer-spezifische Settings für Rekonstruktion
ANALYZER_SETTINGS = {
    'video_llava': {
        'sample_rate': 30,
        'max_frames': None,
        'prompt': "Describe EVERYTHING in extreme detail for perfect reconstruction"
    },
    'object_detection': {
        'sample_rate': 30,
        'confidence_threshold': 0.3,
        'include_all_objects': True
    },
    'face_detection': {
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
        'sample_rate': 15,
        'min_face_size': 20,
        'track_all_faces': True
    },
    'text_overlay': {
        'sample_rate': 15,
        'min_text_size': 10,
<<<<<<< HEAD
        'languages': ['en']
=======
        'languages': ['en', 'de']
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
    },
    'body_pose': {
        'sample_rate': 15,
        'all_keypoints': True,
<<<<<<< HEAD
        'confidence_threshold': 0.5
=======
        'confidence_threshold': 0.3
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
    }
}

# Required exports for analyzer compatibility
# Export 1: FRAME_EXTRACTION_INTERVALS (full config for backward compatibility)
FRAME_EXTRACTION_INTERVALS = PERFORMANCE_CONFIG

# Export 2: MAX_FRAMES_PER_ANALYZER (specific max frames configuration)
MAX_FRAMES_PER_ANALYZER = {
<<<<<<< HEAD
    "qwen2_vl_temporal": 150,
    "object_detection": 300,
    "text_overlay": 150,
    "body_pose": 200,
    "face_emotion": 200,
    "background_segmentation": 150,
    "visual_effects": 150,
    "product_detection": 200,
=======
    "face_detection": 200,
    "object_detection": 300,
    "text_overlay": 150,
    "body_pose": 200,
    "video_analysis": 250,
    "emotion_detection": 150,
    "hand_gesture": 150,
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
    "eye_tracking": 150,
    "speech_transcription": None,  # No limit for audio
    "audio_analysis": None,        # No limit for audio
    "default": 150
}

# Export 3: OPTIMIZED_FRAME_INTERVALS (frame intervals only)
OPTIMIZED_FRAME_INTERVALS = PERFORMANCE_CONFIG['frame_intervals']

# Additional exports for specific use cases
FRAME_INTERVALS = PERFORMANCE_CONFIG['frame_intervals']
BATCH_SIZES = PERFORMANCE_CONFIG['batch_sizes']
QUALITY_SETTINGS = PERFORMANCE_CONFIG['quality_settings']
MAX_FRAMES = PERFORMANCE_CONFIG['max_frames']