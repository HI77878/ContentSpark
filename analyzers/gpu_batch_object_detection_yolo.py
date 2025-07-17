#!/usr/bin/env python3

# GPU Forcing
import torch
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

"""
YOLOv8 Object Detection with TRUE Batch Processing
Replaces DETR for 10x faster processing and better GPU utilization
"""
from ultralytics import YOLO
import torch
import numpy as np
from typing import List, Dict, Any
from analyzers.base_analyzer import GPUBatchAnalyzer
import logging

logger = logging.getLogger(__name__)

# YOLO category mapping for better organization
YOLO_CATEGORIES = {
    'person': ['person'],
    'animal': ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
    'vehicle': ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'],
    'furniture': ['chair', 'couch', 'bed', 'dining table', 'toilet'],
    'electronic': ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster'],
    'kitchen': ['refrigerator', 'sink', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'],
    'sport': ['frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket'],
    'household': ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'potted plant', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'],
    'outdoor': ['bench', 'parking meter', 'stop sign', 'fire hydrant', 'traffic light'],
    'accessory': ['backpack', 'umbrella', 'handbag', 'tie', 'suitcase'],
    'other': ['book', 'clock']
}

class GPUBatchObjectDetectionYOLO(GPUBatchAnalyzer):
    def __init__(self):
        super().__init__(batch_size=64)  # Erhöht von 32 auf 64
        self.device = 'cuda'
        self.model = None
        self.sample_rate = 10  # Sample every 0.33 seconds for DENSE coverage
<<<<<<< HEAD
        self.conf_threshold = 0.2   # Lowered for more detections
=======
        self.conf_threshold = 0.25  # Balanced threshold for quality detections
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
        self.iou_threshold = 0.45  # Standard IOU
        self.track_objects = True  # Enable object tracking across frames
    
    def _load_model_impl(self):
        """Load YOLOv8l for good balance of speed and quality"""
        logger.info("[ObjectDetection-YOLO] Loading YOLOv8x model...")
        
        # Use YOLOv8x for maximum detection quality
        self.model = YOLO('yolov8x.pt')
        self.model.cuda()
        
        # Set to eval mode and optimize
        self.model.model.eval()
        self.model.model.half()  # FP16 for faster inference
        
        # Warm up the model
        dummy_input = torch.zeros((1, 3, 640, 640)).cuda().half()
        with torch.inference_mode():
            _ = self.model.model(dummy_input)
        
        logger.info(f"✅ YOLOv8x loaded on {self.device} with FP16")
    
    def _get_object_category(self, object_name: str) -> str:
        """Get category for a detected object"""
        object_name = object_name.lower()
        for category, items in YOLO_CATEGORIES.items():
            if object_name in items:
                return category
        return 'other'
    
    def _categorize_frame_objects(self, detections: List[Dict]) -> Dict[str, Any]:
        """Categorize objects detected in a frame"""
        category_counts = {}
        categorized_objects = {}
        
        for detection in detections:
            object_name = detection.get('object', 'unknown').lower()
            category = self._get_object_category(object_name)
            
            # Count categories
            if category not in category_counts:
                category_counts[category] = 0
                categorized_objects[category] = []
            
            category_counts[category] += 1
            categorized_objects[category].append({
                'name': object_name,
                'confidence': detection.get('confidence', 0),
                'bbox': detection.get('bbox', [])
            })
        
        return {
            'category_counts': category_counts,
            'categorized_objects': categorized_objects,
            'dominant_category': max(category_counts, key=category_counts.get) if category_counts else None,
            'total_objects': sum(category_counts.values())
        }
        
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> Dict[str, Any]:
        """Process frames in batches for maximum GPU utilization"""
        logger.info(f"[ObjectDetection-YOLO] Processing {len(frames)} frames in batches")
        
        segments = []
        batch_size = self.batch_size  # Use configured batch size (64)
        
        # Process frames in batches
        for batch_start in range(0, len(frames), batch_size):
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]
            batch_times = frame_times[batch_start:batch_end]
            
            logger.info(f"🚀 Processing batch {batch_start//batch_size + 1} with {len(batch_frames)} frames...")
            
            # YOLOv8 can handle batch inference directly
            with torch.no_grad():
                # Process batch - YOLOv8 handles numpy arrays directly
                results = self.model(
                    batch_frames, 
                    device=self.device,
                    batch=len(batch_frames),  # Process all at once
                    verbose=False,
                    conf=self.conf_threshold,  # Use configured threshold
                    iou=self.iou_threshold,
                    half=True,  # Use FP16
                    imgsz=640  # Standard YOLO input size
                )
            
            # Process results - create one segment per frame
            for idx, (result, timestamp) in enumerate(zip(results, batch_times)):
                frame_objects = []
                frame_h, frame_w = batch_frames[idx % len(batch_frames)].shape[:2]
                
                # Process all detected objects in this frame
                if result.boxes is not None and len(result.boxes) > 0:
                    # Convert all boxes data to CPU first to avoid meta tensor issues
                    boxes_cpu = result.boxes.cpu()
                    
                    for i in range(len(boxes_cpu)):
                        # Get box coordinates - safely from CPU tensors
                        try:
                            # Access box data from CPU tensors
                            box_xyxy = boxes_cpu.xyxy[i]
                            x1, y1, x2, y2 = box_xyxy.tolist()
                        except Exception as e:
                            logger.warning(f"Skipping box due to tensor error: {e}")
                            continue
                        
                        # Get class and confidence safely
                        try:
                            cls_idx = int(boxes_cpu.cls[i].item())
                            conf_val = float(boxes_cpu.conf[i].item())
                        except Exception as e:
                            logger.warning(f"Skipping detection due to tensor error: {e}")
                            continue
                            
                        object_name = result.names[cls_idx]
                        category = self._get_object_category(object_name)
                        
                        # Calculate relative position in frame
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Determine position description
                        h_pos = "left" if center_x < frame_w/3 else ("center" if center_x < 2*frame_w/3 else "right")
                        v_pos = "top" if center_y < frame_h/3 else ("middle" if center_y < 2*frame_h/3 else "bottom")
                        position_desc = f"{v_pos}-{h_pos}"
                        
                        # Calculate object size relative to frame
                        obj_area = (x2-x1) * (y2-y1)
                        frame_area = frame_w * frame_h
                        size_ratio = obj_area / frame_area
                        size_desc = "tiny" if size_ratio < 0.01 else ("small" if size_ratio < 0.1 else ("medium" if size_ratio < 0.3 else "large"))
                        
                        # Create object data
                        obj_data = {
                            'object_class': object_name,
                            'confidence_score': conf_val,
                            'bounding_box': {
                                'x': int(x1),
                                'y': int(y1),
                                'width': int(x2 - x1),
                                'height': int(y2 - y1),
                                'center_x': int(center_x),
                                'center_y': int(center_y)
                            },
                            'object_category': category,
                            'object_id': f"obj_{cls_idx}_{int(timestamp*10)}",
                            'position': position_desc,
                            'size': size_desc,
                            'size_ratio': float(size_ratio)
                        }
                        frame_objects.append(obj_data)
                
                # Create segment for this frame (even if no objects detected)
                segment = {
                    'timestamp': float(timestamp),
                    'bbox_exact': frame_objects[0]['bounding_box'] if frame_objects and frame_objects[0]['object_class'] == 'person' else {
                        'x': 0,
                        'y': 0,
                        'width': frame_w,
                        'height': frame_h,
                        'center_x': frame_w // 2,
                        'center_y': frame_h // 2
                    },
                    'position': frame_objects[0]['position'] if frame_objects and frame_objects[0]['object_class'] == 'person' else "middle-center",
                    'size': frame_objects[0]['size'] if frame_objects and frame_objects[0]['object_class'] == 'person' else "full-frame",
                    'size_ratio': frame_objects[0]['size_ratio'] if frame_objects and frame_objects[0]['object_class'] == 'person' else 1.0,
                    'batch_processed': True,
                    'gpu_optimized': True,
                    'detection_method': 'YOLOv8x-Dense',
                    'objects': frame_objects,
                    'objects_detected': len(frame_objects),
                    'has_person': any(obj['object_class'] == 'person' for obj in frame_objects)
                }
                segments.append(segment)
        
        logger.info(f"✅ Detected {len(segments)} objects in {len(frames)} frames")
        return {'segments': segments}
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Analyze video with maximum frame extraction"""
        if not self.model_loaded:
            self.load_model()
            
        # Extract MANY frames for GPU saturation
        from configs.performance_config import MAX_FRAMES_PER_ANALYZER, OPTIMIZED_FRAME_INTERVALS
        
        max_frames = MAX_FRAMES_PER_ANALYZER.get('object_detection', 300)  # Use 300 frames!
        frame_interval = OPTIMIZED_FRAME_INTERVALS.get('object_tracking', 10)  # Every 0.33 seconds for dense coverage
        
        logger.info(f"[ObjectDetection-YOLO] Extracting up to {max_frames} frames with interval {frame_interval}")
        
        # Extract frames
        frames, timestamps = self.extract_frames(
            video_path,
            sample_rate=frame_interval,
            max_frames=max_frames
        )
        
        logger.info(f"[ObjectDetection-YOLO] Extracted {len(frames)} frames for batch processing")
        
        # Process with batching
        result = self.process_batch_gpu(frames, timestamps)
        
        # Aggregate results from new format
        object_counts = {}
        all_detections = []
        for segment in result['segments']:
            for obj in segment.get('objects', []):
                obj_name = obj['object_class']
                if obj_name not in object_counts:
                    object_counts[obj_name] = 0
                object_counts[obj_name] += 1
                # Create detection dict for categorization
                detection = {
                    'object': obj_name,
                    'confidence': obj['confidence_score'],
                    'bbox': [obj['bounding_box']['x'], obj['bounding_box']['y'], 
                            obj['bounding_box']['width'], obj['bounding_box']['height']]
                }
                all_detections.append(detection)
        
        # Categorize all detected objects
        categorization = self._categorize_frame_objects(all_detections)
        
        return {
            'segments': result['segments'],
            'total_frames_processed': len(frames),
            'total_objects_detected': len(result['segments']),
            'unique_objects': len(object_counts),
            'object_categories': categorization,
            'object_counts': object_counts,
            'gpu_batch_size': self.batch_size,
            'optimization': 'TRUE_BATCH_PROCESSING',
            'processing_mode': 'GPU_OPTIMIZED_YOLO'
        }