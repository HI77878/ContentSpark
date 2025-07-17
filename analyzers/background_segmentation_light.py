#!/usr/bin/env python3
"""
Lightweight Background Segmentation mit SegFormer-B0
8x schneller als B5, trotzdem gute Qualität
"""

import torch
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import numpy as np
from typing import List, Dict, Any
from analyzers.base_analyzer import GPUBatchAnalyzer
import logging
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

class GPUBatchBackgroundSegmentationLight(GPUBatchAnalyzer):
    def __init__(self):
<<<<<<< HEAD
        super().__init__(batch_size=16)  # Reduziert auf 16 für Stabilität
        self.model = None
        self.processor = None
        self.sample_rate = 15   # ERHÖHT: Every 0.5 seconds for better temporal coverage
        self.resize_resolution = (256, 256)  # Kleiner als 512x512 für mehr Speed
        # Clear GPU cache to prevent storage pool errors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
=======
        super().__init__(batch_size=32)  # Erhöht von 16 auf 32
        self.model = None
        self.processor = None
        self.sample_rate = 120  # Every 4 seconds for 2x faster processing
        self.resize_resolution = (256, 256)  # Kleiner als 512x512 für mehr Speed
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
        
    def _load_model_impl(self):
        """Load SegFormer-B0 - smallest and fastest"""
        logger.info("[BackgroundSeg-Light] Loading SegFormer-B0...")
        
        # B0 ist 8x kleiner und schneller als B5
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
<<<<<<< HEAD
            torch_dtype=torch.float32  # Use FP32 to avoid dtype issues
=======
            torch_dtype=torch.float16
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
        )
        self.model.cuda()
        self.model.eval()
        
<<<<<<< HEAD
        # Disable torch.compile for stability (causes storage pool errors)
        # if hasattr(torch, 'compile'):
        #     self.model = torch.compile(self.model, mode="reduce-overhead")
=======
        # Enable CUDA optimizations
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode="reduce-overhead")
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
        
        logger.info("✅ SegFormer-B0 loaded - 8x faster!")
    
    def _analyze_impl(self, video_path: str) -> Dict[str, Any]:
        """Main analysis entry point"""
        frames, frame_times = self.extract_frames(video_path, self.sample_rate)
        if not frames:
            return {'segments': []}
        return self.process_batch_gpu(frames, frame_times)
    
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> Dict[str, Any]:
        logger.info(f"[BackgroundSeg-Light] Processing {len(frames)} frames")
        
        segments = []
        
        # Batch processing
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i+self.batch_size]
            batch_times = frame_times[i:i+self.batch_size]
            
            # Resize für mehr Speed
            pil_images = []
            for frame in batch_frames:
<<<<<<< HEAD
                # Ensure proper dtype
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
=======
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
                # Resize BEFORE color conversion (faster)
                frame_resized = cv2.resize(frame, self.resize_resolution, interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(frame_rgb))
            
<<<<<<< HEAD
            # Process batch - FIX: Disable AMP to avoid storage pool error
            with torch.no_grad():
                # Disable autocast to prevent storage pool allocation errors
                with torch.cuda.amp.autocast(enabled=False):
                    inputs = self.processor(images=pil_images, return_tensors="pt")
                    # Keep as FP32
=======
            # Process batch
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    inputs = self.processor(images=pil_images, return_tensors="pt")
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    outputs = self.model(**inputs)
                    # Handle both tuple and object outputs
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
            
            # Simplified segmentation
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            for pred, timestamp in zip(predictions, batch_times):
                # Schnelle Analyse
                unique_segments, counts = np.unique(pred, return_counts=True)
                total_pixels = pred.size
                
                # ADE20K classes mapping (wichtigste)
                class_mapping = {
                    0: 'wall', 1: 'building', 2: 'sky', 3: 'floor', 4: 'tree',
                    5: 'ceiling', 6: 'road', 7: 'bed', 8: 'windowpane', 9: 'grass',
                    10: 'cabinet', 11: 'sidewalk', 12: 'person', 13: 'earth',
                    14: 'door', 15: 'table', 16: 'mountain', 17: 'plant', 18: 'curtain',
                    19: 'chair', 20: 'car', 21: 'water', 22: 'painting', 23: 'sofa',
                    24: 'shelf', 25: 'house', 26: 'sea', 27: 'mirror', 28: 'rug',
                    29: 'field', 30: 'armchair', 31: 'seat', 32: 'fence', 33: 'desk',
                    34: 'rock', 35: 'wardrobe', 36: 'lamp', 37: 'bathtub', 38: 'railing',
                    39: 'cushion', 40: 'base', 41: 'box', 42: 'column', 43: 'signboard',
                    44: 'chest of drawers', 45: 'counter', 46: 'sand', 47: 'sink',
                    48: 'skyscraper', 49: 'fireplace', 50: 'refrigerator', 51: 'grandstand',
                    52: 'path', 53: 'stairs', 54: 'runway', 55: 'case', 56: 'pool table',
                    57: 'pillow', 58: 'screen door', 59: 'stairway', 60: 'river',
                    61: 'bridge', 62: 'bookcase', 63: 'blind', 64: 'coffee table',
                    65: 'toilet', 66: 'flower', 67: 'book', 68: 'hill', 69: 'bench',
                    70: 'countertop', 71: 'stove', 72: 'palm', 73: 'kitchen island',
                    74: 'computer', 75: 'swivel chair', 76: 'boat', 77: 'bar',
                    78: 'arcade machine', 79: 'hovel', 80: 'bus', 81: 'towel',
                    82: 'light', 83: 'truck', 84: 'tower', 85: 'chandelier',
                    86: 'awning', 87: 'streetlight', 88: 'booth', 89: 'television',
                    90: 'airplane', 91: 'dirt track', 92: 'apparel', 93: 'pole',
                    94: 'land', 95: 'bannister', 96: 'escalator', 97: 'ottoman',
                    98: 'bottle', 99: 'buffet', 100: 'poster', 101: 'stage',
                    102: 'van', 103: 'ship', 104: 'fountain', 105: 'conveyor belt',
                    106: 'canopy', 107: 'washer', 108: 'plaything', 109: 'swimming pool',
                    110: 'stool', 111: 'barrel', 112: 'basket', 113: 'waterfall',
                    114: 'tent', 115: 'bag', 116: 'minibike', 117: 'cradle',
                    118: 'oven', 119: 'ball', 120: 'food', 121: 'step', 122: 'tank',
                    123: 'trade name', 124: 'microwave', 125: 'pot', 126: 'animal',
                    127: 'bicycle', 128: 'lake', 129: 'dishwasher', 130: 'screen',
                    131: 'blanket', 132: 'sculpture', 133: 'hood', 134: 'sconce',
                    135: 'vase', 136: 'traffic light', 137: 'tray', 138: 'ashcan',
                    139: 'fan', 140: 'pier', 141: 'crt screen', 142: 'plate',
                    143: 'monitor', 144: 'bulletin board', 145: 'shower', 146: 'radiator',
                    147: 'glass', 148: 'clock', 149: 'flag'
                }
                
                # Detaillierte Analyse
                detected_objects = {}
                background_classes = {0, 1, 2, 3, 5, 6, 9, 13, 16, 21, 26, 29, 46, 60, 68, 94, 113, 128}
                indoor_classes = {3, 5, 7, 10, 14, 15, 19, 23, 24, 30, 33, 35, 36, 39, 44, 45, 47, 50, 57, 62, 65, 70, 71, 73}
                outdoor_classes = {1, 2, 4, 6, 9, 11, 13, 16, 21, 26, 29, 46, 52, 60, 68, 94}
                
                environment_score = {'indoor': 0, 'outdoor': 0}
                
                for seg_id, count in zip(unique_segments, counts):
                    ratio = count / total_pixels
                    if ratio > 0.01:  # Nur Objekte > 1% der Fläche
                        class_name = class_mapping.get(seg_id, f'unknown_{seg_id}')
                        detected_objects[class_name] = float(ratio)
                        
                        # Environment detection
                        if seg_id in indoor_classes:
                            environment_score['indoor'] += ratio
                        elif seg_id in outdoor_classes:
                            environment_score['outdoor'] += ratio
                
                # Bestimme dominante Umgebung
                environment = 'indoor' if environment_score['indoor'] > environment_score['outdoor'] else 'outdoor'
                if abs(environment_score['indoor'] - environment_score['outdoor']) < 0.1:
                    environment = 'mixed'
                
                # Person detection
                person_present = 'person' in detected_objects
                person_ratio = detected_objects.get('person', 0.0)
                
                # Scene complexity
                scene_complexity = 'simple' if len(detected_objects) < 5 else 'medium' if len(detected_objects) < 10 else 'complex'
                
<<<<<<< HEAD
                # Create unique description
                dominant_objects = sorted(detected_objects.items(), key=lambda x: x[1], reverse=True)[:3]
                obj_descriptions = [f"{name} ({ratio:.1%})" for name, ratio in dominant_objects]
                
                if person_present:
                    description = f"{environment.title()} Szene mit Person ({person_ratio:.1%}). Hauptobjekte: {', '.join(obj_descriptions)}. Komplexität: {scene_complexity}"
                else:
                    description = f"{environment.title()} Szene ohne Person. Hauptobjekte: {', '.join(obj_descriptions)}. Komplexität: {scene_complexity}"
                
                segments.append({
                    'start_time': max(0.0, float(timestamp) - 0.5),
                    'end_time': float(timestamp) + 0.5,
                    'timestamp': float(timestamp),
                    'segment_id': f'background_seg_{int(timestamp * 10)}',
                    'description': description,
=======
                segments.append({
                    'timestamp': float(timestamp),
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
                    'detected_objects': detected_objects,
                    'environment': environment,
                    'person_present': person_present,
                    'person_ratio': float(person_ratio),
                    'scene_complexity': scene_complexity,
                    'num_objects': len(detected_objects),
<<<<<<< HEAD
                    'dominant_objects': dominant_objects
=======
                    'dominant_objects': sorted(detected_objects.items(), key=lambda x: x[1], reverse=True)[:5]
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
                })
        
        return {'segments': segments}