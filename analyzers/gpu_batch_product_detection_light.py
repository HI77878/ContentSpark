#!/usr/bin/env python3
"""
YOLOv8s Light Product Detection - 5x schneller
Spezialisiert auf Produkte/Marken mit kleinerem Modell
"""

import torch
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any
from analyzers.base_analyzer import GPUBatchAnalyzer
import logging
import time

logger = logging.getLogger(__name__)

class GPUBatchProductDetectionLight(GPUBatchAnalyzer):
    def __init__(self):
        super().__init__(batch_size=32)
        self.device = 'cuda'
        self.model = None
        self.sample_rate = 30  # Jede Sekunde
        
        # Produkt-relevante COCO Klassen
        self.product_classes = [
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
            'hot dog', 'pizza', 'donut', 'cake', 'potted plant', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase'
        ]
    
    def _load_model_impl(self):
        """Load YOLOv8s - small but accurate"""
        logger.info("[ProductDetection-Light] Loading YOLOv8s...")
        
        self.model = YOLO('yolov8s.pt')  # 22MB vs 136MB
        self.model.cuda()
        self.model.model.eval()
        self.model.model.half()  # FP16
        
        logger.info("✅ YOLOv8s loaded - 5x faster for products!")
    
    def _analyze_impl(self, video_path: str) -> Dict[str, Any]:
        """Main analysis entry point"""
        frames, frame_times = self.extract_frames(video_path, self.sample_rate)
        if not frames:
            return {'segments': []}
        return self.process_batch_gpu(frames, frame_times)
    
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> Dict[str, Any]:
        logger.info(f"[ProductDetection-Light] Processing {len(frames)} frames")
        
        segments = []
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                results = self.model(
                    frames, 
                    device=self.device,
                    batch=len(frames),
                    verbose=False,
                    conf=0.3,  # Höher für Produkte
                    iou=0.5,
                    half=True,
                    max_det=20  # Weniger Detections
                )
        
        for result, timestamp in zip(results, frame_times):
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_cpu = result.boxes.cpu()
                
                for i in range(len(boxes_cpu)):
                    try:
                        cls_idx = int(boxes_cpu.cls[i].item())
                        object_name = result.names[cls_idx]
                        
                        # Nur produkt-relevante Objekte
                        if object_name in self.product_classes:
                            x1, y1, x2, y2 = boxes_cpu.xyxy[i].tolist()
                            conf_val = float(boxes_cpu.conf[i].item())
                            
                            segment = {
                                'timestamp': float(timestamp),
                                'product': object_name,
                                'confidence': conf_val,
                                'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                'category': self._get_product_category(object_name)
                            }
                            segments.append(segment)
                    except:
                        continue
        
        return {'segments': segments}
    
    def _get_product_category(self, product: str) -> str:
        """Kategorisiere Produkte"""
        if product in ['bottle', 'wine glass', 'cup', 'bowl']:
            return 'beverage'
        elif product in ['banana', 'apple', 'sandwich', 'orange', 'pizza', 'donut', 'cake']:
            return 'food'
        elif product in ['tv', 'laptop', 'mouse', 'keyboard', 'cell phone']:
            return 'electronics'
        elif product in ['backpack', 'handbag', 'suitcase']:
            return 'fashion'
        else:
            return 'other'