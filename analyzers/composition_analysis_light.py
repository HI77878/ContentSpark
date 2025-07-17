#!/usr/bin/env python3
"""
Lightweight Composition Analysis mit CLIP Tiny
15x schneller als CLIP Large
"""

import torch
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import List, Dict, Any
from analyzers.base_analyzer import GPUBatchAnalyzer
import logging
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

class GPUBatchCompositionAnalysisLight(GPUBatchAnalyzer):
    def __init__(self):
        super().__init__(batch_size=32)
        self.model = None
        self.processor = None
        self.sample_rate = 30
        
        # Kompositions-Prompts
        self.composition_prompts = [
            "a well composed image",
            "a poorly composed image", 
            "rule of thirds composition",
            "centered composition",
            "dynamic composition",
            "static composition",
            "professional photography",
            "amateur photography"
        ]
        
    def _load_model_impl(self):
        """Load CLIP Base/32 - viel kleiner und schneller"""
        logger.info("[Composition-Light] Loading CLIP Base/32...")
        
        model_name = "openai/clip-vit-base-patch32"  # 15x schneller als Large!
        
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )
        self.model.cuda()
        self.model.eval()
        
        logger.info("✅ CLIP Base/32 loaded - 15x faster!")
    
    def _analyze_impl(self, video_path: str) -> Dict[str, Any]:
        """Main analysis entry point"""
        frames, frame_times = self.extract_frames(video_path, self.sample_rate)
        if not frames:
            return {'segments': []}
        return self.process_batch_gpu(frames, frame_times)
    
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> Dict[str, Any]:
        logger.info(f"[Composition-Light] Processing {len(frames)} frames")
        
        segments = []
        
        # Process in batches
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i+self.batch_size]
            batch_times = frame_times[i:i+self.batch_size]
            
            # Convert to PIL
            pil_images = []
            for frame in batch_frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize für Speed
                frame_small = cv2.resize(frame_rgb, (224, 224))
                pil_images.append(Image.fromarray(frame_small))
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # Encode images and text
                    inputs = self.processor(
                        text=self.composition_prompts,
                        images=pil_images,
                        return_tensors="pt",
                        padding=True
                    )
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    outputs = self.model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1).cpu().numpy()
            
            # Analyze results
            for prob, timestamp in zip(probs, batch_times):
                # Find best matching prompts
                top_indices = np.argsort(prob)[-3:][::-1]
                
                # Simple scoring
                quality_score = prob[0] + prob[6]  # well composed + professional
                
                # Composition type
                if prob[2] > 0.3:
                    comp_type = "rule_of_thirds"
                elif prob[3] > 0.3:
                    comp_type = "centered"
                elif prob[4] > 0.3:
                    comp_type = "dynamic"
                else:
                    comp_type = "standard"
                
                segments.append({
                    'timestamp': float(timestamp),
                    'quality_score': float(quality_score),
                    'composition_type': comp_type,
                    'professional_score': float(prob[6]),
                    'rating': 'good' if quality_score > 0.6 else 'average' if quality_score > 0.4 else 'poor'
                })
        
        return {'segments': segments}