#!/usr/bin/env python3
"""
Qwen2-VL Video Analyzer - KORREKTE VERSION
Basiert auf offiziellen Qwen2-VL Beispielen für Video-Analyse
"""

import torch
import cv2
import numpy as np
from PIL import Image
import logging
import gc
import tempfile
import os
from typing import List, Dict, Any
from pathlib import Path

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from analyzers.base_analyzer import GPUBatchAnalyzer

logger = logging.getLogger(__name__)

class Qwen2VLVideoAnalyzer(GPUBatchAnalyzer):
    """Qwen2-VL Video Analyzer für temporale Beschreibungen"""
    
    def __init__(self, batch_size=1):
        super().__init__(batch_size)
        self.analyzer_name = "qwen2_vl_temporal"
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
        
<<<<<<< HEAD
        # Video processing parameters - OPTIMIZED
        self.max_frames = 24  # Increased for better temporal coverage
        self.segment_duration = 1.0  # Reduced to 1 second for more detailed analysis
        
        # Performance optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
=======
        # Video processing parameters
        self.max_frames = 16  # Optimal für 7B Modell
        self.segment_duration = 2.0  # 2 Sekunden pro Segment
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
        
    def _load_model_impl(self):
        """Load Qwen2-VL model correctly"""
        logger.info(f"[{self.analyzer_name}] Loading Qwen2-VL-7B model...")
        
        # Correct model loading
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better stability
            device_map=self.device,
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model.eval()
        logger.info(f"[{self.analyzer_name}] Model loaded successfully")
    
    def extract_video_frames(self, video_path: str, max_frames: int = 16) -> List[Image.Image]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices for uniform sampling
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / max_frames
            frame_indices = [int(i * step) for i in range(max_frames)]
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)
        
        cap.release()
        return frames
    
    def analyze_video_segment(self, frames: List[Image.Image], start_time: float, end_time: float) -> Dict[str, Any]:
        """Analyze video segment with Qwen2-VL"""
        
        # Create proper video description prompt
        prompt = f"""Describe what happens in this video from {start_time:.1f} to {end_time:.1f} seconds.
Focus on:
- Actions and movements
- People and objects
- Scene changes
- Visual elements
- Any text or graphics

Provide a detailed temporal description of the video content."""

        # Prepare messages for Qwen2-VL
        content = []
        for frame in frames:
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": prompt})
        
        messages = [{
            "role": "user",
            "content": content
        }]
        
        try:
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process vision info
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True
                )
            
            # Decode response
            generated_text = self.processor.decode(
                generated_ids[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            ).strip()
            
            return {
                'segment_id': f"qwen_temporal_{int(start_time)}",
                'start_time': start_time,
                'end_time': end_time,
                'timestamp': start_time,
                'description': generated_text,
                'frames_analyzed': len(frames),
                'confidence': 0.95,
                'analyzer': 'qwen2_vl_temporal'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing segment at {start_time}s: {e}")
            return None
    
    def process_batch_gpu(self, frames, frame_times):
        """Process batch of frames on GPU - required by base class"""
        return []
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Main analysis function"""
        if not self.model_loaded:
            self._load_model_impl()
            self.model_loaded = True
            
        video_path = Path(video_path)
        logger.info(f"[{self.analyzer_name}] Analyzing {video_path.name}")
        
        # Get video duration
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        segments = []
        current_time = 0.0
        
        # Process video in segments
        while current_time < duration:
            end_time = min(current_time + self.segment_duration, duration)
            
            # Extract frames for this segment
            frames = self.extract_frames_for_segment(
                str(video_path), 
                current_time, 
                end_time - current_time
            )
            
            if frames:
                result = self.analyze_video_segment(frames, current_time, end_time)
                if result:
                    segments.append(result)
                    logger.info(f"   Segment {current_time:.1f}s: {result['description'][:60]}...")
            
            current_time = end_time
            
            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
        
        logger.info(f"[{self.analyzer_name}] Completed: {len(segments)} segments")
        
        return {
            'analyzer_name': self.analyzer_name,
            'segments': segments,
            'summary': {
                'total_segments': len(segments),
                'video_duration': duration
            }
        }
    
    def extract_frames_for_segment(self, video_path: str, start_time: float, duration: float) -> List[Image.Image]:
        """Extract frames for a specific segment"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_time * fps)
        end_frame = int((start_time + duration) * fps)
        total_segment_frames = end_frame - start_frame
        
        # Select frames for this segment (max 8 frames per segment)
        max_frames_per_segment = min(8, total_segment_frames)
        
        if total_segment_frames <= max_frames_per_segment:
            frame_indices = list(range(start_frame, end_frame))
        else:
            step = total_segment_frames / max_frames_per_segment
            frame_indices = [int(start_frame + i * step) for i in range(max_frames_per_segment)]
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)
        
        cap.release()
        return frames
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()