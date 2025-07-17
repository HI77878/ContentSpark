#!/usr/bin/env python3
"""
Qwen2-VL Temporal Analyzer - SIMPLE VERSION
- Removes complex deduplication logic
- Uses straightforward multi-image processing
- Focuses on getting it working first
"""

import torch
import cv2
import numpy as np
from PIL import Image
import logging
import gc
from typing import List, Dict, Any, Optional
from pathlib import Path

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from analyzers.base_analyzer import GPUBatchAnalyzer

logger = logging.getLogger(__name__)

class Qwen2VLTemporalFixed(GPUBatchAnalyzer):
    """Simple Qwen2-VL for temporal video analysis"""
    
    def __init__(self, batch_size=1):
        super().__init__(batch_size)
        self.analyzer_name = "qwen2_vl_temporal"
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
        
        # Frame sampling for temporal analysis - SINGLE FRAME ONLY
        self.frames_per_segment = 1  # Single frame to avoid OOM
        self.segment_duration = 1.0  # 1 second segments
        self.segment_overlap = 0.0  # No overlap to save memory
        
    def _load_model_impl(self):
        """Load Qwen2-VL model"""
        logger.info(f"[{self.analyzer_name}] Loading Qwen2-VL-7B model...")
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model.eval()
        logger.info(f"[{self.analyzer_name}] Model loaded successfully")
    
    def analyze_segment(self, frames: List[Image.Image], start_time: float, end_time: float) -> Dict[str, Any]:
        """Analyze a video segment with multiple frames"""
        
        # Simple prompt
        prompt = f"""Describe what happens in this video segment from {start_time:.1f}s to {end_time:.1f}s.
Focus on: actions, people, objects, scene, camera movement, and any text or UI elements.
Be specific and detailed."""
        
        # Use only the first frame to avoid OOM
        content = [
            {"type": "image", "image": frames[0]},
            {"type": "text", "text": prompt}
        ]
        
        messages = [{
            "role": "user",
            "content": content
        }]
        
        try:
            # Process with model
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            image_inputs, _ = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.inference_mode():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        use_cache=True
                    )
            
            # Decode
            generated_text = self.processor.decode(
                generated_ids[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ).strip()
            
            return {
                'segment_id': f"temporal_{int(start_time)}",
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
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def extract_frames_for_segment(self, video_path: str, start_time: float, duration: float) -> List[Image.Image]:
        """Extract frames for a segment"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_time * fps)
        end_frame = int((start_time + duration) * fps)
        total_frames = end_frame - start_frame
        
        # Select evenly distributed frames
        if total_frames <= self.frames_per_segment:
            frame_indices = list(range(start_frame, end_frame))
        else:
            step = total_frames / self.frames_per_segment
            frame_indices = [int(start_frame + i * step) for i in range(self.frames_per_segment)]
        
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
    
    def process_batch_gpu(self, frames, frame_times):
        """Process batch of frames on GPU - required by base class"""
        # Compatibility method
        return []
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Main analysis function"""
        if not self.model_loaded:
            self._load_model_impl()
            self.model_loaded = True
            
        video_path = Path(video_path)
        logger.info(f"[{self.analyzer_name}] Analyzing {video_path.name}")
        
        # Get video info
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        # Process video segments
        segments = []
        current_time = 0.0
        step = self.segment_duration - self.segment_overlap
        
        while current_time < duration:
            seg_duration = min(self.segment_duration, duration - current_time)
            
            # Extract frames
            frames = self.extract_frames_for_segment(
                str(video_path), 
                current_time, 
                seg_duration
            )
            
            if frames:
                # Analyze segment
                result = self.analyze_segment(
                    frames, 
                    current_time, 
                    current_time + seg_duration
                )
                
                if result:
                    segments.append(result)
                    logger.info(f"   Segment {current_time:.1f}s: {result['description'][:50]}...")
            
            current_time += step
            
            # Periodic cleanup
            if len(segments) % 5 == 0:
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
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()