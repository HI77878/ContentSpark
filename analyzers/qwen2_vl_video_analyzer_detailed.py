#!/usr/bin/env python3
"""
Qwen2-VL Video Analyzer - DETAILED VERSION
Optimized for maximum detail and accuracy in video description
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

class Qwen2VLVideoAnalyzerDetailed(GPUBatchAnalyzer):
    """Qwen2-VL Video Analyzer with maximum detail extraction"""
    
    def __init__(self, batch_size=1):
        super().__init__(batch_size)
        self.analyzer_name = "qwen2_vl_temporal"
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
        
        # OPTIMIZED FOR SPEED AND DETAIL BALANCE
        self.segment_duration = 2.0  # 2 seconds per segment - still detailed but faster
        self.segment_overlap = 0.5   # 0.5s overlap to catch transitions
        self.segment_step = self.segment_duration - self.segment_overlap  # 1.5s step
        self.min_segment_duration = 1.0  # Minimum segment length
        
        # Optimized frame sampling
        self.fps_sample = 2.0  # 2 frames per second - good balance
        self.max_frames_per_segment = 3  # Up to 3 frames per segment
        self.max_pixels = 896 * 28 * 28  # Slightly smaller for speed
        
        # Optimized single prompt for speed
        self.prompt = "Describe this 2-second video segment in detail. Include ALL of the following: 1) Person's specific actions and movements in chronological order, 2) What they're wearing and holding, 3) Facial expressions and gestures, 4) Objects being used or touched, 5) Background/environment, 6) Any text visible. Be specific about the sequence - what happens first, middle, and end of these 2 seconds. Include every action and change."
        
    def _load_model_impl(self):
        """Load Qwen2-VL model with quality optimizations"""
        logger.info(f"[{self.analyzer_name}] Loading Qwen2-VL-7B DETAILED mode...")
        
        # Configure pixel limits for higher quality
        min_pixels = 512 * 28 * 28
        max_pixels = self.max_pixels
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True
        )
        
        # Load model optimized for quality
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"[{self.analyzer_name}] Model loaded in DETAILED mode!")
        
    def _extract_frames_for_segment(self, video_path: str, start_time: float, end_time: float) -> List[Image.Image]:
        """Extract frames with higher sampling rate for detail"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices to extract
        segment_duration = end_time - start_time
        num_frames = min(int(segment_duration * self.fps_sample), self.max_frames_per_segment)
        
        if num_frames > 0:
            time_points = np.linspace(start_time, end_time - 0.001, num_frames)
            
            for time_point in time_points:
                frame_idx = int(time_point * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Higher resolution for better detail
                    height, width = frame.shape[:2]
                    if height > 640:
                        scale = 640 / height
                        new_width = int(width * scale)
                        new_height = 640
                        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        return frames
        
    def _generate_detailed_description(self, frames: List[Image.Image], start_time: float, end_time: float) -> str:
        """Generate detailed description with single optimized prompt"""
        if not frames:
            return f"No frames available for segment {start_time:.1f}-{end_time:.1f}s"
        
        try:
            # Create message for Qwen2-VL
            messages = [{
                "role": "user",
                "content": []
            }]
            
            # Add frames
            for frame in frames:
                messages[0]["content"].append({"type": "image", "image": frame})
            
            # Add optimized prompt
            messages[0]["content"].append({
                "type": "text", 
                "text": f"Video segment from {start_time:.1f}s to {end_time:.1f}s. {self.prompt}"
            })
            
            # Process with model
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate with balanced settings
            with torch.cuda.amp.autocast():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=400,  # Slightly less for speed
                    min_new_tokens=150,  # Still ensure good detail
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in 
                zip(inputs.input_ids, generated_ids)
            ]
            
            description = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return description
                
        except Exception as e:
            logger.error(f"Failed to generate description: {e}")
            return f"Analysis failed for segment {start_time:.1f}-{end_time:.1f}s"
        
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Analyze video with detailed segment processing"""
        if self.model is None:
            self._load_model_impl()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        logger.info(f"[{self.analyzer_name}] Analyzing {duration:.1f}s video with {self.segment_step}s steps (DETAILED mode)")
        
        segments = []
        segment_count = 0
        
        # Process video with overlapping segments
        current_time = 0.0
        while current_time < duration:
            # Calculate segment boundaries
            segment_end = min(current_time + self.segment_duration, duration)
            actual_duration = segment_end - current_time
            
            # Skip if segment is too short
            if actual_duration < self.min_segment_duration:
                break
            
            segment_count += 1
            logger.info(f"[{self.analyzer_name}] Segment {segment_count}: {current_time:.1f}-{segment_end:.1f}s")
            
            # Extract frames for this segment
            frames = self._extract_frames_for_segment(video_path, current_time, segment_end)
            
            if frames:
                # Generate detailed description
                description = self._generate_detailed_description(frames, current_time, segment_end)
                
                segments.append({
                    "segment_id": segment_count,
                    "start_time": current_time,
                    "end_time": segment_end,
                    "duration": actual_duration,
                    "timestamp": current_time,
                    "description": description,
                    "frame_count": len(frames),
                    "sampling_fps": self.fps_sample
                })
                
                # Log description length for monitoring
                logger.info(f"[{self.analyzer_name}] Description length: {len(description)} chars")
            
            # Clear GPU memory between segments
            torch.cuda.empty_cache()
            
            # Move to next segment
            current_time += self.segment_step
        
        # Create summary
        summary = {
            "total_segments": len(segments),
            "segment_duration": self.segment_duration,
            "segment_overlap": self.segment_overlap,
            "video_duration_seconds": duration,
            "coverage_percentage": min(100, (len(segments) * self.segment_step / duration) * 100) if duration > 0 else 0,
            "analysis_mode": "DETAILED_FAST",
            "frames_per_second": self.fps_sample,
            "average_description_length": np.mean([len(s["description"]) for s in segments]) if segments else 0
        }
        
        logger.info(f"[{self.analyzer_name}] Completed {len(segments)} segments with {summary['coverage_percentage']:.1f}% coverage")
        logger.info(f"[{self.analyzer_name}] Average description length: {summary['average_description_length']:.0f} chars")
        
        return {
            "analyzer_name": self.analyzer_name,
            "segments": segments,
            "summary": summary
        }
    
    def process_batch_gpu(self, frames, frame_times):
        """Required abstract method - not used in this analyzer"""
        # This analyzer uses its own segment-based processing
        return []