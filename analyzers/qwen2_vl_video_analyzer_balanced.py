#!/usr/bin/env python3
"""
Qwen2-VL Video Analyzer - Balanced Version
Provides a good balance between detail and speed
Target: 10-15s per segment with meaningful descriptions
"""

import os
import sys
import cv2
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import gc
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.base_analyzer import GPUBatchAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class VideoSegment:
    """Represents a video segment for analysis"""
    start_time: float
    end_time: float
    frames: List[np.ndarray]
    frame_indices: List[int]

class Qwen2VLVideoAnalyzerBalanced(GPUBatchAnalyzer):
    """
    Balanced Qwen2-VL analyzer with practical performance
    - 3-second segments with 1s overlap
    - 10-15 seconds processing per segment
    - 400-600 character descriptions
    """
    
    def __init__(self):
        super().__init__()
        # Balanced parameters for good detail/speed tradeoff
        self.segment_duration = 3.0      # 3-second segments
        self.segment_overlap = 1.0       # 1-second overlap
        self.fps_sample = 1.5            # 1.5 FPS sampling
        self.max_frames_per_segment = 5  # Max 5 frames per segment
        self.resize_height = 560         # Medium resolution
        self.max_new_tokens = 300        # Moderate description length
        
        self.model = None
        self.processor = None
        
        # Optimized prompt for balanced detail
        self.prompt = """Describe what happens in this 3-second video segment. Include:
- All major actions and movements
- What people are doing (standing up, putting on clothes, etc)
- Objects being used or interacted with
- Any scene changes or camera movements
Be specific but concise - aim for 5-8 sentences covering the key events."""
        
        logger.info(f"[qwen2_vl_balanced] Initialized with balanced parameters:")
        logger.info(f"  - Segment duration: {self.segment_duration}s")
        logger.info(f"  - Overlap: {self.segment_overlap}s")
        logger.info(f"  - FPS sample: {self.fps_sample}")
        logger.info(f"  - Max tokens: {self.max_new_tokens}")
    
    def _load_model_impl(self):
        """Load Qwen2-VL model with optimizations"""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            model_path = "Qwen/Qwen2-VL-7B-Instruct"
            
            logger.info(f"[qwen2_vl_balanced] Loading model from {model_path}...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Load model with optimizations
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="cuda:0",
                trust_remote_code=True,
                attn_implementation="flash_attention_2"  # Use Flash Attention if available
            )
            
            self.model.eval()
            logger.info("[qwen2_vl_balanced] Model loaded successfully with Flash Attention")
            
        except Exception as e:
            logger.error(f"[qwen2_vl_balanced] Error loading model: {e}")
            # Fallback to standard attention
            try:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="cuda:0",
                    trust_remote_code=True
                )
                
                self.model.eval()
                logger.info("[qwen2_vl_balanced] Model loaded with standard attention")
            except Exception as e2:
                logger.error(f"[qwen2_vl_balanced] Failed to load model: {e2}")
                raise
    
    def _extract_segments(self, video_path: str) -> List[VideoSegment]:
        """Extract video segments with balanced parameters"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        segments = []
        current_time = 0.0
        
        # Calculate segment times with overlap
        while current_time < duration - 0.1:
            end_time = min(current_time + self.segment_duration, duration)
            
            # Extract frames for this segment
            frames = []
            frame_indices = []
            
            # Calculate frame positions
            segment_duration = end_time - current_time
            num_frames = min(
                int(segment_duration * self.fps_sample),
                self.max_frames_per_segment
            )
            
            if num_frames > 0:
                time_points = np.linspace(current_time, end_time - 0.01, num_frames)
                
                for t in time_points:
                    frame_idx = int(t * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        # Resize frame for efficiency
                        height, width = frame.shape[:2]
                        new_height = self.resize_height
                        new_width = int(width * (new_height / height))
                        frame = cv2.resize(frame, (new_width, new_height))
                        
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        frame_indices.append(frame_idx)
                
                if frames:
                    segments.append(VideoSegment(
                        start_time=current_time,
                        end_time=end_time,
                        frames=frames,
                        frame_indices=frame_indices
                    ))
            
            # Move to next segment with overlap
            current_time += (self.segment_duration - self.segment_overlap)
        
        cap.release()
        
        logger.info(f"[qwen2_vl_balanced] Extracted {len(segments)} segments from {duration:.1f}s video")
        return segments
    
    def _analyze_segment(self, segment: VideoSegment) -> Dict[str, Any]:
        """Analyze a single segment with Qwen2-VL"""
        try:
            start_time = time.time()
            
            # Prepare frames for model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        *[{"type": "image", "image": frame} for frame in segment.frames]
                    ]
                }
            ]
            
            # Process with model
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=segment.frames,
                padding=True,
                return_tensors="pt"
            ).to("cuda")
            
            # Generate description
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            description = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            processing_time = time.time() - start_time
            
            logger.info(f"[qwen2_vl_balanced] Segment {segment.start_time:.1f}-{segment.end_time:.1f}s: "
                       f"{len(description)} chars in {processing_time:.1f}s")
            
            return {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.end_time - segment.start_time,
                "description": description.strip(),
                "frame_count": len(segment.frames),
                "processing_time": processing_time,
                "confidence": 0.85  # Balanced confidence
            }
            
        except Exception as e:
            logger.error(f"[qwen2_vl_balanced] Error analyzing segment: {e}")
            return {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.end_time - segment.start_time,
                "description": f"Error analyzing segment: {str(e)}",
                "frame_count": len(segment.frames),
                "error": str(e)
            }
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Main analysis function with balanced processing"""
        try:
            logger.info(f"[qwen2_vl_balanced] Starting balanced analysis of {video_path}")
            start_time = time.time()
            
            # Load model if needed
            if self.model is None:
                self._load_model_impl()
            
            # Extract segments
            segments = self._extract_segments(video_path)
            
            # Analyze each segment
            results = []
            total_processing_time = 0
            
            for i, segment in enumerate(segments):
                logger.info(f"[qwen2_vl_balanced] Processing segment {i+1}/{len(segments)}")
                
                result = self._analyze_segment(segment)
                results.append(result)
                
                if "processing_time" in result:
                    total_processing_time += result["processing_time"]
                
                # Clear GPU cache periodically
                if i % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Calculate statistics
            total_time = time.time() - start_time
            video_duration = segments[-1].end_time if segments else 0
            
            analysis_result = {
                "segments": results,
                "metadata": {
                    "total_segments": len(segments),
                    "video_duration": video_duration,
                    "total_processing_time": total_time,
                    "segment_processing_time": total_processing_time,
                    "average_time_per_segment": total_processing_time / len(segments) if segments else 0,
                    "realtime_factor": total_time / video_duration if video_duration > 0 else 0,
                    "analyzer": "qwen2_vl_balanced",
                    "parameters": {
                        "segment_duration": self.segment_duration,
                        "segment_overlap": self.segment_overlap,
                        "fps_sample": self.fps_sample,
                        "max_frames": self.max_frames_per_segment,
                        "max_tokens": self.max_new_tokens
                    }
                }
            }
            
            logger.info(f"[qwen2_vl_balanced] Completed analysis in {total_time:.1f}s "
                       f"(avg {total_processing_time/len(segments):.1f}s per segment)")
            
            # Clean up
            torch.cuda.empty_cache()
            gc.collect()
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"[qwen2_vl_balanced] Analysis failed: {e}")
            return {
                "error": str(e),
                "segments": []
            }
    
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> Dict[str, Any]:
        """Legacy method for compatibility"""
        # This analyzer uses its own segmentation logic
        return {"segments": [], "error": "Use analyze() method directly"}