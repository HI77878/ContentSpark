#!/usr/bin/env python3
"""
Qwen2-VL Video Analyzer - FAST SECONDS VERSION
Optimiert für schnellere Verarbeitung bei gutem Detail
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

class Qwen2VLVideoAnalyzerSecondsFast(GPUBatchAnalyzer):
    """Qwen2-VL Video Analyzer mit optimierter Geschwindigkeit"""
    
    def __init__(self, batch_size=1):
        super().__init__(batch_size)
        self.analyzer_name = "qwen2_vl_temporal"
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
        
        # OPTIMIZED FOR SPEED WITH GOOD COVERAGE
        self.segment_duration = 2.0  # 2 seconds per segment
        self.segment_overlap = 1.0   # 50% overlap
        self.segment_step = self.segment_duration - self.segment_overlap  # 1s step
        self.min_segment_duration = 1.0  # Minimum segment length
        
        # Reduced frame sampling for speed
        self.fps_sample = 1.0  # Only 1 frame per second
        self.max_frames_per_segment = 3  # Max 3 frames to reduce processing time
        self.max_pixels = 512 * 28 * 28  # Smaller pixel budget
        
    def _load_model_impl(self):
        """Load Qwen2-VL model with speed optimizations"""
        logger.info(f"[{self.analyzer_name}] Loading Qwen2-VL-7B FAST mode...")
        
        # Configure pixel limits
        min_pixels = 256 * 28 * 28
        max_pixels = self.max_pixels
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True
        )
        
        # Load model optimized for speed
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation="eager",  # Standard attention
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"[{self.analyzer_name}] Model loaded in FAST mode!")
        
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Analyze video with optimized segment processing"""
        if self.model is None:
            self._load_model_impl()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        logger.info(f"[{self.analyzer_name}] Analyzing {duration:.1f}s video with {self.segment_step}s steps (FAST mode)")
        
        segments = []
        segment_count = 0
        
        # Process video with overlapping segments
        current_time = 0.0
        while current_time < duration:
            # Calculate segment boundaries
            start_time = current_time
            end_time = min(current_time + self.segment_duration, duration)
            
            # Skip if segment is too short
            if end_time - start_time < self.min_segment_duration:
                break
            
            try:
                # Extract frames for this segment
                frames = self._extract_segment_frames(video_path, start_time, end_time)
                
                if frames:
                    # Analyze segment with simplified prompt
                    segment_result = self._analyze_frame_segment(
                        frames, 
                        start_time, 
                        end_time,
                        segment_count
                    )
                    
                    if segment_result and segment_result.get('description'):
                        segments.append(segment_result)
                        segment_count += 1
                        logger.info(f"[{self.analyzer_name}] Segment {segment_count}: {start_time:.1f}-{end_time:.1f}s")
                
                # Cleanup after each segment
                del frames
                torch.cuda.empty_cache()
                gc.collect()
                    
            except Exception as e:
                logger.error(f"[{self.analyzer_name}] Error in segment {start_time:.1f}-{end_time:.1f}: {e}")
                torch.cuda.empty_cache()
                gc.collect()
            
            # Move to next segment
            current_time += self.segment_step
        
        logger.info(f"[{self.analyzer_name}] Completed: {len(segments)} segments for {duration:.1f}s video")
        
        return {
            'analyzer_name': self.analyzer_name,
            'segments': segments,
            'summary': {
                'total_segments': len(segments),
                'video_duration': duration,
                'segments_per_second': len(segments) / duration if duration > 0 else 0,
                'segment_duration': self.segment_duration,
                'segment_overlap': self.segment_overlap,
                'mode': 'fast'
            }
        }
    
    def _extract_segment_frames(self, video_path: str, start_time: float, end_time: float) -> List[str]:
        """Extract frames from video segment with minimal sampling"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        segment_duration = end_time - start_time
        
        # Calculate frames to extract (minimal)
        target_frames = min(
            int(segment_duration * self.fps_sample),
            self.max_frames_per_segment
        )
        
        if target_frames < 1:
            target_frames = 1
        
        # Calculate frame interval
        total_frames_in_segment = end_frame - start_frame
        if total_frames_in_segment > target_frames:
            frame_interval = total_frames_in_segment // target_frames
        else:
            frame_interval = 1
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_count = 0
            for i in range(target_frames):
                frame_idx = start_frame + (i * frame_interval)
                if frame_idx >= end_frame:
                    break
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame to reduce memory (smaller size)
                height, width = frame.shape[:2]
                if width > 480:  # Smaller resolution for speed
                    scale = 480 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Save frame
                frame_path = os.path.join(temp_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frames.append(frame_path)
                frame_count += 1
            
            cap.release()
            return frames
            
        except Exception as e:
            cap.release()
            # Cleanup on error
            for f in frames:
                if os.path.exists(f):
                    os.remove(f)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
            raise e
    
    def _analyze_frame_segment(self, frame_paths: List[str], start_time: float, end_time: float, 
                              segment_index: int) -> Dict[str, Any]:
        """Analyze segment with simplified prompt for speed"""
        
        # Create message with frames
        content = []
        for i, frame_path in enumerate(frame_paths):
            content.append({"type": "image", "image": frame_path})
        
        # Simplified prompt for faster processing
        prompt = f"""Video segment {start_time:.1f}-{end_time:.1f}s.

Describe: WHO is visible, WHAT they are doing, WHERE this takes place.
Be concise but complete."""
        
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
            
            # Generate response with limited tokens
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=150,  # Reduced for speed
                    do_sample=False,
                    temperature=0.7,
                    use_cache=True,
                    num_beams=1
                )
            
            # Decode response
            generated_text = self.processor.decode(
                generated_ids[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            ).strip()
            
            # Cleanup
            del inputs, generated_ids
            torch.cuda.empty_cache()
            
            # Cleanup temp files
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            
            # Remove temp directory
            if frame_paths:
                temp_dir = os.path.dirname(frame_paths[0])
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            
            return {
                'segment_id': f"qwen_temporal_{segment_index:04d}",
                'segment_index': segment_index,
                'start_time': start_time,
                'end_time': end_time,
                'timestamp': start_time,
                'duration': end_time - start_time,
                'description': generated_text,
                'frames_analyzed': len(frame_paths),
                'confidence': 0.95,
                'analyzer': self.analyzer_name
            }
            
        except Exception as e:
            logger.error(f"[{self.analyzer_name}] Error in frame analysis: {e}")
            
            # Cleanup on error
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            
            return None
    
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> List[Dict[str, Any]]:
        """Not used - we process video segments directly"""
        return []