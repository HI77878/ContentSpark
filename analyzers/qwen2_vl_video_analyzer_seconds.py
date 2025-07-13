#!/usr/bin/env python3
"""
Qwen2-VL Video Analyzer - SECONDS VERSION
Sekundengenaue Analyse mit Overlap für lückenlose Abdeckung
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

class Qwen2VLVideoAnalyzerSeconds(GPUBatchAnalyzer):
    """Qwen2-VL Video Analyzer mit sekundengenauen Segmenten"""
    
    def __init__(self, batch_size=1):
        super().__init__(batch_size)
        self.analyzer_name = "qwen2_vl_temporal"
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
        
        # OPTIMIZED FOR SECOND-BY-SECOND ANALYSIS
        self.segment_duration = 1.0  # 1 second per segment
        self.segment_overlap = 0.5   # 50% overlap for seamless coverage
        self.segment_step = self.segment_duration - self.segment_overlap  # 0.5s step
        self.min_segment_duration = 0.5  # Minimum segment length
        
        # Frame sampling parameters
        self.fps_sample = 4.0  # 4 frames per second for detail
        self.max_frames_per_segment = 8  # Max frames to prevent OOM
        self.max_pixels = 768 * 28 * 28  # Memory limit
        
    def _load_model_impl(self):
        """Load Qwen2-VL model with memory optimizations"""
        logger.info(f"[{self.analyzer_name}] Loading Qwen2-VL-7B for second-by-second analysis...")
        
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
        
        # Load model optimized for many segments
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"[{self.analyzer_name}] Model loaded for detailed temporal analysis!")
        
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Analyze video with second-by-second segments and overlap"""
        if self.model is None:
            self._load_model_impl()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        logger.info(f"[{self.analyzer_name}] Analyzing {duration:.1f}s video with {self.segment_step}s steps")
        
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
                    # Analyze segment with detailed prompt
                    segment_result = self._analyze_frame_segment(
                        frames, 
                        start_time, 
                        end_time,
                        segment_count,
                        segments[-1] if segments else None  # Previous segment for context
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
            
            # Move to next segment with overlap
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
                'segment_overlap': self.segment_overlap
            }
        }
    
    def _extract_segment_frames(self, video_path: str, start_time: float, end_time: float) -> List[str]:
        """Extract frames from video segment with higher sampling rate"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        segment_duration = end_time - start_time
        
        # Calculate frames to extract
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
                
                # Resize frame to reduce memory
                height, width = frame.shape[:2]
                if width > 640:  # Slightly larger for better quality
                    scale = 640 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Save frame
                frame_path = os.path.join(temp_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
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
                              segment_index: int, previous_segment: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze segment with detailed prompt for comprehensive description"""
        
        # Create message with frames
        content = []
        for i, frame_path in enumerate(frame_paths):
            content.append({"type": "image", "image": frame_path})
        
        # Build detailed prompt
        prompt = f"""Analyze this video segment from {start_time:.1f}s to {end_time:.1f}s (segment #{segment_index + 1}).

Please provide a COMPREHENSIVE description including:

1. PEOPLE & ACTIONS:
   - Who is visible (appearance, clothing, distinguishing features)
   - What they are doing (specific actions, movements, gestures)
   - Facial expressions and emotions
   - Body language and posture

2. ENVIRONMENT & LOCATION:
   - Where this takes place (room type, indoor/outdoor)
   - Visible objects and their arrangement
   - Lighting conditions and atmosphere
   - Background elements

3. VISUAL DETAILS:
   - Any text visible on screen
   - Brand names or logos
   - Colors and visual style
   - Camera angle and movement

4. TEMPORAL ASPECTS:
   - What changes during this segment
   - Movement patterns
   - Transitions or cuts"""

        # Add context from previous segment if available
        if previous_segment and 'description' in previous_segment:
            prompt += f"""

5. CONTEXT FROM PREVIOUS SEGMENT ({previous_segment['start_time']:.1f}-{previous_segment['end_time']:.1f}s):
   - Note any continuity or changes from the previous scene
   - Mention if it's the same location/person or different"""

        prompt += "\n\nProvide a detailed, natural description that captures all important visual information."
        
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
            
            # Generate detailed response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=300,  # More tokens for detailed description
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