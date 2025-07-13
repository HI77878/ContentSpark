#!/usr/bin/env python3
"""
Qwen2-VL Video Analyzer - OPTIMIZED VERSION
Mit Frame-Sampling und Memory-Management für lange Videos
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

class Qwen2VLVideoAnalyzerOptimized(GPUBatchAnalyzer):
    """Qwen2-VL Video Analyzer mit Memory-Optimierung"""
    
    def __init__(self, batch_size=1):
        super().__init__(batch_size)
        self.analyzer_name = "qwen2_vl_temporal"
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
        
        # Optimized parameters for long videos
        self.segment_duration = 4.0  # 4 seconds per segment
        self.fps_sample = 1.0  # Only 1 frame per second to reduce memory
        self.max_pixels = 768 * 28 * 28  # Reduced max pixels
        
    def _load_model_impl(self):
        """Load Qwen2-VL model with memory optimizations"""
        logger.info(f"[{self.analyzer_name}] Loading Qwen2-VL-7B model with optimizations...")
        
        # Configure pixel limits for memory efficiency
        min_pixels = 256 * 28 * 28
        max_pixels = self.max_pixels
        
        # Load processor with strict limits
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True
        )
        
        # Load model with memory optimizations
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # Use fp16 instead of bfloat16
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        
        # Clear cache after loading
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"[{self.analyzer_name}] Model loaded with memory optimizations!")
        
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Analyze video with memory-efficient segment processing"""
        if self.model is None:
            self._load_model_impl()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        segments = []
        
        # Process video in segments with frame extraction
        for start_time in range(0, int(duration), int(self.segment_duration)):
            end_time = min(start_time + self.segment_duration, duration)
            
            try:
                # Extract frames for this segment to reduce memory
                frames = self._extract_segment_frames(video_path, start_time, end_time)
                
                if frames:
                    # Analyze segment with extracted frames
                    segment_result = self._analyze_frame_segment(
                        frames, 
                        start_time, 
                        end_time
                    )
                    
                    if segment_result and segment_result.get('description'):
                        segments.append(segment_result)
                        logger.info(f"[{self.analyzer_name}] Processed segment {start_time}-{end_time}s")
                
                # Cleanup after each segment
                del frames
                torch.cuda.empty_cache()
                gc.collect()
                    
            except Exception as e:
                logger.error(f"[{self.analyzer_name}] Error processing segment {start_time}-{end_time}: {e}")
                # Continue with next segment even if one fails
                torch.cuda.empty_cache()
                gc.collect()
                continue
        
        return {
            'analyzer_name': self.analyzer_name,
            'segments': segments,
            'summary': {
                'total_segments': len(segments),
                'video_duration': duration
            }
        }
    
    def _extract_segment_frames(self, video_path: str, start_time: float, end_time: float) -> List[str]:
        """Extract frames from video segment and save as temporary files"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        frame_interval = int(fps / self.fps_sample)  # Sample based on fps_sample
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_count = 0
            for frame_idx in range(start_frame, end_frame, frame_interval):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame to reduce memory
                height, width = frame.shape[:2]
                if width > 560:
                    scale = 560 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Save frame as temporary file
                frame_path = os.path.join(temp_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
                frame_count += 1
                
                # Limit frames to prevent OOM
                if frame_count >= 8:  # Max 8 frames per segment
                    break
            
            cap.release()
            return frames
            
        except Exception as e:
            cap.release()
            # Cleanup temp files on error
            for f in frames:
                if os.path.exists(f):
                    os.remove(f)
            os.rmdir(temp_dir)
            raise e
    
    def _analyze_frame_segment(self, frame_paths: List[str], start_time: float, end_time: float) -> Dict[str, Any]:
        """Analyze segment using extracted frames"""
        
        # Create message with frames as images
        content = []
        for frame_path in frame_paths:
            content.append({"type": "image", "image": frame_path})
        
        content.append({
            "type": "text", 
            "text": f"This is a segment from {start_time:.1f}s to {end_time:.1f}s of a video. Please describe what happens in this video segment. Focus on the main subjects, their actions, and any important visual elements."
        })
        
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
            
            # Prepare inputs with memory limit
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
                    max_new_tokens=200,  # Reduced from 256
                    do_sample=False,
                    temperature=0.7,
                    use_cache=True,
                    num_beams=1  # No beam search to save memory
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
            temp_dir = os.path.dirname(frame_paths[0])
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
            
            return {
                'segment_id': f"qwen_temporal_{int(start_time)}",
                'start_time': start_time,
                'end_time': end_time,
                'timestamp': start_time,
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