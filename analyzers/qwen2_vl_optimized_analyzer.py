#!/usr/bin/env python3
"""
Qwen2-VL Optimized Video Analyzer - Memory-efficient version
Uses dynamic resolution control and single-frame processing to avoid OOM
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import cv2
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import logging
import time
from pathlib import Path
from qwen_vl_utils import process_vision_info
import gc

# Import base analyzer
import sys
sys.path.append('/home/user/tiktok_production')
from analyzers.base_analyzer import GPUBatchAnalyzer

logger = logging.getLogger(__name__)

class Qwen2VLOptimizedAnalyzer(GPUBatchAnalyzer):
    """
    Memory-optimized Qwen2-VL analyzer for second-by-second video descriptions.
    Uses dynamic resolution control and single-frame processing.
    """
    
    def __init__(self):
        # Initialize with batch size 1 for memory efficiency
        super().__init__(batch_size=1)
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
        self.model = None
        self.processor = None
        
        # Memory-optimized settings
        self.min_pixels = 256 * 28 * 28  # ~200k pixels minimum
        self.max_pixels = 512 * 28 * 28  # ~400k pixels maximum (reduced from default)
        
        # Video processing settings
        self.fps_sample = 1.0  # 1 frame per second
        self.max_frames_per_batch = 1  # Process single frames only
        self.max_video_duration = 300  # Max 5 minutes
        
    def _load_model_impl(self):
        """Load Qwen2-VL model with memory optimizations"""
        try:
            logger.info(f"Loading optimized Qwen2-VL model: {self.model_name}")
            
            # Load processor with memory-efficient settings
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                trust_remote_code=True
            )
            logger.info(f"✅ Processor loaded with pixels range: {self.min_pixels} - {self.max_pixels}")
            
            # Load model with optimizations
            logger.info("Loading model with memory optimizations...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map="cuda:0",  # Force single GPU for speed (was "auto")
                torch_dtype=torch.float16,  # Use FP16 instead of 8-bit for better compatibility
                trust_remote_code=True,
                attn_implementation="eager"  # Use standard attention (flash_attn not available)
            )
            
            self.model.eval()
            
            # Log model info
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"✅ Qwen2-VL model loaded successfully")
            logger.info(f"   Total parameters: {total_params / 1e9:.1f}B")
            logger.info(f"   Device: {next(self.model.parameters()).device}")
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"   GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen2-VL model: {e}")
            raise

    def extract_and_resize_frame(self, video_path: str, timestamp: float) -> Optional[Image.Image]:
        """Extract a single frame at timestamp and resize for memory efficiency"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL and resize to reduce memory usage
        pil_frame = Image.fromarray(frame)
        
        # Calculate optimal size within pixel limits
        width, height = pil_frame.size
        pixels = width * height
        
        if pixels > self.max_pixels / (28 * 28):
            # Need to downscale
            scale = np.sqrt((self.max_pixels / (28 * 28)) / pixels)
            new_width = int(width * scale)
            new_height = int(height * scale)
            # Round to nearest multiple of 28, ensure minimum size
            new_width = max(28, (new_width // 28) * 28)
            new_height = max(28, (new_height // 28) * 28)
            pil_frame = pil_frame.resize((new_width, new_height), Image.LANCZOS)
            logger.debug(f"Resized frame from {width}x{height} to {new_width}x{new_height}")
        
        return pil_frame

    def process_single_frame(self, frame: Image.Image, timestamp: float) -> Dict[str, Any]:
        """Process a single frame with Qwen2-VL"""
        try:
            # Create prompt for single frame
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": frame},
                    {"type": "text", "text": f"Describe what is happening in this frame at {int(timestamp)} seconds. Include: people and their actions, objects and positions, environment and background, any text or UI elements visible. Be specific and detailed."}
                ]
            }]
            
            # Prepare inputs
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
            
            # Generate with memory-efficient settings
            with torch.inference_mode():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=150,  # Limit output length
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        use_cache=False  # Don't cache for memory efficiency
                    )
            
            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            # Clear intermediate tensors
            del inputs, generated_ids, generated_ids_trimmed
            torch.cuda.empty_cache()
            
            return {
                'segment_id': f"qwen2vl_opt_{int(timestamp)}",
                'start_time': timestamp,
                'end_time': timestamp + 1.0,
                'timestamp': timestamp,
                'description': output_text.strip(),
                'confidence': 0.9,
                'analyzer': 'qwen2_vl_optimized'
            }
            
        except Exception as e:
            logger.error(f"Error processing frame at {timestamp}s: {e}")
            return {
                'segment_id': f"qwen2vl_opt_{int(timestamp)}",
                'start_time': timestamp,
                'end_time': timestamp + 1.0,
                'timestamp': timestamp,
                'description': f"Error processing frame: {str(e)}",
                'confidence': 0.0,
                'analyzer': 'qwen2_vl_optimized'
            }

    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> List[Dict[str, Any]]:
        """Process batch of frames - not used in optimized version"""
        # This analyzer processes frames one by one, so this method is not used
        # But we need to implement it to satisfy the abstract base class
        return []
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Analyze video frame-by-frame with memory optimization"""
        start_time = time.time()
        
        # Load model if needed
        if self.model is None:
            self.load_model()
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        cap.release()
        
        logger.info(f"Analyzing video: {Path(video_path).name}")
        logger.info(f"Duration: {video_duration:.1f}s, FPS: {fps:.1f}")
        logger.info("Processing frame-by-frame for memory efficiency...")
        
        # Process video second by second
        all_segments = []
        current_time = 0
        
        # Limit to max duration
        max_time = min(video_duration, self.max_video_duration)
        
        while current_time < max_time:
            # Extract and process single frame
            frame = self.extract_and_resize_frame(video_path, current_time)
            
            if frame:
                # Log progress every 10 seconds
                if int(current_time) % 10 == 0:
                    logger.info(f"Processing: {current_time:.0f}s / {max_time:.0f}s")
                    # Log memory usage
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        logger.info(f"  GPU Memory: {allocated:.1f}GB")
                
                # Process single frame
                segment = self.process_single_frame(frame, current_time)
                all_segments.append(segment)
                
                # Clean up
                del frame
                gc.collect()
            
            # Move to next second
            current_time += 1.0
        
        # Build final result
        analysis_time = time.time() - start_time
        
        result = {
            'segments': all_segments,
            'metadata': {
                'analyzer': 'qwen2_vl_optimized',
                'model': self.model_name,
                'video_duration': video_duration,
                'analyzed_duration': min(video_duration, self.max_video_duration),
                'total_segments': len(all_segments),
                'fps_analyzed': self.fps_sample,
                'processing_time': analysis_time,
                'temporal_coverage': len(all_segments) / video_duration if video_duration > 0 else 0,
                'memory_optimizations': {
                    'min_pixels': self.min_pixels,
                    'max_pixels': self.max_pixels,
                    'batch_size': 1,
                    'fp16': True,
                    'use_cache': False
                }
            }
        }
        
        logger.info(f"✅ Qwen2-VL analysis complete: {len(all_segments)} segments in {analysis_time:.1f}s")
        
        # Log sample output
        if all_segments:
            logger.info("Sample descriptions:")
            for seg in all_segments[:3]:
                logger.info(f"  [{seg['timestamp']:.0f}s] {seg['description'][:100]}...")
        
        return result

    def cleanup(self):
        """Clean up GPU memory"""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Qwen2-VL optimized model cleaned up")