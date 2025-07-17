#!/usr/bin/env python3
"""
Qwen2-VL Video Analyzer - FIXED VERSION
Korrekte Implementierung basierend auf offizieller Dokumentation
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

class Qwen2VLVideoAnalyzerFixed(GPUBatchAnalyzer):
    """Qwen2-VL Video Analyzer mit korrekter Video-Verarbeitung"""
    
    def __init__(self, batch_size=1):
        super().__init__(batch_size)
        self.analyzer_name = "qwen2_vl_temporal"
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
        
        # Video processing parameters
        self.max_frames = 16  # Optimal für 7B Modell
        self.segment_duration = 3.0  # 3 Sekunden pro Segment für mehr Kontext
        self.fps_sample = 2.0  # Sample 2 frames per second
        
    def _load_model_impl(self):
        """Load Qwen2-VL model correctly"""
        logger.info(f"[{self.analyzer_name}] Loading Qwen2-VL-7B model...")
        
        # Configure pixel limits
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        
        # Load processor with proper configuration
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True
        )
        
        # Load model with optimizations
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation="eager"  # Use eager for stability
        )
        
        self.model.eval()
        logger.info(f"[{self.analyzer_name}] Model loaded successfully!")
        
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Analyze video with proper segment processing"""
        if self.model is None:
            self._load_model_impl()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        segments = []
        
        # Process video in segments
        for start_time in range(0, int(duration), int(self.segment_duration)):
            end_time = min(start_time + self.segment_duration, duration)
            
            try:
                # Analyze segment using DIRECT VIDEO PATH
                segment_result = self._analyze_video_segment(
                    video_path, 
                    start_time, 
                    end_time
                )
                
                if segment_result and segment_result.get('description'):
                    segments.append(segment_result)
                    logger.info(f"[{self.analyzer_name}] Processed segment {start_time}-{end_time}s")
                    
            except Exception as e:
                logger.error(f"[{self.analyzer_name}] Error processing segment {start_time}-{end_time}: {e}")
                continue
        
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            'analyzer_name': self.analyzer_name,
            'segments': segments,
            'summary': {
                'total_segments': len(segments),
                'video_duration': duration
            }
        }
    
    def _analyze_video_segment(self, video_path: str, start_time: float, end_time: float) -> Dict[str, Any]:
        """Analyze a video segment using NATIVE VIDEO SUPPORT"""
        
        # Create message with video content type
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,  # Direct video path
                        "fps": self.fps_sample,  # Sample rate
                        "resized_height": 280,
                        "resized_width": 280,
                        # Optional: specify time range (if supported)
                        # "start_time": start_time,
                        # "end_time": end_time
                    },
                    {
                        "type": "text", 
                        "text": f"Please describe what happens in this video segment from {start_time:.1f}s to {end_time:.1f}s. Focus on the main subjects, their actions, scene changes, and any important visual elements. Be specific and detailed."
                    }
                ]
            }
        ]
        
        try:
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process vision info - this handles video loading
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
                    temperature=0.7,
                    use_cache=True
                )
            
            # Decode response
            generated_text = self.processor.decode(
                generated_ids[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            ).strip()
            
            # Cleanup tensors
            del inputs, generated_ids
            torch.cuda.empty_cache()
            
            return {
                'segment_id': f"qwen_temporal_{int(start_time)}",
                'start_time': start_time,
                'end_time': end_time,
                'timestamp': start_time,
                'description': generated_text,
                'frames_analyzed': int((end_time - start_time) * self.fps_sample),
                'confidence': 0.95,
                'analyzer': self.analyzer_name
            }
            
        except Exception as e:
            logger.error(f"[{self.analyzer_name}] Error in segment analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> List[Dict[str, Any]]:
        """Not used - we process video directly"""
        return []