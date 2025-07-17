#!/usr/bin/env python3
"""
Qwen2-VL Ultra Detailed Analyzer - Maximum Data Quality & Quantity
Optimized for comprehensive video analysis with rich, detailed descriptions
"""

import os
# Disable torch.compile completely to avoid INT8 conflicts
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging
import gc
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from analyzers.base_analyzer import GPUBatchAnalyzer

logger = logging.getLogger(__name__)

class Qwen2VL2BTestAnalyzer(GPUBatchAnalyzer):
    """
    Ultra-detailed Qwen2-VL analyzer for maximum data quality
    
    Key Features:
    - 5-10 frames per segment for comprehensive analysis
    - 2-3 second segments for meaningful action sequences
    - Multiple detailed prompts for rich descriptions
    - Batch processing for optimal performance
    - No anti-duplication - every moment is unique!
    """
    
    def __init__(self, batch_size=2):
        super().__init__(batch_size)
        self.analyzer_name = "qwen2_vl_2b_test"
        self.model_name = "Qwen/Qwen2-VL-2B-Instruct"  # 2B model for faster performance
        
        # OPTIMIZED SETTINGS FOR QUALITY
        self.frames_per_segment = 4          # Reduced from 8 - 2x faster
        self.segment_duration = 3.0          # Increased from 2.0 - 33% fewer segments
        self.frame_resolution = (336, 336)   # Higher resolution
        self.max_new_tokens = 300            # Much longer descriptions
        
        # DETAILED PROMPT TEMPLATES
        self.prompt_templates = [
            """Analyze this video segment in extreme detail. Describe EVERYTHING:

1. PEOPLE & ACTIONS:
   - Exact body positions, movements, gestures
   - Facial expressions, eye direction, emotions
   - Clothing details, accessories, physical appearance
   - What they are doing, holding, interacting with

2. ENVIRONMENT & OBJECTS:
   - Location/setting (room type, indoor/outdoor)
   - All visible objects and their positions
   - Furniture, decorations, background elements
   - Lighting conditions, time of day

3. CAMERA & COMPOSITION:
   - Camera angle, movement, perspective
   - Shot type (close-up, wide, medium)
   - Visual composition, framing

4. TEXT & UI:
   - Any visible text, signs, labels
   - Screen content, overlays, captions
   - Brand names, logos

5. STORY & CONTEXT:
   - What's happening in this scene
   - The narrative or activity being shown
   - Mood, atmosphere, energy level

Write a comprehensive description (minimum 150 words) covering ALL these aspects. Be extremely specific and detailed.""",

            """You are analyzing a {duration}s video segment with {num_frames} frames.
Study each frame carefully and describe:

- The complete sequence of actions from start to finish
- Every person's movements, expressions, and activities  
- All objects being used, held, or interacted with
- The full environment including background details
- Any changes or progression throughout the segment
- Camera movements or angle changes
- Visible text, UI elements, or overlays

Provide an exhaustive, detailed analysis of everything visible in these frames. 
Minimum 200 words. Miss nothing.""",

            """Frame-by-frame analysis required:

For each visible element, describe:
- WHO: All people, their appearance, expressions, positions
- WHAT: Every action, gesture, movement, interaction  
- WHERE: Complete setting, location details, background
- WHEN: Time indicators, lighting, sequence of events
- HOW: The manner of actions, speed, intensity
- WHY: The apparent purpose or context

Include ALL objects, even minor ones. Describe colors, textures, brands, text.
Note any audio cues visible (mouth movements, instrument playing).
This should be a forensic-level detailed description. Minimum 150 words."""
        ]
        
        # Batch processing queue
        self.processing_queue = []
        self.batch_size = batch_size
        
    def _load_model_impl(self):
        """Load model with optimal settings for detailed analysis"""
        logger.info(f"[{self.analyzer_name}] Loading Qwen2-VL-7B for ultra-detailed analysis...")
        
        # Load processor with high resolution support
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            min_pixels=256*28*28,
            max_pixels=1280*28*28  # Much higher for quality
        )
        
        # INT8 Quantization configuration for 2-4x speedup
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4"
        )
        
        # Load model with INT8 quantization
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation="eager"  # Disable flash attention - not installed
        )
        
        self.model.eval()
        
        # Optimize with torch.compile for additional speedup
        # DISABLED - causing errors with Qwen2VL
        # try:
        #     torch._dynamo.config.suppress_errors = True
        #     self.model = torch.compile(self.model, mode="max-autotune", backend="inductor")
        #     logger.info("✅ Model compiled with torch.compile for maximum performance")
        # except Exception as e:
        #     logger.warning(f"torch.compile not available or failed: {e}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        logger.info(f"✅ [{self.analyzer_name}] Model loaded successfully")
        
    def create_frame_grid(self, frames: List[Image.Image], add_labels: bool = True) -> Image.Image:
        """Create an optimized grid of frames for analysis"""
        n_frames = len(frames)
        
        # Calculate grid dimensions
        if n_frames <= 4:
            cols = 2
        elif n_frames <= 9:
            cols = 3
        else:
            cols = 4
        rows = (n_frames + cols - 1) // cols
        
        # Resize frames
        frame_size = self.frame_resolution
        resized_frames = [f.resize(frame_size, Image.Resampling.LANCZOS) for f in frames]
        
        # Create grid
        grid_width = frame_size[0] * cols
        grid_height = frame_size[1] * rows
        grid = Image.new('RGB', (grid_width, grid_height), 'black')
        
        # Place frames
        for idx, frame in enumerate(resized_frames):
            row = idx // cols
            col = idx % cols
            x = col * frame_size[0]
            y = row * frame_size[1]
            
            grid.paste(frame, (x, y))
            
            # Add frame numbers
            if add_labels:
                draw = ImageDraw.Draw(grid)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                except:
                    font = None
                    
                # Add frame number
                text = f"F{idx+1}"
                draw.rectangle((x, y, x+50, y+30), fill='black')
                draw.text((x+5, y+5), text, fill='white', font=font)
        
        return grid
    
    def process_batch_gpu(self, frames, frame_times):
        """Required method for base class compatibility"""
        # This analyzer uses its own batching in analyze()
        return [{"timestamp": t, "frame_analyzed": True} for t in frame_times]
    
    def analyze_segment_batch(self, segments: List[Dict]) -> List[Dict]:
        """Analyze multiple segments in a batch for efficiency"""
        results = []
        
        for segment in segments:
            frames = segment['frames']
            start_time = segment['start_time']
            end_time = segment['end_time']
            prompt_idx = segment.get('prompt_idx', 0)
            
            # Create grid
            grid = self.create_frame_grid(frames, add_labels=True)
            
            # Select prompt template
            prompt_template = self.prompt_templates[prompt_idx % len(self.prompt_templates)]
            prompt = prompt_template.format(
                duration=end_time - start_time,
                num_frames=len(frames)
            )
            
            # Add temporal context
            prompt += f"\n\nThis segment covers {start_time:.1f}s to {end_time:.1f}s of the video."
            
            # Prepare message
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": grid},
                    {"type": "text", "text": prompt}
                ]
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
                
                # Generate detailed description with optimized settings
                with torch.inference_mode():
                    with torch.cuda.amp.autocast(dtype=torch.float16):  # Match model dtype
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=True,        # Enable sampling for variety
                            temperature=0.8,       # Balanced creativity
                            top_p=0.95,           # Nucleus sampling
                            repetition_penalty=1.1, # Reduce repetition
                            use_cache=True,        # Enable KV cache
                            cache_implementation="static",  # Faster than dynamic
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                
                # Decode
                generated_text = self.processor.decode(
                    generated_ids[0][len(inputs.input_ids[0]):],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                ).strip()
                
                result = {
                    'segment_id': f"segment_{int(start_time)}_{int(end_time)}",
                    'start_time': start_time,
                    'end_time': end_time,
                    'time_range': [start_time, end_time],
                    'duration': end_time - start_time,
                    'frames_analyzed': len(frames),
                    'description': generated_text,
                    'word_count': len(generated_text.split()),
                    'prompt_template_used': prompt_idx,
                    'confidence': 0.95,
                    'analyzer': self.analyzer_name
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error analyzing segment {start_time}-{end_time}: {e}")
                results.append({
                    'segment_id': f"segment_{int(start_time)}_{int(end_time)}",
                    'start_time': start_time,
                    'end_time': end_time,
                    'time_range': [start_time, end_time],
                    'error': str(e)
                })
        
        # Clear GPU cache after batch
        torch.cuda.empty_cache()
        gc.collect()
        
        return results
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Main analysis function with optimized batching"""
        if not self.model_loaded:
            self._load_model_impl()
            self.model_loaded = True
            
        video_path = Path(video_path)
        logger.info(f"[{self.analyzer_name}] Starting ultra-detailed analysis of {video_path.name}")
        
        start_time = time.time()
        
        # Get video info
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        
        # Calculate segments
        segments_to_process = []
        current_time = 0
        prompt_idx = 0
        
        while current_time < duration:
            end_time = min(current_time + self.segment_duration, duration)
            
            # Extract frames for this segment
            frames = self.extract_frames_for_segment(
                str(video_path), 
                current_time, 
                end_time - current_time
            )
            
            if frames:
                segments_to_process.append({
                    'frames': frames,
                    'start_time': current_time,
                    'end_time': end_time,
                    'prompt_idx': prompt_idx
                })
                
                # Rotate prompts for variety
                prompt_idx += 1
            
            current_time = end_time
        
        logger.info(f"[{self.analyzer_name}] Processing {len(segments_to_process)} segments")
        
        # Process in batches
        all_results = []
        for i in range(0, len(segments_to_process), self.batch_size):
            batch = segments_to_process[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(segments_to_process) + self.batch_size - 1)//self.batch_size}")
            
            batch_results = self.analyze_segment_batch(batch)
            all_results.extend(batch_results)
        
        # Calculate statistics
        processing_time = time.time() - start_time
        total_words = sum(r.get('word_count', 0) for r in all_results if 'error' not in r)
        avg_words_per_segment = total_words / len(all_results) if all_results else 0
        
        logger.info(f"[{self.analyzer_name}] Analysis complete in {processing_time:.1f}s")
        logger.info(f"  - Total segments: {len(all_results)}")
        logger.info(f"  - Total words: {total_words}")
        logger.info(f"  - Avg words/segment: {avg_words_per_segment:.1f}")
        
        return {
            'segments': all_results,
            'summary': {
                'total_segments': len(all_results),
                'total_duration': duration,
                'processing_time_seconds': processing_time,
                'total_words_generated': total_words,
                'average_words_per_segment': avg_words_per_segment,
                'frames_per_segment': self.frames_per_segment,
                'segment_duration': self.segment_duration
            },
            'metadata': {
                'analyzer': self.analyzer_name,
                'model': self.model_name,
                'video_path': str(video_path),
                'fps': fps,
                'total_frames': total_frames
            }
        }
    
    def extract_frames_for_segment(self, video_path: str, start_time: float, duration: float) -> List[Image.Image]:
        """Extract frames for a segment with optimal sampling"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_time * fps)
        end_frame = int((start_time + duration) * fps)
        total_frames = end_frame - start_frame
        
        # Calculate frame indices
        if total_frames <= self.frames_per_segment:
            frame_indices = list(range(start_frame, end_frame))
        else:
            # Evenly distributed sampling
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