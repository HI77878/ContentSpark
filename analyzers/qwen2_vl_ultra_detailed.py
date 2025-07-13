#!/usr/bin/env python3
"""
Qwen2-VL Ultra Detailed Analyzer - Maximum Data Quality & Quantity
Optimized for comprehensive video analysis with rich, detailed descriptions
"""

import os
import warnings
import logging

# Disable torch.compile completely to avoid INT8 conflicts
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress all torch.compile warnings
warnings.filterwarnings("ignore", message=".*torch.compile.*")
warnings.filterwarnings("ignore", message=".*Using.*torch.compile.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
# GPU Performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
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

class Qwen2VLUltraDetailedAnalyzer(GPUBatchAnalyzer):
    """
    Ultra-detailed Qwen2-VL analyzer for maximum data quality
    
    Key Features:
    - 5-10 frames per segment for comprehensive analysis
    - 2-3 second segments for meaningful action sequences
    - Multiple detailed prompts for rich descriptions
    - Batch processing for optimal performance
    - No anti-duplication - every moment is unique!
    """
    
    def __init__(self, batch_size=4):  # Increased to 4 for maximum GPU utilization
        super().__init__(batch_size)
        self.analyzer_name = "qwen2_vl_ultra_detailed"
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
        
        # OPTIMIZED SETTINGS FOR QUALITY
        self.frames_per_segment = 3          # 3 frames für Balance zwischen Qualität und Speed
        self.segment_duration = 3.0          # 3s Segmente - weniger Overhead, gute Qualität
        self.frame_resolution = (280, 280)   # Etwas reduziert für Speed
        self.max_new_tokens = 200            # Reduziert für schnellere Generation
        
        # DETAILED PROMPT TEMPLATES
        self.prompt_templates = [
            """Analyze this video segment in CHRONOLOGICAL ORDER. Describe events as they happen:

IMPORTANT: Follow the exact time sequence. If the scene changes or cuts to a new location, clearly indicate this transition.

Frame 1 (0s): Describe what happens at the START of the segment
Frame 2 (1.5s): Describe what happens in the MIDDLE 
Frame 3 (3s): Describe what happens at the END

For each moment, detail:
1. PEOPLE & ACTIONS:
   - Exact body positions, movements, gestures
   - Facial expressions, emotions
   - Clothing details, physical appearance
   - What they are doing, holding, interacting with

2. ENVIRONMENT & SETTING:
   - Location/setting (room type, indoor/outdoor)
   - All visible objects and their positions
   - Furniture, decorations, background elements
   - Lighting conditions

3. SCENE CHANGES:
   - If the location or scene changes between frames, describe it as "The scene changes to..."
   - Note any cuts or transitions between different settings
   - Maintain clear chronological order

Do NOT mention text overlays, UI elements, captions, or on-screen text.
Focus on the visual action and story progression. Minimum 150 words.""",

            """You are analyzing a {duration}s video segment with {num_frames} frames showing a time sequence.

CRITICAL: Describe events in CHRONOLOGICAL ORDER as they unfold from Frame 1 to Frame {num_frames}.

For this segment spanning 0s to {duration}s:
- Frame 1 represents the beginning (0s)
- Frame 2 represents the middle (~{duration:.1f}s/2)  
- Frame 3 represents the end ({duration}s)

Describe in order:
1. FIRST: What happens at the beginning of the segment
2. THEN: What happens next (note if scene changes or cuts)
3. FINALLY: What happens at the end

Include:
- Complete sequence of actions in temporal order
- Every person's movements and activities as they progress
- Environment and setting (note if it changes)
- Any scene transitions or cuts between different locations

If you notice a scene change (different location/setting between frames), explicitly state:
"The scene cuts to..." or "The setting changes to..."

Do NOT describe text overlays or UI elements.
Provide detailed chronological analysis. Minimum 200 words.""",

            """Analyze this video segment frame-by-frame in STRICT CHRONOLOGICAL ORDER:

Frame F1 (Start - 0s):
- WHO: People present, their appearance, position
- WHAT: Actions happening at the beginning
- WHERE: Setting/location at start

Frame F2 (Middle - ~1.5s):
- WHO: Same person or different? Note any changes
- WHAT: Actions progressing or new actions
- WHERE: Same location or scene change?
- TRANSITION: If scene changed, describe as "The scene cuts to a new location..."

Frame F3 (End - 3s):
- WHO: People and their final positions
- WHAT: Concluding actions
- WHERE: Final setting

IMPORTANT INSTRUCTIONS:
- Describe the temporal flow from beginning to end
- Explicitly note scene changes or cuts between frames
- Use phrases like "Initially...", "Then...", "Finally..."
- If location changes between frames, this is a scene cut - describe it clearly
- Focus on visual storytelling and action progression

Do NOT mention any text overlays, UI elements, or on-screen text.
Provide forensic-level detail in chronological order. Minimum 150 words."""
        ]
        
        # Batch processing queue
        self.processing_queue = []
        self.batch_size = batch_size
        
    def _load_model_impl(self):
        """Load model with optimal settings for detailed analysis"""
        # WICHTIG: Model Pre-Loading funktioniert NICHT mit Multiprocessing!
        # Jeder Prozess lädt sein eigenes Model.
        
        logger.info(f"[{self.analyzer_name}] Loading Qwen2-VL-7B for ultra-detailed analysis...")
        
        # Load processor with high resolution support
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            min_pixels=256*28*28,
            max_pixels=1280*28*28  # Much higher for quality
        )
        
        # KEINE Quantisierung für beste Qualität - nutze float16 direkt
        # Quantisierung zerstört die Videoanalyse-Qualität!
        
        # Load model with float16 precision
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # Native float16 ohne Quantisierung
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation="sdpa"  # Use Scaled Dot Product Attention - faster than eager!
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
            
            # Add temporal context with frame timing
            frame_duration = (end_time - start_time) / len(frames)
            frame_times_str = ", ".join([f"F{i+1}={start_time + i*frame_duration:.1f}s" for i in range(len(frames))])
            prompt += f"\n\nThis segment covers {start_time:.1f}s to {end_time:.1f}s of the video."
            prompt += f"\nFrame timings: {frame_times_str}"
            prompt += f"\nREMEMBER: Describe events in chronological order. Note any scene cuts or location changes between frames."
            
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
                            do_sample=False,       # Deterministic für konsistente Qualität
                            temperature=0.7,       # Lower für präzisere Beschreibungen
                            top_p=0.9,            # Nucleus sampling
                            repetition_penalty=1.05, # Leicht reduziert
                            use_cache=True,        # Enable KV cache
                            cache_implementation="static",  # Faster than dynamic
                            pad_token_id=self.processor.tokenizer.pad_token_id
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