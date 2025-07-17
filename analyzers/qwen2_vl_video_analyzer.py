#!/usr/bin/env python3
"""
Qwen2-VL Video Analyzer - KORREKTE VERSION
Basiert auf offiziellen Qwen2-VL Beispielen für Video-Analyse
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

class Qwen2VLVideoAnalyzer(GPUBatchAnalyzer):
    """Qwen2-VL Video Analyzer für temporale Beschreibungen"""
    
    def __init__(self, batch_size=1):
        super().__init__(batch_size)
        self.analyzer_name = "qwen2_vl_temporal"
        # DISABLE INT8 - produces gibberish output!
        # self.model_name = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8"
        # self.fallback_model_name = "Qwen/Qwen2-VL-7B-Instruct"
        
        # Use regular model only - INT8 is broken
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
        self.fallback_model_name = "Qwen/Qwen2-VL-7B-Instruct"
        
        # ULTRA-OPTIMIZED: 2s segments, 1s steps for detailed coverage
        self.segment_duration = 2.0  # 2 seconds per segment
        self.segment_overlap = 1.0   # 50% overlap for continuity  
        self.segment_step = 1.0      # Analyze every second! ~49 segments
        self.fps_sample = 2.0        # 2 fps (official Qwen2-VL recommendation)
        self.max_frames = 4          # 4 frames per 2s segment
        self.max_tokens = 100        # Shorter, more focused descriptions
        self.batch_size = 8          # Process more frames in parallel
        
        # Context tracking for better quality
        self.scene_context = {}
        self.previous_description = ""
        self.person_attributes = {}
        
    def _load_model_impl(self):
        """Load Qwen2-VL model with optimized configuration"""
        logger.info(f"[{self.analyzer_name}] Loading optimized Qwen2-VL...")
        
        # Configure pixel limits for memory efficiency
        min_pixels = 256 * 28 * 28
        max_pixels = 768 * 28 * 28  # Memory limit that worked
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True
        )
        
        # Try Flash Attention 2 first (if available)
        try:
            logger.info(f"[{self.analyzer_name}] Trying Flash Attention 2...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",  # 2-3x faster if available
                low_cpu_mem_usage=True
            )
            logger.info(f"[{self.analyzer_name}] ✅ Loaded with Flash Attention 2!")
        except Exception as e:
            logger.warning(f"[{self.analyzer_name}] Flash Attention 2 not available: {e}")
            # Fallback to eager attention
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
                attn_implementation="eager",
                low_cpu_mem_usage=True
            )
            logger.info(f"[{self.analyzer_name}] ✅ Loaded with eager attention")
        
        self.model.eval()
        logger.info(f"[{self.analyzer_name}] Model loaded successfully")
    
    def extract_video_frames(self, video_path: str, max_frames: int = 16) -> List[Image.Image]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices for uniform sampling
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / max_frames
            frame_indices = [int(i * step) for i in range(max_frames)]
        
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
    
    def analyze_video_segment(self, frames: List[Image.Image], start_time: float, end_time: float) -> Dict[str, Any]:
        """Analyze video segment with Qwen2-VL"""
        
        # Create high-quality German prompt for detailed action analysis
        prompt = f"""Analysiere Sekunde {start_time:.0f}-{end_time:.0f} des Videos.

BESCHREIBE GENAU:
1. WER: Geschlecht, Kleidung, Körperhaltung (z.B. "oberkörperfreier Mann mit Tattoos")
2. WO: Exakter Ort + sichtbare Objekte (z.B. "Badezimmer mit Waschbecken links, Spiegel")  
3. WAS: JEDE Bewegung/Aktion in dieser Sekunde:
   - Körperbewegungen (hebt Arm, dreht Kopf, geht)
   - Handlungen (putzt Zähne, zeigt Daumen hoch)
   - Blickrichtung (schaut in Kamera, nach rechts)
4. DETAILS: Objekte in Händen, Gesichtsausdruck, Beleuchtung

KURZ & PRÄZISE (40 Wörter). Was passiert GENAU in diesen 2 Sekunden?"""

        # Prepare messages for Qwen2-VL
        content = []
        for frame in frames:
            content.append({"type": "image", "image": frame})
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
            
            # Generate response with optimized parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,  # 100 tokens
                    do_sample=False,  # Greedy decoding for speed
                    use_cache=True,   # KV cache optimization
                    temperature=0.0,  # Deterministic
                    num_beams=1,      # No beam search for speed
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            # Decode response
            generated_text = self.processor.decode(
                generated_ids[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            ).strip()
            
            # Update context tracking
            self.previous_description = generated_text
            self._update_scene_context(generated_text)
            
            return {
                'segment_id': f"qwen_temporal_{int(start_time * 10)}",
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
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _update_scene_context(self, description: str):
        """Update scene context from description"""
        # Extract location
        locations = ["Badezimmer", "Küche", "Auto", "Büro", "Fitnessstudio", "Straße"]
        for loc in locations:
            if loc.lower() in description.lower():
                self.scene_context['location'] = loc
                break
        
        # Extract person attributes
        if "Mann" in description:
            self.person_attributes['gender'] = 'Mann'
        elif "Frau" in description:
            self.person_attributes['gender'] = 'Frau'
            
        if "tätowiert" in description or "Tattoo" in description:
            self.person_attributes['tattoos'] = True
    
    def process_batch_gpu(self, frames, frame_times):
        """Process batch of frames on GPU - required by base class"""
        return []
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Main analysis function"""
        if not self.model_loaded:
            self._load_model_impl()
            self.model_loaded = True
            
        video_path = Path(video_path)
        logger.info(f"[{self.analyzer_name}] Analyzing {video_path.name}")
        
        # Get video duration
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        segments = []
        current_time = 0.0
        
        # Process video with optimized segments (~16 for 49s video)
        while current_time < duration:
            # Calculate segment boundaries  
            start_time = current_time
            end_time = min(current_time + self.segment_duration, duration)
            
            # Extract frames for this segment using working parameters
            frames = self.extract_frames_for_segment_working(
                str(video_path), 
                start_time, 
                end_time - start_time
            )
            
            if frames and len(frames) > 0:
                result = self.analyze_video_segment(frames, start_time, end_time)
                if result:
                    segments.append(result)
                    logger.info(f"   Segment {start_time:.1f}s: {result['description'][:60]}...")
                    logger.debug(f"[{self.analyzer_name}] Segment added to list, total now: {len(segments)}")
                else:
                    logger.warning(f"[{self.analyzer_name}] No result for segment {start_time:.1f}s")
            else:
                logger.warning(f"[{self.analyzer_name}] No frames extracted for segment {start_time:.1f}s")
            
            # Move to next segment
            current_time += self.segment_step
            
            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
        
        logger.info(f"[{self.analyzer_name}] Completed: {len(segments)} segments")
        
        # CRITICAL FIX: Ensure segments are properly returned
        result = {
            'analyzer_name': self.analyzer_name,
            'segments': segments,
            'summary': {
                'total_segments': len(segments),
                'video_duration': duration
            }
        }
        
        # DEBUG LOGGING: Verify result before return
        logger.info(f"[{self.analyzer_name}] Return result keys: {list(result.keys())}")
        logger.info(f"[{self.analyzer_name}] Segments in result: {len(result.get('segments', []))}")
        
        if segments and len(segments) > 0:
            logger.info(f"[{self.analyzer_name}] First segment sample: {segments[0]}")
            logger.info(f"[{self.analyzer_name}] Last segment sample: {segments[-1]}")
        else:
            logger.error(f"[{self.analyzer_name}] ERROR: No segments to return!")
        
        return result
    
    def extract_frames_for_segment(self, video_path: str, start_time: float, duration: float) -> List[Image.Image]:
        """Extract frames for a specific segment"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_time * fps)
        end_frame = int((start_time + duration) * fps)
        total_segment_frames = end_frame - start_frame
        
        # Select frames for this segment (max 8 frames per segment)
        max_frames_per_segment = min(8, total_segment_frames)
        
        if total_segment_frames <= max_frames_per_segment:
            frame_indices = list(range(start_frame, end_frame))
        else:
            step = total_segment_frames / max_frames_per_segment
            frame_indices = [int(start_frame + i * step) for i in range(max_frames_per_segment)]
        
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
    
    def extract_frames_for_segment_working(self, video_path: str, start_time: float, duration: float) -> List[Image.Image]:
        """Extract frames using working 97-segment parameters"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_time * fps)
        end_frame = int((start_time + duration) * fps)
        
        # Use 2.0 fps sampling like working version
        sample_frames_in_segment = int(duration * self.fps_sample)
        sample_frames_in_segment = min(sample_frames_in_segment, self.max_frames)
        sample_frames_in_segment = max(1, sample_frames_in_segment)  # At least 1 frame
        
        if end_frame - start_frame <= sample_frames_in_segment:
            frame_indices = list(range(start_frame, end_frame))
        else:
            step = (end_frame - start_frame) / sample_frames_in_segment
            frame_indices = [int(start_frame + i * step) for i in range(sample_frames_in_segment)]
        
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
    
    def analyze_video_segment_working(self, frames: List[Image.Image], start_time: float, end_time: float) -> Dict[str, Any]:
        """Analyze video segment using working 97-segment approach"""
        
        # Shorter prompt like working version
        prompt = f"Describe what happens from {start_time:.1f}s to {end_time:.1f}s. Focus on actions, objects, and scene details."
        
        # Prepare messages for Qwen2-VL
        content = []
        for frame in frames:
            content.append({"type": "image", "image": frame})
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
            
            # Generate response with working parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                    use_cache=True
                )
            
            # Decode response
            generated_text = self.processor.decode(
                generated_ids[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            ).strip()
            
            return {
                'segment_id': f"qwen_temporal_{int(start_time*10)}",  # More granular IDs
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
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()