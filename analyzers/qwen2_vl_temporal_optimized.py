#!/usr/bin/env python3
"""
Ultra-optimized Qwen2-VL mit Batching für <3s pro Segment
Ziel: Von 120s auf <15s für 5 Segmente
"""

import torch
import time
import logging
import numpy as np
import cv2
import os
import sys
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Performance optimizations
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)

class Qwen2VLTemporalOptimized:
    """Ultra-optimized Qwen2-VL mit Batching für <3s pro Segment"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.segment_duration = 3.0  # Längere Segmente = weniger Aufrufe
        self.max_new_tokens = 80     # Reduziert für Speed
        self.target_resolution = (480, 270)  # Kleinere Auflösung für weniger tokens
        
    def _load_model(self):
        """Lazy load optimized model"""
        if self.model is None:
            try:
                logger.info("🚀 Loading optimized Qwen2-VL model...")
                
                from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
                
                # Use standard model with optimizations instead of GPTQ (compatibility)
                model_name = "Qwen/Qwen2-VL-7B-Instruct"
                
                # Load with aggressive optimizations
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    attn_implementation="eager"  # Use eager instead of flash_attention_2
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    min_pixels=128*28*28,   # Minimale visual tokens
                    max_pixels=320*28*28    # Stark reduziert für Speed
                )
                
                # Enable memory efficient attention if available
                try:
                    self.model.enable_xformers_memory_efficient_attention()
                    logger.info("✅ xformers memory efficient attention enabled")
                except:
                    logger.info("⚠️ xformers not available, using standard attention")
                
                # Compile model for faster inference
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("✅ Model compiled with torch.compile")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")
                
                logger.info("✅ Optimized Qwen2-VL loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load optimized model: {e}")
                raise
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Ultra-fast analysis with batching"""
        start_time = time.time()
        
        # Load model if needed
        self._load_model()
        
        # Extract video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Calculate fewer segments for speed
        num_segments = max(1, int(np.ceil(duration / self.segment_duration)))
        logger.info(f"🔥 Processing {num_segments} segments with ultra-optimized pipeline")
        
        # Extract frames efficiently
        frames = []
        segment_info = []
        
        for i in range(num_segments):
            segment_start = i * self.segment_duration
            segment_end = min((i + 1) * self.segment_duration, duration)
            
            # Extract middle frame of segment
            frame_idx = int((segment_start + segment_end) / 2 * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Aggressive downscaling for speed
                frame_small = cv2.resize(frame, self.target_resolution, interpolation=cv2.INTER_LINEAR)
                frames.append(frame_small)
                segment_info.append({
                    'start_time': segment_start,
                    'end_time': segment_end,
                    'frame_idx': frame_idx
                })
        
        cap.release()
        
        if not frames:
            return {
                'segments': [],
                'error': 'No frames extracted',
                'metadata': {'duration': duration, 'fps': fps}
            }
        
        # Process all frames in a single batch
        try:
            logger.info(f"🚀 Running batch inference for {len(frames)} frames...")
            batch_start = time.time()
            
            # Create batch messages
            messages_batch = []
            for i, frame in enumerate(frames):
                segment_start = segment_info[i]['start_time']
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame},
                        {"type": "text", "text": f"Describe what happens at {segment_start:.1f}s in this video in 1-2 sentences."}
                    ]
                }]
                messages_batch.append(messages)
            
            # Process batch
            from qwen_vl_utils import process_vision_info
            
            # Prepare batch inputs
            texts = []
            all_image_inputs = []
            
            for messages in messages_batch:
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                texts.append(text)
                
                image_inputs, _ = process_vision_info(messages)
                all_image_inputs.extend(image_inputs)
            
            # Tokenize batch
            inputs = self.processor(
                text=texts,
                images=all_image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate with optimized settings
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # Greedy for speed
                    temperature=0.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True
                )
            
            # Decode outputs
            generated_texts = self.tokenizer.batch_decode(
                outputs[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            batch_time = time.time() - batch_start
            logger.info(f"✅ Batch inference complete in {batch_time:.1f}s")
            
            # Create results
            segments = []
            for i, (info, text) in enumerate(zip(segment_info, generated_texts)):
                description = text.strip()
                
                # Clean up the response
                if description.startswith("The video"):
                    description = description
                elif description.startswith("At"):
                    description = description
                else:
                    description = f"The video shows {description}"
                
                segments.append({
                    'start_time': info['start_time'],
                    'end_time': info['end_time'],
                    'description': description[:200]  # Limit length
                })
            
            elapsed = time.time() - start_time
            logger.info(f"🎯 Ultra-optimized Qwen2-VL analysis complete: {len(segments)} segments in {elapsed:.1f}s")
            
            return {
                'segments': segments,
                'metadata': {
                    'duration': duration,
                    'fps': fps,
                    'total_frames': total_frames,
                    'processing_time': elapsed,
                    'batch_inference_time': batch_time,
                    'frames_processed': len(frames),
                    'target_resolution': self.target_resolution,
                    'segment_duration': self.segment_duration
                }
            }
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            return {
                'segments': [],
                'error': str(e),
                'metadata': {
                    'duration': duration,
                    'fps': fps,
                    'processing_time': time.time() - start_time
                }
            }
    
    def cleanup(self):
        """Clean up GPU resources"""
        if self.model is not None:
            del self.model
            self.model = None
            
        if hasattr(self, 'processor'):
            del self.processor
            self.processor = None
            
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None
            
        torch.cuda.empty_cache()
        logger.info("🧹 Qwen2-VL optimized model cleaned up")

# For compatibility with registry
Qwen2VLTemporalAnalyzer = Qwen2VLTemporalOptimized

def test_optimized_qwen():
    """Test the optimized Qwen2-VL analyzer"""
    
    video_path = "/home/user/tiktok_production/test_videos/test1.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Test video not found: {video_path}")
        return
    
    print("🚀 Testing ultra-optimized Qwen2-VL...")
    
    analyzer = Qwen2VLTemporalOptimized()
    result = analyzer.analyze(video_path)
    
    segments = len(result.get('segments', []))
    processing_time = result.get('metadata', {}).get('processing_time', 0)
    error = result.get('error', '')
    
    if segments > 0:
        print(f"✅ Optimized Qwen2-VL works: {segments} segments in {processing_time:.1f}s")
        print(f"📊 Performance: {processing_time/segments:.1f}s per segment")
        print(f"🎯 Target was <3s per segment")
        
        # Show sample
        if result['segments']:
            print(f"📝 Sample: {result['segments'][0]['description'][:100]}...")
    else:
        print(f"❌ Optimized Qwen2-VL failed: {error}")
    
    # Cleanup
    analyzer.cleanup()
    
    return result

if __name__ == "__main__":
    test_optimized_qwen()