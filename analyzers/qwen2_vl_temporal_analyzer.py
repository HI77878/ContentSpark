#!/usr/bin/env python3
"""
ULTRA-OPTIMIZED Qwen2-VL - Batch Processing für <15s total
"""
import torch
import time
import logging
import numpy as np
import cv2
from typing import Dict, List, Any
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Base class placeholder
class BaseVideoAnalyzer:
    """Minimal base class for compatibility"""
    def __init__(self):
        pass

logger = logging.getLogger(__name__)

# GLOBAL MODEL LOADING - NUR EINMAL!
MODEL_LOADED = False
model = None
tokenizer = None
processor = None

def load_model_once():
    global MODEL_LOADED, model, tokenizer, processor
    if MODEL_LOADED:
        return
    
    logger.info("🚀 Loading Qwen2-VL OPTIMIZED - ONLY ONCE!")
    
    # Standard model (nicht GPTQ wegen Problemen)
    MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
    
    # FORCE GPU und optimierungen
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda:0",  # FORCE auf GPU!
        attn_implementation="eager"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=256*28*28,   # Weniger tokens
        max_pixels=512*28*28    # Stark reduziert für Speed!
    )
    
    # GPU Optimierungen
    model.eval()
    torch.cuda.empty_cache()
    
    MODEL_LOADED = True
    logger.info("✅ Qwen2-VL loaded on GPU - ready for BATCH processing!")

class Qwen2VLTemporalAnalyzer(BaseVideoAnalyzer):
    """BATCH-OPTIMIZED für <3s pro Segment"""
    
    def __init__(self):
        super().__init__()
        load_model_once()  # Load model if not loaded
        self.segment_duration = 2.0
        self.max_new_tokens = 200  # KEEP quality!
        
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """BATCH process ALL segments in ONE GPU call"""
        start_time = time.time()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        num_segments = int(np.ceil(duration / self.segment_duration))
        logger.info(f"🔥 BATCH processing {num_segments} segments on GPU")
        
        # Collect ALL frames at once
        frames_batch = []
        segment_infos = []
        
        for i in range(num_segments):
            segment_start = i * self.segment_duration
            segment_end = min((i + 1) * self.segment_duration, duration)
            frame_idx = int(segment_start * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize for speed but keep quality
                frame_resized = cv2.resize(frame, (512, 384))
                frames_batch.append(frame_resized)
                segment_infos.append({
                    'start_time': segment_start,
                    'end_time': segment_end,
                    'index': i
                })
        
        cap.release()
        
        if len(frames_batch) == 0:
            return {'segments': [], 'error': 'No frames extracted'}
        
        # BATCH INFERENCE - All at once!
        logger.info(f"🚀 Running BATCH inference for {len(frames_batch)} frames...")
        batch_start = time.time()
        
        # Prepare messages for batch
        messages_batch = []
        for i, frame in enumerate(frames_batch):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": frame},
                    {"type": "text", "text": f"Describe in detail what is happening at {segment_infos[i]['start_time']:.1f} seconds in this video. Include all visual elements, colors, positions, movements, text, and any other details visible."}
                ]
            }]
            messages_batch.append(messages)
        
        # Process all messages
        texts = []
        all_images = []
        
        for messages in messages_batch:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            # Extract image directly from messages
            for msg in messages:
                for content in msg.get('content', []):
                    if content.get('type') == 'image':
                        all_images.append(content['image'])
        
        # Tokenize batch
        inputs = processor(
            text=texts,
            images=all_images,
            padding=True,
            return_tensors="pt"
        ).to("cuda")
        
        # Generate ALL outputs in one go!
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Mixed precision for speed
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,  # Greedy for speed
                    pad_token_id=tokenizer.pad_token_id
                )
        
        # Decode all outputs
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        batch_time = time.time() - batch_start
        logger.info(f"✅ BATCH inference complete in {batch_time:.1f}s!")
        
        # Create results
        segments = []
        for info, text in zip(segment_infos, generated_texts):
            # Extract answer part
            if "assistant" in text:
                answer = text.split("assistant")[-1].strip()
            else:
                answer = text.split("Describe in detail")[-1].strip()
            
            segments.append({
                'start_time': info['start_time'],
                'end_time': info['end_time'],
                'description': answer[:1000]  # Keep long descriptions!
            })
        
        elapsed = time.time() - start_time
        logger.info(f"🎯 Qwen2-VL OPTIMIZED complete: {len(segments)} segments in {elapsed:.1f}s")
        
        return {
            'segments': segments,
            'metadata': {
                'duration': duration,
                'fps': fps,
                'total_frames': total_frames,
                'processing_time': elapsed,
                'batch_time': batch_time,
                'per_segment_time': elapsed / num_segments if num_segments > 0 else 0
            }
        }