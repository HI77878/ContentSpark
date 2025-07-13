#!/usr/bin/env python3
"""
Ray-based Analyzers with Model Sharing
All analyzers use Ray actors for efficient GPU/model sharing
"""

import ray
import torch
import whisper
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any, Optional
import logging
import librosa
from pathlib import Path

logger = logging.getLogger(__name__)

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Speech Transcription with Ray
@ray.remote(num_gpus=0.2)
class SpeechTranscriptionActor:
    """Speech Transcription with Whisper"""
    
    def __init__(self):
        logger.info("Loading Whisper model in Ray actor...")
        self.model = whisper.load_model("base", device="cuda")
        logger.info("✅ Whisper loaded in Ray actor")
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio from video"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        audio_normalized = audio / (np.max(np.abs(audio)) + 1e-10)
        audio_fp32 = audio_normalized.astype(np.float32)
        
        # Transcribe
        result = self.model.transcribe(
            audio_fp32,
            language=None,
            task='transcribe',
            verbose=False,
            temperature=0.0,
            word_timestamps=True,
            beam_size=5,
            best_of=5,
            fp16=True
        )
        
        # Format segments
        segments = []
        for seg in result.get('segments', []):
            segments.append({
                'start_time': float(seg['start']),
                'end_time': float(seg['end']),
                'text': seg['text'].strip(),
                'language': result.get('language', 'unknown'),
                'confidence': seg.get('confidence', 0.95)
            })
        
        return {
            'segments': segments,
            'language': result.get('language'),
            'text': result.get('text', ''),
            'total_segments': len(segments)
        }

# Comment CTA Detection with Ray
@ray.remote(num_gpus=0)
class CommentCTAActor:
    """CTA Detection optimized for Marc Gebauer patterns"""
    
    def __init__(self):
        import re
        self.re = re
        
        # Marc Gebauer specific patterns
        self.marc_patterns = [
            'noch mal bestellen',
            'was nun',
            'verstehe die frage nicht',
            'einfach nochmal bestellen',
            'nochmal bestellen'
        ]
        
        # General CTA patterns
        self.cta_patterns = {
            'comment_below': [
                r'kommentier.*unten',
                r'schreib.*kommentar',
                r'lass.*kommentar',
                r'comment.*below',
                r'was.*meinst.*du',
                r'was.*denkst.*du',
                r'eure.*meinung',
                r'schreibt.*mir'
            ],
            'question': [
                r'was.*würd.*ihr',
                r'was.*würd.*du', 
                r'was.*nun\?',
                r'.*\?.*kommentar',
                r'noch.*mal.*bestellen',
                r'versteh.*frage.*nicht'
            ],
            'like_share': [
                r'lik.*teil',
                r'vergiss.*nicht.*lik',
                r'like.*share',
                r'daumen.*hoch'
            ],
            'follow': [
                r'folg.*mir',
                r'folg.*für.*mehr',
                r'abonnier'
            ]
        }
        
        logger.info("✅ Comment CTA Actor initialized")
    
    def detect_ctas(self, speech_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect CTAs in speech segments"""
        cta_segments = []
        
        for seg in speech_segments:
            text = seg.get('text', '')
            if not text:
                continue
            
            text_lower = text.lower()
            ctas_found = []
            
            # Check Marc Gebauer patterns FIRST
            for pattern in self.marc_patterns:
                if pattern in text_lower:
                    ctas_found.append({
                        'type': 'marc_gebauer_cta',
                        'pattern': pattern,
                        'confidence': 0.99
                    })
            
            # Check general patterns
            for cta_type, patterns in self.cta_patterns.items():
                for pattern in patterns:
                    if self.re.search(pattern, text_lower):
                        ctas_found.append({
                            'type': cta_type,
                            'pattern': pattern,
                            'confidence': 0.9
                        })
            
            if ctas_found:
                cta_segments.append({
                    'start_time': seg['start_time'],
                    'end_time': seg['end_time'],
                    'text': text,
                    'cta_detected': True,
                    'cta_types': list(set(cta['type'] for cta in ctas_found)),
                    'confidence': max(cta['confidence'] for cta in ctas_found),
                    'is_marc_gebauer': any(cta['type'] == 'marc_gebauer_cta' for cta in ctas_found)
                })
        
        return {
            'segments': cta_segments,
            'total_ctas': len(cta_segments),
            'cta_types_found': list(set(
                cta_type 
                for seg in cta_segments 
                for cta_type in seg.get('cta_types', [])
            )),
            'marc_gebauer_cta_detected': any(seg.get('is_marc_gebauer', False) for seg in cta_segments)
        }

# Qwen2-VL with Ray
@ray.remote(num_gpus=0.7)
class Qwen2VLActor:
    """Qwen2-VL for video understanding"""
    
    def __init__(self):
        logger.info("Loading Qwen2-VL in Ray actor...")
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            min_pixels=256*28*28,
            max_pixels=1280*28*28
        )
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
        self.model.eval()
        self.process_vision_info = process_vision_info
        logger.info("✅ Qwen2-VL loaded in Ray actor")
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Analyze full video with temporal understanding"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        segments = []
        segment_duration = 3.0
        current_time = 0
        
        while current_time < duration:
            end_time = min(current_time + segment_duration, duration)
            
            # Extract 3 frames per segment
            frames = []
            for i in range(3):
                frame_time = current_time + (i * (end_time - current_time) / 3)
                frame_idx = int(frame_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    frames.append(pil_frame)
            
            if frames:
                # Analyze segment
                prompt = f"""Analyze this video segment from {current_time:.1f}s to {end_time:.1f}s in CHRONOLOGICAL ORDER.

Frame 1 (start): What happens at the beginning
Frame 2 (middle): What happens in the middle  
Frame 3 (end): What happens at the end

Describe:
1. People, actions, movements in order
2. Environment and setting
3. Any scene changes or cuts

If the scene/location changes between frames, clearly state "The scene changes to..."

Focus on visual narrative. Minimum 100 words."""

                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame} for frame in frames
                    ] + [{"type": "text", "text": prompt}]
                }]
                
                # Process
                text = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                image_inputs, _ = self.process_vision_info(messages)
                
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to("cuda")
                
                # Generate
                with torch.inference_mode():
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=150,
                            do_sample=False,
                            temperature=0.7,
                            pad_token_id=self.processor.tokenizer.pad_token_id
                        )
                
                # Decode
                generated_text = self.processor.decode(
                    generated_ids[0][len(inputs.input_ids[0]):],
                    skip_special_tokens=True
                ).strip()
                
                segments.append({
                    'start_time': current_time,
                    'end_time': end_time,
                    'description': generated_text,
                    'frames_analyzed': len(frames)
                })
            
            current_time = end_time
        
        cap.release()
        
        return {
            'segments': segments,
            'total_segments': len(segments),
            'duration': duration
        }

# Object Detection with Ray
@ray.remote(num_gpus=0.3)
class ObjectDetectionActor:
    """YOLOv8 Object Detection"""
    
    def __init__(self):
        logger.info("Loading YOLOv8 in Ray actor...")
        from ultralytics import YOLO
        self.model = YOLO('yolov8x.pt')
        self.model.to('cuda')
        logger.info("✅ YOLOv8 loaded in Ray actor")
    
    def detect_objects(self, video_path: str) -> Dict[str, Any]:
        """Detect objects in video"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample every 30 frames
        frame_interval = 30
        segments = []
        
        for frame_idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Run detection
            results = self.model(frame, verbose=False)
            
            # Process results
            timestamp = frame_idx / fps
            objects = []
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        objects.append({
                            'class': self.model.names[int(box.cls)],
                            'confidence': float(box.conf),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'timestamp': timestamp
                        })
            
            if objects:
                segments.append({
                    'timestamp': timestamp,
                    'frame_number': frame_idx,
                    'objects': objects,
                    'object_count': len(objects)
                })
        
        cap.release()
        
        return {
            'segments': segments,
            'total_objects_detected': sum(seg['object_count'] for seg in segments),
            'unique_classes': list(set(
                obj['class'] 
                for seg in segments 
                for obj in seg['objects']
            ))
        }

# Ray-based analyzer wrapper for API integration
class RayAnalyzerSystem:
    """Manages all Ray actors for video analysis"""
    
    def __init__(self):
        self.actors = {}
        self._initialize_actors()
    
    def _initialize_actors(self):
        """Initialize all Ray actors"""
        logger.info("Initializing Ray actor system...")
        
        # Create actors
        self.actors['speech_transcription'] = SpeechTranscriptionActor.remote()
        self.actors['comment_cta'] = CommentCTAActor.remote()
        self.actors['qwen2_vl'] = Qwen2VLActor.remote()
        self.actors['object_detection'] = ObjectDetectionActor.remote()
        
        logger.info("✅ All Ray actors initialized")
    
    async def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Analyze video with all Ray actors"""
        results = {}
        
        # Speech transcription
        try:
            speech_result = await self.actors['speech_transcription'].transcribe.remote(video_path)
            results['speech_transcription'] = ray.get(speech_result)
            
            # CTA detection on speech results
            if results['speech_transcription']['segments']:
                cta_result = await self.actors['comment_cta'].detect_ctas.remote(
                    results['speech_transcription']['segments']
                )
                results['comment_cta_detection'] = ray.get(cta_result)
        except Exception as e:
            logger.error(f"Speech/CTA analysis failed: {e}")
            results['speech_transcription'] = {"error": str(e)}
            results['comment_cta_detection'] = {"error": str(e)}
        
        # Qwen2-VL analysis
        try:
            qwen_result = await self.actors['qwen2_vl'].analyze_video.remote(video_path)
            results['qwen2_vl_temporal'] = ray.get(qwen_result)
        except Exception as e:
            logger.error(f"Qwen2-VL analysis failed: {e}")
            results['qwen2_vl_temporal'] = {"error": str(e)}
        
        # Object detection
        try:
            object_result = await self.actors['object_detection'].detect_objects.remote(video_path)
            results['object_detection'] = ray.get(object_result)
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            results['object_detection'] = {"error": str(e)}
        
        return results