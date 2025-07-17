#!/usr/bin/env python3
"""
Ray-based Model Server für effizientes Model Sharing
Löst das Problem von Model Pre-Loading über Prozessgrenzen
"""

import ray
from ray import serve
import torch
import whisper
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import logging
import asyncio
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Initialize Ray
ray.init(ignore_reinit_error=True)

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1}
)
class WhisperModelServer:
    """Whisper Model Server using Ray Serve"""
    
    def __init__(self):
        logger.info("Loading Whisper model in Ray actor...")
        self.model = whisper.load_model("base", device="cuda")
        logger.info("✅ Whisper model loaded in Ray actor")
    
    async def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio file"""
        try:
            result = self.model.transcribe(audio_path)
            return {
                "text": result["text"],
                "segments": result["segments"],
                "language": result["language"]
            }
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {"error": str(e)}

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1}
)
class Qwen2VLModelServer:
    """Qwen2-VL Model Server using Ray Serve"""
    
    def __init__(self):
        logger.info("Loading Qwen2-VL model in Ray actor...")
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            min_pixels=256*28*28,
            max_pixels=1280*28*28
        )
        
        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
        self.model.eval()
        logger.info("✅ Qwen2-VL model loaded in Ray actor")
    
    async def analyze(self, messages: list, **kwargs) -> Dict[str, Any]:
        """Analyze video frames"""
        try:
            from qwen_vl_utils import process_vision_info
            
            # Process input
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
            ).to("cuda")
            
            # Generate
            with torch.inference_mode():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=kwargs.get("max_new_tokens", 200),
                        do_sample=False,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.05,
                        use_cache=True,
                        pad_token_id=self.processor.tokenizer.pad_token_id
                    )
            
            # Decode
            generated_text = self.processor.decode(
                generated_ids[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ).strip()
            
            return {"description": generated_text}
            
        except Exception as e:
            logger.error(f"Qwen2-VL analysis error: {e}")
            return {"error": str(e)}

class RayModelClient:
    """Client to interact with Ray model servers"""
    
    def __init__(self):
        self.whisper_handle = None
        self.qwen_handle = None
        
    async def initialize(self):
        """Deploy and get handles to model servers"""
        # Deploy Whisper
        serve.start(detached=True)
        WhisperModelServer.deploy()
        self.whisper_handle = WhisperModelServer.get_handle()
        
        # Deploy Qwen2-VL
        Qwen2VLModelServer.deploy()
        self.qwen_handle = Qwen2VLModelServer.get_handle()
        
        logger.info("✅ Ray model servers deployed")
    
    async def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio using Ray server"""
        if not self.whisper_handle:
            await self.initialize()
        
        result = await self.whisper_handle.transcribe.remote(audio_path)
        return result
    
    async def analyze_frames(self, messages: list, **kwargs) -> Dict[str, Any]:
        """Analyze frames using Ray server"""
        if not self.qwen_handle:
            await self.initialize()
        
        result = await self.qwen_handle.analyze.remote(messages, **kwargs)
        return result

# Global client instance
ray_model_client = RayModelClient()

if __name__ == "__main__":
    # Test the servers
    async def test():
        client = RayModelClient()
        await client.initialize()
        
        # Test Whisper
        result = await client.transcribe_audio("/path/to/audio.wav")
        print(f"Transcription: {result}")
    
    asyncio.run(test())