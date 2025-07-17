#!/usr/bin/env python3
"""
Video-LLaVA Service - FastAPI wrapper for LLaVA-NeXT-Video
Provides REST API for video analysis with pre-loaded model
"""
import os
import sys
import time
import torch
import numpy as np
import av
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from transformers import (
    LlavaNextVideoProcessor,
    LlavaNextVideoForConditionalGeneration,
    BitsAndBytesConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instances
processor = None
model = None
model_loaded = False
load_lock = asyncio.Lock()

# Model configuration
MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"
CACHE_DIR = "/app/models"
MAX_FRAMES = 8  # Reduced for performance
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class VideoAnalysisRequest(BaseModel):
    """Request model for video analysis"""
    video_path: str = Field(..., description="Path to video file")
    prompt: Optional[str] = Field(
        default="Describe this video in detail, including all visible objects, people, actions, and scene changes.",
        description="Prompt for video analysis"
    )
    max_frames: Optional[int] = Field(default=8, ge=1, le=32, description="Maximum frames to analyze")
    temperature: Optional[float] = Field(default=0.7, ge=0.1, le=1.0, description="Generation temperature")

class VideoAnalysisResponse(BaseModel):
    """Response model for video analysis"""
    status: str
    video_path: str
    analysis: Dict[str, Any]
    processing_time: float
    model_info: Dict[str, Any]

async def load_model():
    """Load the model once at startup"""
    global processor, model, model_loaded
    
    if model_loaded:
        return
    
    async with load_lock:
        if model_loaded:  # Double check after acquiring lock
            return
        
        logger.info("="*80)
        logger.info("Loading LLaVA-NeXT-Video-7B model...")
        logger.info("="*80)
        
        start_time = time.time()
        
        try:
            # Load processor
            logger.info("Loading processor...")
            processor = LlavaNextVideoProcessor.from_pretrained(
                MODEL_ID,
                cache_dir=CACHE_DIR
            )
            
            # Configure 4-bit quantization
            logger.info("Configuring 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load model
            logger.info("Loading model weights...")
            model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                cache_dir=CACHE_DIR,
                low_cpu_mem_usage=True
            )
            
            model.eval()
            
            # Enable Flash Attention if available
            if hasattr(model.config, 'use_flash_attention_2'):
                model.config.use_flash_attention_2 = True
                logger.info("✅ Flash Attention enabled")
            
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded successfully in {load_time:.1f}s")
            logger.info(f"✅ Using device: {DEVICE}")
            logger.info(f"✅ GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
            
            model_loaded = True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

def read_video_pyav(video_path: str, num_frames: int = 8) -> np.ndarray:
    """Read video frames using PyAV"""
    try:
        container = av.open(video_path)
        
        # Get total frames
        total_frames = container.streams.video[0].frames
        if total_frames == 0:
            # Count frames manually for some formats
            total_frames = sum(1 for _ in container.decode(video=0))
            container.close()
            container = av.open(video_path)
        
        # Calculate frame indices to sample
        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        
        frames = []
        container.seek(0)
        frame_idx = 0
        sample_idx = 0
        
        for frame in container.decode(video=0):
            if sample_idx < len(indices) and frame_idx == indices[sample_idx]:
                # Convert to RGB and resize for efficiency
                frame_rgb = frame.to_ndarray(format="rgb24")
                frames.append(frame_rgb)
                sample_idx += 1
            
            frame_idx += 1
            if sample_idx >= len(indices):
                break
        
        container.close()
        
        # Pad if needed
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((336, 336, 3), dtype=np.uint8))
        
        return np.stack(frames[:num_frames])
        
    except Exception as e:
        logger.error(f"Error reading video {video_path}: {e}")
        raise ValueError(f"Failed to read video: {e}")

async def analyze_video(request: VideoAnalysisRequest) -> Dict[str, Any]:
    """Analyze a video using LLaVA-NeXT"""
    if not model_loaded:
        await load_model()
    
    start_time = time.time()
    
    try:
        # Read video frames
        logger.info(f"Reading video: {request.video_path}")
        video_frames = read_video_pyav(request.video_path, request.max_frames)
        logger.info(f"Extracted {len(video_frames)} frames")
        
        # Create conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.prompt},
                    {"type": "video"},
                ],
            },
        ]
        
        # Apply chat template
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Process inputs
        inputs = processor(
            text=prompt,
            videos=video_frames,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate response
        logger.info("Generating analysis...")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=request.temperature,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        # Decode response
        generated_text = processor.decode(output[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "Assistant:" in generated_text:
            response = generated_text.split("Assistant:")[-1].strip()
        else:
            response = generated_text[len(prompt):].strip()
        
        # Clear GPU memory
        del inputs, output
        torch.cuda.empty_cache()
        
        processing_time = time.time() - start_time
        
        # Structure the analysis
        analysis = {
            "description": response,
            "frame_count": len(video_frames),
            "processing_details": {
                "frames_analyzed": request.max_frames,
                "temperature": request.temperature,
                "processing_time": processing_time
            }
        }
        
        logger.info(f"✅ Analysis completed in {processing_time:.1f}s")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Create FastAPI app with lifespan for model loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    logger.info("Starting Video-LLaVA service...")
    await load_model()
    yield
    logger.info("Shutting down Video-LLaVA service...")

app = FastAPI(
    title="Video-LLaVA Service",
    description="REST API for LLaVA-NeXT-Video analysis",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        }
    
    return {
        "status": "healthy" if model_loaded else "loading",
        "model_loaded": model_loaded,
        "model_id": MODEL_ID,
        "device": DEVICE,
        "gpu_info": gpu_info,
        "timestamp": time.time()
    }

@app.post("/analyze", response_model=VideoAnalysisResponse)
async def analyze_endpoint(request: VideoAnalysisRequest):
    """Analyze a video"""
    logger.info(f"Received analysis request for: {request.video_path}")
    
    # Check if file exists
    if not Path(request.video_path).exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {request.video_path}")
    
    try:
        # Perform analysis
        start_time = time.time()
        analysis = await analyze_video(request)
        processing_time = time.time() - start_time
        
        return VideoAnalysisResponse(
            status="success",
            video_path=request.video_path,
            analysis=analysis,
            processing_time=processing_time,
            model_info={
                "model_id": MODEL_ID,
                "device": DEVICE,
                "quantization": "4-bit"
            }
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Video-LLaVA Analysis Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST)",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    # Run the service
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8004,
        log_level="info",
        access_log=True
    )