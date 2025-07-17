# video_captioning_detailed.py - Enhanced BLIP-2 for detailed descriptions
import torch
import cv2
import numpy as np
import sys
import os
import json
from typing import List, Dict, Tuple
import traceback
from pathlib import Path
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Configuration
DEFAULT_VIDEO_PATH = "/videos/input.mp4"
OUTPUT_DIR = "/app/output/"
FRAME_RATE = 0.5  # Every 0.5 seconds for detailed coverage

# Multiple prompts for comprehensive descriptions
DETAILED_PROMPTS = [
    "Describe this image in extreme detail:",
    "Question: What objects are visible and where are they located? Answer:",
    "Question: What actions or movements are happening? Answer:",
    "Question: Describe the colors, lighting, and atmosphere. Answer:",
    "Question: What is the setting and background? Answer:",
]

def load_video(video_path, frame_rate=0.5):
    """Load video and extract frames at specified rate"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None

    frames = []
    timestamps = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        fps = 30
    
    # Calculate frame interval
    frame_interval = int(fps / (1/frame_rate))
    if frame_interval < 1:
        frame_interval = 1

    print(f"Processing video: {video_path}")
    print(f"  FPS: {fps:.2f}, Total frames: {total_frames}")
    print(f"  Extracting every {frame_interval} frames (every {frame_interval/fps:.2f}s)")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            timestamps.append(frame_count / fps)
        frame_count += 1

    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames, timestamps

def initialize_model():
    """Initialize BLIP-2 model"""
    print("Loading BLIP-2 model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = "Salesforce/blip2-opt-2.7b"
    
    try:
        # Load with 8-bit quantization if possible
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        print("Model loaded with 8-bit quantization")
    except:
        # Fallback to fp16
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model loaded with fp16")
    
    model.eval()
    return model, processor, device

def generate_detailed_description(model, processor, frame, timestamp, device):
    """Generate comprehensive description using multiple prompts"""
    pil_image = Image.fromarray(frame)
    descriptions = {}
    
    for prompt in DETAILED_PROMPTS:
        try:
            inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=3,
                    temperature=0.7,
                    do_sample=False
                )
            
            description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clean the description
            description = description.replace(prompt, "").strip()
            
            # Categorize by prompt type
            if "objects" in prompt.lower():
                descriptions["objects"] = description
            elif "actions" in prompt.lower() or "movements" in prompt.lower():
                descriptions["actions"] = description
            elif "colors" in prompt.lower() or "lighting" in prompt.lower():
                descriptions["atmosphere"] = description
            elif "setting" in prompt.lower() or "background" in prompt.lower():
                descriptions["setting"] = description
            else:
                descriptions["general"] = description
                
        except Exception as e:
            print(f"Error with prompt '{prompt[:30]}...': {e}")
            
    # Combine all descriptions
    combined = []
    if "general" in descriptions:
        combined.append(descriptions["general"])
    if "objects" in descriptions:
        combined.append(f"Objects: {descriptions['objects']}")
    if "actions" in descriptions:
        combined.append(f"Actions: {descriptions['actions']}")
    if "setting" in descriptions:
        combined.append(f"Setting: {descriptions['setting']}")
    if "atmosphere" in descriptions:
        combined.append(f"Atmosphere: {descriptions['atmosphere']}")
    
    return {
        "timestamp": timestamp,
        "combined_description": " | ".join(combined) if combined else "No description available",
        "detailed_descriptions": descriptions
    }

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        sys.exit(1)
    
    # Initialize model
    model, processor, device = initialize_model()
    
    # Load video
    frames, timestamps = load_video(video_path, frame_rate=FRAME_RATE)
    if frames is None:
        print("Error loading video")
        sys.exit(1)
    
    # Generate descriptions
    print(f"\nGenerating detailed descriptions for {len(frames)} frames...")
    descriptions_data = []
    
    for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        print(f"  Processing frame {i+1}/{len(frames)} at {timestamp:.2f}s...", end="", flush=True)
        
        result = generate_detailed_description(model, processor, frame, timestamp, device)
        descriptions_data.append(result)
        
        print(" ✓")
        
        # Show sample output for first frame
        if i == 0:
            print(f"    Sample: {result['combined_description'][:100]}...")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    video_basename = os.path.basename(video_path)
    json_filename = os.path.join(OUTPUT_DIR, f"{video_basename}_detailed_analysis.json")
    
    output_data = {
        "video_path": video_path,
        "video_name": video_basename,
        "frame_rate": FRAME_RATE,
        "total_frames_analyzed": len(frames),
        "model": "BLIP-2 OPT-2.7B (Multi-Prompt Detailed)",
        "prompts_used": DETAILED_PROMPTS,
        "segments": descriptions_data
    }
    
    try:
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Analysis successful! Results saved to: {json_filename}")
    except Exception as e:
        print(f"\nError saving results: {e}")
    
    # Also save as readable text
    txt_filename = os.path.join(OUTPUT_DIR, f"{video_basename}_detailed_timeline.txt")
    try:
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(f"DETAILED VIDEO ANALYSIS: {video_path}\n")
            f.write(f"Model: BLIP-2 with Multiple Prompts\n")
            f.write(f"Analyzed {len(frames)} frames at {FRAME_RATE}s intervals\n")
            f.write("="*80 + "\n\n")
            
            for segment in descriptions_data:
                f.write(f"[{segment['timestamp']:.2f}s]\n")
                f.write(f"{segment['combined_description']}\n")
                f.write("-"*40 + "\n")
                
        print(f"✅ Timeline saved to: {txt_filename}")
    except Exception as e:
        print(f"Error saving timeline: {e}")

if __name__ == "__main__":
    main()