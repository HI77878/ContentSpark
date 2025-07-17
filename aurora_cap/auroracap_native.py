#!/usr/bin/env python3
"""
Native AuroraCap inference without Docker
"""
import os
import sys
import json
import torch
import cv2
import time
from pathlib import Path
from PIL import Image
from typing import List, Dict

# Prevent deepspeed issues
os.environ['NO_DEEPSPEED'] = '1'

def load_auroracap_model():
    """Load AuroraCap model with compatible settings"""
    print("Loading AuroraCap model...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
    from transformers import BitsAndBytesConfig
    
    model_id = "BAAI/Emu2-Chat"  # AuroraCap is based on Emu2
    
    # Try 8-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16
    )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        image_processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        print("AuroraCap model loaded successfully!")
        return model, tokenizer, image_processor
        
    except Exception as e:
        print(f"Error loading AuroraCap: {e}")
        print("Falling back to BLIP-2...")
        return load_blip2_fallback()

def load_blip2_fallback():
    """Fallback to BLIP-2 if AuroraCap fails"""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    
    model_name = "Salesforce/blip2-opt-2.7b"
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto"
    )
    
    return model, processor, None

def extract_frames(video_path: str, interval: float = 1.0) -> List[Dict]:
    """Extract frames from video at given interval"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video: {duration:.1f}s, {total_frames} frames at {fps:.1f} fps")
    
    frames = []
    frame_interval = int(fps * interval)
    
    for i in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        frames.append({
            'image': pil_image,
            'timestamp': i / fps,
            'frame_number': i
        })
    
    cap.release()
    return frames

def generate_auroracap_description(model, tokenizer, image_processor, frame_data: Dict) -> str:
    """Generate detailed description using AuroraCap"""
    image = frame_data['image']
    timestamp = frame_data['timestamp']
    
    # Detailed prompt for maximum information
    prompt = f"""[Time: {timestamp:.1f}s]
Describe this video frame in extreme detail. Include:
1. All visible objects and their exact positions
2. People: appearance, clothing, expressions, actions
3. Environment: location, lighting, colors, atmosphere
4. Any text, logos, or UI elements
5. Camera angle and movement
6. What is happening in this exact moment
Provide enough detail to recreate this exact frame."""
    
    try:
        # Process with AuroraCap
        inputs = image_processor(images=image, return_tensors="pt").to("cuda")
        text_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **text_inputs,
                max_new_tokens=300,
                num_beams=3,
                temperature=0.7,
                do_sample=False
            )
        
        description = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return description.replace(prompt, "").strip()
        
    except:
        # Fallback to BLIP-2 style
        if hasattr(model, 'generate'):
            # BLIP-2 processor
            processor = tokenizer  # In fallback, tokenizer is actually the processor
            inputs = processor(image, text="Describe this image in detail:", return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=150, num_beams=3)
            
            return processor.decode(outputs[0], skip_special_tokens=True)
    
    return "Description generation failed"

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else '/videos/input.mp4'
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    
    # Load model
    model, tokenizer, image_processor = load_auroracap_model()
    
    # Extract frames
    print(f"\nAnalyzing video: {video_path}")
    frames = extract_frames(video_path, interval=0.5)  # Every 0.5 seconds
    
    # Generate descriptions
    print(f"\nGenerating detailed descriptions for {len(frames)} frames...")
    results = []
    
    for i, frame_data in enumerate(frames):
        print(f"Processing frame {i+1}/{len(frames)} at {frame_data['timestamp']:.1f}s...", end="", flush=True)
        
        description = generate_auroracap_description(model, tokenizer, image_processor, frame_data)
        
        results.append({
            'timestamp': frame_data['timestamp'],
            'frame_number': frame_data['frame_number'],
            'description': description
        })
        
        print(" ✓")
        
        # Show sample
        if i == 0:
            print(f"  Sample: {description[:100]}...")
    
    # Save results
    output_dir = '/home/user/tiktok_production/aurora_cap/output'
    os.makedirs(output_dir, exist_ok=True)
    
    video_name = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{video_name}_auroracap_native.json")
    
    with open(output_path, 'w') as f:
        json.dump({
            'video_path': video_path,
            'model': 'AuroraCap (Native)',
            'frame_interval': 0.5,
            'total_frames': len(frames),
            'descriptions': results
        }, f, indent=2)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_path}")
    
    # Also save as timeline
    timeline_path = os.path.join(output_dir, f"{video_name}_auroracap_timeline.txt")
    with open(timeline_path, 'w') as f:
        f.write(f"AURORACAP DETAILED VIDEO ANALYSIS\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Analyzed {len(frames)} frames\n")
        f.write("="*80 + "\n\n")
        
        for result in results:
            f.write(f"[{result['timestamp']:.2f}s] Frame {result['frame_number']}\n")
            f.write(f"{result['description']}\n")
            f.write("-"*40 + "\n")
    
    print(f"✅ Timeline saved to: {timeline_path}")

if __name__ == "__main__":
    main()