#!/usr/bin/env python3
"""
Fixed AuroraCap inference script with proper model loading and rope_scaling handling
"""
import os
import sys
import json
import torch
import cv2
import time
import traceback
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any
import numpy as np

# Ensure no deepspeed dependency
os.environ['NO_DEEPSPEED'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def fix_rope_scaling_config(config_path: str):
    """Fix rope_scaling configuration for compatibility"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Fix rope_scaling if present
        if 'rope_scaling' in config and isinstance(config['rope_scaling'], dict):
            # Keep only 'type' and 'factor' fields
            rope_config = config['rope_scaling']
            fixed_rope = {
                'type': rope_config.get('rope_type', rope_config.get('type', 'dynamic')),
                'factor': rope_config.get('factor', 8.0)
            }
            config['rope_scaling'] = fixed_rope
            
            # Write back
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Fixed rope_scaling configuration in {config_path}")

def load_auroracap_model(model_path: str = "wchai/AuroraCap-7B-VID-xtuner"):
    """Load AuroraCap model with proper configuration"""
    print(f"Loading AuroraCap model from {model_path}...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
    from transformers import BitsAndBytesConfig
    
    # Download and fix config if needed
    from huggingface_hub import snapshot_download
    cache_dir = "/home/user/tiktok_production/aurora_cap/.cache"
    local_dir = os.path.join(cache_dir, "models", model_path.replace("/", "_"))
    
    if not os.path.exists(local_dir):
        print("Downloading model...")
        snapshot_download(repo_id=model_path, local_dir=local_dir, cache_dir=cache_dir)
    
    # Fix rope_scaling in config
    config_path = os.path.join(local_dir, "config.json")
    fix_rope_scaling_config(config_path)
    
    # Load with 8-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=True
    )
    
    try:
        # Load tokenizer and processor
        tokenizer = AutoTokenizer.from_pretrained(
            local_dir,
            trust_remote_code=True,
            use_fast=False
        )
        
        # For vision models, we might need an image processor
        try:
            image_processor = AutoImageProcessor.from_pretrained(
                local_dir,
                trust_remote_code=True
            )
        except:
            print("No image processor found, using default image preprocessing")
            image_processor = None
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        model.eval()
        print("✅ AuroraCap model loaded successfully!")
        return model, tokenizer, image_processor
        
    except Exception as e:
        print(f"Error loading AuroraCap: {e}")
        traceback.print_exc()
        return None, None, None

def extract_video_frames(video_path: str, interval: float = 1.0, max_frames: int = 120) -> List[Dict]:
    """Extract frames from video at specified interval"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video info: {duration:.1f}s, {total_frames} frames at {fps:.1f} fps")
    
    frames = []
    frame_interval = max(1, int(fps * interval))
    
    for i in range(0, min(total_frames, max_frames * frame_interval), frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        frames.append({
            'image': pil_image,
            'frame_number': i,
            'timestamp': i / fps if fps > 0 else 0
        })
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames

def generate_frame_caption(model, tokenizer, image_processor, frame_data: Dict, 
                          token_kept_ratio: float = 0.3) -> str:
    """Generate detailed caption for a single frame"""
    image = frame_data['image']
    timestamp = frame_data['timestamp']
    
    # Prepare prompt
    prompt = f"""<image>
Describe this video frame at {timestamp:.1f} seconds in extreme detail. Include:
1. All visible objects, people, and their positions
2. Actions or movements occurring
3. Facial expressions and clothing details
4. Background elements and setting
5. Colors, lighting, and atmosphere
6. Any text or UI elements visible
7. Camera angle and composition

Provide enough detail to recreate this exact frame."""
    
    try:
        # Process image
        if image_processor:
            pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(model.device, dtype=torch.float16)
        else:
            # Manual preprocessing if no processor
            image = image.resize((336, 336))
            image_array = np.array(image).astype(np.float32) / 255.0
            image_array = (image_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            pixel_values = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
            pixel_values = pixel_values.to(model.device, dtype=torch.float16)
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        # Generate caption
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=300,
                num_beams=3,
                temperature=0.7,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from output
        if prompt in caption:
            caption = caption.replace(prompt, "").strip()
        
        return caption
        
    except Exception as e:
        print(f"Error generating caption: {e}")
        traceback.print_exc()
        return f"Error processing frame at {timestamp:.1f}s"

def analyze_video(video_path: str, output_dir: str = None) -> Dict[str, Any]:
    """Main function to analyze video with AuroraCap"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Load model
    model, tokenizer, image_processor = load_auroracap_model()
    
    if model is None:
        print("Failed to load AuroraCap model, falling back to BLIP-2")
        return analyze_with_blip2_fallback(video_path, output_dir)
    
    # Extract frames
    print(f"\nAnalyzing video: {video_path}")
    frames = extract_video_frames(video_path, interval=0.5, max_frames=120)  # Every 0.5s
    
    # Generate captions
    print("\nGenerating detailed captions...")
    results = []
    
    for i, frame_data in enumerate(frames):
        print(f"Processing frame {i+1}/{len(frames)} at {frame_data['timestamp']:.1f}s...", end="", flush=True)
        
        caption = generate_frame_caption(model, tokenizer, image_processor, frame_data)
        
        results.append({
            'timestamp': frame_data['timestamp'],
            'frame_number': frame_data['frame_number'],
            'description': caption
        })
        
        print(" ✓")
        
        # Show sample
        if i == 0:
            print(f"  Sample: {caption[:100]}...")
    
    # Prepare output
    video_name = Path(video_path).stem
    analysis_data = {
        'video_path': video_path,
        'video_name': video_name,
        'model': 'AuroraCap-7B-VID',
        'frame_interval': 0.5,
        'total_frames_analyzed': len(frames),
        'segments': results
    }
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON
        json_path = os.path.join(output_dir, f"{video_name}_auroracap_analysis.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Analysis saved to: {json_path}")
        
        # Save readable timeline
        timeline_path = os.path.join(output_dir, f"{video_name}_auroracap_timeline.txt")
        with open(timeline_path, 'w', encoding='utf-8') as f:
            f.write(f"AURORACAP DETAILED VIDEO ANALYSIS\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Model: AuroraCap-7B-VID\n")
            f.write(f"Analyzed {len(frames)} frames at 0.5s intervals\n")
            f.write("="*80 + "\n\n")
            
            for segment in results:
                f.write(f"[{segment['timestamp']:.2f}s] Frame {segment['frame_number']}\n")
                f.write(f"{segment['description']}\n")
                f.write("-"*40 + "\n")
        
        print(f"✅ Timeline saved to: {timeline_path}")
    
    return analysis_data

def analyze_with_blip2_fallback(video_path: str, output_dir: str = None) -> Dict[str, Any]:
    """Fallback to BLIP-2 if AuroraCap fails"""
    print("\nFalling back to BLIP-2 analysis...")
    
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    
    # Load BLIP-2
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        load_in_8bit=True,
        device_map="auto"
    )
    model.eval()
    
    # Extract frames
    frames = extract_video_frames(video_path, interval=1.0, max_frames=60)
    
    # Process frames
    results = []
    for i, frame_data in enumerate(frames):
        print(f"Processing frame {i+1}/{len(frames)}...", end="", flush=True)
        
        image = frame_data['image']
        
        # Generate caption
        inputs = processor(image, text="a photo of", return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        results.append({
            'timestamp': frame_data['timestamp'],
            'frame_number': frame_data['frame_number'],
            'description': caption
        })
        print(" ✓")
    
    # Return similar structure
    return {
        'video_path': video_path,
        'model': 'BLIP-2 (Fallback)',
        'segments': results
    }

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python auroracap_inference_fixed.py <video_path> [output_dir]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/home/user/tiktok_production/aurora_cap/output"
    
    try:
        analysis = analyze_video(video_path, output_dir)
        print(f"\n✅ Analysis complete! Found {len(analysis['segments'])} segments")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()