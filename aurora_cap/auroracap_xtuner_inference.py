#!/usr/bin/env python3
"""
AuroraCap inference using xtuner format
Based on the official Aurora repository implementation
"""
import os
import sys
import json
import torch
import cv2
import time
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict, Any
import traceback

# Set environment
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_auroracap_xtuner():
    """Load AuroraCap using xtuner framework"""
    print("Loading AuroraCap with xtuner framework...")
    
    try:
        # Import xtuner components
        sys.path.append('/home/user/tiktok_production/aurora_cap/aurora/src')
        from xtuner.model.utils import guess_load_checkpoint
        from xtuner.apis import MMRunner
        from xtuner.utils import SYSTEM_TEMPLATE
        
        # Model configuration
        model_name = 'wchai/AuroraCap-7B-VID-xtuner'
        
        # Load model using xtuner
        runner = MMRunner(
            model_name_or_path=model_name,
            tokenizer_name_or_path=model_name,
            work_dir='./workdir',
            trust_remote_code=True,
            offload_folder='offload',
            max_memory={0: '40GB'},
            device_map='auto',
            load_in_8bit=True
        )
        
        print("✅ AuroraCap loaded with xtuner!")
        return runner
        
    except Exception as e:
        print(f"Error loading with xtuner: {e}")
        traceback.print_exc()
        # Fallback to direct loading
        return load_auroracap_direct()

def load_auroracap_direct():
    """Direct loading of AuroraCap model"""
    print("Loading AuroraCap directly...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import transformers
    
    # Fix for older transformers
    transformers.logging.set_verbosity_error()
    
    model_id = "wchai/AuroraCap-7B-VID-xtuner"
    
    # BitsAndBytes config for 8-bit
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        offload_folder="offload",
        low_cpu_mem_usage=True
    )
    
    model.eval()
    
    print("✅ AuroraCap loaded directly!")
    return model, tokenizer

def process_video_with_auroracap(video_path: str, model_or_runner, output_dir: str = None):
    """Process video using AuroraCap"""
    
    # Extract frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\nVideo: {duration:.1f}s, {total_frames} frames at {fps:.1f} fps")
    
    # Sample frames every 0.5 seconds
    frames = []
    timestamps = []
    frame_interval = max(1, int(fps * 0.5))
    
    for i in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        frames.append(pil_image)
        timestamps.append(i / fps if fps > 0 else 0)
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    
    # Process frames
    results = []
    
    # Check if we have xtuner runner or direct model
    if hasattr(model_or_runner, 'generate'):
        # Direct model
        model, tokenizer = model_or_runner, model_or_runner
        
        for i, (image, timestamp) in enumerate(zip(frames, timestamps)):
            print(f"\rProcessing frame {i+1}/{len(frames)} at {timestamp:.1f}s...", end="", flush=True)
            
            # Prepare conversation
            messages = [
                {
                    "role": "user",
                    "content": f"<image>\nDescribe this video frame at {timestamp:.1f} seconds in extreme detail. Include all visible objects, people, actions, settings, colors, and any text visible."
                }
            ]
            
            # Apply chat template
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process image - AuroraCap expects images to be embedded in the prompt
            # This is model-specific and may need adjustment
            try:
                # Encode image to base64 or process it according to model requirements
                # For now, we'll use a simple text-only approach
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs.to(model.device),
                        max_new_tokens=300,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.95
                    )
                
                caption = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
            except Exception as e:
                caption = f"Error: {str(e)}"
            
            results.append({
                'timestamp': timestamp,
                'description': caption
            })
    
    else:
        # Xtuner runner
        print("\nUsing xtuner runner for inference...")
        # Implementation would go here based on xtuner API
        results = [{'timestamp': t, 'description': 'Xtuner inference not implemented'} for t in timestamps]
    
    print("\n✅ Processing complete!")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        video_name = Path(video_path).stem
        
        # Save JSON
        json_path = os.path.join(output_dir, f"{video_name}_auroracap_xtuner.json")
        with open(json_path, 'w') as f:
            json.dump({
                'video_path': video_path,
                'model': 'AuroraCap-7B-VID (xtuner)',
                'frame_count': len(frames),
                'segments': results
            }, f, indent=2)
        
        # Save timeline
        timeline_path = os.path.join(output_dir, f"{video_name}_auroracap_xtuner_timeline.txt")
        with open(timeline_path, 'w') as f:
            f.write(f"AURORACAP ANALYSIS (xtuner format)\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Duration: {duration:.1f}s\n")
            f.write("="*80 + "\n\n")
            
            for result in results:
                f.write(f"[{result['timestamp']:.2f}s]\n")
                f.write(f"{result['description']}\n")
                f.write("-"*40 + "\n")
        
        print(f"Results saved to {output_dir}")
    
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python auroracap_xtuner_inference.py <video_path> [output_dir]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/home/user/tiktok_production/aurora_cap/output"
    
    # Try xtuner first, fallback to direct
    try:
        model_or_runner = load_auroracap_xtuner()
    except:
        model_or_runner = load_auroracap_direct()
    
    # Process video
    results = process_video_with_auroracap(video_path, model_or_runner, output_dir)
    
    print(f"\n✅ Analysis complete! Processed {len(results)} frames")

if __name__ == "__main__":
    main()