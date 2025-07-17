#!/usr/bin/env python3
"""
Simplified AuroraCap inference script that avoids deepspeed dependency
"""
import os
import sys
import json
import torch
import time
from pathlib import Path

# Set environment to avoid deepspeed
os.environ['NO_DEEPSPEED'] = '1'
os.environ['CUDA_HOME'] = '/usr/local/cuda'

# Try importing the required modules
try:
    # First try the standard transformers approach
    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Using transformers directly for AuroraCap")
    USE_XTUNER = False
except:
    print("Transformers approach failed, trying xtuner...")
    USE_XTUNER = True

def analyze_with_transformers(video_path):
    """Use transformers library directly"""
    import cv2
    from PIL import Image
    
    print("Loading AuroraCap model via transformers...")
    
    # Load model and processor
    model_name = "wchai/AuroraCap-7B-VID-xtuner"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video: {duration:.1f}s, {total_frames} frames, {fps:.1f} fps")
    
    # Process video in segments
    segments = []
    frame_interval = int(fps)  # 1 frame per second
    
    for i in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Generate caption
        prompt = "Describe this frame in extreme detail including all objects, actions, and settings."
        inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=200, num_beams=3)
        
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        segments.append({
            "timestamp": i / fps,
            "description": caption
        })
        
        print(f"  [{i/fps:.1f}s] {caption[:80]}...")
    
    cap.release()
    return segments

def analyze_fallback(video_path):
    """Fallback to BLIP-2 if AuroraCap fails"""
    print("Falling back to BLIP-2 for analysis...")
    
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    from PIL import Image
    import cv2
    
    # Load BLIP-2
    model_name = "Salesforce/blip2-opt-2.7b"
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video: {duration:.1f}s, {total_frames} frames, {fps:.1f} fps")
    
    # Process video
    segments = []
    frame_interval = int(fps)  # 1 frame per second
    
    for i in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Generate multiple captions for detail
        prompts = [
            "a photo of",
            "Question: What is happening in this image? Answer:",
            "Question: Describe the setting and objects in detail. Answer:"
        ]
        
        captions = []
        for prompt in prompts:
            inputs = processor(pil_image, text=prompt, return_tensors="pt").to("cuda")
            generated_ids = model.generate(**inputs, max_new_tokens=50, num_beams=3)
            caption = processor.decode(generated_ids[0], skip_special_tokens=True)
            caption = caption.replace(prompt, "").strip()
            captions.append(caption)
        
        # Combine captions
        combined = " | ".join(captions)
        
        segments.append({
            "timestamp": i / fps,
            "description": combined
        })
        
        print(f"  [{i/fps:.1f}s] {combined[:80]}...")
    
    cap.release()
    return segments

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else '/app/videos/input.mp4'
    output_dir = '/app/output'
    
    print(f"Analyzing video: {video_path}")
    
    start_time = time.time()
    
    try:
        # Try AuroraCap first
        if USE_XTUNER:
            raise Exception("Skipping xtuner approach due to deepspeed issues")
        segments = analyze_with_transformers(video_path)
    except Exception as e:
        print(f"AuroraCap failed: {e}")
        # Fallback to BLIP-2
        segments = analyze_fallback(video_path)
    
    analysis_time = time.time() - start_time
    
    # Save results
    output_data = {
        "video_path": video_path,
        "model": "AuroraCap (or BLIP-2 fallback)",
        "analysis_time": analysis_time,
        "segments": segments
    }
    
    video_name = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{video_name}_auroracap_analysis.json")
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Analysis complete in {analysis_time:.1f}s")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()