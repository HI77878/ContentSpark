#!/usr/bin/env python3
"""
AuroraCap Alternative - Enhanced BLIP-2 with AuroraCap-style prompting
Provides detailed frame-by-frame video descriptions
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
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig

class AuroraCapAlternative:
    """Enhanced BLIP-2 that mimics AuroraCap's detailed descriptions"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load BLIP-2 with optimal settings"""
        print("Loading enhanced BLIP-2 model...")
        
        model_name = "Salesforce/blip2-opt-2.7b"
        
        # 8-bit quantization for efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True
        )
        
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        self.model.eval()
        print("✅ Model loaded successfully!")
        
    def extract_frames(self, video_path: str, interval: float = 0.5) -> List[Dict]:
        """Extract frames at specified interval"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video: {duration:.1f}s, {total_frames} frames at {fps:.1f} fps")
        
        frames = []
        frame_interval = max(1, int(fps * interval))
        
        for i in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
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
    
    def generate_detailed_caption(self, image: Image.Image, timestamp: float) -> Dict[str, str]:
        """Generate AuroraCap-style detailed caption using multiple prompts"""
        
        # Multiple analysis passes for comprehensive description
        prompts = [
            # Basic scene understanding
            ("a photo of", "scene"),
            
            # Detailed questions
            ("Question: What specific objects are visible and where are they located? Answer:", "objects"),
            ("Question: Describe any people, their appearance, clothing, and expressions. Answer:", "people"),
            ("Question: What actions or movements are happening? Answer:", "actions"),
            ("Question: Describe the setting, location, and background. Answer:", "setting"),
            ("Question: What are the colors, lighting conditions, and atmosphere? Answer:", "atmosphere"),
            ("Question: Is there any text, UI elements, or brands visible? Answer:", "text"),
            ("Question: From what angle or perspective is this filmed? Answer:", "camera")
        ]
        
        descriptions = {}
        
        for prompt, key in prompts:
            try:
                inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=80,
                        num_beams=3,
                        temperature=0.9,
                        do_sample=False
                    )
                
                caption = self.processor.decode(out[0], skip_special_tokens=True)
                
                # Clean the caption
                if prompt in caption:
                    caption = caption.replace(prompt, "").strip()
                
                # Filter out generic responses
                if len(caption) > 10 and not caption.startswith("http"):
                    descriptions[key] = caption
                    
            except Exception as e:
                print(f"Error with prompt '{key}': {e}")
        
        return descriptions
    
    def combine_descriptions(self, descriptions: Dict[str, str], timestamp: float) -> str:
        """Combine multiple descriptions into AuroraCap-style narrative"""
        
        # Start with timestamp
        parts = [f"At {timestamp:.1f} seconds:"]
        
        # Add scene overview
        if 'scene' in descriptions:
            parts.append(f"The frame shows {descriptions['scene']}.")
        
        # Add detailed elements
        if 'people' in descriptions:
            parts.append(f"People: {descriptions['people']}")
        
        if 'objects' in descriptions:
            parts.append(f"Objects: {descriptions['objects']}")
        
        if 'actions' in descriptions:
            parts.append(f"Actions: {descriptions['actions']}")
        
        if 'setting' in descriptions:
            parts.append(f"Setting: {descriptions['setting']}")
        
        if 'atmosphere' in descriptions:
            parts.append(f"Visual atmosphere: {descriptions['atmosphere']}")
        
        if 'text' in descriptions:
            parts.append(f"Text/UI: {descriptions['text']}")
        
        if 'camera' in descriptions:
            parts.append(f"Camera: {descriptions['camera']}")
        
        # Join with proper formatting
        combined = " ".join(parts)
        
        # Ensure detailed description
        if len(combined) < 100:
            combined += " The frame captures a moment in the video sequence."
        
        return combined
    
    def analyze_video(self, video_path: str, output_dir: str = None) -> Dict[str, Any]:
        """Main analysis function"""
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Load model if needed
        if self.model is None:
            self.load_model()
        
        # Extract frames
        print(f"\nAnalyzing video: {video_path}")
        frames = self.extract_frames(video_path, interval=0.5)
        
        # Process frames
        print("\nGenerating detailed descriptions...")
        results = []
        
        for i, frame_data in enumerate(frames):
            print(f"\rProcessing frame {i+1}/{len(frames)} at {frame_data['timestamp']:.1f}s...", end="", flush=True)
            
            # Generate multi-aspect description
            descriptions = self.generate_detailed_caption(frame_data['image'], frame_data['timestamp'])
            
            # Combine into narrative
            combined = self.combine_descriptions(descriptions, frame_data['timestamp'])
            
            results.append({
                'timestamp': frame_data['timestamp'],
                'frame_number': frame_data['frame_number'],
                'description': combined,
                'detailed_aspects': descriptions
            })
            
            # Clear cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
        print("\n✅ Analysis complete!")
        
        # Prepare output
        video_name = Path(video_path).stem
        analysis_data = {
            'video_path': video_path,
            'video_name': video_name,
            'model': 'AuroraCap Alternative (Enhanced BLIP-2)',
            'frame_interval': 0.5,
            'total_frames_analyzed': len(frames),
            'segments': results
        }
        
        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save JSON
            json_path = os.path.join(output_dir, f"{video_name}_auroracap_alternative.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {json_path}")
            
            # Save readable timeline
            timeline_path = os.path.join(output_dir, f"{video_name}_auroracap_alternative_timeline.txt")
            with open(timeline_path, 'w', encoding='utf-8') as f:
                f.write(f"AURORACAP-STYLE DETAILED VIDEO ANALYSIS\n")
                f.write(f"Video: {video_path}\n")
                f.write(f"Model: Enhanced BLIP-2 with Multi-Aspect Analysis\n")
                f.write(f"Analyzed {len(frames)} frames at 0.5s intervals\n")
                f.write("="*80 + "\n\n")
                
                for segment in results:
                    f.write(f"[{segment['timestamp']:.2f}s] Frame {segment['frame_number']}\n")
                    f.write(f"{segment['description']}\n")
                    
                    # Add detailed aspects
                    if segment.get('detailed_aspects'):
                        f.write("\nDetailed aspects:\n")
                        for aspect, desc in segment['detailed_aspects'].items():
                            f.write(f"  - {aspect}: {desc}\n")
                    
                    f.write("-"*60 + "\n\n")
            
            print(f"Timeline saved to: {timeline_path}")
        
        return analysis_data

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python auroracap_alternative.py <video_path> [output_dir]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/home/user/tiktok_production/aurora_cap/output"
    
    try:
        analyzer = AuroraCapAlternative()
        analysis = analyzer.analyze_video(video_path, output_dir)
        
        print(f"\n✅ Success! Analyzed {len(analysis['segments'])} segments")
        print(f"First segment: {analysis['segments'][0]['description'][:100]}...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()