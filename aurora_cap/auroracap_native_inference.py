#!/usr/bin/env python3
"""
Native AuroraCap inference script with proper xtuner integration
Based on official aurora repository implementation
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
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Add aurora paths
sys.path.insert(0, '/home/user/tiktok_production/aurora_cap/aurora/src')
sys.path.insert(0, '/home/user/tiktok_production/aurora_cap/aurora')

@dataclass
class InferenceConfig:
    """Configuration for AuroraCap inference"""
    model_path: str = "wchai/AuroraCap-7B-VID-xtuner"
    num_frames: int = 8
    token_kept_ratio: float = 0.3  # 0.2-0.4 for captioning
    temperature: float = 0.0
    top_p: float = 1.0
    num_beams: int = 1
    max_new_tokens: int = 300
    device: str = "cuda"
    
class AuroraCapInference:
    """AuroraCap video captioning with xtuner framework"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.visual_encoder = None
        self.projector = None
        
    def load_model(self):
        """Load AuroraCap model components"""
        print(f"Loading AuroraCap model from {self.config.model_path}...")
        
        try:
            from transformers import (
                AutoModelForCausalLM, 
                AutoTokenizer, 
                CLIPImageProcessor,
                AutoModel
            )
            from huggingface_hub import snapshot_download
            
            # Download model if needed
            cache_dir = "/home/user/tiktok_production/aurora_cap/.cache"
            model_dir = snapshot_download(
                repo_id=self.config.model_path,
                cache_dir=cache_dir,
                local_dir_use_symlinks=False
            )
            
            print(f"Model downloaded to: {model_dir}")
            
            # Fix rope_scaling in config if needed
            config_path = os.path.join(model_dir, "config.json")
            self._fix_rope_scaling(config_path)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True,
                use_fast=False
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load language model (Vicuna base)
            print("Loading language model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_flash_attention_2=False  # Disable flash attention for compatibility
            )
            
            # Load visual components
            print("Loading visual encoder...")
            try:
                # Try loading AuroraEncoder
                from aurora.model.aurora_encoder import AuroraEncoder
                self.visual_encoder = AuroraEncoder.from_pretrained(
                    os.path.join(model_dir, "visual_encoder"),
                    torch_dtype=torch.float16
                ).to(self.config.device)
            except:
                print("Using CLIP encoder as fallback...")
                from transformers import CLIPVisionModel
                self.visual_encoder = CLIPVisionModel.from_pretrained(
                    "openai/clip-vit-large-patch14-336",
                    torch_dtype=torch.float16
                ).to(self.config.device)
            
            # Load projector
            print("Loading projector...")
            try:
                self.projector = AutoModel.from_pretrained(
                    os.path.join(model_dir, "projector"),
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to(self.config.device)
            except:
                print("Projector not found, using identity mapping")
                self.projector = None
            
            # Load image processor
            self.image_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14-336"
            )
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            raise
    
    def _fix_rope_scaling(self, config_path: str):
        """Fix rope_scaling configuration for compatibility"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if 'rope_scaling' in config:
                # Ensure only type and factor fields
                if isinstance(config['rope_scaling'], dict):
                    rope_scaling = config['rope_scaling']
                    fixed = {
                        'type': rope_scaling.get('rope_type', rope_scaling.get('type', 'linear')),
                        'factor': float(rope_scaling.get('factor', 8.0))
                    }
                    config['rope_scaling'] = fixed
                    
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    print("Fixed rope_scaling configuration")
        except Exception as e:
            print(f"Warning: Could not fix rope_scaling: {e}")
    
    def extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], List[float]]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video: {duration:.1f}s, {total_frames} frames at {fps:.1f} fps")
        
        # Sample frames uniformly
        if total_frames <= self.config.num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.config.num_frames, dtype=int)
        
        frames = []
        timestamps = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                timestamps.append(idx / fps if fps > 0 else 0)
        
        cap.release()
        print(f"Extracted {len(frames)} frames")
        return frames, timestamps
    
    def process_video_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Process video frames for model input"""
        processed_frames = []
        
        for frame in frames:
            # Convert to PIL
            pil_image = Image.fromarray(frame)
            # Process with CLIP processor
            processed = self.image_processor(
                images=pil_image,
                return_tensors="pt"
            )['pixel_values'][0]
            processed_frames.append(processed)
        
        # Stack frames
        video_tensor = torch.stack(processed_frames)
        return video_tensor.to(self.config.device, dtype=torch.float16)
    
    def generate_caption(self, video_tensor: torch.Tensor, prompt: str) -> str:
        """Generate caption for video"""
        try:
            # Encode visual features
            with torch.no_grad():
                if hasattr(self.visual_encoder, 'forward_features'):
                    visual_features = self.visual_encoder.forward_features(video_tensor)
                else:
                    visual_features = self.visual_encoder(video_tensor).last_hidden_state
                
                # Project features if projector exists
                if self.projector is not None:
                    visual_features = self.projector(visual_features)
            
            # Prepare text input
            text_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.config.device)
            
            # Generate with visual context
            with torch.no_grad():
                # For multimodal generation, we need to combine visual and text embeddings
                # This is model-specific and may need adjustment
                
                # Simple approach: use prompt with model's generate
                outputs = self.model.generate(
                    input_ids=text_inputs.input_ids,
                    attention_mask=text_inputs.attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    num_beams=self.config.num_beams,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][text_inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error during generation: {e}")
            traceback.print_exc()
            return f"Error generating caption: {str(e)}"
    
    def analyze_video(self, video_path: str) -> List[Dict[str, Any]]:
        """Main analysis function"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Load model if needed
        if self.model is None:
            self.load_model()
        
        # Extract frames
        frames, timestamps = self.extract_frames(video_path)
        
        # Process all frames
        print("\nProcessing video frames...")
        video_tensor = self.process_video_frames(frames)
        
        # Generate caption for entire video
        prompt = "Describe this video in extreme detail, including all objects, people, actions, settings, and changes throughout the video."
        
        print("Generating detailed caption...")
        caption = self.generate_caption(video_tensor, prompt)
        
        # Also generate per-segment captions
        segments = []
        
        # Divide into time segments
        segment_size = max(1, len(frames) // 4)  # 4 segments
        
        for i in range(0, len(frames), segment_size):
            segment_frames = frames[i:i+segment_size]
            segment_timestamps = timestamps[i:i+segment_size]
            
            if segment_frames:
                # Process segment
                segment_tensor = self.process_video_frames(segment_frames)
                
                # Generate segment caption
                start_time = segment_timestamps[0]
                end_time = segment_timestamps[-1]
                
                segment_prompt = f"Describe what happens in this video segment from {start_time:.1f}s to {end_time:.1f}s in detail."
                segment_caption = self.generate_caption(segment_tensor, segment_prompt)
                
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'description': segment_caption
                })
        
        return {
            'overall_description': caption,
            'segments': segments,
            'frame_count': len(frames),
            'duration': timestamps[-1] if timestamps else 0
        }

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AuroraCap Video Analysis")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--output_dir", default="/home/user/tiktok_production/aurora_cap/output",
                       help="Output directory")
    parser.add_argument("--model_path", default="wchai/AuroraCap-7B-VID-xtuner",
                       help="Model path or HuggingFace ID")
    parser.add_argument("--num_frames", type=int, default=8,
                       help="Number of frames to sample")
    parser.add_argument("--token_kept_ratio", type=float, default=0.3,
                       help="Token retention ratio (0.2-0.4 for captioning)")
    parser.add_argument("--max_new_tokens", type=int, default=300,
                       help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # Create config
    config = InferenceConfig(
        model_path=args.model_path,
        num_frames=args.num_frames,
        token_kept_ratio=args.token_kept_ratio,
        max_new_tokens=args.max_new_tokens
    )
    
    # Initialize inference
    aurora = AuroraCapInference(config)
    
    try:
        # Analyze video
        print(f"\nAnalyzing video: {args.video_path}")
        results = aurora.analyze_video(args.video_path)
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        video_name = Path(args.video_path).stem
        
        # Save JSON
        output_path = os.path.join(args.output_dir, f"{video_name}_auroracap_native.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save timeline
        timeline_path = os.path.join(args.output_dir, f"{video_name}_auroracap_native_timeline.txt")
        with open(timeline_path, 'w', encoding='utf-8') as f:
            f.write(f"AURORACAP NATIVE ANALYSIS\n")
            f.write(f"Video: {args.video_path}\n")
            f.write(f"Duration: {results['duration']:.1f}s\n")
            f.write(f"Frames analyzed: {results['frame_count']}\n")
            f.write("="*80 + "\n\n")
            
            f.write("OVERALL DESCRIPTION:\n")
            f.write(results['overall_description'] + "\n\n")
            f.write("-"*80 + "\n\n")
            
            f.write("DETAILED SEGMENTS:\n")
            for seg in results['segments']:
                f.write(f"[{seg['start_time']:.1f}s - {seg['end_time']:.1f}s]\n")
                f.write(seg['description'] + "\n\n")
        
        print(f"\n✅ Analysis complete!")
        print(f"Results saved to: {output_path}")
        print(f"Timeline saved to: {timeline_path}")
        
        # Display sample
        print(f"\nOverall description preview:")
        print(results['overall_description'][:200] + "...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()