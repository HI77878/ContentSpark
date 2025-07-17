#!/usr/bin/env python3
"""
Final AuroraCap inference script v3 - with all import issues resolved
Based on the official Aurora repository implementation
"""
import os
import os.path as osp
import sys
import json
import torch
import argparse
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

# Add aurora paths
sys.path.insert(0, '/home/user/tiktok_production/aurora_cap/aurora/src')
sys.path.insert(0, '/home/user/tiktok_production/aurora_cap/aurora')

# Define constants directly to avoid import issues
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN_INDEX = 0
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = '<image>'

# Import essential components
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
from huggingface_hub import snapshot_download

# Try importing Aurora components
try:
    from xtuner.xtuner.tools.load_video import read_video_pyav
    USE_AURORA_VIDEO_LOADER = True
except:
    USE_AURORA_VIDEO_LOADER = False
    print("Warning: Could not import Aurora video loader, using fallback")

def load_video_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """Load video frames with fallback method"""
    if USE_AURORA_VIDEO_LOADER:
        try:
            return read_video_pyav(video_path, num_frames)
        except:
            pass
    
    # Fallback video loading
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames

def get_vicuna_prompt(instruction: str) -> str:
    """Get Vicuna prompt format"""
    return f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} ASSISTANT:"""

def process_text(inputs, tokenizer):
    """Process text with image tokens according to Aurora's requirements"""
    chunk_encode = []
    for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
        if idx == 0:
            cur_encode = tokenizer.encode(chunk)
        else:
            cur_encode = tokenizer.encode(chunk, add_special_tokens=False)
        chunk_encode.append(cur_encode)
    
    ids = []
    for idx, cur_chunk_encode in enumerate(chunk_encode):
        ids.extend(cur_chunk_encode)
        if idx != len(chunk_encode) - 1:
            ids.append(IMAGE_TOKEN_INDEX)
    
    ids = torch.tensor(ids).cuda().unsqueeze(0)
    return ids

class AuroraCapVideoAnalyzer:
    """AuroraCap video analyzer with robust implementation"""
    
    def __init__(self, model_path='wchai/AuroraCap-7B-VID-xtuner'):
        self.model_path = model_path
        self.llm = None
        self.visual_encoder = None
        self.projector = None
        self.tokenizer = None
        self.image_processor = None
        
    def load_model(self):
        """Load AuroraCap model components"""
        print(f"Loading AuroraCap model from {self.model_path}...")
        
        # Download model
        if not osp.isdir(self.model_path):
            pretrained_pth = snapshot_download(repo_id=self.model_path)
        else:
            pretrained_pth = self.model_path
            
        print(f"Model path: {pretrained_pth}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_pth,
            trust_remote_code=True,
            padding_side='right',
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load LLM
        print("Loading language model...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            pretrained_pth,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.llm.eval()
        
        # Load visual encoder
        visual_encoder_path = os.path.join(pretrained_pth, "visual_encoder")
        print(f"Loading visual encoder from {visual_encoder_path}...")
        
        self.visual_encoder = AutoModel.from_pretrained(
            visual_encoder_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).cuda()
        self.visual_encoder.eval()
        
        # Load projector
        projector_path = os.path.join(pretrained_pth, "projector")
        print(f"Loading projector from {projector_path}...")
        
        self.projector = AutoModel.from_pretrained(
            projector_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).cuda()
        self.projector.eval()
        
        # Initialize image processor
        print("Loading image processor...")
        self.image_processor = CLIPImageProcessor.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            size=378,
            crop_size=378,
        )
        
        print("✅ All models loaded successfully!")
    
    def process_video_features(self, video_frames: List[Image.Image]) -> torch.Tensor:
        """Process video frames and extract features"""
        # Process frames with image processor
        processed = self.image_processor(video_frames, return_tensors='pt')
        pixel_values = processed['pixel_values'].to(dtype=torch.float16).cuda()
        
        # Extract visual features
        with torch.no_grad():
            # Process through visual encoder
            if hasattr(self.visual_encoder, 'vision_model'):
                # Standard CLIP structure
                outputs = self.visual_encoder(pixel_values, output_hidden_states=True)
                # Use the second to last layer features and remove CLS token
                visual_features = outputs.hidden_states[-2][:, 1:]
            else:
                # Aurora encoder
                outputs = self.visual_encoder(pixel_values, output_hidden_states=True)
                visual_features = outputs.hidden_states[-2][:, 1:]
            
            # Reshape for batch processing
            b = 1  # batch size
            f = len(video_frames)  # number of frames
            n = visual_features.shape[1]  # number of patches
            c = visual_features.shape[2]  # feature dimension
            
            # Combine frame features
            visual_features = visual_features.reshape(b, f * n, c)
            
            # Project features to LLM space
            projected_features = self.projector(visual_features)
            
            # Reshape back to separate frames
            projected_features = projected_features.reshape(b, f, n, -1)
        
        return projected_features
    
    def generate_description(self, visual_features: torch.Tensor, prompt: str) -> str:
        """Generate description from visual features"""
        b, f, n, c = visual_features.shape
        
        # Flatten visual features for sequence
        visual_embeds = visual_features.reshape(b, f * n, c)
        
        # Tokenize prompt
        text_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to('cuda')
        
        # Get text embeddings
        with torch.no_grad():
            text_embeds = self.llm.get_input_embeddings()(text_inputs.input_ids)
        
        # Combine visual and text embeddings
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        # Create attention mask
        visual_attention = torch.ones(
            visual_embeds.shape[0], visual_embeds.shape[1],
            dtype=torch.long, device='cuda'
        )
        combined_attention = torch.cat([visual_attention, text_inputs.attention_mask], dim=1)
        
        # Generate response
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention,
                max_new_tokens=2048,
                temperature=0.0,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][combined_embeds.shape[1]:]
        description = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return description.strip()
    
    def analyze_video(self, video_path: str, num_frames: int = 8) -> Dict[str, Any]:
        """Main video analysis function"""
        if self.llm is None:
            self.load_model()
        
        print(f"\nAnalyzing video: {video_path}")
        print(f"Extracting {num_frames} frames...")
        
        # Load video frames
        video_frames = load_video_frames(video_path, num_frames)
        print(f"Loaded {len(video_frames)} frames")
        
        # Process video features
        print("Processing video features...")
        visual_features = self.process_video_features(video_frames)
        
        # Generate overall description
        print("Generating comprehensive video description...")
        
        # Create prompt with image tokens
        image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * len(video_frames))
        instruction = f"{image_tokens}\nProvide an extremely detailed description of this video. Include all visual elements, actions, people, objects, settings, camera movements, lighting, colors, and any text or graphics visible. Describe the progression of events chronologically. Be as comprehensive and specific as possible to enable full reconstruction of the video content."
        
        prompt = get_vicuna_prompt(instruction)
        overall_description = self.generate_description(visual_features, prompt)
        
        # Generate frame-by-frame analysis
        print("Generating frame-by-frame analysis...")
        segments = []
        
        for i in range(len(video_frames)):
            # Process single frame
            frame_features = visual_features[:, i:i+1, :, :]
            
            frame_instruction = f"{DEFAULT_IMAGE_TOKEN}\nDescribe this specific frame in detail, including all visible elements, people, objects, text, and actions."
            frame_prompt = get_vicuna_prompt(frame_instruction)
            
            frame_description = self.generate_description(frame_features, frame_prompt)
            
            segments.append({
                'frame_index': i,
                'timestamp': i / max(len(video_frames) - 1, 1),
                'description': frame_description
            })
            
            print(f"Processed frame {i+1}/{len(video_frames)}")
        
        return {
            'overall_description': overall_description,
            'segments': segments,
            'metadata': {
                'video_path': video_path,
                'num_frames_analyzed': len(video_frames),
                'model': 'AuroraCap-7B-VID',
                'implementation': 'v3-robust'
            }
        }

def main():
    parser = argparse.ArgumentParser(description="AuroraCap Video Analysis - Final Implementation v3")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--model_path", default="wchai/AuroraCap-7B-VID-xtuner", help="Model path")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to sample")
    parser.add_argument("--output_dir", default="/home/user/tiktok_production/aurora_cap/output", help="Output directory")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = AuroraCapVideoAnalyzer(model_path=args.model_path)
    
    try:
        # Analyze video
        results = analyzer.analyze_video(
            video_path=args.video_path,
            num_frames=args.num_frames
        )
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        video_name = Path(args.video_path).stem
        
        # Save JSON
        output_path = os.path.join(args.output_dir, f"{video_name}_auroracap_final.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save human-readable report
        report_path = os.path.join(args.output_dir, f"{video_name}_auroracap_final_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("AURORACAP VIDEO ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("VIDEO INFORMATION:\n")
            f.write(f"  Path: {args.video_path}\n")
            f.write(f"  Frames analyzed: {results['metadata']['num_frames_analyzed']}\n")
            f.write(f"  Model: {results['metadata']['model']}\n")
            f.write(f"  Implementation: {results['metadata']['implementation']}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            f.write("COMPREHENSIVE VIDEO DESCRIPTION:\n")
            f.write("-" * 80 + "\n")
            f.write(results['overall_description'] + "\n\n")
            
            f.write("=" * 80 + "\n\n")
            f.write("FRAME-BY-FRAME ANALYSIS:\n")
            f.write("-" * 80 + "\n\n")
            
            for seg in results['segments']:
                f.write(f"[Frame {seg['frame_index']+1} - Timestamp: {seg['timestamp']:.2f}]\n")
                f.write(seg['description'] + "\n\n")
                f.write("-" * 40 + "\n\n")
        
        print(f"\n✅ Analysis complete!")
        print(f"Results saved to: {output_path}")
        print(f"Report saved to: {report_path}")
        
        # Display preview
        print("\n" + "=" * 80)
        print("DESCRIPTION PREVIEW:")
        print("-" * 80)
        preview_length = 1000
        preview = results['overall_description'][:preview_length]
        print(preview + "..." if len(results['overall_description']) > preview_length else preview)
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()