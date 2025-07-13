#!/usr/bin/env python3
"""
Final AuroraCap inference script with correct architecture and implementation
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
from typing import List, Dict, Any, Tuple

# Add aurora paths
sys.path.insert(0, '/home/user/tiktok_production/aurora_cap/aurora/src')
sys.path.insert(0, '/home/user/tiktok_production/aurora_cap/aurora')

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
from huggingface_hub import snapshot_download

from xtuner.xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, PROMPT_TEMPLATE
from xtuner.xtuner.model.aurora import AuroraEncoder, AuroraModel
from xtuner.xtuner.tools.load_video import read_video_pyav

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
    """AuroraCap video analyzer with correct model architecture"""
    
    def __init__(self, model_path='wchai/AuroraCap-7B-VID-xtuner'):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        
    def load_model(self):
        """Load AuroraCap model with correct architecture"""
        print(f"Loading AuroraCap model from {self.model_path}...")
        
        # Download model if needed
        pretrained_pth = snapshot_download(repo_id=self.model_path) if not osp.isdir(self.model_path) else self.model_path
        pretrained_vit = osp.join(pretrained_pth, "visual_encoder")
        projector_path = osp.join(pretrained_pth, "projector")
        
        print(f"Model path: {pretrained_pth}")
        print(f"Visual encoder path: {pretrained_vit}")
        print(f"Projector path: {projector_path}")
        
        # Initialize AuroraModel with correct components
        self.model = AuroraModel(
            llm=AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=pretrained_pth,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ),
            visual_encoder=AuroraEncoder.from_pretrained(
                pretrained_model_name_or_path=pretrained_vit,
                torch_dtype=torch.float16,
            ),
        ).cuda()
        
        # Load projector separately
        self.model.projector = AutoModel.from_pretrained(
            projector_path, 
            torch_dtype=torch.float16, 
            trust_remote_code=True
        ).cuda()
        
        # Initialize image processor with correct model
        self.image_processor = CLIPImageProcessor.from_pretrained(
            pretrained_model_name_or_path="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            trust_remote_code=True,
            size=378,
            crop_size=378,
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_pth,
            trust_remote_code=True,
            padding_side='right',
        )
        
        print("✅ Model loaded successfully!")
        
    def analyze_video(self, video_path: str, num_frames: int = 8, token_kept_ratio: float = 0.2) -> Dict[str, Any]:
        """Analyze video and generate detailed description"""
        if self.model is None:
            self.load_model()
        
        print(f"\nAnalyzing video: {video_path}")
        print(f"Sampling {num_frames} frames with token_kept_ratio={token_kept_ratio}")
        
        # Load video frames using Aurora's video loader
        video_frames = read_video_pyav(video_path, num_frames)
        print(f"Loaded {len(video_frames)} frames")
        
        # Process frames
        image_tensor = self.image_processor(video_frames, return_tensors='pt')['pixel_values']
        image_tensor = [_image.to(dtype=torch.float16).cuda() for _image in image_tensor]
        
        # Prepare data dict
        data = dict()
        data["pixel_values"] = torch.stack(image_tensor).unsqueeze(0)
        
        # Create image tokens for each frame
        image_tokens = [DEFAULT_IMAGE_TOKEN] * len(video_frames)
        image_tokens = " ".join(image_tokens)
        
        # Create prompt with proper Vicuna format
        prompt = "Provide an extremely detailed description of this video. Include all visual elements, actions, people, objects, settings, camera movements, lighting, colors, and any text or graphics visible. Describe the progression of events frame by frame. Be as comprehensive and specific as possible to enable full reconstruction of the video content."
        
        text_input = image_tokens + "\n" + prompt
        prompt_text = PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(input=text_input, round=1)
        
        # Process text input
        data["input_ids"] = process_text(prompt_text, self.tokenizer).cuda()
        
        # Set token merge ratio
        self.model.visual_encoder.reset_tome_r(token_kept_ratio)
        
        # Run inference
        print("Generating description...")
        output = self.model(data, mode="inference")
        
        # Generate text
        with torch.no_grad():
            cont = self.model.llm.generate(
                **output,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                num_beams=1,
                max_new_tokens=2048,
            )
        
        # Decode output
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        
        # Extract only the generated part (remove the prompt)
        if "ASSISTANT:" in text_outputs:
            description = text_outputs.split("ASSISTANT:")[-1].strip()
        else:
            description = text_outputs
        
        # Create frame-by-frame analysis
        segments = []
        frame_interval = len(video_frames)
        
        # Analyze individual frames for timeline
        for i, frame in enumerate(video_frames):
            # Prepare single frame
            frame_tensor = self.image_processor([frame], return_tensors='pt')['pixel_values']
            frame_tensor = frame_tensor.to(dtype=torch.float16).cuda()
            
            frame_data = {
                "pixel_values": frame_tensor.unsqueeze(0),
                "input_ids": process_text(
                    PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(
                        input=f"{DEFAULT_IMAGE_TOKEN}\nDescribe this specific frame in detail, including all visible elements.",
                        round=1
                    ), 
                    self.tokenizer
                ).cuda()
            }
            
            # Set higher token ratio for single frame
            self.model.visual_encoder.reset_tome_r(0.8)
            
            frame_output = self.model(frame_data, mode="inference")
            
            with torch.no_grad():
                frame_cont = self.model.llm.generate(
                    **frame_output,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=300,
                )
            
            frame_description = self.tokenizer.batch_decode(frame_cont, skip_special_tokens=True)[0]
            if "ASSISTANT:" in frame_description:
                frame_description = frame_description.split("ASSISTANT:")[-1].strip()
            
            segments.append({
                'frame_index': i,
                'timestamp': i / frame_interval,  # Normalized timestamp
                'description': frame_description
            })
            
            print(f"Processed frame {i+1}/{len(video_frames)}")
        
        return {
            'overall_description': description,
            'segments': segments,
            'metadata': {
                'num_frames_analyzed': len(video_frames),
                'token_kept_ratio': token_kept_ratio,
                'model': 'AuroraCap-7B-VID',
                'video_path': video_path
            }
        }

def main():
    parser = argparse.ArgumentParser(description="AuroraCap Video Analysis - Final Implementation")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--model_path", default="wchai/AuroraCap-7B-VID-xtuner", help="Model path")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to sample")
    parser.add_argument("--token_kept_ratio", type=float, default=0.2, help="Token merge ratio (0.2-0.4 for captioning)")
    parser.add_argument("--output_dir", default="/home/user/tiktok_production/aurora_cap/output", help="Output directory")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = AuroraCapVideoAnalyzer(model_path=args.model_path)
    
    try:
        # Analyze video
        results = analyzer.analyze_video(
            video_path=args.video_path,
            num_frames=args.num_frames,
            token_kept_ratio=args.token_kept_ratio
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
            f.write(f"Video: {args.video_path}\n")
            f.write(f"Frames analyzed: {results['metadata']['num_frames_analyzed']}\n")
            f.write(f"Token kept ratio: {results['metadata']['token_kept_ratio']}\n")
            f.write(f"Model: {results['metadata']['model']}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            f.write("OVERALL VIDEO DESCRIPTION:\n")
            f.write("-" * 80 + "\n")
            f.write(results['overall_description'] + "\n\n")
            
            f.write("FRAME-BY-FRAME ANALYSIS:\n")
            f.write("-" * 80 + "\n")
            for seg in results['segments']:
                f.write(f"\n[Frame {seg['frame_index']+1}]\n")
                f.write(seg['description'] + "\n")
        
        print(f"\n✅ Analysis complete!")
        print(f"Results saved to: {output_path}")
        print(f"Report saved to: {report_path}")
        
        # Display preview
        print(f"\nOverall description preview:")
        print("-" * 80)
        preview = results['overall_description'][:500]
        print(preview + "..." if len(results['overall_description']) > 500 else preview)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()