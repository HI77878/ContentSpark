#!/usr/bin/env python3
"""
Final AuroraCap inference script v2 - with import fixes
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

# Import only what we need to avoid deepspeed dependency
from xtuner.xtuner.utils.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from xtuner.xtuner.utils.templates import PROMPT_TEMPLATE

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
from huggingface_hub import snapshot_download

# Import Aurora model components
try:
    from xtuner.xtuner.model.aurora import AuroraEncoder, AuroraModel
except ImportError as e:
    print(f"Warning: Could not import Aurora components directly: {e}")
    # We'll use a fallback approach

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
        self.visual_encoder = None
        self.projector = None
        self.llm = None
        
    def load_model_fallback(self):
        """Fallback model loading approach"""
        print(f"Using fallback loading approach for {self.model_path}...")
        
        # Download model
        pretrained_pth = snapshot_download(repo_id=self.model_path)
        print(f"Model downloaded to: {pretrained_pth}")
        
        # Load LLM
        print("Loading language model...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            pretrained_pth,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Load visual encoder
        visual_encoder_path = os.path.join(pretrained_pth, "visual_encoder")
        print(f"Loading visual encoder from {visual_encoder_path}...")
        
        # Try loading with AutoModel first
        try:
            self.visual_encoder = AutoModel.from_pretrained(
                visual_encoder_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).cuda()
        except Exception as e:
            print(f"Could not load visual encoder with AutoModel: {e}")
            # Try alternative loading
            from transformers import CLIPVisionModel
            self.visual_encoder = CLIPVisionModel.from_pretrained(
                visual_encoder_path,
                torch_dtype=torch.float16
            ).cuda()
        
        # Load projector
        projector_path = os.path.join(pretrained_pth, "projector")
        print(f"Loading projector from {projector_path}...")
        self.projector = AutoModel.from_pretrained(
            projector_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).cuda()
        
        # Initialize image processor
        self.image_processor = CLIPImageProcessor.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            size=378,
            crop_size=378,
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_pth,
            trust_remote_code=True,
            padding_side='right',
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Model loaded successfully using fallback method!")
        
    def load_model(self):
        """Load AuroraCap model with correct architecture"""
        print(f"Loading AuroraCap model from {self.model_path}...")
        
        try:
            # Try the proper Aurora loading first
            pretrained_pth = snapshot_download(repo_id=self.model_path) if not osp.isdir(self.model_path) else self.model_path
            pretrained_vit = osp.join(pretrained_pth, "visual_encoder")
            projector_path = osp.join(pretrained_pth, "projector")
            
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
            
            self.model.projector = AutoModel.from_pretrained(
                projector_path, 
                torch_dtype=torch.float16, 
                trust_remote_code=True
            ).cuda()
            
            self.image_processor = CLIPImageProcessor.from_pretrained(
                pretrained_model_name_or_path="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                trust_remote_code=True,
                size=378,
                crop_size=378,
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=pretrained_pth,
                trust_remote_code=True,
                padding_side='right',
            )
            
            print("✅ Model loaded successfully with Aurora architecture!")
            
        except Exception as e:
            print(f"Could not load with Aurora architecture: {e}")
            print("Falling back to alternative loading method...")
            self.load_model_fallback()
    
    def generate_with_fallback(self, video_tensor, prompt):
        """Generate caption using fallback method"""
        # Process video through visual encoder
        with torch.no_grad():
            if hasattr(self.visual_encoder, 'vision_model'):
                # CLIP model structure
                visual_outputs = self.visual_encoder(video_tensor)
                visual_features = visual_outputs.last_hidden_state[:, 1:]  # Remove CLS token
            else:
                visual_outputs = self.visual_encoder(video_tensor)
                visual_features = visual_outputs.last_hidden_state
            
            # Project features
            b, f = video_tensor.shape[0], video_tensor.shape[1]
            visual_features = visual_features.view(b * f, -1, visual_features.shape[-1])
            visual_features = visual_features.view(b, -1, visual_features.shape[-1])
            
            visual_embeds = self.projector(visual_features)
            
            # Prepare text
            text_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to('cuda')
            
            # Get text embeddings
            text_embeds = self.llm.get_input_embeddings()(text_inputs.input_ids)
            
            # Combine embeddings
            combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            
            # Create attention mask
            visual_attention = torch.ones(
                visual_embeds.shape[0], visual_embeds.shape[1], 
                dtype=torch.long, device='cuda'
            )
            combined_attention_mask = torch.cat([visual_attention, text_inputs.attention_mask], dim=1)
            
            # Generate
            outputs = self.llm.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                max_new_tokens=2048,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode only generated part
            generated_tokens = outputs[0][combined_embeds.shape[1]:]
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    def analyze_video(self, video_path: str, num_frames: int = 8, token_kept_ratio: float = 0.2) -> Dict[str, Any]:
        """Analyze video and generate detailed description"""
        if self.model is None and self.llm is None:
            self.load_model()
        
        print(f"\nAnalyzing video: {video_path}")
        print(f"Sampling {num_frames} frames with token_kept_ratio={token_kept_ratio}")
        
        # Load video frames
        video_frames = read_video_pyav(video_path, num_frames)
        print(f"Loaded {len(video_frames)} frames")
        
        # Process frames
        image_tensor = self.image_processor(video_frames, return_tensors='pt')['pixel_values']
        image_tensor = torch.stack([img.to(dtype=torch.float16).cuda() for img in image_tensor])
        
        # Generate description
        if self.model is not None:
            # Use proper Aurora model
            data = {
                "pixel_values": image_tensor.unsqueeze(0)
            }
            
            image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * len(video_frames))
            prompt = "Provide an extremely detailed description of this video. Include all visual elements, actions, people, objects, settings, camera movements, lighting, colors, and any text or graphics visible. Describe the progression of events frame by frame. Be as comprehensive and specific as possible to enable full reconstruction of the video content."
            
            text_input = image_tokens + "\n" + prompt
            prompt_text = PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(input=text_input, round=1)
            
            data["input_ids"] = process_text(prompt_text, self.tokenizer).cuda()
            
            # Set token merge ratio if available
            if hasattr(self.model.visual_encoder, 'reset_tome_r'):
                self.model.visual_encoder.reset_tome_r(token_kept_ratio)
            
            print("Generating description...")
            output = self.model(data, mode="inference")
            
            with torch.no_grad():
                cont = self.model.llm.generate(
                    **output,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    num_beams=1,
                    max_new_tokens=2048,
                )
            
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            
            if "ASSISTANT:" in text_outputs:
                description = text_outputs.split("ASSISTANT:")[-1].strip()
            else:
                description = text_outputs
                
        else:
            # Use fallback method
            print("Using fallback generation method...")
            prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image> Provide an extremely detailed description of this video. Include all visual elements, actions, people, objects, settings, camera movements, lighting, colors, and any text or graphics visible. ASSISTANT:"
            description = self.generate_with_fallback(image_tensor.unsqueeze(0), prompt)
        
        # Create simple segments for now
        segments = []
        for i in range(len(video_frames)):
            segments.append({
                'frame_index': i,
                'timestamp': i / len(video_frames),
                'description': f"Frame {i+1} of {len(video_frames)}"
            })
        
        return {
            'overall_description': description,
            'segments': segments,
            'metadata': {
                'num_frames_analyzed': len(video_frames),
                'token_kept_ratio': token_kept_ratio,
                'model': 'AuroraCap-7B-VID',
                'video_path': video_path,
                'loading_method': 'aurora' if self.model is not None else 'fallback'
            }
        }

def main():
    parser = argparse.ArgumentParser(description="AuroraCap Video Analysis - Final Implementation v2")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--model_path", default="wchai/AuroraCap-7B-VID-xtuner", help="Model path")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to sample")
    parser.add_argument("--token_kept_ratio", type=float, default=0.3, help="Token merge ratio")
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
        
        # Save report
        report_path = os.path.join(args.output_dir, f"{video_name}_auroracap_final_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("AURORACAP VIDEO ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Video: {args.video_path}\n")
            f.write(f"Frames analyzed: {results['metadata']['num_frames_analyzed']}\n")
            f.write(f"Model: {results['metadata']['model']}\n")
            f.write(f"Loading method: {results['metadata']['loading_method']}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            f.write("VIDEO DESCRIPTION:\n")
            f.write("-" * 80 + "\n")
            f.write(results['overall_description'] + "\n")
        
        print(f"\n✅ Analysis complete!")
        print(f"Results saved to: {output_path}")
        print(f"Report saved to: {report_path}")
        
        # Display preview
        print(f"\nDescription preview:")
        print("-" * 80)
        preview = results['overall_description'][:800]
        print(preview + "..." if len(results['overall_description']) > 800 else preview)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()