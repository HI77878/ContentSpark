#!/usr/bin/env python3
"""
Simplified AuroraCap inference without complex token merging
Focus on getting basic functionality working first
"""
import os
import sys
import json
import torch
import argparse
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any

# Constants
DEFAULT_IMAGE_TOKEN = '<image>'
IMAGE_TOKEN_INDEX = -200

def load_video_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """Load frames from video"""
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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of frames')
    parser.add_argument('--output_dir', default='/home/user/tiktok_production/aurora_cap/output')
    args = parser.parse_args()
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor, AutoModel
        from huggingface_hub import snapshot_download
        
        print("Loading AuroraCap model...")
        
        # Model path
        model_id = "wchai/AuroraCap-7B-VID-xtuner"
        model_path = snapshot_download(repo_id=model_id)
        
        # Fix config if needed
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if 'rope_scaling' in config and isinstance(config['rope_scaling'], dict):
            config['rope_scaling'] = {
                'type': config['rope_scaling'].get('type', 'linear'),
                'factor': float(config['rope_scaling'].get('factor', 4.0))
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load LLM
        print("Loading language model...")
        llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Load visual encoder
        print("Loading visual encoder...")
        visual_encoder = AutoModel.from_pretrained(
            os.path.join(model_path, "visual_encoder"),
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).cuda()
        
        # Load projector
        print("Loading projector...")
        projector = AutoModel.from_pretrained(
            os.path.join(model_path, "projector"),
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).cuda()
        
        # Image processor
        image_processor = CLIPImageProcessor.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            size=378,
            crop_size=378
        )
        
        # Load video
        print(f"Loading video: {args.video_path}")
        frames = load_video_frames(args.video_path, args.num_frames)
        print(f"Loaded {len(frames)} frames")
        
        # Process frames
        pixel_values = image_processor(frames, return_tensors='pt')['pixel_values']
        pixel_values = pixel_values.to(dtype=torch.float16).cuda()
        
        # Extract features
        print("Extracting visual features...")
        with torch.no_grad():
            # Get visual features
            visual_outputs = visual_encoder(pixel_values, output_hidden_states=True)
            
            # Use second-to-last layer, remove CLS token
            if hasattr(visual_outputs, 'hidden_states'):
                visual_features = visual_outputs.hidden_states[-2][:, 1:, :]
            else:
                visual_features = visual_outputs.last_hidden_state[:, 1:, :]
            
            print(f"Visual features shape: {visual_features.shape}")
            
            # Project features
            # Flatten all frames together
            b, n, c = visual_features.shape
            visual_features_flat = visual_features.reshape(1, b * n, c)
            
            print("Projecting features...")
            visual_embeds = projector(visual_features_flat)
            print(f"Projected shape: {visual_embeds.shape}")
            
            # Create prompt
            # Use simple format with one image token per frame
            image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * len(frames))
            
            prompt = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {image_tokens} Provide a detailed description of this video. Include all visual elements, actions, settings, and any text visible. ASSISTANT:"""
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to('cuda')
            
            # Get text embeddings
            text_embeds = llm.get_input_embeddings()(inputs.input_ids)
            print(f"Text embeds shape: {text_embeds.shape}")
            
            # Find where to insert visual embeddings
            # Look for IMAGE_TOKEN_INDEX or insert after "USER:"
            input_ids_list = inputs.input_ids[0].tolist()
            
            # Find insertion points
            image_positions = []
            for i, token_id in enumerate(input_ids_list):
                if token_id == IMAGE_TOKEN_INDEX:
                    image_positions.append(i)
            
            print(f"Found {len(image_positions)} image token positions")
            
            if len(image_positions) == 0:
                # No image tokens found, insert after "USER:"
                # Find "USER:" in the prompt
                user_tokens = tokenizer.encode("USER:", add_special_tokens=False)
                insert_pos = None
                
                for i in range(len(input_ids_list) - len(user_tokens) + 1):
                    if input_ids_list[i:i+len(user_tokens)] == user_tokens:
                        insert_pos = i + len(user_tokens) + 1  # After "USER: "
                        break
                
                if insert_pos is None:
                    insert_pos = 10  # Fallback
                
                print(f"Inserting visual features at position {insert_pos}")
                
                # Insert all visual features at once
                combined_embeds = torch.cat([
                    text_embeds[:, :insert_pos],
                    visual_embeds,
                    text_embeds[:, insert_pos:]
                ], dim=1)
                
                # Update attention mask
                visual_attention = torch.ones(1, visual_embeds.shape[1], dtype=torch.long, device='cuda')
                combined_attention = torch.cat([
                    inputs.attention_mask[:, :insert_pos],
                    visual_attention,
                    inputs.attention_mask[:, insert_pos:]
                ], dim=1)
            else:
                # Replace image tokens with visual features
                # Distribute visual features among image tokens
                features_per_token = visual_embeds.shape[1] // len(image_positions)
                remaining = visual_embeds.shape[1] % len(image_positions)
                
                # Build new embeddings
                new_embeds = []
                last_pos = 0
                
                for i, pos in enumerate(image_positions):
                    # Add text up to this position
                    new_embeds.append(text_embeds[:, last_pos:pos])
                    
                    # Add visual features for this token
                    start_idx = i * features_per_token
                    end_idx = start_idx + features_per_token
                    if i < remaining:
                        end_idx += 1
                    
                    new_embeds.append(visual_embeds[:, start_idx:end_idx])
                    last_pos = pos + 1
                
                # Add remaining text
                new_embeds.append(text_embeds[:, last_pos:])
                
                # Concatenate all
                combined_embeds = torch.cat(new_embeds, dim=1)
                
                # Create attention mask
                combined_attention = torch.ones(1, combined_embeds.shape[1], dtype=torch.long, device='cuda')
            
            print(f"Combined embeds shape: {combined_embeds.shape}")
            
            # Generate
            print("Generating description...")
            outputs = llm.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "ASSISTANT:" in generated:
                description = generated.split("ASSISTANT:")[-1].strip()
            else:
                description = generated.strip()
            
            print(f"\nGenerated description ({len(description)} chars):")
            print("-" * 80)
            print(description[:500] + "..." if len(description) > 500 else description)
            print("-" * 80)
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        video_name = Path(args.video_path).stem
        
        result = {
            'overall_description': description,
            'metadata': {
                'num_frames': len(frames),
                'model': 'AuroraCap-7B-VID',
                'video_path': args.video_path,
                'description_length': len(description)
            }
        }
        
        output_path = os.path.join(args.output_dir, f"{video_name}_auroracap_simple.json")
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()