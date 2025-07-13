#!/usr/bin/env python3
import os
import sys
import json
import torch
import argparse
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# Simple constants
DEFAULT_IMAGE_TOKEN = '<image>'
IMAGE_TOKEN_INDEX = -200

def load_video_frames(video_path, num_frames=8):
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
    parser.add_argument('--output', help='Output JSON path', required=True)
    parser.add_argument('--num_frames', type=int, default=8)
    args = parser.parse_args()
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor, AutoModel
        from huggingface_hub import snapshot_download
        
        # Model path
        model_id = "wchai/AuroraCap-7B-VID-xtuner"
        model_path = snapshot_download(repo_id=model_id)
        
        # Load components
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        visual_encoder = AutoModel.from_pretrained(
            os.path.join(model_path, "visual_encoder"),
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).cuda()
        
        projector = AutoModel.from_pretrained(
            os.path.join(model_path, "projector"),
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).cuda()
        
        image_processor = CLIPImageProcessor.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            size=378,
            crop_size=378
        )
        
        # Load video
        frames = load_video_frames(args.video_path, args.num_frames)
        
        # Process frames
        pixel_values = image_processor(frames, return_tensors='pt')['pixel_values']
        pixel_values = pixel_values.to(dtype=torch.float16).cuda()
        
        # Extract features
        with torch.no_grad():
            visual_outputs = visual_encoder(pixel_values, output_hidden_states=True)
            visual_features = visual_outputs.hidden_states[-2][:, 1:]  # Remove CLS
            
            # Project
            b = 1
            f = len(frames)
            n = visual_features.shape[1]
            c = visual_features.shape[2]
            
            visual_features = visual_features.reshape(b, f * n, c)
            visual_embeds = projector(visual_features)
            
            # Create prompt
            prompt = "A chat between a curious user and an artificial intelligence assistant. "
            prompt += "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            prompt += f"USER: {DEFAULT_IMAGE_TOKEN} "
            prompt += "Provide a detailed description of this video, including all visual elements, "
            prompt += "actions, settings, and any text visible. ASSISTANT: "
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
            
            # Replace image token with visual embeddings
            input_ids = inputs.input_ids
            image_token_mask = input_ids == IMAGE_TOKEN_INDEX
            
            # If no image token found, insert embeddings at beginning
            if not image_token_mask.any():
                # Find where to insert (after "USER: ")
                user_pos = (input_ids[0] == tokenizer.encode("USER:", add_special_tokens=False)[0]).nonzero(as_tuple=True)[0]
                if len(user_pos) > 0:
                    insert_pos = user_pos[0].item() + 2  # After "USER: "
                else:
                    insert_pos = 10  # Fallback position
                
                # Get embeddings
                text_embeds = llm.get_input_embeddings()(input_ids)
                
                # Insert visual embeddings
                combined_embeds = torch.cat([
                    text_embeds[:, :insert_pos],
                    visual_embeds,
                    text_embeds[:, insert_pos:]
                ], dim=1)
                
                # Create attention mask
                visual_attention = torch.ones(1, visual_embeds.shape[1], dtype=torch.long, device='cuda')
                combined_attention = torch.cat([
                    inputs.attention_mask[:, :insert_pos],
                    visual_attention,
                    inputs.attention_mask[:, insert_pos:]
                ], dim=1)
            else:
                # Standard processing with image token
                text_embeds = llm.get_input_embeddings()(input_ids)
                combined_embeds = text_embeds  # Simplified for now
                combined_attention = inputs.attention_mask
            
            # Generate
            outputs = llm.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
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
        
        # Create result
        result = {
            'overall_description': description,
            'segments': [{
                'timestamp': 0.0,
                'end_timestamp': len(frames) / 30.0,  # Assume 30fps
                'description': description
            }],
            'metadata': {
                'num_frames': len(frames),
                'model': 'AuroraCap-7B-VID'
            }
        }
        
        # Save result
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
            
        print("Success")
        
    except Exception as e:
        # Save error result
        error_result = {
            'overall_description': f'Error: {str(e)}',
            'segments': [],
            'metadata': {'error': str(e)}
        }
        with open(args.output, 'w') as f:
            json.dump(error_result, f, indent=2)
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
