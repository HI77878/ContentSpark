#!/usr/bin/env python3
"""
AuroraCap inference script for detailed video captioning
Based on the original inference.py but adapted for batch processing
"""
import os
import os.path as osp
import sys
import json
import torch
import time
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image
import numpy as np
from pathlib import Path

# Import AuroraCap modules
from src.xtuner.xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, PROMPT_TEMPLATE
from src.xtuner.xtuner.model.aurora import AuroraEncoder, AuroraModel
from src.xtuner.xtuner.tools.load_video import read_video_pyav

def process_text(inputs, tokenizer):
    """Process text input with image tokens"""
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

def generate_caption_for_segment(model, tokenizer, image_processor, visual_input, start_frame, end_frame, 
                                prompt="Describe this video segment in extreme detail.", 
                                token_kept_ratio=0.3, temperature=0.0, top_p=1.0, 
                                num_beams=1, max_new_tokens=300):
    """Generate detailed caption for a video segment"""
    
    # Sample frames from segment
    num_frames = min(8, end_frame - start_frame)  # Max 8 frames per segment
    if num_frames <= 0:
        return "No frames in segment"
    
    # Read video frames
    container, sample_fps = read_video_pyav(visual_input, num_frm=num_frames, 
                                           start_time=start_frame/30.0,  # Assuming 30fps
                                           end_time=end_frame/30.0)
    
    # Process frames
    video = [image_processor.preprocess(frame, return_tensors='pt')['pixel_values'][0] for frame in container]
    pixel_values = torch.stack(video).cuda()  # [num_frames, 3, H, W]
    num_patches = pixel_values.shape[0]  # Number of frames
    
    # Prepare text input
    question = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    template = PROMPT_TEMPLATE['internlm2_chat']
    question = template['INSTRUCTION'].format(input=question)
    
    # Tokenize
    input_ids = process_text(question, tokenizer)
    
    # Generate caption
    with torch.cuda.amp.autocast():
        generation_output = model.generate(
            pixel_values=pixel_values.to(dtype=torch.float16),
            input_ids=input_ids,
            do_sample=temperature > 0.0,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            num_patches_list=[num_patches],
            token_kept_ratio=token_kept_ratio,
        )
    
    # Decode output
    response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
    response = response.split(template['ASSISTANT'])[-1].strip()
    
    return response

def main():
    # Configuration
    model_path = os.environ.get('AURORACAP_MODEL', 'wchai/AuroraCap-7B-VID-xtuner')
    video_path = sys.argv[1] if len(sys.argv) > 1 else '/app/videos/input.mp4'
    output_dir = '/app/output'
    
    # Detailed prompt for video description
    detailed_prompt = """Describe this video segment in extreme detail. Include:
- All visible objects and their precise locations
- Every action and movement occurring
- Facial expressions and emotions of people
- Visual effects, transitions, or camera movements
- Colors, lighting, and atmosphere
- Any text overlays or graphics
- Background elements and setting details
Be specific and comprehensive."""
    
    print(f"Loading AuroraCap model: {model_path}")
    
    # Download and setup model paths
    pretrained_pth = snapshot_download(repo_id=model_path) if not osp.isdir(model_path) else model_path
    pretrained_vit = osp.join(pretrained_pth, "visual_encoder")
    projector_path = osp.join(pretrained_pth, "projector")
    
    # Initialize model
    auroracap = AuroraModel(
        llm=AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_pth,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map='cuda',
        ),
        visual_encoder=AuroraEncoder(
            encoder=AutoModel.from_pretrained(
                pretrained_model_name_or_path=pretrained_vit,
                trust_remote_code=True,
                device_map='cuda',
            ).model,
            projector_path=projector_path
        ),
        freeze_llm=True,
        freeze_visual_encoder=True,
    )
    auroracap = auroracap.eval().cuda()
    
    # Load tokenizer and image processor
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_pth,
        trust_remote_code=True,
    )
    image_processor = CLIPImageProcessor.from_pretrained(pretrained_vit)
    
    print(f"Processing video: {video_path}")
    
    # Get video info
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    print(f"Video info: {duration:.1f}s, {total_frames} frames, {fps:.1f} fps")
    
    # Process video in 1-second segments
    segments = []
    segment_duration = 1.0  # 1 second per segment
    
    start_time = time.time()
    
    for seg_idx in range(int(duration)):
        start_frame = int(seg_idx * fps)
        end_frame = min(int((seg_idx + 1) * fps), total_frames)
        
        print(f"\nProcessing segment {seg_idx + 1}/{int(duration)} [{seg_idx}s - {seg_idx + 1}s]")
        
        try:
            caption = generate_caption_for_segment(
                auroracap, tokenizer, image_processor,
                video_path, start_frame, end_frame,
                prompt=detailed_prompt,
                token_kept_ratio=0.3,  # Keep 30% of tokens for detailed captions
                temperature=0.0,
                max_new_tokens=300
            )
            
            segments.append({
                "timestamp": float(seg_idx),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "description": caption
            })
            
            print(f"  Generated: {caption[:100]}...")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            segments.append({
                "timestamp": float(seg_idx),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "description": f"Error processing segment: {str(e)}"
            })
    
    # Calculate analysis time
    analysis_time = time.time() - start_time
    
    # Save results
    output_data = {
        "video_path": video_path,
        "model": model_path,
        "duration": duration,
        "fps": fps,
        "total_frames": total_frames,
        "analysis_time": analysis_time,
        "segments": segments
    }
    
    video_name = Path(video_path).stem
    output_path = osp.join(output_dir, f"{video_name}_auroracap_detailed.json")
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Analysis complete!")
    print(f"   Processed {len(segments)} segments in {analysis_time:.1f}s")
    print(f"   Results saved to: {output_path}")

if __name__ == "__main__":
    main()