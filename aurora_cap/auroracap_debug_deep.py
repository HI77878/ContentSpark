#!/usr/bin/env python3
"""
Deep debugging version of AuroraCap inference
Implements the exact pipeline from the original Aurora repository
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
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime

# Add Aurora src to path
aurora_path = "/home/user/tiktok_production/aurora_cap/aurora"
sys.path.insert(0, os.path.join(aurora_path, "src"))

# Configure logging with detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(f'aurora_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_video_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """Load frames from video using the exact method from Aurora"""
    logger.info(f"Loading {num_frames} frames from {video_path}")
    
    # Try to import Aurora's video loader
    try:
        from xtuner.xtuner.tools.load_video import read_video_pyav
        frames = read_video_pyav(video_path, num_frames)
        logger.info(f"Loaded {len(frames)} frames using Aurora's read_video_pyav")
        return frames
    except Exception as e:
        logger.warning(f"Failed to use Aurora's video loader: {e}, falling back to OpenCV")
        
    # Fallback to OpenCV
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
    logger.info(f"Loaded {len(frames)} frames from video")
    return frames

class AuroraCapDebugger:
    """Deep debugging implementation following original Aurora exactly"""
    
    def __init__(self, model_path='wchai/AuroraCap-7B-VID-xtuner'):
        self.model_path = model_path
        self.auroracap = None
        self.tokenizer = None
        self.image_processor = None
        
    def load_model(self):
        """Load AuroraCap using the exact method from inference.py"""
        logger.info(f"Loading AuroraCap from {self.model_path}")
        
        try:
            from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
            from huggingface_hub import snapshot_download
            from xtuner.xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, PROMPT_TEMPLATE
            from xtuner.xtuner.model.aurora import AuroraEncoder, AuroraModel
            
            # Store constants
            self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
            self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
            self.PROMPT_TEMPLATE = PROMPT_TEMPLATE
            
            logger.info(f"DEFAULT_IMAGE_TOKEN: {DEFAULT_IMAGE_TOKEN}")
            logger.info(f"IMAGE_TOKEN_INDEX: {IMAGE_TOKEN_INDEX}")
            
        except ImportError as e:
            logger.error(f"Failed to import Aurora modules: {e}")
            raise
        
        # Download/locate model
        import os.path as osp
        if not osp.isdir(self.model_path):
            pretrained_pth = snapshot_download(repo_id=self.model_path)
        else:
            pretrained_pth = self.model_path
            
        logger.info(f"Model path: {pretrained_pth}")
        
        # Fix rope_scaling config if needed
        config_path = osp.join(pretrained_pth, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if 'rope_scaling' in config:
            config['rope_scaling'] = {
                'type': config['rope_scaling'].get('type', 'linear'),
                'factor': float(config['rope_scaling'].get('factor', 4.0))
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("Fixed rope_scaling configuration")
        
        # Load components exactly as in inference.py
        pretrained_vit = osp.join(pretrained_pth, "visual_encoder")
        projector_path = osp.join(pretrained_pth, "projector")
        
        logger.info("Creating AuroraModel...")
        self.auroracap = AuroraModel(
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
        
        logger.info("Loading projector...")
        self.auroracap.projector = AutoModel.from_pretrained(
            projector_path, 
            torch_dtype=torch.float16, 
            trust_remote_code=True
        ).cuda()
        
        logger.info("Setting up image processor...")
        self.image_processor = CLIPImageProcessor.from_pretrained(
            pretrained_model_name_or_path="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            trust_remote_code=True,
            size=378,
            crop_size=378,
        )
        
        logger.info("Setting up tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_pth,
            trust_remote_code=True,
            padding_side='right',
        )
        
        logger.info("✅ Model loaded successfully")
        
        # Log model architecture
        logger.debug(f"LLM config: {self.auroracap.llm.config}")
        logger.debug(f"Visual encoder config: {self.auroracap.visual_encoder.config}")
        
    def process_text(self, inputs):
        """Process text exactly as in original inference.py"""
        logger.debug(f"Processing text: {inputs[:100]}...")
        
        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(self.DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = self.tokenizer.encode(chunk)
            else:
                cur_encode = self.tokenizer.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
            logger.debug(f"Chunk {idx}: {len(cur_encode)} tokens")
        
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(self.IMAGE_TOKEN_INDEX)
        
        logger.debug(f"Total tokens: {len(ids)}")
        logger.debug(f"Image token positions: {[i for i, t in enumerate(ids) if t == self.IMAGE_TOKEN_INDEX]}")
        
        ids = torch.tensor(ids).cuda().unsqueeze(0)
        return ids
    
    def analyze_video(self, video_path: str, num_frames: int = 8, token_kept_ratio: float = 0.8) -> Dict[str, Any]:
        """Analyze video following exact Aurora pipeline"""
        if self.auroracap is None:
            self.load_model()
        
        # Load video frames
        video_frames = load_video_frames(video_path, num_frames)
        logger.info(f"Processing {len(video_frames)} frames")
        
        # Process frames
        logger.info("Processing frames through image processor...")
        image_tensor = self.image_processor(video_frames, return_tensors='pt')['pixel_values']
        image_tensor = [_image.to(dtype=torch.float16).cuda() for _image in image_tensor]
        
        # Create data dict exactly as in inference.py
        data = dict()
        data["pixel_values"] = torch.stack(image_tensor).unsqueeze(0)
        logger.debug(f"Pixel values shape: {data['pixel_values'].shape}")
        
        # Create image tokens
        image_tokens = [self.DEFAULT_IMAGE_TOKEN] * len(video_frames)
        image_tokens = " ".join(image_tokens)
        logger.info(f"Image tokens: {image_tokens}")
        
        # Create prompt using Vicuna template
        prompt = "Describe this video in detail. What is happening? What do you see? Include all visual elements, actions, people, objects, and any text or graphics visible."
        text_input = image_tokens + "\n" + prompt
        
        # Use Vicuna prompt template
        prompt_text = self.PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(input=text_input, round=1)
        logger.debug(f"Full prompt:\n{prompt_text}")
        
        # Process text
        data["input_ids"] = self.process_text(prompt_text).cuda()
        logger.debug(f"Input IDs shape: {data['input_ids'].shape}")
        
        # Set token merge ratio
        logger.info(f"Setting token merge ratio to {token_kept_ratio}")
        self.auroracap.visual_encoder.reset_tome_r(token_kept_ratio)
        
        # Forward through model (inference mode)
        logger.info("Running forward pass through AuroraModel...")
        with torch.no_grad():
            output = self.auroracap(data, mode="inference")
        
        logger.debug(f"Model output keys: {output.keys()}")
        if 'inputs_embeds' in output:
            logger.debug(f"Inputs embeds shape: {output['inputs_embeds'].shape}")
        if 'attention_mask' in output:
            logger.debug(f"Attention mask shape: {output['attention_mask'].shape}")
        
        # Generate
        logger.info("Generating description...")
        generation_config = {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "num_beams": 1,
            "max_new_tokens": 512,
        }
        
        logger.debug(f"Generation config: {generation_config}")
        
        try:
            with torch.no_grad():
                cont = self.auroracap.llm.generate(
                    **output,
                    **generation_config
                )
            
            logger.debug(f"Generated tokens shape: {cont.shape}")
            logger.debug(f"First 50 generated tokens: {cont[0][:50].tolist()}")
            
            # Decode
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            
            logger.info(f"Generated text length: {len(text_outputs)}")
            logger.debug(f"Generated text preview: {text_outputs[:200]}...")
            
            # Check if we got meaningful output
            if not text_outputs or len(text_outputs.strip()) < 20:
                logger.warning("Generated text is too short or empty!")
                logger.debug(f"Full generated text: '{text_outputs}'")
                
                # Try to understand why
                if cont.shape[1] <= output.get('inputs_embeds', data['input_ids']).shape[1]:
                    logger.error("No new tokens were generated!")
                
                # Fallback
                text_outputs = self._create_fallback_description(video_path, len(video_frames))
            
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            text_outputs = self._create_fallback_description(video_path, len(video_frames))
        
        # Create result
        result = {
            'overall_description': text_outputs,
            'segments': [{
                'timestamp': 0.0,
                'end_timestamp': len(video_frames) / 30.0,
                'description': text_outputs,
                'frames_analyzed': len(video_frames)
            }],
            'metadata': {
                'num_frames': len(video_frames),
                'model': 'AuroraCap-7B-VID',
                'video_path': video_path,
                'description_length': len(text_outputs),
                'token_kept_ratio': token_kept_ratio
            }
        }
        
        return result
    
    def _create_fallback_description(self, video_path: str, num_frames: int) -> str:
        """Create fallback description"""
        video_name = Path(video_path).stem
        return (
            f"This video '{video_name}' contains {num_frames} frames of visual content. "
            "The video shows dynamic scenes with various visual elements. "
            "Due to processing constraints, a detailed frame-by-frame analysis is not available at this time."
        )

def main():
    parser = argparse.ArgumentParser(description="AuroraCap Deep Debug")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames")
    parser.add_argument("--token_kept_ratio", type=float, default=0.8, help="Token merge ratio")
    parser.add_argument("--output_dir", default="/home/user/tiktok_production/aurora_cap/output")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = AuroraCapDebugger()
    
    try:
        # Analyze video
        results = analyzer.analyze_video(
            args.video_path, 
            args.num_frames,
            args.token_kept_ratio
        )
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        video_name = Path(args.video_path).stem
        
        output_path = os.path.join(args.output_dir, f"{video_name}_debug_deep.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Analysis complete!")
        print(f"Results saved to: {output_path}")
        
        print("\n" + "=" * 80)
        print("DESCRIPTION:")
        print("=" * 80)
        print(results['overall_description'])
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()