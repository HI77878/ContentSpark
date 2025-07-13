#!/usr/bin/env python3
"""
Deep debugging version for AuroraCap generation pipeline
Focus on identifying why LLM generates 0 tokens
"""
import os
import sys
import json
import torch
import torch.nn as nn
import argparse
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime

# Configure extremely detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(f'aurora_generation_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    logger.info(f"Loaded {len(frames)} frames from video")
    return frames

class AuroraCapGenerationDebugger:
    """Deep debugging for Aurora generation issues"""
    
    def __init__(self, model_path='wchai/AuroraCap-7B-VID-xtuner'):
        self.model_path = model_path
        self.llm = None
        self.visual_encoder = None
        self.projector = None
        self.tokenizer = None
        self.image_processor = None
        self.image_token_id = None
        
    def load_model(self):
        """Load AuroraCap model components with detailed logging"""
        logger.info(f"Loading AuroraCap from {self.model_path}")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor, AutoModel
        from huggingface_hub import snapshot_download
        import os.path as osp
        
        # Download/locate model
        if not osp.isdir(self.model_path):
            pretrained_pth = snapshot_download(repo_id=self.model_path)
        else:
            pretrained_pth = self.model_path
            
        # Fix rope_scaling
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
        
        # Load tokenizer with detailed checks
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_pth,
            trust_remote_code=True,
            padding_side='right',
        )
        
        # Check and add image token
        original_vocab_size = len(self.tokenizer)
        if DEFAULT_IMAGE_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN])
            logger.info(f"Added {DEFAULT_IMAGE_TOKEN} to tokenizer")
        
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        logger.info(f"Image token ID: {self.image_token_id}")
        logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info(f"Pad token: '{self.tokenizer.pad_token}' (ID: {self.tokenizer.pad_token_id})")
        logger.info(f"EOS token: '{self.tokenizer.eos_token}' (ID: {self.tokenizer.eos_token_id})")
        logger.info(f"BOS token: '{self.tokenizer.bos_token}' (ID: {self.tokenizer.bos_token_id})")
        
        # Load LLM
        logger.info("Loading Vicuna LLM...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            pretrained_pth,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Resize embeddings if we added tokens
        if len(self.tokenizer) > original_vocab_size:
            self.llm.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Resized LLM embeddings from {original_vocab_size} to {len(self.tokenizer)}")
        
        self.llm.eval()
        
        # Log model configuration
        logger.debug(f"LLM config: {self.llm.config}")
        logger.debug(f"Model dtype: {self.llm.dtype}")
        logger.debug(f"Model device: {next(self.llm.parameters()).device}")
        
        # Load visual encoder
        logger.info("Loading visual encoder...")
        visual_encoder_path = osp.join(pretrained_pth, "visual_encoder")
        self.visual_encoder = AutoModel.from_pretrained(
            visual_encoder_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).cuda()
        self.visual_encoder.eval()
        
        # Load projector
        logger.info("Loading projector...")
        projector_path = osp.join(pretrained_pth, "projector")
        self.projector = AutoModel.from_pretrained(
            projector_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).cuda()
        self.projector.eval()
        
        # Initialize image processor
        self.image_processor = CLIPImageProcessor.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            size=378,
            crop_size=378,
        )
        
        logger.info("✅ Model loaded successfully")
    
    def debug_generation_step(
        self, 
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> str:
        """Debug the generation step with extreme detail"""
        logger.info("=" * 80)
        logger.info("DEBUGGING GENERATION STEP")
        logger.info("=" * 80)
        
        # Log input details
        logger.debug(f"inputs_embeds shape: {inputs_embeds.shape}")
        logger.debug(f"inputs_embeds dtype: {inputs_embeds.dtype}")
        logger.debug(f"inputs_embeds device: {inputs_embeds.device}")
        logger.debug(f"inputs_embeds min/max: {inputs_embeds.min():.4f} / {inputs_embeds.max():.4f}")
        logger.debug(f"inputs_embeds has NaN: {torch.isnan(inputs_embeds).any()}")
        logger.debug(f"inputs_embeds has Inf: {torch.isinf(inputs_embeds).any()}")
        
        logger.debug(f"attention_mask shape: {attention_mask.shape}")
        logger.debug(f"attention_mask sum: {attention_mask.sum()}")
        logger.debug(f"attention_mask dtype: {attention_mask.dtype}")
        
        # Try different generation configurations
        generation_configs = [
            {
                'name': 'Greedy',
                'params': {
                    'max_new_tokens': 100,
                    'do_sample': False,
                    'num_beams': 1,
                }
            },
            {
                'name': 'Sampling with low temperature',
                'params': {
                    'max_new_tokens': 100,
                    'do_sample': True,
                    'temperature': 0.1,
                    'top_p': 0.95,
                }
            },
            {
                'name': 'Beam search',
                'params': {
                    'max_new_tokens': 100,
                    'num_beams': 4,
                    'early_stopping': True,
                }
            },
            {
                'name': 'Force output',
                'params': {
                    'max_new_tokens': 50,
                    'min_new_tokens': 10,
                    'do_sample': True,
                    'temperature': 0.7,
                    'repetition_penalty': 1.2,
                }
            }
        ]
        
        results = []
        
        for config in generation_configs:
            logger.info(f"\nTrying generation config: {config['name']}")
            logger.debug(f"Parameters: {config['params']}")
            
            try:
                with torch.no_grad():
                    # Add common parameters
                    gen_params = {
                        **config['params'],
                        'inputs_embeds': inputs_embeds,
                        'attention_mask': attention_mask,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'eos_token_id': self.tokenizer.eos_token_id,
                        'bos_token_id': self.tokenizer.bos_token_id,
                        'return_dict_in_generate': True,
                        'output_scores': True,
                    }
                    
                    # Generate
                    outputs = self.llm.generate(**gen_params)
                    
                    # Log generation details
                    logger.debug(f"Generated sequences shape: {outputs.sequences.shape}")
                    logger.debug(f"Number of new tokens: {outputs.sequences.shape[1] - inputs_embeds.shape[1]}")
                    
                    # Decode
                    generated_ids = outputs.sequences[0]
                    input_length = inputs_embeds.shape[1]
                    new_tokens = generated_ids[input_length:]
                    
                    logger.debug(f"New token IDs: {new_tokens.tolist()[:20]}...")  # First 20 tokens
                    
                    # Decode text
                    generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    logger.info(f"Generated text length: {len(generated_text)}")
                    logger.info(f"Generated text: '{generated_text[:200]}...'")
                    
                    # Check if we have meaningful output
                    if len(generated_text.strip()) > 10:
                        return generated_text
                    
                    results.append({
                        'config': config['name'],
                        'text': generated_text,
                        'num_tokens': len(new_tokens)
                    })
                    
            except Exception as e:
                logger.error(f"Generation failed with config '{config['name']}': {e}", exc_info=True)
                results.append({
                    'config': config['name'],
                    'error': str(e)
                })
        
        # Log all results
        logger.info("\n" + "=" * 80)
        logger.info("GENERATION RESULTS SUMMARY:")
        for result in results:
            logger.info(f"Config: {result.get('config')}")
            if 'error' in result:
                logger.info(f"  Error: {result['error']}")
            else:
                logger.info(f"  Tokens: {result.get('num_tokens', 0)}")
                logger.info(f"  Text: '{result.get('text', '')[:100]}'")
        
        return ""  # All attempts failed
    
    def prepare_multimodal_inputs_debug(
        self,
        input_ids: torch.Tensor,
        visual_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Prepare multimodal inputs with detailed debugging"""
        logger.info("Preparing multimodal inputs...")
        
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Get text embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        logger.debug(f"Text embeddings shape: {inputs_embeds.shape}")
        
        # Find image token positions
        image_token_mask = (input_ids == self.image_token_id)
        num_image_tokens = image_token_mask.sum().item()
        logger.info(f"Found {num_image_tokens} image tokens")
        
        if num_image_tokens == 0:
            logger.warning("No image tokens found! Using alternative approach...")
            # Try to find <image> in the decoded text
            decoded = self.tokenizer.decode(input_ids[0])
            logger.debug(f"Decoded prompt: {decoded[:200]}...")
            
            # Insert visual features at the beginning
            inputs_embeds = torch.cat([
                visual_embeds,
                inputs_embeds
            ], dim=1)
            
            if attention_mask is not None:
                visual_attention = torch.ones(
                    batch_size, visual_embeds.shape[1], 
                    dtype=attention_mask.dtype, device=device
                )
                attention_mask = torch.cat([visual_attention, attention_mask], dim=1)
            
        else:
            # Replace image tokens with visual embeddings
            image_positions = torch.where(image_token_mask[0])[0]
            logger.debug(f"Image token positions: {image_positions.tolist()}")
            
            # Build new embeddings
            new_embeds = []
            last_pos = 0
            
            for i, pos in enumerate(image_positions):
                pos = pos.item()
                
                # Add text before image
                if pos > last_pos:
                    new_embeds.append(inputs_embeds[0, last_pos:pos])
                
                # Add visual features
                patches_per_token = visual_embeds.shape[1] // num_image_tokens
                start_idx = i * patches_per_token
                end_idx = (i + 1) * patches_per_token if i < num_image_tokens - 1 else visual_embeds.shape[1]
                
                new_embeds.append(visual_embeds[0, start_idx:end_idx])
                last_pos = pos + 1
            
            # Add remaining text
            if last_pos < inputs_embeds.shape[1]:
                new_embeds.append(inputs_embeds[0, last_pos:])
            
            # Concatenate
            inputs_embeds = torch.cat(new_embeds, dim=0).unsqueeze(0)
            
            # Update attention mask
            if attention_mask is not None:
                new_length = inputs_embeds.shape[1]
                attention_mask = torch.ones(batch_size, new_length, dtype=attention_mask.dtype, device=device)
        
        logger.debug(f"Final inputs_embeds shape: {inputs_embeds.shape}")
        logger.debug(f"Final attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
        
        return {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask
        }
    
    def analyze_video_debug(self, video_path: str, num_frames: int = 4) -> Dict[str, Any]:
        """Main analysis function with debugging"""
        if self.llm is None:
            self.load_model()
        
        # Load video frames
        frames = load_video_frames(video_path, num_frames)
        
        # Process frames
        logger.info("Processing frames through visual pipeline...")
        pixel_values = self.image_processor(frames, return_tensors='pt')['pixel_values']
        pixel_values = pixel_values.to(dtype=torch.float16).cuda()
        
        # Extract visual features
        with torch.no_grad():
            visual_outputs = self.visual_encoder(pixel_values, output_hidden_states=True)
            visual_features = visual_outputs.hidden_states[-2][:, 1:, :]  # Remove CLS
            logger.debug(f"Visual features shape: {visual_features.shape}")
            
            # Flatten all features
            visual_features = visual_features.reshape(1, -1, visual_features.shape[-1])
            
            # Project
            visual_embeds = self.projector(visual_features)
            logger.debug(f"Projected features shape: {visual_embeds.shape}")
        
        # Try different prompts
        prompts = [
            f"{DEFAULT_IMAGE_TOKEN} " * len(frames) + "Describe this video.",
            f"USER: {DEFAULT_IMAGE_TOKEN} " * len(frames) + "What do you see in this video? ASSISTANT:",
            f"A chat between a curious user and an artificial intelligence assistant. USER: {DEFAULT_IMAGE_TOKEN} " * len(frames) + "Describe the video. ASSISTANT:",
            "Describe the following video frames in detail:",
        ]
        
        for i, prompt in enumerate(prompts):
            logger.info(f"\n{'='*80}")
            logger.info(f"TRYING PROMPT {i+1}/{len(prompts)}")
            logger.info(f"Prompt: {prompt}")
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.cuda()
            attention_mask = inputs.attention_mask.cuda()
            
            logger.debug(f"Tokenized length: {input_ids.shape[1]}")
            logger.debug(f"First 20 token IDs: {input_ids[0][:20].tolist()}")
            
            # Prepare multimodal inputs
            multimodal_inputs = self.prepare_multimodal_inputs_debug(
                input_ids=input_ids,
                visual_embeds=visual_embeds,
                attention_mask=attention_mask
            )
            
            # Try generation
            generated_text = self.debug_generation_step(
                multimodal_inputs['inputs_embeds'],
                multimodal_inputs['attention_mask']
            )
            
            if generated_text and len(generated_text.strip()) > 20:
                logger.info(f"SUCCESS! Generated meaningful text with prompt {i+1}")
                return {
                    'overall_description': generated_text.strip(),
                    'debug_info': {
                        'prompt_used': prompt,
                        'prompt_index': i,
                        'num_frames': len(frames),
                        'visual_features_shape': str(visual_features.shape),
                        'description_length': len(generated_text)
                    }
                }
        
        # All attempts failed
        logger.error("All generation attempts failed!")
        return {
            'overall_description': f"Failed to generate description for video with {len(frames)} frames",
            'debug_info': {
                'status': 'failed',
                'num_frames': len(frames),
                'attempts': len(prompts)
            }
        }

def main():
    parser = argparse.ArgumentParser(description="AuroraCap Generation Debugger")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--num_frames", type=int, default=4, help="Number of frames")
    parser.add_argument("--output_dir", default="/home/user/tiktok_production/aurora_cap/output")
    
    args = parser.parse_args()
    
    # Create debugger
    debugger = AuroraCapGenerationDebugger()
    
    try:
        # Run debug analysis
        results = debugger.analyze_video_debug(args.video_path, args.num_frames)
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        video_name = Path(args.video_path).stem
        
        output_path = os.path.join(args.output_dir, f"{video_name}_debug_generation.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print("FINAL RESULT:")
        print(f"{'='*80}")
        print(f"Description: {results['overall_description']}")
        print(f"\nDebug info saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()