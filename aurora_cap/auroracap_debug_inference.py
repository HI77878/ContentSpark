#!/usr/bin/env python3
"""
Debug version of AuroraCap inference with extensive logging
Identifies and fixes the issue with empty descriptions
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
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'/home/user/tiktok_production/aurora_cap/debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_IMAGE_TOKEN = '<image>'
IMAGE_TOKEN_INDEX = -200

def load_video_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """Load frames from video with logging"""
    logger.info(f"Loading video from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Video info: {total_frames} frames, {fps} fps")
    
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
            logger.debug(f"Loaded frame {len(frames)} at position {idx}")
    
    cap.release()
    logger.info(f"Successfully loaded {len(frames)} frames")
    return frames

class DebugAuroraCapInference:
    """Debug version with extensive logging"""
    
    def __init__(self, model_path='wchai/AuroraCap-7B-VID-xtuner'):
        self.model_path = model_path
        self.llm = None
        self.visual_encoder = None
        self.projector = None
        self.tokenizer = None
        self.image_processor = None
        
    def load_model(self):
        """Load AuroraCap model with debug logging"""
        logger.info(f"Starting model loading from {self.model_path}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor, AutoModel
            from huggingface_hub import snapshot_download
            
            # Download/locate model
            if not os.path.isdir(self.model_path):
                logger.info("Downloading model from HuggingFace...")
                pretrained_pth = snapshot_download(repo_id=self.model_path)
            else:
                pretrained_pth = self.model_path
                
            logger.info(f"Model path: {pretrained_pth}")
            
            # Fix rope_scaling
            config_path = os.path.join(pretrained_pth, "config.json")
            self._fix_rope_scaling(config_path)
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_pth,
                trust_remote_code=True,
                padding_side='right',
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            # Check special tokens
            logger.info(f"Tokenizer special tokens:")
            logger.info(f"  pad_token: {self.tokenizer.pad_token} (id: {self.tokenizer.pad_token_id})")
            logger.info(f"  eos_token: {self.tokenizer.eos_token} (id: {self.tokenizer.eos_token_id})")
            logger.info(f"  vocab_size: {self.tokenizer.vocab_size}")
            
            # Check if image token exists
            image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
            logger.info(f"Image token '{DEFAULT_IMAGE_TOKEN}' id: {image_token_id}")
            
            # Load LLM
            logger.info("Loading language model...")
            self.llm = AutoModelForCausalLM.from_pretrained(
                pretrained_pth,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.llm.eval()
            logger.info(f"LLM loaded, config: {self.llm.config}")
            logger.info(f"LLM hidden size: {self.llm.config.hidden_size}")
            
            # Load visual encoder
            visual_encoder_path = os.path.join(pretrained_pth, "visual_encoder")
            logger.info(f"Loading visual encoder from {visual_encoder_path}")
            
            self.visual_encoder = AutoModel.from_pretrained(
                visual_encoder_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).cuda()
            self.visual_encoder.eval()
            
            # Check visual encoder config
            if hasattr(self.visual_encoder, 'config'):
                logger.info(f"Visual encoder config: {self.visual_encoder.config}")
                logger.info(f"Visual encoder hidden size: {getattr(self.visual_encoder.config, 'hidden_size', 'unknown')}")
            
            # Load projector
            projector_path = os.path.join(pretrained_pth, "projector")
            logger.info(f"Loading projector from {projector_path}")
            
            self.projector = AutoModel.from_pretrained(
                projector_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).cuda()
            self.projector.eval()
            
            # Test projector dimensions
            logger.info("Testing projector dimensions...")
            with torch.no_grad():
                test_input = torch.randn(1, 1, 1280, dtype=torch.float16).cuda()  # CLIP hidden size
                test_output = self.projector(test_input)
                logger.info(f"Projector test: input {test_input.shape} -> output {test_output.shape}")
            
            # Initialize image processor
            logger.info("Loading image processor...")
            self.image_processor = CLIPImageProcessor.from_pretrained(
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                size=378,
                crop_size=378,
            )
            
            logger.info("✅ All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise
    
    def _fix_rope_scaling(self, config_path):
        """Fix rope_scaling configuration"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if 'rope_scaling' in config:
                if isinstance(config['rope_scaling'], dict):
                    rope_scaling = config['rope_scaling']
                    config['rope_scaling'] = {
                        'type': rope_scaling.get('type', 'linear'),
                        'factor': float(rope_scaling.get('factor', 4.0))
                    }
                    
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    logger.info("Fixed rope_scaling configuration")
        except Exception as e:
            logger.warning(f"Could not fix rope_scaling: {e}")
    
    def process_frames_to_features(self, frames: List[Image.Image]) -> torch.Tensor:
        """Process frames and extract visual features with detailed logging"""
        logger.info(f"Processing {len(frames)} frames...")
        
        # Process frames with image processor
        pixel_values = self.image_processor(frames, return_tensors='pt')['pixel_values']
        pixel_values = pixel_values.to(dtype=torch.float16).cuda()
        logger.info(f"Pixel values shape: {pixel_values.shape}")
        
        # Extract visual features
        with torch.no_grad():
            logger.info("Extracting visual features...")
            visual_outputs = self.visual_encoder(pixel_values, output_hidden_states=True)
            
            # Log all available outputs
            if hasattr(visual_outputs, 'hidden_states'):
                logger.info(f"Visual encoder has {len(visual_outputs.hidden_states)} hidden states")
                for i, hs in enumerate(visual_outputs.hidden_states):
                    if hs is not None:
                        logger.debug(f"  Hidden state {i}: shape {hs.shape}")
            
            # Get features from second-to-last layer
            if hasattr(visual_outputs, 'hidden_states') and len(visual_outputs.hidden_states) > 1:
                visual_features = visual_outputs.hidden_states[-2]
                logger.info(f"Using hidden state -2 with shape: {visual_features.shape}")
            else:
                visual_features = visual_outputs.last_hidden_state
                logger.info(f"Using last_hidden_state with shape: {visual_features.shape}")
            
            # Remove CLS token if present
            if visual_features.shape[1] > 1:
                visual_features = visual_features[:, 1:, :]
                logger.info(f"Removed CLS token, new shape: {visual_features.shape}")
            
            # Project features
            logger.info("Projecting visual features...")
            b, n, c = visual_features.shape
            visual_features_flat = visual_features.reshape(1, b * n, c)
            logger.info(f"Flattened features shape: {visual_features_flat.shape}")
            
            projected_features = self.projector(visual_features_flat)
            logger.info(f"Projected features shape: {projected_features.shape}")
            
            return projected_features
    
    def generate_with_visual_features(self, visual_features: torch.Tensor, prompt: str) -> str:
        """Generate text with visual features - detailed debug version"""
        logger.info("Starting generation with visual features...")
        logger.info(f"Visual features shape: {visual_features.shape}")
        logger.info(f"Prompt: {prompt[:100]}...")
        
        # Tokenize prompt
        tokens = self.tokenizer.tokenize(prompt)
        logger.debug(f"Tokenized prompt ({len(tokens)} tokens): {tokens[:20]}...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs.input_ids.to('cuda')
        logger.info(f"Input IDs shape: {input_ids.shape}")
        logger.debug(f"First 20 input IDs: {input_ids[0][:20].tolist()}")
        
        # Get text embeddings
        with torch.no_grad():
            text_embeds = self.llm.get_input_embeddings()(input_ids)
            logger.info(f"Text embeddings shape: {text_embeds.shape}")
        
        # Find where to insert visual features
        # Look for "USER:" pattern
        user_token_ids = self.tokenizer.encode("USER:", add_special_tokens=False)
        logger.info(f"USER: token IDs: {user_token_ids}")
        
        # Find position after "USER:"
        input_list = input_ids[0].tolist()
        insert_pos = None
        for i in range(len(input_list) - len(user_token_ids) + 1):
            if input_list[i:i+len(user_token_ids)] == user_token_ids:
                insert_pos = i + len(user_token_ids) + 1  # After "USER: "
                break
        
        if insert_pos is None:
            insert_pos = 20  # Fallback
            logger.warning(f"Could not find USER: pattern, using fallback position {insert_pos}")
        else:
            logger.info(f"Found USER: pattern, inserting at position {insert_pos}")
        
        # Create combined embeddings
        logger.info("Creating combined embeddings...")
        combined_embeds = torch.cat([
            text_embeds[:, :insert_pos],
            visual_features,
            text_embeds[:, insert_pos:]
        ], dim=1)
        logger.info(f"Combined embeddings shape: {combined_embeds.shape}")
        
        # Create attention mask
        visual_attention = torch.ones(1, visual_features.shape[1], dtype=torch.long, device='cuda')
        combined_attention = torch.cat([
            inputs.attention_mask[:, :insert_pos],
            visual_attention,
            inputs.attention_mask[:, insert_pos:]
        ], dim=1)
        logger.info(f"Combined attention mask shape: {combined_attention.shape}")
        
        # Generate
        logger.info("Starting generation...")
        generation_config = {
            'max_new_tokens': 512,
            'temperature': 0.7,
            'do_sample': True,
            'top_p': 0.95,
            'num_beams': 1,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        logger.info(f"Generation config: {generation_config}")
        
        try:
            with torch.no_grad():
                outputs = self.llm.generate(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_attention,
                    **generation_config
                )
            
            logger.info(f"Generated output shape: {outputs.shape}")
            logger.info(f"Generated {outputs.shape[1] - combined_embeds.shape[1]} new tokens")
            
            # Decode full output
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Full decoded output: {full_output[:200]}...")
            
            # Extract assistant response
            if "ASSISTANT:" in full_output:
                response = full_output.split("ASSISTANT:")[-1].strip()
            else:
                response = full_output.strip()
            
            logger.info(f"Extracted response: {response[:200]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            return ""
    
    def analyze_video(self, video_path: str, num_frames: int = 8) -> Dict[str, Any]:
        """Main analysis function with debugging"""
        logger.info(f"\n{'='*80}")
        logger.info(f"ANALYZING VIDEO: {video_path}")
        logger.info(f"{'='*80}\n")
        
        if self.llm is None:
            self.load_model()
        
        # Load video frames
        frames = load_video_frames(video_path, num_frames)
        
        # Extract visual features
        visual_features = self.process_frames_to_features(frames)
        
        # Create prompts and generate
        # Try different prompt formats
        prompts = [
            # Vicuna format
            "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Describe this video in detail, including all visual elements, actions, and events. ASSISTANT:",
            
            # Simple format
            "USER: What is happening in this video? Please provide a detailed description. ASSISTANT:",
            
            # Direct instruction
            "Describe the video content:",
        ]
        
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"\nTrying prompt format {i+1}...")
            description = self.generate_with_visual_features(visual_features, prompt)
            results.append({
                'prompt': prompt,
                'description': description,
                'length': len(description)
            })
            
            if description and len(description) > 20:
                logger.info(f"✅ Got valid description with prompt {i+1}")
                break
        
        # Select best result
        best_result = max(results, key=lambda x: x['length'])
        description = best_result['description']
        
        # Log all results
        logger.info("\nGeneration results summary:")
        for i, r in enumerate(results):
            logger.info(f"  Prompt {i+1}: {r['length']} chars")
            if r['description']:
                logger.info(f"    Preview: {r['description'][:100]}...")
        
        # Create final result
        result = {
            'overall_description': description,
            'segments': [{
                'timestamp': 0.0,
                'description': description,
                'frames_analyzed': len(frames)
            }],
            'metadata': {
                'num_frames': len(frames),
                'model': 'AuroraCap-7B-VID',
                'video_path': video_path,
                'description_length': len(description),
                'visual_features_shape': list(visual_features.shape),
                'debug_mode': True
            }
        }
        
        return result

def main():
    parser = argparse.ArgumentParser(description="AuroraCap Debug Inference")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames")
    parser.add_argument("--output_dir", default="/home/user/tiktok_production/aurora_cap/output")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = DebugAuroraCapInference()
    
    try:
        # Analyze video
        results = analyzer.analyze_video(args.video_path, args.num_frames)
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        video_name = Path(args.video_path).stem
        
        output_path = os.path.join(args.output_dir, f"{video_name}_debug.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✅ Analysis complete!")
        logger.info(f"Results saved to: {output_path}")
        
        # Display results
        logger.info("\n" + "="*80)
        logger.info("FINAL RESULTS:")
        logger.info("="*80)
        logger.info(f"Description length: {results['metadata']['description_length']} characters")
        if results['overall_description']:
            logger.info(f"Description preview:")
            logger.info(results['overall_description'][:500] + "..." if len(results['overall_description']) > 500 else results['overall_description'])
        else:
            logger.error("❌ No description generated!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()