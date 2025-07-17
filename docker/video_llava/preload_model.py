#!/usr/bin/env python3
"""
Preload LLaVA-NeXT-Video model during Docker build
This ensures the model is cached and ready for use
"""
import os
import torch
from transformers import (
    LlavaNextVideoProcessor,
    LlavaNextVideoForConditionalGeneration,
    BitsAndBytesConfig
)

def preload_model():
    """Download and cache the model"""
    print("="*80)
    print("Preloading LLaVA-NeXT-Video-7B model...")
    print("="*80)
    
    model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    cache_dir = "/app/models"
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Download processor
        print("1. Downloading processor...")
        processor = LlavaNextVideoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir
        )
        print("✅ Processor downloaded")
        
        # Configure 4-bit quantization for optimal performance
        print("2. Configuring model with 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Download model
        print("3. Downloading model (this may take several minutes)...")
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True
        )
        print("✅ Model downloaded and configured")
        
        # Test model loading
        print("4. Testing model inference...")
        model.eval()
        
        # Create dummy input to test
        dummy_text = "Describe this video"
        inputs = processor(
            text=dummy_text,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            # Just ensure forward pass works
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
        
        print("✅ Model inference test passed")
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
        
        print("\n" + "="*80)
        print("✅ Model preloading complete!")
        print(f"✅ Model cached in: {cache_dir}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during preloading: {e}")
        print("The model will be downloaded on first use instead.")
        # Don't fail the build if preloading fails
        pass

if __name__ == "__main__":
    preload_model()