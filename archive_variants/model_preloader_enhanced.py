#!/usr/bin/env python3
"""
Enhanced Model Preloader for Production
Loads ALL analyzer models at startup for maximum GPU utilization
"""

import torch
import gc
import logging
from threading import Lock
import time
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class EnhancedModelPreloader:
    """Enhanced preloader that loads all analyzer models at startup"""
    
    _instance = None
    _models = {}
    _lock = Lock()
    _analyzer_models = {}  # Maps analyzer names to their models
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Enhanced ModelPreloader initialized with device: {self.device}")
            
            # Model configurations for each analyzer
            self.model_configs = {
                'qwen2_vl_temporal': {
                    'model_name': 'Qwen/Qwen2-VL-7B-Instruct',
                    'model_type': 'qwen2vl',
                    'memory_mb': 16000
                },
                'whisper_base': {
                    'model_name': 'whisper-base',
                    'model_type': 'whisper',
                    'memory_mb': 1500
                },
                'yolov8': {
                    'model_name': 'yolov8l.pt',
                    'model_type': 'yolo',
                    'memory_mb': 600
                },
                'yolov8_pose': {
                    'model_name': 'yolov8x-pose.pt',
                    'model_type': 'yolo',
                    'memory_mb': 800
                },
                'easyocr': {
                    'model_name': 'easyocr',
                    'model_type': 'easyocr',
                    'memory_mb': 2000
                },
                'segformer': {
                    'model_name': 'nvidia/segformer-b0-finetuned-ade-512-512',
                    'model_type': 'segformer',
                    'memory_mb': 500
                }
            }
    
    def preload_all_models(self):
        """Preload all analyzer models at startup"""
        logger.info("🚀 Starting comprehensive model preloading...")
        start_time = time.time()
        total_memory = 0
        
        # Check available GPU memory
        if torch.cuda.is_available():
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
            used_memory = torch.cuda.memory_allocated() / 1024**2
            available_memory = total_gpu_memory - used_memory
            logger.info(f"GPU Memory: {used_memory:.0f}/{total_gpu_memory:.0f} MB used, {available_memory:.0f} MB available")
        
        # Load models in order of importance
        load_order = [
            'qwen2_vl_temporal',  # Most important - 16GB
            'whisper_base',       # Speech - 1.5GB  
            'yolov8',            # Object detection - 0.6GB
            'easyocr',           # Text overlay - 2GB
            'segformer',         # Background - 0.5GB
            'yolov8_pose'        # Body pose - 0.8GB
        ]
        
        for model_key in load_order:
            if model_key not in self.model_configs:
                continue
                
            config = self.model_configs[model_key]
            required_memory = config['memory_mb']
            
            # Check if we have enough memory
            if torch.cuda.is_available():
                current_used = torch.cuda.memory_allocated() / 1024**2
                if current_used + required_memory > total_gpu_memory * 0.9:  # Keep 10% free
                    logger.warning(f"⚠️ Skipping {model_key} - would exceed 90% GPU memory")
                    continue
            
            try:
                logger.info(f"Loading {model_key} ({required_memory} MB)...")
                self._load_specific_model(model_key, config)
                total_memory += required_memory
                
                # Log current GPU usage
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / 1024**2
                    logger.info(f"✅ Loaded {model_key} - GPU memory: {current_memory:.0f} MB")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load {model_key}: {e}")
        
        load_time = time.time() - start_time
        logger.info(f"✅ Model preloading completed in {load_time:.1f}s")
        logger.info(f"📊 Total models loaded: {len(self._analyzer_models)}")
        logger.info(f"💾 Estimated memory usage: {total_memory} MB")
        
        # Run warmup
        self._warmup_models()
        
        return self._analyzer_models
    
    def _load_specific_model(self, model_key: str, config: Dict[str, Any]):
        """Load a specific model based on its type"""
        model_type = config['model_type']
        model_name = config['model_name']
        
        if model_type == 'qwen2vl':
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cuda",
                trust_remote_code=True,
                attn_implementation="eager"  # More stable than flash_attention
            )
            model.eval()
            self._analyzer_models[model_key] = {'model': model, 'processor': processor}
            
        elif model_type == 'whisper':
            import whisper
            model = whisper.load_model("base", device="cuda", download_root="/home/user/.cache/whisper")
            self._analyzer_models[model_key] = {'model': model, 'processor': None}
            
        elif model_type == 'yolo':
            from ultralytics import YOLO
            model = YOLO(model_name)
            model.to('cuda')
            self._analyzer_models[model_key] = {'model': model, 'processor': None}
            
        elif model_type == 'easyocr':
            import easyocr
            # Note: EasyOCR will be loaded on-demand per analyzer due to its initialization pattern
            self._analyzer_models[model_key] = {'model': 'easyocr_placeholder', 'processor': None}
            
        elif model_type == 'segformer':
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
            processor = SegformerImageProcessor.from_pretrained(model_name)
            model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            ).to('cuda')
            model.eval()
            self._analyzer_models[model_key] = {'model': model, 'processor': processor}
    
    def _warmup_models(self):
        """Warmup models to ensure consistent performance"""
        logger.info("🔥 Running model warmup...")
        
        # Create dummy inputs for warmup
        with torch.inference_mode():
            # Warmup GPU
            for _ in range(5):
                dummy = torch.randn(1, 3, 224, 224).cuda().half()
                _ = dummy * 2
                torch.cuda.synchronize()
            
            # Warmup YOLO if loaded
            if 'yolov8' in self._analyzer_models:
                try:
                    import numpy as np
                    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
                    model = self._analyzer_models['yolov8']['model']
                    _ = model(dummy_image, verbose=False)
                except:
                    pass
        
        logger.info("✅ Model warmup completed")
    
    def get_model(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get a preloaded model"""
        return self._analyzer_models.get(model_key)
    
    def get_all_models(self) -> Dict[str, Any]:
        """Get all preloaded models"""
        return self._analyzer_models
    
    def cleanup_memory(self):
        """Cleanup GPU memory"""
        torch.cuda.empty_cache()
        gc.collect()
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        if not torch.cuda.is_available():
            return {}
            
        return {
            'total_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2,
            'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
            'free_mb': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2
        }

# Global instance
enhanced_model_preloader = EnhancedModelPreloader()