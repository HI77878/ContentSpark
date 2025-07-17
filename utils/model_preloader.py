import torch
import gc
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import logging
from threading import Lock
import time

logger = logging.getLogger(__name__)

class ModelPreloader:
    """Persistent Model Loading - Modelle werden EINMAL geladen und wiederverwendet"""
    
    _instance = None
    _models = {}
    _lock = Lock()
    
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
            logger.info(f"ModelPreloader initialized with device: {self.device}")
    
    def get_model(self, model_name, model_class=None, processor_class=None, model_kwargs=None, **kwargs):
        """Get or load a model with caching"""
        with self._lock:
            # Check if model already loaded
            if model_name in self._models:
                logger.info(f"Model {model_name} already loaded, returning cached version")
                return self._models[model_name]
            
            # Load model
            logger.info(f"Loading model {model_name} for the first time...")
            start_time = time.time()
            
            if model_kwargs:
                # Special handling for models like Whisper
                self._load_special_model(model_name, model_class, model_kwargs)
            else:
                # Standard HuggingFace-style loading
                self._load_model(model_name, model_class, processor_class, **kwargs)
            
            load_time = time.time() - start_time
            logger.info(f"Model {model_name} loaded in {load_time:.1f}s")
            
            return self._models[model_name]
    
    def _load_model(self, model_name, model_class, processor_class, **kwargs):
        """Lädt Model und Processor"""
        try:
            # Load processor if specified
            processor = None
            if processor_class:
                processor = processor_class.from_pretrained(model_name, trust_remote_code=True)
            
            # Load model
            if model_class:
                # Entferne device_map aus kwargs wenn vorhanden
                kwargs_copy = kwargs.copy()
                kwargs_copy.pop('device_map', None)
                
                model = model_class.from_pretrained(
                    model_name,
                    device_map="cuda",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    **kwargs_copy
                )
                model.eval()  # Set to evaluation mode
            else:
                # Fallback für andere Model-Typen
                model = None
            
            self._models[model_name] = {
                'model': model,
                'processor': processor,
                'device': self.device
            }
            
            logger.info(f"Successfully loaded: {model_name}")
            
            # Log memory usage
            if torch.cuda.is_available():
                logger.info(f"GPU memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _load_special_model(self, model_name, model_class, model_kwargs):
        """Load models with special loading patterns (e.g., Whisper)"""
        try:
            # Load model using provided function and kwargs
            model = model_class(**model_kwargs)
            
            self._models[model_name] = {
                'model': model,
                'processor': None,
                'device': self.device
            }
            
            logger.info(f"Successfully loaded special model: {model_name}")
            
            # Log memory usage
            if torch.cuda.is_available():
                logger.info(f"GPU memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
                
        except Exception as e:
            logger.error(f"Failed to load special model {model_name}: {e}")
            raise
    
    def cleanup_memory(self):
        """Cleanup GPU memory nach Analyse"""
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cleaned")
    
    def unload_model(self, model_name):
        """Entlädt spezifisches Model aus Cache"""
        with self._lock:
            if model_name in self._models:
                del self._models[model_name]
                self.cleanup_memory()
                logger.info(f"Unloaded model: {model_name}")

# Globale Instanz
model_preloader = ModelPreloader()