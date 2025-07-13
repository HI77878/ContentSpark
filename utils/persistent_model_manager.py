#!/usr/bin/env python3
"""Persistent Model Manager für GPU-Optimierung

Hält ML-Modelle im GPU-Speicher für schnelleren Zugriff zwischen Analysen.
WICHTIG: Datenqualität darf NICHT leiden - nur Performance-Optimierung!
"""

import torch
import gc
import logging
import threading
from typing import Dict, Any, Optional, Tuple
import time

logger = logging.getLogger(__name__)

class PersistentModelManager:
    """Hält Modelle im GPU-Speicher für schnelleren Zugriff"""
    
    def __init__(self, max_memory_gb: float = 40.0):
        """
        Initialize the model manager.
        
        Args:
            max_memory_gb: Maximum GPU memory to use (in GB)
        """
        self.models: Dict[str, Any] = {}
        self.model_locks: Dict[str, threading.Lock] = {}
        self.model_usage: Dict[str, float] = {}  # Track last usage time
        self.max_memory = max_memory_gb * 1024  # Convert to MB
        self._lock = threading.Lock()
        
        logger.info(f"Initialized PersistentModelManager with {max_memory_gb}GB limit")
        
    def get_analyzer(self, analyzer_name: str, analyzer_class, *args, **kwargs) -> Tuple[Any, threading.Lock]:
        """
        Lädt Analyzer einmal und hält ihn im Speicher.
        
        Args:
            analyzer_name: Name des Analyzers
            analyzer_class: Klasse des Analyzers
            *args, **kwargs: Argumente für Analyzer-Konstruktor
            
        Returns:
            Tuple von (analyzer_instance, lock)
        """
        with self._lock:
            # Update usage time
            self.model_usage[analyzer_name] = time.time()
            
            if analyzer_name not in self.models:
                logger.info(f"Loading {analyzer_name} into persistent cache")
                
                # Check available GPU memory
                if not self._check_memory():
                    logger.warning(f"Low GPU memory, cleaning up before loading {analyzer_name}")
                    self._cleanup_least_used()
                
                try:
                    # Create analyzer instance
                    analyzer = analyzer_class(*args, **kwargs)
                    
                    # Set model to eval mode if available
                    if hasattr(analyzer, 'model') and hasattr(analyzer.model, 'eval'):
                        analyzer.model.eval()
                        logger.info(f"Set {analyzer_name} model to eval mode")
                        
                        # Pin to GPU if not already
                        if hasattr(analyzer.model, 'cuda') and torch.cuda.is_available():
                            try:
                                # Check if model is already on GPU
                                if hasattr(analyzer.model, 'device'):
                                    current_device = str(analyzer.model.device)
                                    if 'cuda' not in current_device:
                                        analyzer.model = analyzer.model.cuda()
                                        logger.info(f"Moved {analyzer_name} model to GPU")
                                else:
                                    analyzer.model = analyzer.model.cuda()
                                    logger.info(f"Moved {analyzer_name} model to GPU")
                            except Exception as e:
                                logger.warning(f"Could not move {analyzer_name} to GPU: {e}")
                    
                    # Store in cache
                    self.models[analyzer_name] = analyzer
                    self.model_locks[analyzer_name] = threading.Lock()
                    
                    # Log memory usage
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                        logger.info(f"GPU memory after loading {analyzer_name}: {allocated:.1f}MB")
                    
                except Exception as e:
                    logger.error(f"Failed to load {analyzer_name}: {e}")
                    raise
            else:
                logger.debug(f"Reusing cached {analyzer_name}")
                
            return self.models[analyzer_name], self.model_locks[analyzer_name]
    
    def _check_memory(self) -> bool:
        """
        Prüft ob genug GPU-Speicher verfügbar ist.
        
        Returns:
            True wenn genug Speicher verfügbar (>5GB frei)
        """
        if not torch.cuda.is_available():
            return True
            
        torch.cuda.empty_cache()
        free_memory = torch.cuda.mem_get_info()[0] / 1024**2  # MB
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        
        logger.debug(f"GPU memory check - Free: {free_memory:.1f}MB, Allocated: {allocated:.1f}MB")
        
        # Keep at least 5GB free for processing
        return free_memory > 5000
    
    def _cleanup_least_used(self):
        """Entfernt das am wenigsten genutzte Modell"""
        if not self.models:
            return
            
        # Find least recently used model
        lru_model = min(self.model_usage.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Removing least used model: {lru_model}")
        
        with self.model_locks[lru_model]:
            # Remove model
            model = self.models.pop(lru_model)
            
            # Clean up GPU memory if model has CUDA tensors
            if hasattr(model, 'model'):
                try:
                    # Move to CPU first
                    if hasattr(model.model, 'cpu'):
                        model.model.cpu()
                    # Delete model
                    del model.model
                except:
                    pass
            
            del model
            
        # Remove from tracking
        self.model_locks.pop(lru_model)
        self.model_usage.pop(lru_model)
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log memory after cleanup
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            logger.info(f"GPU memory after cleanup: {allocated:.1f}MB")
    
    def clear_all(self):
        """Entfernt alle gecachten Modelle"""
        with self._lock:
            logger.info("Clearing all cached models")
            
            for name in list(self.models.keys()):
                with self.model_locks[name]:
                    model = self.models.pop(name)
                    
                    # Clean up GPU memory
                    if hasattr(model, 'model'):
                        try:
                            if hasattr(model.model, 'cpu'):
                                model.model.cpu()
                            del model.model
                        except:
                            pass
                    
                    del model
            
            self.model_locks.clear()
            self.model_usage.clear()
            
            # Force cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("All models cleared from cache")
    
    def get_status(self) -> Dict[str, Any]:
        """Gibt Status des Model Managers zurück"""
        with self._lock:
            status = {
                "cached_models": list(self.models.keys()),
                "model_count": len(self.models),
                "gpu_available": torch.cuda.is_available()
            }
            
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                free = torch.cuda.mem_get_info()[0] / 1024**2  # MB
                status.update({
                    "gpu_memory_allocated_mb": allocated,
                    "gpu_memory_free_mb": free,
                    "gpu_memory_total_mb": self.max_memory
                })
            
            return status

# Globale Instanz für die gesamte Anwendung
model_manager = PersistentModelManager(max_memory_gb=40.0)

# Convenience functions
def get_cached_analyzer(analyzer_name: str, analyzer_class, *args, **kwargs):
    """Wrapper function für einfache Nutzung"""
    return model_manager.get_analyzer(analyzer_name, analyzer_class, *args, **kwargs)

def clear_model_cache():
    """Clear all cached models"""
    model_manager.clear_all()

def get_model_cache_status():
    """Get cache status"""
    return model_manager.get_status()