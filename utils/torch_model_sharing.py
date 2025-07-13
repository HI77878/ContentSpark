#!/usr/bin/env python3
"""
PyTorch Model Sharing mit torch.multiprocessing
Nutzt shared memory für effizientes Model Sharing
"""

import torch
import torch.multiprocessing as mp
import whisper
import numpy as np
import logging
import pickle
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

# WICHTIG: Set start method für CUDA
mp.set_start_method('spawn', force=True)

class WhisperSharedModel:
    """Whisper Model mit Shared Memory Support"""
    
    def __init__(self):
        self.model_state = None
        self.model_config = {
            "name": "base",
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
    def load_and_share(self):
        """Load model and prepare for sharing"""
        logger.info("Loading Whisper model for sharing...")
        
        # Load model on CPU first for sharing
        model = whisper.load_model(self.model_config["name"], device="cpu")
        
        # Get state dict
        state_dict = model.state_dict()
        
        # Convert to shared tensors
        shared_state = {}
        for key, tensor in state_dict.items():
            shared_tensor = tensor.share_memory_()
            shared_state[key] = shared_tensor
        
        self.model_state = shared_state
        self.dims = model.dims
        
        logger.info("✅ Whisper model prepared for sharing")
        
    def get_model_in_process(self, device="cuda"):
        """Reconstruct model in worker process"""
        if self.model_state is None:
            raise RuntimeError("Model not loaded for sharing")
        
        # Create model architecture
        model = whisper.load_model(self.model_config["name"], device="cpu")
        
        # Load shared state
        model.load_state_dict(self.model_state)
        
        # Move to desired device
        if device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
        
        return model

class TorchModelPool:
    """Model pool using torch.multiprocessing"""
    
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers = []
        self.shared_whisper = WhisperSharedModel()
        
    def start(self):
        """Start worker processes"""
        # Load and share models
        self.shared_whisper.load_and_share()
        
        # Start workers
        for i in range(self.num_workers):
            p = mp.Process(
                target=self._worker_loop,
                args=(i, self.task_queue, self.result_queue, self.shared_whisper)
            )
            p.start()
            self.workers.append(p)
        
        logger.info(f"✅ Started {self.num_workers} worker processes")
    
    @staticmethod
    def _worker_loop(worker_id, task_queue, result_queue, shared_whisper):
        """Worker process loop"""
        # Set CUDA device for this worker
        if torch.cuda.is_available():
            torch.cuda.set_device(worker_id % torch.cuda.device_count())
        
        # Load model in this process
        whisper_model = shared_whisper.get_model_in_process()
        
        logger.info(f"Worker {worker_id} ready")
        
        while True:
            task = task_queue.get()
            if task is None:  # Shutdown signal
                break
            
            task_type = task["type"]
            task_id = task["id"]
            
            try:
                if task_type == "transcribe":
                    audio_path = task["audio_path"]
                    result = whisper_model.transcribe(audio_path)
                    result_queue.put({
                        "id": task_id,
                        "result": result,
                        "error": None
                    })
                    
            except Exception as e:
                result_queue.put({
                    "id": task_id,
                    "result": None,
                    "error": str(e)
                })
    
    def transcribe(self, audio_path: str, task_id: str) -> None:
        """Submit transcription task"""
        self.task_queue.put({
            "type": "transcribe",
            "id": task_id,
            "audio_path": audio_path
        })
    
    def get_result(self, timeout=None) -> Dict[str, Any]:
        """Get result from queue"""
        return self.result_queue.get(timeout=timeout)
    
    def shutdown(self):
        """Shutdown workers"""
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        
        for p in self.workers:
            p.join()

# Alternative: Einfacher Shared Memory Ansatz
class SimpleSharedWhisper:
    """Einfacher Ansatz mit serialisiertem Model State"""
    
    def __init__(self):
        self.model_buffer = None
        
    def prepare_model(self):
        """Prepare model for sharing"""
        logger.info("Preparing Whisper model for sharing...")
        
        # Load model
        model = whisper.load_model("base", device="cpu")
        
        # Serialize state dict to bytes
        import io
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        
        # Convert to shared tensor
        data = buffer.read()
        self.model_buffer = torch.frombuffer(
            bytearray(data), 
            dtype=torch.uint8
        ).share_memory_()
        
        logger.info(f"✅ Model serialized to {len(data)/1024/1024:.1f}MB shared buffer")
    
    def load_in_worker(self, device="cuda"):
        """Load model in worker process"""
        import io
        
        # Deserialize from shared buffer
        buffer = io.BytesIO(self.model_buffer.numpy().tobytes())
        state_dict = torch.load(buffer, map_location="cpu")
        
        # Create model and load state
        model = whisper.load_model("base", device="cpu")
        model.load_state_dict(state_dict)
        
        # Move to device
        if device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
        
        return model

# Beispiel-Verwendung
if __name__ == "__main__":
    # Test torch multiprocessing pool
    pool = TorchModelPool(num_workers=2)
    pool.start()
    
    # Submit tasks
    pool.transcribe("/path/to/audio1.wav", "task1")
    pool.transcribe("/path/to/audio2.wav", "task2")
    
    # Get results
    for _ in range(2):
        result = pool.get_result(timeout=30)
        print(f"Task {result['id']}: {result['result']}")
    
    pool.shutdown()