"""
Sequential Processing Configuration for Qwen2-VL
Ensures Qwen2-VL runs alone with full GPU access
"""

import torch

# Phase-based processing configuration
SEQUENTIAL_PHASES = {
    # Phase 1: Qwen2-VL runs alone with full GPU
    'phase_1_exclusive': {
        'analyzers': ['qwen2_vl_temporal'],
        'gpu_config': {
            'device_map': 'auto',  # Use full GPU
            'max_memory': {0: '42GB'},  # Allow up to 42GB (leaving 2.5GB for system)
            'torch_dtype': torch.bfloat16,  # Better precision for Qwen2-VL
            'low_cpu_mem_usage': True,
            'offload_folder': '/tmp/offload',
            'offload_state_dict': True
        },
        'batch_size': 1,
        'max_concurrent': 1,
        'clear_cache_before': True,
        'clear_cache_after': True
    },
    
    # Phase 2: Other GPU analyzers
    'phase_2_gpu_batch': {
        'analyzers': [
            'object_detection',
            'text_overlay', 
            'background_segmentation',
            'camera_analysis',
            'scene_segmentation',
            'color_analysis',
            'body_pose',
            'age_estimation',
            'content_quality',
            'eye_tracking',
            'cut_analysis'
        ],
        'gpu_config': {
            'max_memory_mb': 15000,  # Share 15GB among all
            'torch_dtype': torch.float16
        },
        'batch_sizes': {
            'object_detection': 32,
            'text_overlay': 16,
            'background_segmentation': 8,
            'default': 16
        },
        'max_concurrent': 4,
        'clear_cache_before': True,
        'clear_cache_after': False
    },
    
    # Phase 3: CPU analyzers (can run in parallel)
    'phase_3_cpu': {
        'analyzers': [
            'speech_transcription',
            'audio_analysis',
            'audio_environment',
            'speech_emotion',
            'temporal_flow',
            'speech_flow',
            'cross_analyzer_intelligence'
        ],
        'max_concurrent': 8,
        'device': 'cpu'
    }
}

# Qwen2-VL specific optimizations for full GPU usage
QWEN2_VL_FULL_GPU_CONFIG = {
    'model_kwargs': {
        'torch_dtype': torch.bfloat16,
        'device_map': 'auto',
        'max_memory': {0: '42GB'},
        'offload_folder': '/tmp/offload',
        'offload_state_dict': True,
        'low_cpu_mem_usage': True
    },
    'generation_config': {
        'max_new_tokens': 512,  # More tokens when running alone
        'temperature': 0.7,
        'do_sample': True,
        'top_p': 0.9,
        'repetition_penalty': 1.1
    },
    'processing': {
        'segment_duration': 2.0,  # Process 2-second segments
        'fps_sample': 2.0,  # Sample at 2 FPS
        'max_frames_per_segment': 16,  # More frames when running alone
        'resize_height': 720  # Higher resolution when running alone
    }
}

# Memory management for sequential processing
MEMORY_CONFIG = {
    'pre_phase_cleanup': True,
    'inter_phase_cleanup': True,
    'aggressive_cleanup': True,
    'memory_fraction': 0.95,  # Use 95% of GPU when Qwen2-VL runs alone
    'enable_amp': True
}

def get_phase_config(phase_name):
    """Get configuration for a specific phase"""
    return SEQUENTIAL_PHASES.get(phase_name, {})

def prepare_gpu_for_phase(phase_name):
    """Prepare GPU for a specific phase"""
    import gc
    
    # Clear everything first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
    
    # Set memory fraction based on phase
    if phase_name == 'phase_1_exclusive':
        # Allow Qwen2-VL to use almost all GPU memory
        torch.cuda.set_per_process_memory_fraction(0.95)
    else:
        # Restrict memory for shared phases
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    return True

def cleanup_after_phase(phase_name):
    """Cleanup after phase completion"""
    import gc
    
    if torch.cuda.is_available():
        # Force cleanup all GPU memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Extra aggressive cleanup for Qwen2-VL
        if phase_name == 'phase_1_exclusive':
            # Clear all cached allocations
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
    gc.collect()
    return True