import concurrent.futures
import threading
import torch
import time
from typing import Dict, List, Any, Optional
import logging
import os
import sys

# Add project to path
sys.path.append('/home/user/tiktok_production')

logger = logging.getLogger(__name__)

class ParallelAnalyzerProcessor:
    def __init__(self):
        # GPU Lock für sequential GPU processing
        self.gpu_lock = threading.Lock()
        
        # Frame cache (shared between analyzers)
        self.frame_cache = {}
        self.frame_cache_lock = threading.Lock()
        
        # Analyzer Groups nach Resource-Typ
        self.groups = {
            'cpu_parallel': [
                'audio_analysis',
                'audio_environment', 
                'speech_emotion',
                'speech_rate',
                'sound_effects',
                'scene_description',
                'temporal_flow',
                'speech_analysis_music_aware'
            ],
            'gpu_heavy_sequential': [
                'vid2seq',
                'blip2_video_analyzer',
                'background_segmentation'
            ],
            'gpu_medium_batch': [
                'object_detection',
                'face_detection',
                'emotion_detection',
                'body_pose',
                'hand_gesture',
                'eye_tracking',
                'age_estimation'
            ],
            'gpu_light_batch': [
                'text_overlay',
                'color_analysis',
                'composition_analysis',
                'visual_effects',
                'content_quality',
                'facial_details',
                'gesture_recognition',
                'product_detection'
            ],
            'gpu_special': [
                'speech_transcription',  # Needs special handling
                'cut_analysis',
                'scene_segmentation',
                'camera_analysis'
            ]
        }
        
        # Preloaded models cache
        self.model_cache = {}
        self.model_cache_lock = threading.Lock()
        
    def analyze_parallel(self, video_path: str, requested_analyzers: Optional[List[str]] = None) -> Dict:
        """Führt Analyzer parallel aus mit optimaler Resource-Nutzung"""
        
        results = {}
        errors = {}
        
        # Get all analyzers if none specified
        if requested_analyzers is None:
            from ml_analyzer_registry_complete import ML_ANALYZERS
            requested_analyzers = list(ML_ANALYZERS.keys())
        
        # Filter groups to only requested analyzers
        active_groups = {}
        for group_name, analyzers in self.groups.items():
            active = [a for a in analyzers if a in requested_analyzers]
            if active:
                active_groups[group_name] = active
        
        start_time = time.time()
        logger.info(f"Starting parallel analysis of {video_path} with {len(requested_analyzers)} analyzers")
        
        # 1. Start CPU analyzers in parallel (they can all run simultaneously)
        cpu_futures = {}
        cpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        
        if 'cpu_parallel' in active_groups:
            for analyzer in active_groups['cpu_parallel']:
                future = cpu_executor.submit(self._run_analyzer_safe, analyzer, video_path, use_gpu=False)
                cpu_futures[future] = analyzer
                logger.info(f"Started CPU analyzer: {analyzer}")
        
        try:
            # 2. Run GPU Heavy analyzers sequentially (too much VRAM for parallel)
            if 'gpu_heavy_sequential' in active_groups:
                logger.info("Running GPU Heavy analyzers sequentially...")
                for analyzer in active_groups['gpu_heavy_sequential']:
                    try:
                        with self.gpu_lock:
                            # Clear GPU cache before heavy analyzer
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            result = self._run_analyzer_safe(analyzer, video_path, use_gpu=True)
                            results[analyzer] = result
                            logger.info(f"✅ {analyzer} completed (GPU Heavy)")
                            
                            # Clear GPU cache after heavy analyzer
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    except Exception as e:
                        errors[analyzer] = str(e)
                        logger.error(f"❌ {analyzer} failed: {e}")
            
            # 3. Run GPU Medium analyzers in batches
            if 'gpu_medium_batch' in active_groups:
                logger.info("Running GPU Medium analyzers in batches...")
                batch_results = self._run_gpu_batch(
                    active_groups['gpu_medium_batch'], 
                    video_path,
                    batch_size=2  # 2 at a time
                )
                results.update(batch_results['results'])
                errors.update(batch_results['errors'])
            
            # 4. Run GPU Light analyzers in larger batches
            if 'gpu_light_batch' in active_groups:
                logger.info("Running GPU Light analyzers in batches...")
                batch_results = self._run_gpu_batch(
                    active_groups['gpu_light_batch'],
                    video_path, 
                    batch_size=3  # 3 at a time
                )
                results.update(batch_results['results'])
                errors.update(batch_results['errors'])
            
            # 5. Run special GPU analyzers
            if 'gpu_special' in active_groups:
                logger.info("Running special GPU analyzers...")
                for analyzer in active_groups['gpu_special']:
                    try:
                        with self.gpu_lock:
                            result = self._run_analyzer_safe(analyzer, video_path, use_gpu=True)
                            results[analyzer] = result
                            logger.info(f"✅ {analyzer} completed (GPU Special)")
                    except Exception as e:
                        errors[analyzer] = str(e)
                        logger.error(f"❌ {analyzer} failed: {e}")
            
            # 6. Collect CPU results
            for future in concurrent.futures.as_completed(cpu_futures):
                analyzer = cpu_futures[future]
                try:
                    result = future.result(timeout=300)  # 5 min timeout
                    results[analyzer] = result
                    logger.info(f"✅ {analyzer} completed (CPU)")
                except Exception as e:
                    errors[analyzer] = str(e)
                    logger.error(f"❌ {analyzer} failed: {e}")
                    
        finally:
            cpu_executor.shutdown(wait=True)
        
        total_time = time.time() - start_time
        
        # Clear frame cache
        self.frame_cache.clear()
        
        logger.info(f"Parallel analysis completed in {total_time:.1f}s")
        logger.info(f"Successful: {len(results)}, Failed: {len(errors)}")
        
        return {
            'results': results,
            'errors': errors,
            'total_time': total_time,
            'successful': len(results),
            'failed': len(errors)
        }
    
    def _run_gpu_batch(self, analyzers: List[str], video_path: str, batch_size: int = 2) -> Dict:
        """Run GPU analyzers in batches"""
        results = {}
        errors = {}
        
        # Process in batches
        for i in range(0, len(analyzers), batch_size):
            batch = analyzers[i:i+batch_size]
            
            with self.gpu_lock:
                # Run batch parallel on GPU
                with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                    futures = {}
                    
                    for analyzer in batch:
                        future = executor.submit(self._run_analyzer_safe, analyzer, video_path, use_gpu=True)
                        futures[future] = analyzer
                    
                    for future in concurrent.futures.as_completed(futures):
                        analyzer = futures[future]
                        try:
                            result = future.result(timeout=180)  # 3 min timeout
                            results[analyzer] = result
                            logger.info(f"✅ {analyzer} completed (GPU Batch)")
                        except Exception as e:
                            errors[analyzer] = str(e)
                            logger.error(f"❌ {analyzer} failed: {e}")
                
                # Clear cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return {'results': results, 'errors': errors}
    
    def _run_analyzer_safe(self, analyzer_name: str, video_path: str, use_gpu: bool = True) -> Dict:
        """Run single analyzer with error handling"""
        try:
            # Import here to avoid circular imports
            from ml_analyzer_registry_complete import ML_ANALYZERS
            
            if analyzer_name not in ML_ANALYZERS:
                raise ValueError(f"Analyzer {analyzer_name} not found")
            
            # Check if model is preloaded
            with self.model_cache_lock:
                if analyzer_name in self.model_cache:
                    analyzer = self.model_cache[analyzer_name]
                    logger.info(f"Using preloaded model for {analyzer_name}")
                else:
                    # Create new instance
                    analyzer = ML_ANALYZERS[analyzer_name]()
            
            # Run analysis
            result = analyzer.analyze(video_path)
            
            # Store successful result
            return result
            
        except Exception as e:
            logger.error(f"Error in {analyzer_name}: {str(e)}")
            raise
    
    def preload_models(self, model_names: List[str]):
        """Preload models for faster processing"""
        from ml_analyzer_registry_complete import ML_ANALYZERS
        
        logger.info(f"Preloading {len(model_names)} models...")
        
        for name in model_names:
            if name in ML_ANALYZERS:
                try:
                    with self.model_cache_lock:
                        if name not in self.model_cache:
                            self.model_cache[name] = ML_ANALYZERS[name]()
                            logger.info(f"Preloaded {name}")
                except Exception as e:
                    logger.error(f"Failed to preload {name}: {e}")

# Test if running directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    processor = ParallelAnalyzerProcessor()
    
    # Test with sample video
    test_video = "/home/user/test_video_with_text.mp4"
    
    print("Testing Parallel Processor...")
    result = processor.analyze_parallel(
        video_path=test_video,
        requested_analyzers=['object_detection', 'face_detection', 'audio_analysis', 'color_analysis']
    )
    
    print(f"\nResults:")
    print(f"  Total time: {result['total_time']:.1f}s")
    print(f"  Successful: {result['successful']}")
    print(f"  Failed: {result['failed']}")
    
    for analyzer, data in result['results'].items():
        segments = len(data.get('segments', []))
        print(f"  - {analyzer}: {segments} segments")