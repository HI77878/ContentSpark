"""
Production Pipeline
Simple wrapper for the production system
"""
import sys
sys.path.append('/home/user/tiktok_production')

from utils.multiprocess_gpu_executor_final import MultiprocessGPUExecutorFinal

class ProductionPipeline:
    """Production pipeline using multiprocess executor"""
    
    def __init__(self):
        self.executor = MultiprocessGPUExecutorFinal(num_gpu_processes=3)
        
    def process_video(self, video_path, analyzers=None):
        """Process a video through the pipeline"""
        if analyzers is None:
            # Use default set
            from ml_analyzer_registry_complete import ML_ANALYZERS
            from configs.gpu_groups_config import DISABLED_ANALYZERS, GPU_ANALYZER_GROUPS
            
            analyzers = []
            for group_name, analyzer_list in GPU_ANALYZER_GROUPS.items():
                for analyzer in analyzer_list:
                    if analyzer not in DISABLED_ANALYZERS and analyzer in ML_ANALYZERS:
                        if analyzer not in analyzers:
                            analyzers.append(analyzer)
            
            analyzers = analyzers[:21]  # Top 21
            
        return self.executor.execute_parallel(video_path, analyzers)