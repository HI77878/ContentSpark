"""
Optimized Batch Processor
Wrapper for the multiprocess GPU executor
"""
import sys
sys.path.append('/home/user/tiktok_production')

from utils.multiprocess_gpu_executor_final import MultiprocessGPUExecutorFinal
from ml_analyzer_registry_complete import ML_ANALYZERS
from configs.gpu_groups_config import DISABLED_ANALYZERS, GPU_ANALYZER_GROUPS

class OptimizedBatchProcessor:
    """Optimized batch processor using multiprocess execution"""
    
    def __init__(self, num_gpu_processes=3):
        self.executor = MultiprocessGPUExecutorFinal(num_gpu_processes=num_gpu_processes)
        
        # Get active analyzers
        self.active_analyzers = []
        for group_name, analyzer_list in GPU_ANALYZER_GROUPS.items():
            for analyzer in analyzer_list:
                if analyzer not in DISABLED_ANALYZERS and analyzer in ML_ANALYZERS:
                    if analyzer not in self.active_analyzers:
                        self.active_analyzers.append(analyzer)
        
    def process_video(self, video_path, selected_analyzers=None):
        """Process a video with selected analyzers"""
        if selected_analyzers is None:
            selected_analyzers = self.active_analyzers[:21]  # Use top 21
            
        return self.executor.execute_parallel(video_path, selected_analyzers)
    
    def get_active_analyzers(self):
        """Get list of active analyzers"""
        return self.active_analyzers