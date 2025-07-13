#!/usr/bin/env python3
"""
Wrapper for Narrative Analysis that handles analyzer output collection
"""

from analyzers.narrative_analysis_advanced import NarrativeAnalysisAdvanced
from typing import Dict, Any
import json
import os
import logging

logger = logging.getLogger(__name__)

class NarrativeAnalysisWrapper(NarrativeAnalysisAdvanced):
    """Wrapper that collects analyzer outputs for narrative analysis"""
    
    def __init__(self):
        super().__init__()
        self.analyzer_outputs = {}
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """
        Override analyze to collect analyzer outputs from result file
        """
        # Try to find corresponding result file
        analyzer_outputs = self._load_analyzer_outputs(video_path)
        
        if analyzer_outputs:
            # Use parent class with analyzer outputs
            return super().analyze(video_path, analyzer_outputs)
        else:
            # Fallback to limited analysis
            logger.warning(f"[NarrativeWrapper] No analyzer outputs found for {video_path}")
            return self._create_limited_analysis(video_path)
    
    def _load_analyzer_outputs(self, video_path: str) -> Dict[str, Any]:
        """Load analyzer outputs from result file"""
        # Extract video ID from path
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        
        # Look for result files
        results_dir = "/home/user/tiktok_production/results"
        
        # Try to find most recent result file for this video
        matching_files = []
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                if video_id in filename and filename.endswith('.json'):
                    matching_files.append(os.path.join(results_dir, filename))
        
        if not matching_files:
            return {}
        
        # Use most recent file
        matching_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        result_file = matching_files[0]
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                return data.get('analyzer_results', {})
        except Exception as e:
            logger.error(f"[NarrativeWrapper] Failed to load results from {result_file}: {e}")
            return {}
    
    def set_analyzer_outputs(self, outputs: Dict[str, Any]):
        """Manually set analyzer outputs for testing"""
        self.analyzer_outputs = outputs
    
    def analyze_with_outputs(self, video_path: str, analyzer_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze with provided outputs"""
        return super().analyze(video_path, analyzer_outputs)