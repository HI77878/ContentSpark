"""
Validate Reconstruction Capability
Simple validation for analyzer results
"""
import json
import os
from pathlib import Path

def validate_reconstruction(results_file):
    """
    Validate reconstruction capability from results file
    
    Args:
        results_file: Path to JSON results file
        
    Returns:
        dict with validation results
    """
    if not os.path.exists(results_file):
        return {
            'valid': False,
            'error': 'Results file not found',
            'reconstruction_score': 0
        }
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return {
            'valid': False,
            'error': f'Failed to load JSON: {e}',
            'reconstruction_score': 0
        }
    
    # Check for analyzer results
    if 'analyzer_results' not in data:
        return {
            'valid': False,
            'error': 'No analyzer_results found',
            'reconstruction_score': 0
        }
    
    analyzer_results = data['analyzer_results']
    total_analyzers = len(analyzer_results)
    successful_analyzers = 0
    
    # Key analyzers for reconstruction
    key_analyzers = ['video_llava', 'object_detection', 'speech_transcription', 
                     'text_overlay', 'camera_analysis']
    key_present = 0
    
    # Validate each analyzer
    for analyzer_name, result in analyzer_results.items():
        if isinstance(result, dict) and 'error' not in result:
            # Check if it has actual data
            if 'segments' in result or 'frames' in result or 'speed_analysis' in result:
                successful_analyzers += 1
                if analyzer_name in key_analyzers:
                    key_present += 1
    
    reconstruction_score = (successful_analyzers / total_analyzers * 100) if total_analyzers > 0 else 0
    
    # Check BLIP-2 quality
    blip2_quality = False
    if 'video_llava' in analyzer_results:
        blip2 = analyzer_results['video_llava']
        if 'segments' in blip2 and len(blip2['segments']) > 0:
            # Check description length
            desc_lens = [len(s.get('description', '')) for s in blip2['segments']]
            avg_len = sum(desc_lens) / len(desc_lens) if desc_lens else 0
            blip2_quality = avg_len > 150
    
    return {
        'valid': reconstruction_score >= 90,
        'reconstruction_score': reconstruction_score,
        'total_analyzers': total_analyzers,
        'successful_analyzers': successful_analyzers,
        'key_analyzers_present': key_present,
        'blip2_quality': blip2_quality,
        'details': {
            'missing_key_analyzers': [k for k in key_analyzers if k not in analyzer_results],
            'failed_analyzers': [k for k, v in analyzer_results.items() if 'error' in v]
        }
    }

if __name__ == "__main__":
    # Test with latest results
    results_dir = Path("/home/user/tiktok_production/results")
    if results_dir.exists():
        results_files = sorted(results_dir.glob("*.json"))
        if results_files:
            latest = results_files[-1]
            print(f"Validating: {latest}")
            result = validate_reconstruction(str(latest))
            print(json.dumps(result, indent=2))