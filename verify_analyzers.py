#!/usr/bin/env python3
"""Verify all 21 active analyzers are working correctly"""

import sys
sys.path.append('/home/user/tiktok_production')

import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from ml_analyzer_registry_complete import ML_ANALYZERS
from configs.gpu_groups_config import DISABLED_ANALYZERS

def verify_analyzers():
    """Verify all active analyzers can be imported and initialized"""
    
    # Get active analyzers
    active_analyzers = [name for name in ML_ANALYZERS.keys() if name not in DISABLED_ANALYZERS]
    
    print(f"Total analyzers: {len(ML_ANALYZERS)}")
    print(f"Active analyzers: {len(active_analyzers)}")
    print(f"Disabled analyzers: {len(DISABLED_ANALYZERS)}")
    print("\n" + "="*60 + "\n")
    
    # Test each analyzer
    results = {
        'success': [],
        'failed': []
    }
    
    for i, name in enumerate(active_analyzers, 1):
        print(f"[{i}/{len(active_analyzers)}] Testing {name}...")
        try:
            # Get analyzer class
            analyzer_class = ML_ANALYZERS[name]
            
            # Try to instantiate
            analyzer = analyzer_class()
            
            # Check for analyze method
            if not hasattr(analyzer, 'analyze'):
                raise AttributeError(f"Analyzer {name} missing analyze() method")
            
            results['success'].append(name)
            print(f"  ✓ {name} - OK")
            
        except Exception as e:
            results['failed'].append((name, str(e)))
            print(f"  ✗ {name} - FAILED: {str(e)}")
    
    # Summary
    print("\n" + "="*60)
    print(f"\nSUMMARY:")
    print(f"  Successful: {len(results['success'])}/{len(active_analyzers)}")
    print(f"  Failed: {len(results['failed'])}/{len(active_analyzers)}")
    
    if results['failed']:
        print("\nFAILED ANALYZERS:")
        for name, error in results['failed']:
            print(f"  - {name}: {error}")
    
    return results

if __name__ == "__main__":
    results = verify_analyzers()
    
    # Exit with error if any failed
    if results['failed']:
        sys.exit(1)