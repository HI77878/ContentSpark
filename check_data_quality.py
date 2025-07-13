#!/usr/bin/env python3
"""Check data quality of latest analysis"""
import json
from pathlib import Path
from datetime import datetime

# Find newest analysis
results_dirs = [
    "/home/user/tiktok_production/results",
    "/home/user/tiktok_downloads",
    "/home/user/archived_analyses/tiktok_analysis_results"
]

latest_file = None
for dir_path in results_dirs:
    if Path(dir_path).exists():
        files = list(Path(dir_path).glob("*.json"))
        if files:
            for f in files:
                if not latest_file or f.stat().st_mtime > latest_file.stat().st_mtime:
                    latest_file = f

if latest_file:
    print(f"Latest analysis: {latest_file}")
    print(f"Modified: {datetime.fromtimestamp(latest_file.stat().st_mtime)}")
    
    with open(latest_file) as f:
        data = json.load(f)
    
    print("\nANALYZER DATA QUALITY:")
    print("-" * 60)
    
    analyzer_results = data.get('analyzer_results', {})
    
    for analyzer, result in sorted(analyzer_results.items()):
        quality = "❓"
        details = ""
        
        if isinstance(result, dict):
            if 'error' in result:
                quality = "❌ ERROR"
                details = result['error'][:50]
            elif 'segments' in result and len(result.get('segments', [])) > 0:
                # Check for generic data
                sample = str(result['segments'][0]).lower()
                if any(term in sample for term in ['balanced', 'natural', 'moderate', 'normal']):
                    quality = "⚠️ GENERIC"
                    details = f"{len(result['segments'])} segments"
                else:
                    quality = "✅ SPECIFIC"
                    details = f"{len(result['segments'])} segments"
            else:
                quality = "⚠️ NO SEGMENTS"
                if 'summary' in result:
                    details = "has summary"
        else:
            quality = "❌ NOT DICT"
            details = type(result).__name__
        
        print(f"  {analyzer:30} {quality:15} {details}")
    
    # Check metadata
    print("\nMETADATA:")
    meta = data.get('metadata', {})
    print(f"  Duration: {meta.get('duration', 'N/A')}s")
    print(f"  Processing time: {meta.get('processing_time_seconds', meta.get('processing_time', 'N/A'))}s")
    print(f"  Successful analyzers: {meta.get('successful_analyzers', 'N/A')}/{meta.get('total_analyzers', 'N/A')}")
else:
    print("No analysis files found!")