#!/usr/bin/env python3
"""
Robust JSON encoder for handling all data types from ML analyzers
"""
import json
import numpy as np
from datetime import datetime
from pathlib import Path


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types and other non-serializable objects"""
    
    def default(self, obj):
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
            
        # Handle datetime
        elif isinstance(obj, datetime):
            return obj.isoformat()
            
        # Handle Path objects
        elif isinstance(obj, Path):
            return str(obj)
            
        # Handle bytes
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
            
        # Handle sets
        elif isinstance(obj, set):
            return list(obj)
            
        # Handle tuples (convert to list for JSON)
        elif isinstance(obj, tuple):
            return list(obj)
            
        # For any other type, try to convert to string
        try:
            return str(obj)
        except:
            # If all else fails, return a placeholder
            return f"<non-serializable: {type(obj).__name__}>"


def safe_json_dump(data, file_path=None, **kwargs):
    """Safely dump data to JSON with custom encoder"""
    kwargs['cls'] = NumpyJSONEncoder
    kwargs.setdefault('indent', 2)
    kwargs.setdefault('ensure_ascii', False)
    
    if file_path:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, **kwargs)
    else:
        return json.dumps(data, **kwargs)


def clean_analyzer_results(results):
    """Clean analyzer results to ensure JSON compatibility"""
    if isinstance(results, dict):
        cleaned = {}
        for key, value in results.items():
            # Skip None values
            if value is None:
                continue
                
            # Recursively clean nested dicts
            if isinstance(value, dict):
                cleaned[key] = clean_analyzer_results(value)
            # Clean lists
            elif isinstance(value, (list, tuple)):
                cleaned[key] = [clean_analyzer_results(item) for item in value]
            # Convert numpy arrays
            elif isinstance(value, np.ndarray):
                cleaned[key] = value.tolist()
            # Convert numpy scalars
            elif isinstance(value, (np.integer, np.floating)):
                cleaned[key] = float(value) if isinstance(value, np.floating) else int(value)
            # Keep other values as-is
            else:
                cleaned[key] = value
        return cleaned
    elif isinstance(results, (list, tuple)):
        return [clean_analyzer_results(item) for item in results]
    else:
        return results