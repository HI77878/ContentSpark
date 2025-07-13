"""
JSON Serializer for safe JSON dumps
Handles numpy arrays and other non-serializable objects
"""
import json
import numpy as np
from datetime import datetime
from pathlib import Path

def safe_json_dump(data, filepath, indent=2):
    """
    Safely dump data to JSON file, handling non-serializable objects
    
    Args:
        data: Data to serialize
        filepath: Path to save JSON file
        indent: JSON indentation
    """
    
    class SafeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (datetime,)):
                return obj.isoformat()
            elif isinstance(obj, Path):
                return str(obj)
            elif hasattr(obj, '__dict__'):
                return str(obj)
            return super().default(obj)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=SafeEncoder, indent=indent)

def safe_json_dumps(data, indent=2):
    """Return JSON string with safe encoding"""
    class SafeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (datetime,)):
                return obj.isoformat()
            elif isinstance(obj, Path):
                return str(obj)
            elif hasattr(obj, '__dict__'):
                return str(obj)
            return super().default(obj)
    
    return json.dumps(data, cls=SafeEncoder, indent=indent)