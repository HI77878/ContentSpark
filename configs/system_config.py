# CLEAN SYSTEM CONFIGURATION
# Using LOCAL installation (no Docker)

# API Ports (no conflicts!)
API_PORTS = {
    'stable_production': 8004,  # Main API
    'vid2seq': 8024,           # Vid2Seq service  
    'test_api': 8005,          # Test endpoint
}

# System Mode
SYSTEM_MODE = 'local'  # 'local' or 'docker', NOT both!

# Resource Limits
MAX_CONCURRENT_ANALYZERS = 10  # Prevent memory overflow
MAX_GPU_MEMORY_FRACTION = 0.8   # Leave some GPU memory free

# Model Loading
PRELOAD_MODELS = True
MODEL_CACHE_DIR = '/home/user/.cache/tiktok_models'