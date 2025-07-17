"""System Configuration"""

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8003

# Processing Settings
MAX_CONCURRENT_VIDEOS = 2
VIDEO_TIMEOUT = 600  # 10 minutes max per video

# Storage Settings  
SUPABASE_URL = ""  # To be configured
SUPABASE_KEY = ""  # To be configured

# GPU Settings
FORCE_GPU = True
GPU_MEMORY_FRACTION = 0.9
