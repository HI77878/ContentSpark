#!/usr/bin/env python3
"""
Patch script to fix rope_scaling compatibility in existing container
"""
import subprocess
import sys

def patch_container():
    """Run a patched version with compatible transformers"""
    
    # Create a wrapper script that downgrades transformers on the fly
    wrapper_script = """
import subprocess
import sys

# Downgrade transformers to compatible version
print("Installing compatible transformers version...")
subprocess.run([sys.executable, "-m", "pip", "install", "transformers==4.36.2", "--quiet"])

# Now run the actual inference
import auroracap_simple
auroracap_simple.main()
"""
    
    # Write wrapper to temp file
    with open('/tmp/auroracap_wrapper.py', 'w') as f:
        f.write(wrapper_script)
    
    # Run with Docker
    video_path = sys.argv[1] if len(sys.argv) > 1 else '/videos/input.mp4'
    
    cmd = [
        'docker', 'run', '--rm',
        '--gpus', 'all',
        '-v', f'{video_path}:/app/videos/input.mp4:ro',
        '-v', '/home/user/tiktok_production/aurora_cap/output:/app/output',
        '-v', '/tmp/auroracap_wrapper.py:/app/wrapper.py:ro',
        'aurora-cap:latest',
        'python', '/app/wrapper.py'
    ]
    
    print("Running AuroraCap with patched transformers...")
    subprocess.run(cmd)

if __name__ == "__main__":
    patch_container()