#!/usr/bin/env python3
"""
Run enhanced BLIP-2 with better prompting strategy
"""
import subprocess
import sys
import os

def run_enhanced_blip2(video_path):
    """Run the working enhanced BLIP-2 Docker container"""
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    
    output_dir = "/home/user/tiktok_production/aurora_cap/output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running enhanced BLIP-2 analysis on: {video_path}")
    print(f"Output directory: {output_dir}")
    
    # Run the Docker container
    cmd = [
        'docker', 'run', '--rm',
        '--gpus', 'all',
        '-v', f'{video_path}:/videos/input.mp4:ro',
        '-v', f'{output_dir}:/app/output',
        'blip2-detailed:latest',
        'python', 'video_captioning_detailed.py', '/videos/input.mp4'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n✅ Analysis completed successfully!")
            print("\nOutput files:")
            # List output files
            for file in os.listdir(output_dir):
                if file.endswith('.json') or file.endswith('.txt'):
                    print(f"  - {file}")
        else:
            print(f"\n❌ Analysis failed with code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
    except Exception as e:
        print(f"Error running Docker: {e}")

if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "/videos/input.mp4"
    run_enhanced_blip2(video_path)