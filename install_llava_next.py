#!/usr/bin/env python3
"""
Install script for LLaVA-NeXT Video dependencies
"""
import subprocess
import sys

def install_packages():
    """Install required packages for LLaVA-NeXT Video"""
    
    packages = [
        # Core dependencies
        "transformers>=4.37.0",
        "accelerate>=0.25.0", 
        "bitsandbytes>=0.41.0",
        "av",  # For video processing
        "sentencepiece",  # For tokenization
        "protobuf",  # For model loading
        
        # Already installed but ensure versions
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
    ]
    
    print("Installing LLaVA-NeXT Video dependencies...")
    
    for package in packages:
        print(f"\nInstalling {package}...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                check=True
            )
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            
    print("\n✅ All dependencies installed!")
    
    # Test imports
    print("\nTesting imports...")
    try:
        import transformers
        import accelerate
        import bitsandbytes
        import av
        print("✅ All imports successful!")
        
        # Print versions
        print(f"\nVersions:")
        print(f"Transformers: {transformers.__version__}")
        print(f"Accelerate: {accelerate.__version__}")
        print(f"Bitsandbytes: {bitsandbytes.__version__}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        
if __name__ == "__main__":
    install_packages()