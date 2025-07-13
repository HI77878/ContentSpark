#!/usr/bin/env python3
"""
Download and analyze TikTok video with proper metadata storage
"""
import sys
import json
import time
import requests
from pathlib import Path
from mass_processing.tiktok_downloader import TikTokDownloader

def download_and_analyze(tiktok_url: str):
    """Download TikTok video and analyze it with metadata"""
    print(f"\n🎬 Processing TikTok URL: {tiktok_url}")
    
    # Step 1: Download video
    print("\n📥 Downloading video...")
    downloader = TikTokDownloader()
    result = downloader.download_video(tiktok_url)
    
    if not result['success']:
        print(f"❌ Download failed: {result.get('error', 'Unknown error')}")
        return False
    
    video_path = result['video_path']
    metadata = result['metadata']
    
    print(f"✅ Downloaded: {video_path}")
    print(f"   Creator: @{metadata.get('author', 'unknown')}")
    print(f"   Description: {metadata.get('description', 'N/A')[:100]}...")
    print(f"   Views: {metadata.get('play_count', 0):,}")
    print(f"   Likes: {metadata.get('digg_count', 0):,}")
    
    # Step 2: Analyze with metadata
    print("\n🔍 Starting analysis with metadata...")
    
    api_url = "http://localhost:8003/analyze"
    payload = {
        "video_path": video_path,
        "tiktok_url": tiktok_url,
        "creator_username": metadata.get('author', '')
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=600)
        response.raise_for_status()
        
        result = response.json()
        
        if result['status'] == 'success':
            print(f"\n✅ Analysis complete!")
            print(f"   Processing time: {result['processing_time']:.1f}s")
            print(f"   Successful analyzers: {result['successful_analyzers']}/{result['total_analyzers']}")
            print(f"   Results saved to: {result['results_file']}")
            
            # Verify metadata was saved
            with open(result['results_file']) as f:
                saved_data = json.load(f)
                saved_meta = saved_data.get('metadata', {})
                
            print(f"\n📋 Saved metadata:")
            print(f"   TikTok URL: {'✅' if saved_meta.get('tiktok_url') else '❌'}")
            print(f"   Creator: {'✅' if saved_meta.get('creator_username') else '❌'}")
            print(f"   Video ID: {'✅' if saved_meta.get('tiktok_video_id') else '❌'}")
            print(f"   Duration: {'✅' if saved_meta.get('duration') else '❌'}")
            
            return True
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 download_and_analyze.py <tiktok_url>")
        print("Example: python3 download_and_analyze.py https://www.tiktok.com/@username/video/1234567890")
        sys.exit(1)
    
    tiktok_url = sys.argv[1]
    
    # Validate URL
    if not tiktok_url.startswith('https://www.tiktok.com/'):
        print("❌ Invalid TikTok URL. Must start with https://www.tiktok.com/")
        sys.exit(1)
    
    # Check if API is running
    try:
        health = requests.get("http://localhost:8003/health", timeout=5)
        if health.status_code != 200:
            print("❌ API is not healthy. Please start it first:")
            print("   cd /home/user/tiktok_production")
            print("   source fix_ffmpeg_env.sh")
            print("   python3 api/stable_production_api_multiprocess.py &")
            sys.exit(1)
    except:
        print("❌ API is not running. Please start it first:")
        print("   cd /home/user/tiktok_production")
        print("   source fix_ffmpeg_env.sh")
        print("   python3 api/stable_production_api_multiprocess.py &")
        sys.exit(1)
    
    # Process video
    success = download_and_analyze(tiktok_url)
    
    if success:
        print("\n✅ Complete! All analyzer data saved with TikTok metadata.")
    else:
        print("\n❌ Processing failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()