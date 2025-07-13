#!/usr/bin/env python3
"""TikTok Analyzer Workflow - Download and analyze TikTok videos"""

import sys
import os
import time
import json
import argparse
import requests
from pathlib import Path

sys.path.append('/home/user/tiktok_production')

from mass_processing.tiktok_downloader import TikTokDownloader

def download_tiktok_video(url):
    """Download TikTok video"""
    print(f"\n📥 Downloading TikTok video from: {url}")
    
    downloader = TikTokDownloader()
    result = downloader.download_video(url)
    
    if result and result.get('video_path'):
        print(f"✅ Downloaded: {result['video_path']}")
        return result['video_path']
    else:
        print(f"❌ Failed to download video")
        return None

def analyze_video(video_path, api_url="http://localhost:8003"):
    """Analyze video using production API"""
    print(f"\n🔍 Analyzing video: {video_path}")
    
    start_time = time.time()
    
    # Call API
    response = requests.post(
        f"{api_url}/analyze",
        json={"video_path": video_path}
    )
    
    if response.status_code == 200:
        result = response.json()
        elapsed = time.time() - start_time
        
        print(f"✅ Analysis completed in {elapsed:.1f}s")
        print(f"   Successful analyzers: {result.get('successful_analyzers')}/{result.get('total_analyzers')}")
        print(f"   Results saved to: {result.get('results_file')}")
        
        return result
    else:
        print(f"❌ Analysis failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None

def main():
    parser = argparse.ArgumentParser(description='TikTok Video Analyzer Workflow')
    parser.add_argument('url', help='TikTok video URL')
    parser.add_argument('--api-url', default='http://localhost:8003', help='API URL')
    
    args = parser.parse_args()
    
    print("🚀 TikTok Video Analyzer Workflow")
    print("=" * 50)
    
    # Download video
    video_path = download_tiktok_video(args.url)
    if not video_path:
        print("Failed to download video")
        sys.exit(1)
    
    # Analyze video
    result = analyze_video(video_path, args.api_url)
    if not result:
        print("Failed to analyze video")
        sys.exit(1)
    
    print("\n✅ Workflow completed successfully!")
    
    # Load and display summary
    if result.get('results_file'):
        with open(result['results_file'], 'r') as f:
            analysis = json.load(f)
            
        print("\n📊 Analysis Summary:")
        print(f"   Video duration: {analysis.get('metadata', {}).get('duration', 0):.1f}s")
        print(f"   Processing time: {analysis.get('processing_time', {}).get('total', 0):.1f}s")
        print(f"   Realtime factor: {analysis.get('performance_metrics', {}).get('realtime_factor', 0):.2f}x")
        
        # Show BLIP-2 description
        if 'video_llava' in analysis.get('analysis_results', {}):
            descriptions = analysis['analysis_results']['video_llava'].get('descriptions', [])
            if descriptions:
                print(f"\n📝 Video Description (BLIP-2):")
                print(f"   {descriptions[0].get('text', 'No description')}")

if __name__ == "__main__":
    main()