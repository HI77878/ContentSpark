#!/usr/bin/env python3
"""
Test Ray Production API with Marc Gebauer video
"""

import asyncio
import requests
import json
import time
import sys

async def test_ray_api():
    """Test the Ray API"""
    
    # Marc Gebauer video with CTA
    video_path = "/home/user/tiktok_videos/videos/7525171065367104790.mp4"
    api_url = "http://localhost:8004/analyze"
    
    print("🚀 Testing Ray Production API...")
    print(f"Video: {video_path}")
    
    # Check if API is running
    try:
        health = requests.get("http://localhost:8004/")
        print(f"API Status: {health.json()}")
    except:
        print("❌ API not running. Starting it...")
        return
    
    # Send analysis request
    print("\n📤 Sending analysis request...")
    start_time = time.time()
    
    response = requests.post(api_url, json={
        "video_path": video_path,
        "tiktok_url": "https://www.tiktok.com/@marc.gebauer/video/7525171065367104790",
        "creator_username": "marc.gebauer"
    })
    
    if response.status_code == 200:
        result = response.json()
        processing_time = result['processing_time']
        results_file = result['results_file']
        
        print(f"✅ Analysis complete in {processing_time:.1f}s")
        print(f"📁 Results saved to: {results_file}")
        
        # Load and check results
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        print("\n📊 Analysis Results:")
        
        # Check speech transcription
        if 'speech_transcription' in data['analyzer_results']:
            speech = data['analyzer_results']['speech_transcription']
            print(f"\n🎤 Speech Transcription:")
            print(f"   Segments: {len(speech.get('segments', []))}")
            print(f"   Language: {speech.get('language', 'unknown')}")
            
            # Print transcribed text
            for seg in speech.get('segments', [])[:3]:
                print(f"   [{seg['start_time']:.1f}s]: {seg['text']}")
        
        # Check CTA detection
        if 'comment_cta_detection' in data['analyzer_results']:
            cta = data['analyzer_results']['comment_cta_detection']
            print(f"\n📢 CTA Detection:")
            print(f"   Total CTAs: {cta.get('total_ctas', 0)}")
            print(f"   Marc Gebauer CTA: {cta.get('marc_gebauer_cta_detected', False)}")
            
            for seg in cta.get('segments', []):
                print(f"   [{seg['start_time']:.1f}s]: {seg['text']}")
                print(f"      Types: {seg['cta_types']}")
                print(f"      Marc Pattern: {seg.get('is_marc_gebauer', False)}")
        
        # Check Qwen2-VL
        if 'qwen2_vl_ultra' in data['analyzer_results']:
            qwen = data['analyzer_results']['qwen2_vl_ultra']
            print(f"\n🤖 Qwen2-VL Analysis:")
            print(f"   Segments: {len(qwen.get('segments', []))}")
            
            for seg in qwen.get('segments', [])[:2]:
                desc = seg.get('description', seg.get('error', 'No description'))
                print(f"   [{seg['start_time']:.1f}-{seg['end_time']:.1f}s]: {desc[:100]}...")
        
        # Performance metrics
        print(f"\n⚡ Performance:")
        print(f"   Processing time: {processing_time:.1f}s")
        print(f"   Video duration: 9.3s")
        print(f"   Realtime factor: {processing_time / 9.3:.2f}x")
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # First start the API
    print("Starting Ray Production API...")
    import subprocess
    import os
    
    # Fix FFmpeg
    subprocess.run(["bash", "-c", "source /home/user/tiktok_production/fix_ffmpeg_env.sh"])
    
    # Start API in background
    api_process = subprocess.Popen([
        sys.executable, 
        "/home/user/tiktok_production/api/ray_production_api.py"
    ])
    
    print("Waiting for API to start...")
    time.sleep(10)
    
    try:
        asyncio.run(test_ray_api())
    finally:
        print("\nStopping API...")
        api_process.terminate()
        api_process.wait()