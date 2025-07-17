#!/usr/bin/env python3
"""
Test script for parallel video processing
Tests 2-3 concurrent videos to maximize GPU utilization
"""

import requests
import json
import time
import sys
from pathlib import Path

API_URL = "http://localhost:8003"

def test_parallel_processing():
    """Test parallel video processing with multiple videos"""
    
    # Find available test videos
    video_dir = Path("/home/user/tiktok_videos/videos")
    video_files = list(video_dir.glob("*.mp4"))[:3]  # Get up to 3 videos
    
    if not video_files:
        print("❌ No video files found for testing")
        return
    
    print(f"🎬 Found {len(video_files)} videos for testing:")
    for video in video_files:
        print(f"   - {video.name}")
    
    # Submit batch analysis request
    print("\n📤 Submitting batch analysis request...")
    
    batch_request = {
        "video_paths": [str(video) for video in video_files],
        "max_concurrent": 3
    }
    
    try:
        response = requests.post(f"{API_URL}/analyze_batch", json=batch_request)
        response.raise_for_status()
        
        batch_result = response.json()
        job_ids = batch_result['job_ids']
        
        print(f"✅ Batch request submitted successfully!")
        print(f"📋 Job IDs: {job_ids}")
        
        # Monitor jobs
        print("\n🔄 Monitoring job progress...")
        start_time = time.time()
        
        completed_jobs = []
        failed_jobs = []
        
        while True:
            # Get all jobs status
            jobs_response = requests.get(f"{API_URL}/jobs")
            jobs_data = jobs_response.json()
            
            # Print GPU stats
            gpu_stats = jobs_data.get('gpu_stats', {})
            print(f"\n⚡ GPU Status:")
            print(f"   - Utilization: {gpu_stats.get('utilization_percent', 0):.1f}%")
            print(f"   - Memory: {gpu_stats.get('used_memory_mb', 0):.0f}/{gpu_stats.get('total_memory_mb', 0):.0f} MB")
            print(f"   - Active Videos: {gpu_stats.get('active_videos', 0)}")
            print(f"   - Queued Videos: {gpu_stats.get('queued_videos', 0)}")
            
            # Check individual job status
            print("\n📊 Job Status:")
            all_completed = True
            
            for job_id in job_ids:
                job_response = requests.get(f"{API_URL}/job/{job_id}")
                if job_response.status_code == 200:
                    job_data = job_response.json()
                    status = job_data['status']
                    progress = job_data['progress']
                    video_name = Path(job_data['video_path']).name
                    
                    if status == 'completed':
                        if job_id not in completed_jobs:
                            completed_jobs.append(job_id)
                            print(f"   ✅ {video_name}: COMPLETED in {job_data.get('processing_time', 0):.1f}s")
                    elif status == 'failed':
                        if job_id not in failed_jobs:
                            failed_jobs.append(job_id)
                            print(f"   ❌ {video_name}: FAILED - {job_data.get('error', 'Unknown error')}")
                    else:
                        all_completed = False
                        print(f"   ⏳ {video_name}: {status} ({progress:.0f}%)")
            
            # Check if all jobs are done
            if all_completed or len(completed_jobs) + len(failed_jobs) == len(job_ids):
                break
            
            # Wait before next check
            time.sleep(5)
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🏁 Parallel Processing Complete!")
        print(f"   - Total Time: {total_time:.1f}s")
        print(f"   - Videos Processed: {len(video_files)}")
        print(f"   - Successful: {len(completed_jobs)}")
        print(f"   - Failed: {len(failed_jobs)}")
        print(f"   - Average Time per Video: {total_time/len(video_files):.1f}s")
        
        # Calculate speedup
        sequential_time = total_time * len(video_files) / max(gpu_stats.get('active_videos', 1), 1)
        speedup = sequential_time / total_time
        print(f"   - Speedup: {speedup:.2f}x vs sequential processing")
        
        # Show result paths
        if completed_jobs:
            print("\n📁 Result Files:")
            for job_id in completed_jobs:
                job_response = requests.get(f"{API_URL}/job/{job_id}")
                if job_response.status_code == 200:
                    job_data = job_response.json()
                    if job_data.get('result_path'):
                        print(f"   - {job_data['result_path']}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return

def check_gpu_utilization():
    """Check current GPU utilization"""
    try:
        response = requests.get(f"{API_URL}/jobs")
        jobs_data = response.json()
        gpu_stats = jobs_data.get('gpu_stats', {})
        
        print("\n📊 Current GPU Utilization:")
        print(f"   - GPU Memory: {gpu_stats.get('utilization_percent', 0):.1f}%")
        print(f"   - Active Videos: {gpu_stats.get('active_videos', 0)}")
        
        return gpu_stats.get('utilization_percent', 0)
    except:
        return 0

if __name__ == "__main__":
    print("🚀 TikTok Video Parallel Processing Test")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        print("✅ API is running")
    except:
        print("❌ API is not running. Please start it first.")
        sys.exit(1)
    
    # Run the test
    test_parallel_processing()
    
    # Final GPU check
    check_gpu_utilization()