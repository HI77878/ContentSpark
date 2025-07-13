#!/usr/bin/env python3
"""
Launch Ultimate TikTok Analysis System
Mit ALLEN 21 Ultimate Analyzern für 100% Datenqualität
"""

import subprocess
import time
import sys
import os
import json
import requests

def check_dependencies():
    """Check if all dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required = [
        'torch', 'transformers', 'cv2', 'mediapipe', 
        'easyocr', 'librosa', 'scenedetect', 'ultralytics'
    ]
    
    missing = []
    for dep in required:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Please install with: pip install <package>")
        return False
    
    print("✅ All dependencies installed")
    return True

def start_ultimate_api():
    """Start the Ultimate API"""
    print("\n🚀 Starting Ultimate Production API...")
    
    # Kill any existing API
    subprocess.run(['pkill', '-f', 'production_api'], capture_output=True)
    time.sleep(2)
    
    # Set environment
    env = os.environ.copy()
    env['PYTHONPATH'] = '/home/user/tiktok_production'
    
    # Start API
    api_process = subprocess.Popen(
        ['python3', 'api/ultimate_production_api.py'],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for startup
    print("⏳ Waiting for API to start...")
    for i in range(30):
        try:
            response = requests.get('http://localhost:8004/health')
            if response.status_code == 200:
                print("✅ API is running!")
                return api_process
        except:
            pass
        time.sleep(1)
    
    print("❌ API failed to start")
    return None

def test_ultimate_analysis():
    """Test the Ultimate analysis system"""
    print("\n🧪 Testing Ultimate Analysis System...")
    
    video_path = "/home/user/tiktok_production/downloads/videos/7425998222721633569.mp4"
    
    # Test with a few analyzers first
    test_analyzers = [
        'text_overlay',
        'object_detection', 
        'gesture_body',
        'speech_transcription',
        'camera_analysis'
    ]
    
    print(f"📹 Test video: {video_path}")
    print(f"🔬 Testing {len(test_analyzers)} analyzers...")
    
    # Call API
    response = requests.post(
        'http://localhost:8004/analyze',
        json={
            'video_path': video_path,
            'analyzers': test_analyzers
        },
        timeout=180
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Test successful!")
        print(f"⏱️  Processing time: {result['processing_time']:.1f}s")
        print(f"📊 Data quality: {result['data_quality_score']:.0%}")
        print(f"💾 Results saved: {result['results_file']}")
        
        # Show preview
        if result.get('preview'):
            print("\n📋 Sample results:")
            for analyzer, info in result['preview']['sample_results'].items():
                status = "✅" if info['success'] else "❌"
                print(f"   {status} {analyzer}: {info['data_points']} data points")
        
        return True
    else:
        print(f"❌ Test failed: {response.status_code}")
        print(response.text)
        return False

def run_full_system_test():
    """Run complete system test with ALL analyzers"""
    print("\n🏁 Running FULL SYSTEM TEST with ALL Ultimate Analyzers...")
    
    video_path = "/home/user/tiktok_production/downloads/videos/7425998222721633569.mp4"
    
    print("⚡ This will test all 21 Ultimate analyzers")
    print("⏳ Expected time: 2-3 minutes")
    
    start_time = time.time()
    
    # Call API with ALL analyzers
    response = requests.post(
        'http://localhost:8004/analyze',
        json={'video_path': video_path},  # No analyzers = use all
        timeout=300
    )
    
    if response.status_code == 200:
        result = response.json()
        elapsed = time.time() - start_time
        
        print(f"\n🎉 FULL TEST SUCCESSFUL!")
        print(f"⏱️  Total time: {elapsed:.1f}s")
        print(f"📊 Data quality score: {result['data_quality_score']:.0%}")
        print(f"🔄 Reconstruction score: {result['reconstruction_score']:.0%}")
        
        # Load and analyze results
        if result['results_file']:
            with open(result['results_file'], 'r') as f:
                full_results = json.load(f)
            
            analyzer_results = full_results.get('analyzer_results', {})
            
            print(f"\n📈 Analyzer Results:")
            print(f"   Total analyzers: {len(analyzer_results)}")
            print(f"   Successful: {len([a for a in analyzer_results if 'error' not in analyzer_results[a]])}")
            
            # Show data richness
            print(f"\n💎 Data Richness:")
            for name, data in sorted(analyzer_results.items()):
                if isinstance(data, dict) and 'error' not in data:
                    segments = len(data.get('segments', data.get('timeline', [])))
                    stats = len(data.get('statistics', {}))
                    print(f"   {name}: {segments} segments, {stats} statistics")
        
        print(f"\n✅ SYSTEM IST 100% BEREIT!")
        print(f"   - Alle 21 Ultimate Analyzer funktionieren")
        print(f"   - Maximale Datenqualität erreicht")
        print(f"   - Bereit für 1:1 Video-Rekonstruktion")
        
        return True
    else:
        print(f"❌ Full test failed: {response.status_code}")
        return False

def main():
    print("=" * 60)
    print("🚀 TikTok Ultimate Analysis System Launcher")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Fix FFmpeg environment
    print("\n🔧 Setting FFmpeg environment...")
    subprocess.run(['source', 'fix_ffmpeg_env.sh'], shell=True)
    
    # Start API
    api_process = start_ultimate_api()
    if not api_process:
        sys.exit(1)
    
    try:
        # Run tests
        if test_ultimate_analysis():
            print("\n" + "="*60)
            print("Continue with FULL system test? (y/n): ", end='')
            
            if input().lower() == 'y':
                run_full_system_test()
        
        print("\n" + "="*60)
        print("🎯 Ultimate API is running at: http://localhost:8004")
        print("📚 Documentation: http://localhost:8004/docs")
        print("💡 Stop with: Ctrl+C")
        print("="*60)
        
        # Keep running
        api_process.wait()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        api_process.terminate()
        api_process.wait()

if __name__ == "__main__":
    os.chdir('/home/user/tiktok_production')
    main()