#!/usr/bin/env python3
"""
TikTok Analyzer Workflow - Automated end-to-end video analysis
Handles download, analysis, and result storage
"""
import sys
import os
import time
import json
import requests
import argparse
from datetime import datetime
import subprocess
import re
from pathlib import Path

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mass_processing.tiktok_downloader import TikTokDownloader

class TikTokAnalyzerWorkflow:
    def __init__(self, api_url="http://localhost:8003"):
        self.api_url = api_url
        self.downloader = TikTokDownloader()
        self.results_dir = Path("/home/user/tiktok_production/results")
        self.results_dir.mkdir(exist_ok=True)
        
    def extract_video_id(self, url: str) -> str:
        """Extract TikTok video ID from URL"""
        # Pattern for TikTok URLs
        patterns = [
            r'video/(\d+)',
            r'v/(\d+)',
            r'item/(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # If no pattern matches, use timestamp
        return str(int(time.time()))
    
    def check_api_health(self) -> bool:
        """Check if API is running and healthy"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get('status') == 'healthy'
        except:
            pass
        return False
    
    def download_video(self, tiktok_url: str) -> tuple:
        """Download TikTok video"""
        print(f"📥 Downloading video from: {tiktok_url}")
        
        try:
            result = self.downloader.download_video(tiktok_url)
            
            if result['success']:
                video_path = result['video_path']
                metadata = result.get('metadata', {})
                print(f"✅ Downloaded successfully: {video_path}")
                return video_path, metadata
            else:
                print(f"❌ Download failed: {result.get('error', 'Unknown error')}")
                return None, None
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None, None
    
    def analyze_video(self, video_path: str) -> dict:
        """Call analysis API"""
        print(f"🔬 Analyzing video: {video_path}")
        
        payload = {
            'video_path': video_path
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        try:
            # Start analysis
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/analyze",
                json=payload,
                headers=headers,
                timeout=300  # 5 minute timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'success':
                    analysis_time = time.time() - start_time
                    print(f"✅ Analysis completed in {analysis_time:.1f}s")
                    
                    # Load full results
                    results_file = data.get('results_file')
                    if results_file and os.path.exists(results_file):
                        with open(results_file, 'r') as f:
                            full_results = json.load(f)
                        return full_results
                    else:
                        return data
                else:
                    print(f"❌ Analysis failed: {data.get('error', 'Unknown error')}")
                    return None
            else:
                print(f"❌ API error: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print("❌ Analysis timeout (>5 minutes)")
            return None
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return None
    
    def save_results(self, results: dict, video_id: str, metadata: dict = None) -> str:
        """Save analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tiktok_{video_id}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Add workflow metadata
        results['workflow_metadata'] = {
            'video_id': video_id,
            'analysis_timestamp': timestamp,
            'tiktok_metadata': metadata or {},
            'workflow_version': '1.0'
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filepath}")
        return str(filepath)
    
    def generate_summary(self, results: dict) -> None:
        """Generate and print analysis summary"""
        print("\n" + "="*60)
        print("📊 ANALYSIS SUMMARY")
        print("="*60)
        
        # Metadata
        metadata = results.get('metadata', {})
        print(f"Processing time: {metadata.get('processing_time_seconds', 0):.1f}s")
        print(f"Realtime factor: {metadata.get('realtime_factor', 0):.2f}x")
        print(f"Analyzers: {metadata.get('successful_analyzers', 0)}/{metadata.get('total_analyzers', 0)}")
        
        # Key findings
        analyzer_results = results.get('analyzer_results', {})
        
        # Speech transcription
        if 'speech_transcription' in analyzer_results:
            st = analyzer_results['speech_transcription']
            segments = st.get('segments', [])
            if segments:
                print(f"\n🗣️ Speech: {st.get('language', 'unknown')} - {len(segments)} segments")
                print(f"   Sample: \"{segments[0].get('text', '')[:100]}...\"")
        
        # Objects detected
        if 'object_detection' in analyzer_results:
            od = analyzer_results['object_detection']
            unique_objects = od.get('unique_objects', 0)
            print(f"\n👁️ Objects: {unique_objects} unique types detected")
        
        # Text overlays
        if 'text_overlay' in analyzer_results:
            to = analyzer_results['text_overlay']
            segments = to.get('segments', [])
            with_text = [s for s in segments if s.get('text')]
            if with_text:
                print(f"\n📝 Text overlays: {len(with_text)} detected")
        
        # Camera movements
        if 'camera_analysis' in analyzer_results:
            ca = analyzer_results['camera_analysis']
            movements = ca.get('statistics', {}).get('unique_movements', 0)
            print(f"\n🎥 Camera: {movements} movement types")
        
        # Scene types
        if 'scene_segmentation' in analyzer_results:
            ss = analyzer_results['scene_segmentation']
            scenes = ss.get('statistics', {}).get('unique_scene_types', 0)
            print(f"\n🎬 Scenes: {scenes} different scene types")
        
        print("\n" + "="*60)
    
    def run(self, tiktok_url: str) -> dict:
        """Run complete workflow"""
        print(f"\n🚀 Starting TikTok analysis workflow")
        print(f"   URL: {tiktok_url}")
        
        # Check API health
        if not self.check_api_health():
            print("❌ API is not running! Please start it first:")
            print("   cd /home/user/tiktok_production")
            print("   source fix_ffmpeg_env.sh")
            print("   python3 api/stable_production_api_multiprocess.py")
            return {'error': 'API not running'}
        
        # Extract video ID
        video_id = self.extract_video_id(tiktok_url)
        print(f"   Video ID: {video_id}")
        
        # Download video
        video_path, metadata = self.download_video(tiktok_url)
        if not video_path:
            return {'error': 'Download failed'}
        
        # Analyze video
        results = self.analyze_video(video_path)
        if not results:
            return {'error': 'Analysis failed'}
        
        # Save results
        output_path = self.save_results(results, video_id, metadata)
        
        # Generate summary
        self.generate_summary(results)
        
        return {
            'success': True,
            'video_id': video_id,
            'video_path': video_path,
            'results_path': output_path,
            'summary': {
                'processing_time': results['metadata'].get('processing_time_seconds', 0),
                'realtime_factor': results['metadata'].get('realtime_factor', 0),
                'analyzers_count': results['metadata'].get('successful_analyzers', 0)
            }
        }

def main():
    parser = argparse.ArgumentParser(description='TikTok Video Analysis Workflow')
    parser.add_argument('url', help='TikTok video URL')
    parser.add_argument('--api-url', default='http://localhost:8003', help='API URL')
    
    args = parser.parse_args()
    
    # Initialize workflow
    workflow = TikTokAnalyzerWorkflow(api_url=args.api_url)
    
    # Run analysis
    result = workflow.run(args.url)
    
    if result.get('success'):
        print(f"\n✅ Workflow completed successfully!")
        print(f"   Results: {result['results_path']}")
        sys.exit(0)
    else:
        print(f"\n❌ Workflow failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()