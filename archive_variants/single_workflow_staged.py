#!/usr/bin/env python3
"""
DER EINE WORKFLOW - TikTok Analyzer Production System mit Staged GPU Execution
Löst CUDA OOM durch sequenzielle Stage-Verarbeitung
"""

import os
import sys
import json
import subprocess
import time
import gc
import random
from datetime import datetime
from pathlib import Path
import threading
import yt_dlp

# Add current directory to path
sys.path.insert(0, '/home/user/tiktok_production')

# Fix FFmpeg environment
os.environ['LD_LIBRARY_PATH'] = '/home/user/ffmpeg-install/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PATH'] = '/home/user/ffmpeg-install/bin:' + os.environ.get('PATH', '')
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "protocol_whitelist=file,http,https,tcp,tls"
os.environ['OPENCV_VIDEOIO_PRIORITY_BACKEND'] = "4"  # cv2.CAP_GSTREAMER
os.environ['OPENCV_FFMPEG_MULTITHREADED'] = "0"
os.environ['OPENCV_FFMPEG_DEBUG'] = "1"

# Import staged executor
from utils.staged_gpu_executor import StagedGPUExecutor
from ml_analyzer_registry_complete import ML_ANALYZERS

class TikTokProductionWorkflow:
    def __init__(self):
        self.base_dir = Path("/home/user/tiktok_production")
        self.downloads_dir = self.base_dir / "downloads"
        self.results_dir = self.base_dir / "results"
        self.downloads_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize staged GPU executor
        self.gpu_executor = StagedGPUExecutor()
        
        # Proxy list for rotation
        self.proxies = [
            "http://sp8id82g23:MLrb7ft04WsUg~xjc2@gate.decodo.com:10001",
            "http://sp8id82g23:MLrb7ft04WsUg~xjc2@gate.decodo.com:10002",
            "http://sp8id82g23:MLrb7ft04WsUg~xjc2@gate.decodo.com:10003",
            "http://sp8id82g23:MLrb7ft04WsUg~xjc2@gate.decodo.com:10004",
            "http://sp8id82g23:MLrb7ft04WsUg~xjc2@gate.decodo.com:10005",
            "http://sp8id82g23:MLrb7ft04WsUg~xjc2@gate.decodo.com:10006",
            "http://sp8id82g23:MLrb7ft04WsUg~xjc2@gate.decodo.com:10007",
            "http://sp8id82g23:MLrb7ft04WsUg~xjc2@gate.decodo.com:10008",
            "http://sp8id82g23:MLrb7ft04WsUg~xjc2@gate.decodo.com:10009",
            "http://sp8id82g23:MLrb7ft04WsUg~xjc2@gate.decodo.com:10010"
        ]
        
        # yt-dlp configuration
        self.ydl_opts = {
            'outtmpl': str(self.downloads_dir / '%(id)s.%(ext)s'),
            'format': 'best[ext=mp4]/best',
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
            'socket_timeout': 30,
            'retries': 3,
            'fragment_retries': 3,
            'concurrent_fragment_downloads': 4,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'proxy': random.choice(self.proxies),
        }
        
        # Get all active analyzers from registry
        self.target_analyzers = list(ML_ANALYZERS.keys())
        
        print(f"🚀 TikTok Production Workflow initialized (Staged GPU)")
        print(f"📁 Base directory: {self.base_dir}")
        print(f"🔍 Loaded {len(self.target_analyzers)} analyzers")
        print(f"🎯 Using Staged GPU Executor (CUDA OOM prevention)")
    
    def download_tiktok(self, url):
        """Download with yt-dlp using proxy rotation"""
        print(f"📥 Downloading: {url}")
        
        # Try download with different proxies
        errors = []
        for attempt in range(3):
            proxy = random.choice(self.proxies)
            print(f"🔄 Attempt {attempt+1}/3 using proxy: {proxy.split('@')[1]}")
            
            ydl_opts = self.ydl_opts.copy()
            ydl_opts['proxy'] = proxy
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    video_id = info.get('id', url.split('/')[-1].split('?')[0])
                    
                    video_path = self.downloads_dir / f"{video_id}.mp4"
                    if not video_path.exists():
                        for ext in ['webm', 'mov', 'avi']:
                            alt_path = self.downloads_dir / f"{video_id}.{ext}"
                            if alt_path.exists():
                                video_path = alt_path
                                break
                        else:
                            raise Exception(f"Downloaded file not found: {video_path}")
                    
                    print(f"✅ Downloaded: {video_path} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")
                    return str(video_path), video_id
                    
            except Exception as e:
                error_msg = str(e)
                errors.append(f"Proxy {proxy.split('@')[1]}: {error_msg}")
                print(f"⚠️ Attempt {attempt+1} failed: {error_msg}")
                
                if "blocked" in error_msg.lower() or "403" in error_msg:
                    continue
        
        print(f"❌ Download failed after 3 attempts")
        print(f"Errors: {'; '.join(errors)}")
        raise Exception(f"Download failed: {errors[-1] if errors else 'Unknown error'}")
    
    def get_video_metadata(self, video_path):
        """Get basic video metadata"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return {"duration": 0, "width": 0, "height": 0}
            
            data = json.loads(result.stdout)
            format_info = data.get('format', {})
            video_stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), {})
            
            return {
                "duration": float(format_info.get('duration', 0)),
                "width": int(video_stream.get('width', 0)),
                "height": int(video_stream.get('height', 0)),
                "fps": float(video_stream.get('r_frame_rate', '0/1').split('/')[0]) / max(1, float(video_stream.get('r_frame_rate', '0/1').split('/')[1]))
            }
        except:
            return {"duration": 0, "width": 0, "height": 0, "fps": 0}
    
    def analyze_video(self, video_path, video_id):
        """Staged analysis with all analyzers using GPU Executor"""
        print(f"🔍 Analyzing video: {video_path}")
        print(f"🎯 Using Staged GPU Executor to prevent CUDA OOM")
        total_start = time.time()
        
        # Get video metadata
        metadata = self.get_video_metadata(video_path)
        
        # Execute all analyzers using staged GPU executor
        analyzer_results = self.gpu_executor.execute_all_stages(video_path, self.target_analyzers)
        
        # Get performance metrics
        performance_metrics = self.gpu_executor.get_performance_metrics()
        
        # Format results
        results = {
            'video_id': video_id,
            'video_path': video_path,
            'timestamp': datetime.now().isoformat(),
            'video_metadata': metadata,
            'analyzers': {},
            'summary': {
                'total_analyzers': len(self.target_analyzers),
                'successful': performance_metrics.get('successful_analyzers', 0),
                'failed': len(self.target_analyzers) - performance_metrics.get('successful_analyzers', 0),
                'total_time': performance_metrics.get('total_time', 0)
            }
        }
        
        # Convert analyzer results to expected format
        for analyzer_name, result in analyzer_results.items():
            if 'error' in result:
                results['analyzers'][analyzer_name] = {
                    'status': 'error',
                    'duration': 0,
                    'error': result['error']
                }
            else:
                results['analyzers'][analyzer_name] = {
                    'status': 'success',
                    'duration': 0,  # Not tracked individually in staged executor
                    'result': result
                }
        
        total_time = time.time() - total_start
        results['summary']['total_time'] = total_time
        results['performance'] = {
            'total_seconds': total_time,
            'realtime_ratio': total_time / max(1, metadata.get('duration', 1)),
            'analyzers_per_second': len(self.target_analyzers) / total_time
        }
        
        print(f"\n📊 Analysis Summary:")
        print(f"   ✅ Successful: {results['summary']['successful']}")
        print(f"   ❌ Failed: {results['summary']['failed']}")
        print(f"   ⏱️ Total Time: {total_time:.1f}s")
        print(f"   🎯 Performance: {total_time / max(1, metadata.get('duration', 1)):.1f}x realtime")
        
        return results
    
    def save_results(self, results):
        """Save all results in ONE JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{results['video_id']}_complete_{timestamp}.json"
        filepath = self.results_dir / filename
        
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                return super().default(obj)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
        
        file_size = filepath.stat().st_size / 1024 / 1024
        print(f"💾 Results saved: {filepath} ({file_size:.1f} MB)")
        
        return filepath
    
    def run(self, tiktok_url):
        """DER EINE WORKFLOW mit Staged GPU Execution"""
        print(f"\n🚀 STARTING TIKTOK ANALYSIS WORKFLOW (STAGED GPU)")
        print(f"📍 URL: {tiktok_url}")
        print(f"⏰ Target: <3 minutes")
        print(f"🎯 Analyzers: {len(self.target_analyzers)}")
        print(f"🔧 Mode: Staged GPU Execution (CUDA OOM Prevention)")
        print("="*60)
        
        workflow_start = time.time()
        
        try:
            # 1. Download
            print(f"\n📥 PHASE 1: DOWNLOAD")
            video_path, video_id = self.download_tiktok(tiktok_url)
            
            # 2. Analyze with staged GPU
            print(f"\n🔍 PHASE 2: STAGED GPU ANALYSIS")
            results = self.analyze_video(video_path, video_id)
            
            # 3. Save
            print(f"\n💾 PHASE 3: SAVE RESULTS")
            output_path = self.save_results(results)
            
            total_time = time.time() - workflow_start
            
            print(f"\n" + "="*60)
            print(f"✅ WORKFLOW COMPLETE!")
            print(f"📊 Analyzers: {results['summary']['successful']}/{len(self.target_analyzers)}")
            print(f"⏱️ Total Time: {total_time:.1f}s")
            print(f"🎯 Success Rate: {results['summary']['successful']/len(self.target_analyzers)*100:.1f}%")
            print(f"📁 Results: {output_path}")
            
            if total_time < 180:  # 3 minutes
                print(f"🏆 TARGET ACHIEVED: <3 minutes!")
            else:
                print(f"⚠️ Target missed: {total_time/60:.1f} minutes")
            
            return {
                'status': 'success',
                'output_path': str(output_path),
                'total_time': total_time,
                'analyzers_successful': results['summary']['successful'],
                'analyzers_total': len(self.target_analyzers)
            }
            
        except Exception as e:
            total_time = time.time() - workflow_start
            print(f"\n❌ WORKFLOW FAILED after {total_time:.1f}s")
            print(f"Error: {str(e)}")
            
            return {
                'status': 'error',
                'error': str(e),
                'total_time': total_time
            }

    def run_local(self, video_path):
        """Run workflow with local video (skip download)"""
        print(f"\n🚀 STARTING LOCAL VIDEO ANALYSIS (STAGED GPU)")
        print(f"📍 Video: {video_path}")
        print(f"⏰ Target: <3 minutes")
        print(f"🎯 Analyzers: {len(self.target_analyzers)}")
        print(f"🔧 Mode: Staged GPU Execution (CUDA OOM Prevention)")
        print("="*60)
        
        workflow_start = time.time()
        video_id = Path(video_path).stem
        
        try:
            # Analyze with staged GPU
            print(f"\n🔍 PHASE 1: STAGED GPU ANALYSIS")
            results = self.analyze_video(video_path, video_id)
            
            # Save
            print(f"\n💾 PHASE 2: SAVE RESULTS")
            output_path = self.save_results(results)
            
            total_time = time.time() - workflow_start
            
            print(f"\n" + "="*60)
            print(f"✅ WORKFLOW COMPLETE!")
            print(f"📊 Analyzers: {results['summary']['successful']}/{len(self.target_analyzers)}")
            print(f"⏱️ Total Time: {total_time:.1f}s")
            print(f"🎯 Success Rate: {results['summary']['successful']/len(self.target_analyzers)*100:.1f}%")
            print(f"📁 Results: {output_path}")
            
            if total_time < 180:  # 3 minutes
                print(f"🏆 TARGET ACHIEVED: <3 minutes!")
            else:
                print(f"⚠️ Target missed: {total_time/60:.1f} minutes")
            
            return output_path
            
        except Exception as e:
            total_time = time.time() - workflow_start
            print(f"\n❌ WORKFLOW FAILED after {total_time:.1f}s")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# CLI Interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python single_workflow_staged.py <tiktok_url>")
        print("  python single_workflow_staged.py --local <video_path>")
        sys.exit(1)
    
    workflow = TikTokProductionWorkflow()
    
    if sys.argv[1] == "--local" and len(sys.argv) >= 3:
        # Local video mode
        result = workflow.run_local(sys.argv[2])
        sys.exit(0 if result else 1)
    else:
        # TikTok URL mode
        result = workflow.run(sys.argv[1])
        sys.exit(0 if result['status'] == 'success' else 1)