#!/usr/bin/env python3
"""
DER EINE WORKFLOW - TikTok Analyzer Production System
Keine Duplikate, keine Tests, nur Production!
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

class TikTokProductionWorkflow:
    def __init__(self):
        self.base_dir = Path("/home/user/tiktok_production")
        self.downloads_dir = self.base_dir / "downloads"
        self.results_dir = self.base_dir / "results"
        self.downloads_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
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
        
        # yt-dlp configuration (wie im funktionierenden System)
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
            'proxy': random.choice(self.proxies),  # Random proxy selection
        }
        
        # Registry der 19 funktionierenden Analyzer
        self.analyzers = {
            'qwen2_vl_temporal': {
                'module': 'analyzers.qwen2_vl_temporal_analyzer',
                'class': 'Qwen2VLTemporalAnalyzer',
                'gpu_stage': 1,
                'priority': 1
            },
            'object_detection': {
                'module': 'analyzers.gpu_batch_object_detection_yolo',
                'class': 'GPUBatchObjectDetectionYOLO',
                'gpu_stage': 2,
                'priority': 1
            },
            'text_overlay': {
                'module': 'analyzers.text_overlay_tiktok_fixed',
                'class': 'TikTokTextOverlayAnalyzer',
                'gpu_stage': 2,
                'priority': 1
            },
            'speech_transcription': {
                'module': 'analyzers.speech_transcription_ultimate',
                'class': 'UltimateSpeechTranscription',
                'gpu_stage': 2,
                'priority': 1
            },
            'body_pose': {
                'module': 'analyzers.body_pose_yolov8',
                'class': 'BodyPoseYOLOv8',
                'gpu_stage': 3,
                'priority': 2
            },
            'background_segmentation': {
                'module': 'analyzers.background_segmentation_light',
                'class': 'GPUBatchBackgroundSegmentationLight',
                'gpu_stage': 3,
                'priority': 2
            },
            'camera_analysis': {
                'module': 'analyzers.camera_analysis_fixed',
                'class': 'GPUBatchCameraAnalysisFixed',
                'gpu_stage': 3,
                'priority': 2
            },
            'scene_segmentation': {
                'module': 'analyzers.scene_segmentation_fixed',
                'class': 'SceneSegmentationFixedAnalyzer',
                'gpu_stage': 3,
                'priority': 2
            },
            'color_analysis': {
                'module': 'analyzers.gpu_batch_color_analysis',
                'class': 'GPUBatchColorAnalysis',
                'gpu_stage': 3,
                'priority': 2
            },
            'content_quality': {
                'module': 'analyzers.gpu_batch_content_quality_fixed',
                'class': 'GPUBatchContentQualityFixed',
                'gpu_stage': 3,
                'priority': 2
            },
            'cut_analysis': {
                'module': 'analyzers.cut_analysis_fixed',
                'class': 'CutAnalysisFixedAnalyzer',
                'gpu_stage': 3,
                'priority': 2
            },
            'age_estimation': {
                'module': 'analyzers.age_gender_insightface',
                'class': 'AgeGenderInsightFace',
                'gpu_stage': 3,
                'priority': 2
            },
            'eye_tracking': {
                'module': 'analyzers.gpu_batch_eye_tracking',
                'class': 'GPUBatchEyeTracking',
                'gpu_stage': 3,
                'priority': 2
            },
            'audio_analysis': {
                'module': 'analyzers.audio_analysis_ultimate',
                'class': 'GPUBatchAudioAnalysisEnhanced',
                'gpu_stage': 4,
                'priority': 3
            },
            'audio_environment': {
                'module': 'analyzers.audio_environment_enhanced',
                'class': 'AudioEnvironmentEnhanced',
                'gpu_stage': 4,
                'priority': 3
            },
            'speech_emotion': {
                'module': 'analyzers.gpu_batch_speech_emotion',
                'class': 'GPUBatchSpeechEmotion',
                'gpu_stage': 4,
                'priority': 3
            },
            'speech_flow': {
                'module': 'analyzers.gpu_batch_speech_flow',
                'class': 'GPUBatchSpeechFlow',
                'gpu_stage': 4,
                'priority': 3
            },
            'temporal_flow': {
                'module': 'analyzers.narrative_analysis_wrapper',
                'class': 'NarrativeAnalysisWrapper',
                'gpu_stage': 4,
                'priority': 3
            },
            'cross_analyzer_intelligence': {
                'module': 'analyzers.cross_analyzer_intelligence',
                'class': 'CrossAnalyzerIntelligence',
                'gpu_stage': 5,
                'priority': 4
            }
        }
        
        print(f"🚀 TikTok Production Workflow initialized")
        print(f"📁 Base directory: {self.base_dir}")
        print(f"🔍 Loaded {len(self.analyzers)} analyzers")
    
    def download_tiktok(self, url):
        """Download with yt-dlp using proxy rotation"""
        print(f"📥 Downloading: {url}")
        
        # Try download with different proxies
        errors = []
        for attempt in range(3):  # Try 3 different proxies
            proxy = random.choice(self.proxies)
            print(f"🔄 Attempt {attempt+1}/3 using proxy: {proxy.split('@')[1]}")
            
            # Update proxy in options
            ydl_opts = self.ydl_opts.copy()
            ydl_opts['proxy'] = proxy
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Extract info first
                    info = ydl.extract_info(url, download=True)
                    
                    # Get video ID
                    video_id = info.get('id', url.split('/')[-1].split('?')[0])
                    
                    # Find downloaded file
                    video_path = self.downloads_dir / f"{video_id}.mp4"
                    
                    # Check if file exists
                    if not video_path.exists():
                        # Try alternative extensions
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
                
                # If it's just blocked IP, try next proxy
                if "blocked" in error_msg.lower() or "403" in error_msg:
                    continue
                # For other errors, might still try next proxy
        
        # All attempts failed
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
    
    def run_analyzer(self, analyzer_name, analyzer_config, video_path):
        """Run single analyzer"""
        try:
            print(f"🔄 Starting {analyzer_name}...")
            start_time = time.time()
            
            # Dynamic import
            module_name = analyzer_config['module']
            class_name = analyzer_config['class']
            
            module = __import__(module_name, fromlist=[class_name])
            analyzer_class = getattr(module, class_name)
            
            # Create and run analyzer
            analyzer = analyzer_class()
            result = analyzer.analyze(video_path)
            
            # Cleanup
            if hasattr(analyzer, 'cleanup'):
                analyzer.cleanup()
            del analyzer
            
            # GPU cleanup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            gc.collect()
            
            duration = time.time() - start_time
            print(f"✅ {analyzer_name} completed in {duration:.1f}s")
            
            return {
                'status': 'success',
                'duration': duration,
                'result': result
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ {analyzer_name} failed after {duration:.1f}s: {str(e)}")
            return {
                'status': 'error',
                'duration': duration,
                'error': str(e)
            }
    
    def analyze_video(self, video_path, video_id):
        """Staged analysis with all 19 analyzers"""
        print(f"🔍 Analyzing video: {video_path}")
        total_start = time.time()
        
        # Get video metadata
        metadata = self.get_video_metadata(video_path)
        
        results = {
            'video_id': video_id,
            'video_path': video_path,
            'timestamp': datetime.now().isoformat(),
            'video_metadata': metadata,
            'analyzers': {},
            'summary': {
                'total_analyzers': len(self.analyzers),
                'successful': 0,
                'failed': 0,
                'total_time': 0
            }
        }
        
        # Group analyzers by stage
        stages = {}
        for name, config in self.analyzers.items():
            stage = config['gpu_stage']
            if stage not in stages:
                stages[stage] = []
            stages[stage].append((name, config))
        
        # Process each stage sequentially
        for stage_num in sorted(stages.keys()):
            stage_analyzers = stages[stage_num]
            print(f"\n🎯 Processing Stage {stage_num}: {len(stage_analyzers)} analyzers")
            
            # Run stage analyzers in parallel (within GPU limits)
            max_workers = 2 if stage_num <= 2 else 4  # Limit GPU workers
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                
                for analyzer_name, analyzer_config in stage_analyzers:
                    future = executor.submit(self.run_analyzer, analyzer_name, analyzer_config, video_path)
                    futures[future] = analyzer_name
                
                # Collect results
                for future in as_completed(futures):
                    analyzer_name = futures[future]
                    result = future.result()
                    
                    results['analyzers'][analyzer_name] = result
                    
                    if result['status'] == 'success':
                        results['summary']['successful'] += 1
                    else:
                        results['summary']['failed'] += 1
            
            # Cleanup between stages
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            gc.collect()
            
            print(f"✅ Stage {stage_num} completed")
        
        # Final summary
        total_time = time.time() - total_start
        results['summary']['total_time'] = total_time
        results['performance'] = {
            'total_seconds': total_time,
            'realtime_ratio': total_time / max(1, metadata.get('duration', 1)),
            'analyzers_per_second': len(self.analyzers) / total_time
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
        
        # Custom JSON encoder for better formatting
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
        """DER EINE WORKFLOW"""
        print(f"\n🚀 STARTING TIKTOK ANALYSIS WORKFLOW")
        print(f"📍 URL: {tiktok_url}")
        print(f"⏰ Target: <3 minutes")
        print(f"🎯 Analyzers: {len(self.analyzers)}")
        print("="*60)
        
        workflow_start = time.time()
        
        try:
            # 1. Download
            print(f"\n📥 PHASE 1: DOWNLOAD")
            video_path, video_id = self.download_tiktok(tiktok_url)
            
            # 2. Analyze
            print(f"\n🔍 PHASE 2: ANALYSIS")
            results = self.analyze_video(video_path, video_id)
            
            # 3. Save
            print(f"\n💾 PHASE 3: SAVE RESULTS")
            output_path = self.save_results(results)
            
            total_time = time.time() - workflow_start
            
            print(f"\n" + "="*60)
            print(f"✅ WORKFLOW COMPLETE!")
            print(f"📊 Analyzers: {results['summary']['successful']}/{len(self.analyzers)}")
            print(f"⏱️ Total Time: {total_time:.1f}s")
            print(f"🎯 Success Rate: {results['summary']['successful']/len(self.analyzers)*100:.1f}%")
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
                'analyzers_total': len(self.analyzers)
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


# CLI Interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python single_workflow.py <tiktok_url>")
        print("Example: python single_workflow.py 'https://www.tiktok.com/@username/video/1234567890'")
        sys.exit(1)
    
    workflow = TikTokProductionWorkflow()
    result = workflow.run(sys.argv[1])
    
    if result['status'] == 'success':
        sys.exit(0)
    else:
        sys.exit(1)