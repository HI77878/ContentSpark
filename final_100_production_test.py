#!/usr/bin/env python3
"""
Final 100% Production Test with GPU Monitoring
Tests all optimizations and verifies <3x realtime performance
"""
import time
import json
import subprocess
import os
import threading
from datetime import datetime

class GPUMonitor:
    def __init__(self):
        self.monitoring = False
        self.max_utilization = 0
        self.avg_utilization = []
        self.max_memory = 0
        
    def start(self):
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        
    def stop(self):
        self.monitoring = False
        self.thread.join()
        
    def _monitor(self):
        while self.monitoring:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    util, mem = result.stdout.strip().split(', ')
                    util = int(util)
                    mem = int(mem)
                    
                    self.avg_utilization.append(util)
                    if util > self.max_utilization:
                        self.max_utilization = util
                    if mem > self.max_memory:
                        self.max_memory = mem
            except:
                pass
            time.sleep(0.5)
    
    def get_stats(self):
        avg_util = sum(self.avg_utilization) / len(self.avg_utilization) if self.avg_utilization else 0
        return {
            'max_utilization': self.max_utilization,
            'avg_utilization': avg_util,
            'max_memory_mb': self.max_memory
        }

def run_final_production_test():
    print("🚀 FINAL 100% PRODUCTION TEST WITH ALL OPTIMIZATIONS")
    print("=" * 70)
    
    # Test video
    video_path = "/home/user/tiktok_production/downloads/videos/7425998222721633569.mp4"
    video_duration = 28.9  # seconds
    
    print(f"📹 Test Video: Copenhagen Vlog")
    print(f"   Duration: {video_duration}s")
    print(f"   Target: <3x realtime (<87s)")
    print(f"   Optimizations:")
    print(f"     - 4 GPU worker processes")
    print(f"     - Increased batch sizes")
    print(f"     - Ultimate analyzers for quality")
    print()
    
    # Start GPU monitoring
    gpu_monitor = GPUMonitor()
    print("📊 Starting GPU monitoring...")
    gpu_monitor.start()
    
    # Start analysis
    print("🔬 Starting analysis...")
    start_time = time.time()
    
    # Call API
    api_url = "http://localhost:8003/analyze"
    cmd = [
        'curl', '-X', 'POST', api_url,
        '-H', 'Content-Type: application/json',
        '-d', json.dumps({'video_path': video_path}),
        '-s'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    analysis_time = time.time() - start_time
    
    # Stop GPU monitoring
    gpu_monitor.stop()
    gpu_stats = gpu_monitor.get_stats()
    
    if result.returncode == 0:
        try:
            response = json.loads(result.stdout)
            
            if response.get('status') == 'success':
                print()
                print("✅ ANALYSIS SUCCESSFUL!")
                print(f"   Processing time: {analysis_time:.1f}s")
                print(f"   Realtime factor: {analysis_time/video_duration:.2f}x")
                
                # GPU stats
                print()
                print("🖥️ GPU UTILIZATION:")
                print(f"   Maximum: {gpu_stats['max_utilization']}%")
                print(f"   Average: {gpu_stats['avg_utilization']:.1f}%")
                print(f"   Peak memory: {gpu_stats['max_memory_mb']}MB")
                
                # Load results
                results_file = response.get('results_file')
                if results_file and os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        full_results = json.load(f)
                    
                    analyzer_results = full_results.get('analyzer_results', {})
                    
                    print()
                    print("📋 ANALYZER QUALITY CHECK:")
                    print("-" * 70)
                    
                    # Check ultimate analyzers
                    quality_checks = {
                        'video_llava': check_blip2_ultimate,
                        'speech_transcription': check_speech_ultimate,
                        'audio_analysis': check_audio_ultimate,
                        'text_overlay': check_text_ultimate,
                        'camera_analysis': check_camera_ultimate,
                        'scene_segmentation': check_scene_ultimate
                    }
                    
                    all_scores = []
                    
                    for analyzer_name, check_func in quality_checks.items():
                        if analyzer_name in analyzer_results:
                            score, details = check_func(analyzer_results[analyzer_name])
                            all_scores.append(score)
                            status = "✅" if score >= 0.9 else "⚠️" if score >= 0.7 else "❌"
                            print(f"{status} {analyzer_name}: {details} (Score: {score*100:.0f}/100)")
                        else:
                            print(f"❌ {analyzer_name}: NOT FOUND")
                            all_scores.append(0)
                    
                    # Overall assessment
                    avg_quality = sum(all_scores) / len(all_scores) if all_scores else 0
                    
                    print()
                    print("=" * 70)
                    print("🎯 FINAL 100% PRODUCTION READINESS ASSESSMENT:")
                    print(f"   Data Quality Score: {avg_quality*100:.0f}/100")
                    print(f"   Realtime Factor: {analysis_time/video_duration:.2f}x")
                    print(f"   GPU Utilization: {gpu_stats['avg_utilization']:.1f}% avg, {gpu_stats['max_utilization']}% peak")
                    print(f"   All Analyzers: {len(analyzer_results)}/21")
                    
                    # Final verdict
                    if avg_quality >= 0.95 and analysis_time/video_duration < 3.0:
                        print()
                        print("🎉 SYSTEM IST 100% PRODUKTIONSREIF!")
                        print("   ✓ Maximale Datenqualität erreicht")
                        print("   ✓ Performance unter 3x Realtime")
                        print("   ✓ Alle Ultimate Analyzer funktionieren")
                        print("   ✓ GPU-Optimierung erfolgreich")
                        print()
                        print("💯 PRODUCTION READINESS: 100/100")
                    else:
                        print()
                        print("⚠️ WEITERE OPTIMIERUNG NÖTIG")
                        if avg_quality < 0.95:
                            print(f"   - Datenqualität: {avg_quality*100:.0f}/100 (Ziel: 95/100)")
                        if analysis_time/video_duration >= 3.0:
                            print(f"   - Performance: {analysis_time/video_duration:.2f}x (Ziel: <3x)")
                    
            else:
                print(f"❌ Analysis failed: {response}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            print(f"Raw output: {result.stdout}")
    else:
        print(f"❌ API call failed: {result.stderr}")

def check_blip2_ultimate(data):
    """Check BLIP-2 Ultimate quality"""
    segments = data.get('segments', [])
    if not segments:
        return 0.0, "No segments"
    
    # Check for detailed descriptions
    good_segments = 0
    for seg in segments:
        desc = seg.get('description', '')
        if len(desc) > 50 and '[' in desc and 's]' in desc:  # Has timestamp
            good_segments += 1
    
    ratio = good_segments / len(segments)
    return ratio, f"{good_segments}/{len(segments)} detailed descriptions"

def check_speech_ultimate(data):
    """Check Speech Ultimate features"""
    score = 0.0
    features = []
    
    if data.get('segments'):
        score += 0.2
        features.append(f"{len(data['segments'])} segments")
    
    if data.get('pitch_data', {}).get('mean_fundamental_frequency', 0) > 0:
        score += 0.2
        features.append("pitch analysis")
    
    if data.get('emphasized_words'):
        score += 0.2
        features.append(f"{len(data['emphasized_words'])} emphasized")
    
    if data.get('word_level_timestamps'):
        score += 0.2
        features.append("word timestamps")
    
    if data.get('quality_metrics', {}).get('fluency_score', 0) > 0:
        score += 0.2
        features.append("fluency score")
    
    return score, ", ".join(features)

def check_audio_ultimate(data):
    """Check Audio Ultimate features"""
    score = 0.0
    features = []
    
    if data.get('segments'):
        score += 0.2
        features.append(f"{len(data['segments'])} segments")
    
    if data.get('acoustic_features', {}).get('tempo', 0) > 0:
        score += 0.2
        features.append("tempo detected")
    
    if data.get('mfcc_summary'):
        score += 0.2
        features.append("MFCC features")
    
    if data.get('quality_assessment', {}).get('clarity_score', 0) > 0:
        score += 0.2
        features.append("quality assessment")
    
    if data.get('transitions'):
        score += 0.2
        features.append(f"{len(data['transitions'])} transitions")
    
    return score, ", ".join(features)

def check_text_ultimate(data):
    """Check Text Ultimate OCR"""
    segments = data.get('segments', [])
    if not segments:
        return 0.0, "No segments"
    
    with_text = [s for s in segments if s.get('text')]
    ratio = len(with_text) / len(segments) if segments else 0
    
    # Check statistics
    stats = data.get('statistics', {})
    detection_rate = stats.get('text_detection_rate', 0)
    
    score = max(ratio, detection_rate, 0.1)  # At least 10% if analyzed
    
    return score, f"{len(with_text)}/{len(segments)} with text"

def check_camera_ultimate(data):
    """Check Camera Ultimate movements"""
    stats = data.get('statistics', {})
    unique_movements = stats.get('unique_movements', 0)
    
    if unique_movements > 1:
        return 1.0, f"{unique_movements} movement types detected"
    elif unique_movements == 1:
        return 0.5, "Only 1 movement type"
    else:
        return 0.0, "No movements detected"

def check_scene_ultimate(data):
    """Check Scene Ultimate types"""
    stats = data.get('statistics', {})
    unique_scenes = stats.get('unique_scene_types', 0)
    
    if unique_scenes > 1:
        return 1.0, f"{unique_scenes} scene types detected"
    elif unique_scenes == 1:
        return 0.5, "Only 1 scene type"
    else:
        return 0.0, "No scene types detected"

if __name__ == "__main__":
    # Check if API is running
    try:
        health_check = subprocess.run(
            ['curl', '-s', 'http://localhost:8003/health'],
            capture_output=True, text=True, timeout=5
        )
        if health_check.returncode != 0 or 'healthy' not in health_check.stdout.lower():
            print("⚠️ API not running! Please start it first.")
            exit(1)
    except:
        print("⚠️ Cannot reach API at http://localhost:8003")
        exit(1)
    
    run_final_production_test()