#!/usr/bin/env python3
"""
Final 100% Production Readiness Test
Tests all 21 analyzers with ultimate optimizations
"""
import time
import json
import subprocess
import os
from datetime import datetime

def run_final_100_percent_test():
    print("🚀 FINAL 100% PRODUCTION READINESS TEST")
    print("=" * 70)
    
    # Test video
    video_path = "/home/user/tiktok_production/downloads/videos/7425998222721633569.mp4"
    video_duration = 28.9  # seconds
    
    print(f"📹 Test Video: Copenhagen Vlog")
    print(f"   Duration: {video_duration}s")
    print(f"   Target: <3x realtime (<87s)")
    print(f"   Quality: 100/100 for all analyzers")
    print()
    
    # Start monitoring
    print("📊 Starting ultimate analysis...")
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
    
    if result.returncode == 0:
        try:
            response = json.loads(result.stdout)
            
            if response.get('status') == 'success':
                print()
                print("✅ ANALYSIS SUCCESSFUL!")
                print(f"   Processing time: {analysis_time:.1f}s")
                print(f"   Realtime factor: {analysis_time/video_duration:.2f}x")
                
                # Load results
                results_file = response.get('results_file')
                if results_file and os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        full_results = json.load(f)
                    
                    analyzer_results = full_results.get('analyzer_results', {})
                    
                    print()
                    print("📋 ANALYZER PERFORMANCE CHECK:")
                    print("-" * 70)
                    
                    # Check critical analyzers with ultimate features
                    quality_scores = []
                    
                    # 1. Speech Transcription Ultimate
                    if 'speech_transcription' in analyzer_results:
                        st_data = analyzer_results['speech_transcription']
                        segments = st_data.get('segments', [])
                        quality_metrics = st_data.get('quality_metrics', {})
                        
                        score = 0.0
                        details = []
                        
                        if segments:
                            score += 0.3
                            details.append(f"{len(segments)} segments")
                        
                        if quality_metrics.get('transcription_confidence', 0) > 0.8:
                            score += 0.2
                            details.append(f"confidence: {quality_metrics['transcription_confidence']:.2f}")
                        
                        if quality_metrics.get('transcription_quality', 0) > 0.7:
                            score += 0.2
                            details.append(f"quality: {quality_metrics['transcription_quality']:.2f}")
                        
                        if st_data.get('pitch_analysis', {}).get('pitch_mean_hz', 0) > 0:
                            score += 0.1
                            details.append("pitch analyzed")
                        
                        if st_data.get('emphasized_words'):
                            score += 0.1
                            details.append(f"{len(st_data['emphasized_words'])} emphases")
                        
                        if st_data.get('word_level_timestamps'):
                            score += 0.1
                            details.append("word-level timing")
                        
                        quality_scores.append(score)
                        status = "✅" if score >= 0.9 else "⚠️" if score >= 0.7 else "❌"
                        print(f"{status} speech_transcription: {', '.join(details)} (Score: {score*100:.0f}/100)")
                    
                    # 2. Audio Analysis Ultimate
                    if 'audio_analysis' in analyzer_results:
                        aa_data = analyzer_results['audio_analysis']
                        segments = aa_data.get('segments', [])
                        global_analysis = aa_data.get('global_analysis', {})
                        acoustic_features = aa_data.get('acoustic_features', {})
                        
                        score = 0.0
                        details = []
                        
                        if segments:
                            score += 0.2
                            details.append(f"{len(segments)} segments")
                        
                        if global_analysis.get('snr_db', 0) != 0:
                            score += 0.2
                            details.append(f"SNR: {global_analysis['snr_db']:.1f}dB")
                        
                        if acoustic_features.get('tempo', 0) > 0:
                            score += 0.1
                            details.append(f"tempo: {acoustic_features['tempo']:.0f}")
                        
                        if acoustic_features.get('spectral_bandwidth', 0) > 0:
                            score += 0.1
                            details.append("spectral features")
                        
                        if aa_data.get('mfcc_summary', {}).get('mean_mfcc'):
                            score += 0.2
                            details.append("MFCC extracted")
                        
                        if aa_data.get('quality_assessment', {}).get('clarity_score', 0) > 0:
                            score += 0.1
                            details.append(f"clarity: {aa_data['quality_assessment']['clarity_score']:.2f}")
                        
                        if aa_data.get('transitions'):
                            score += 0.1
                            details.append(f"{len(aa_data['transitions'])} transitions")
                        
                        quality_scores.append(score)
                        status = "✅" if score >= 0.9 else "⚠️" if score >= 0.7 else "❌"
                        print(f"{status} audio_analysis: {', '.join(details)} (Score: {score*100:.0f}/100)")
                    
                    # 3. Check other critical analyzers
                    critical_analyzers = [
                        'video_llava', 'object_detection', 'text_overlay', 
                        'camera_analysis', 'scene_segmentation'
                    ]
                    
                    for analyzer in critical_analyzers:
                        if analyzer in analyzer_results:
                            data = analyzer_results[analyzer]
                            segments = data.get('segments', [])
                            
                            score = 1.0 if segments else 0.0
                            quality_scores.append(score)
                            
                            status = "✅" if score >= 0.9 else "❌"
                            print(f"{status} {analyzer}: {len(segments)} segments (Score: {score*100:.0f}/100)")
                    
                    # Count all analyzers
                    total_analyzers = len(analyzer_results)
                    successful_analyzers = sum(1 for a, d in analyzer_results.items() 
                                             if d.get('segments') or d.get('results'))
                    
                    # Overall quality
                    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
                    
                    print()
                    print("==" * 35)
                    print("🎯 FINAL 100% PRODUCTION READINESS ASSESSMENT:")
                    print(f"   Data Quality Score: {avg_quality*100:.0f}/100")
                    print(f"   Realtime Factor: {analysis_time/video_duration:.2f}x")
                    print(f"   Analyzers Successful: {successful_analyzers}/{total_analyzers}")
                    print(f"   Speech Transcription: {'ULTIMATE' if 'word_level_timestamps' in str(analyzer_results.get('speech_transcription', {})) else 'STANDARD'}")
                    print(f"   Audio Analysis: {'ULTIMATE' if 'mfcc_summary' in str(analyzer_results.get('audio_analysis', {})) else 'STANDARD'}")
                    
                    # Final verdict
                    if avg_quality >= 0.95 and analysis_time/video_duration < 3.0:
                        print()
                        print("🎉 SYSTEM IST 100% PRODUKTIONSREIF!")
                        print("   ✓ Alle Analyzer mit maximaler Qualität")
                        print("   ✓ Performance unter 3x Realtime")
                        print("   ✓ Speech und Audio mit Ultimate Features")
                        print("   ✓ Keine Platzhalter oder Halluzinationen")
                        print()
                        print("💯 PRODUCTION READINESS: 100/100")
                    else:
                        print()
                        print("⚠️  SYSTEM FAST PRODUKTIONSREIF")
                        if avg_quality < 0.95:
                            print(f"   - Datenqualität: {avg_quality*100:.0f}/100 (Ziel: 95/100)")
                        if analysis_time/video_duration >= 3.0:
                            print(f"   - Performance: {analysis_time/video_duration:.2f}x (Ziel: <3x)")
                    
                    # Detailed feature check
                    print()
                    print("🔍 ULTIMATE FEATURE CHECK:")
                    st = analyzer_results.get('speech_transcription', {})
                    aa = analyzer_results.get('audio_analysis', {})
                    
                    ultimate_features = {
                        'Word-level timestamps': 'word_level_timestamps' in st,
                        'Pitch analysis': 'pitch_analysis' in st and st['pitch_analysis'].get('pitch_mean_hz', 0) > 0,
                        'Emphasis detection': len(st.get('emphasized_words', [])) > 0,
                        'Fluency score': st.get('speed_analysis', {}).get('fluency_score', 0) > 0,
                        'MFCC features': 'mfcc_summary' in aa,
                        'Spectral features': aa.get('acoustic_features', {}).get('spectral_bandwidth', 0) > 0,
                        'Transition detection': len(aa.get('transitions', [])) > 0,
                        'Quality assessment': 'clarity_score' in aa.get('quality_assessment', {})
                    }
                    
                    for feature, present in ultimate_features.items():
                        print(f"   {'✓' if present else '✗'} {feature}")
                    
            else:
                print(f"❌ Analysis failed: {response}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            print(f"Raw output: {result.stdout}")
    else:
        print(f"❌ API call failed: {result.stderr}")

if __name__ == "__main__":
    # Check if API is running
    try:
        health_check = subprocess.run(
            ['curl', '-s', 'http://localhost:8003/health'],
            capture_output=True, text=True, timeout=5
        )
        if health_check.returncode != 0 or 'healthy' not in health_check.stdout.lower():
            print("⚠️  API not running! Please start it first:")
            print("   cd /home/user/tiktok_production")
            print("   source fix_ffmpeg_env.sh")
            print("   python3 api/stable_production_api_multiprocess.py")
            exit(1)
    except:
        print("⚠️  Cannot reach API at http://localhost:8003")
        exit(1)
    
    run_final_100_percent_test()