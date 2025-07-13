#!/usr/bin/env python3
"""
Validate analyzer fixes by checking existing results
"""

import json
import sys
from pathlib import Path
from utils.output_normalizer import AnalyzerOutputNormalizer, create_unified_field_extractor

def validate_chase_ridgeway_results():
    """Validate the Chase Ridgeway analysis results"""
    
    # Load the latest analysis
    results_file = "/home/user/tiktok_production/results/7522589683939921165_multiprocess_20250708_142055.json"
    
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("🔍 VALIDATING ANALYZER FIXES\n")
    
    # 1. Check Qwen2-VL for repetitions
    print("1. QWEN2-VL TEMPORAL ANALYZER:")
    qwen = data['analyzer_results'].get('qwen2_vl_temporal', {})
    segments = qwen.get('segments', [])
    
    if segments:
        # Count repetitions
        repetitive_count = 0
        repetitive_phrases = ["possibly to pick something up", "bending over and reaching down"]
        
        for seg in segments:
            desc = seg.get('description', '').lower()
            if any(phrase in desc for phrase in repetitive_phrases):
                repetitive_count += 1
        
        repetition_rate = repetitive_count / len(segments) * 100
        
        print(f"   Total segments: {len(segments)}")
        print(f"   Repetitive segments: {repetitive_count} ({repetition_rate:.1f}%)")
        
        # Show problematic area
        print("\n   Segments 20-30 (where repetition started):")
        for i in range(20, min(30, len(segments))):
            seg = segments[i]
            desc = seg['description'][:80]
            is_repetitive = any(phrase in desc.lower() for phrase in repetitive_phrases)
            marker = "❌" if is_repetitive else "✅"
            print(f"   {marker} [{seg['start_time']:.1f}s]: {desc}...")
        
        if repetition_rate > 40:
            print(f"\n   ❌ FAILED: High repetition rate ({repetition_rate:.1f}%) - Fix not applied yet")
        else:
            print(f"\n   ✅ SUCCESS: Low repetition rate ({repetition_rate:.1f}%)")
    
    # 2. Check data normalization
    print("\n2. DATA NORMALIZATION:")
    
    # Initialize normalizer and field extractors
    normalizer = AnalyzerOutputNormalizer()
    extractors = create_unified_field_extractor()
    
    # Test eye tracking
    print("\n   Eye Tracking:")
    eye_data = data['analyzer_results'].get('eye_tracking', {})
    if eye_data.get('segments'):
        seg = eye_data['segments'][0]
        print(f"   Raw fields: {list(seg.keys())}")
        
        # Try to extract gaze direction
        gaze = extractors['gaze_direction'](seg)
        print(f"   Extracted gaze_direction: {gaze}")
        
        if gaze != 'unknown':
            print("   ✅ Gaze direction extractable")
        else:
            print("   ❌ Gaze direction not found")
    
    # Test speech rate (pitch)
    print("\n   Speech Rate (Pitch):")
    speech_data = data['analyzer_results'].get('speech_rate', {})
    if speech_data.get('segments'):
        seg = speech_data['segments'][0]
        print(f"   Raw fields: {list(seg.keys())[:10]}...")
        
        # Check for pitch data
        pitch = extractors['pitch'](seg)
        if 'average_pitch' in seg:
            pitch = seg['average_pitch']
        
        print(f"   Extracted pitch: {pitch:.1f} Hz")
        
        if pitch > 0:
            print("   ✅ Pitch data available")
        else:
            print("   ❌ Pitch data not found")
    
    # 3. Check performance
    print("\n3. PERFORMANCE METRICS:")
    metadata = data['metadata']
    
    print(f"   Processing time: {metadata['processing_time_seconds']:.1f}s")
    print(f"   Video duration: ~68.4s")
    print(f"   Realtime factor: {metadata['realtime_factor']:.2f}x")
    print(f"   Reconstruction score: {metadata['reconstruction_score']:.1f}%")
    
    if metadata['realtime_factor'] < 3.0:
        print(f"   ✅ Performance target achieved (<3x)")
    else:
        print(f"   ❌ Performance target not met ({metadata['realtime_factor']:.2f}x >= 3x)")
    
    # 4. Summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    issues = []
    
    if repetition_rate > 40:
        issues.append("Qwen2-VL still has high repetition rate (fix not applied)")
    
    if metadata['realtime_factor'] >= 3.0:
        issues.append(f"Performance is {metadata['realtime_factor']:.2f}x (target <3x)")
    
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n💡 The fixes need to be applied by re-running the analysis")
    else:
        print("✅ All fixes validated successfully!")
    
    print("\nNOTE: The current results are from BEFORE the fixes were implemented.")
    print("To see the improvements, run a new analysis with the fixed code.")


if __name__ == "__main__":
    validate_chase_ridgeway_results()