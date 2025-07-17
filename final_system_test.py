#\!/usr/bin/env python3
"""
Final system test to validate ALL fixes are working
"""

import json
import time
from pathlib import Path

print("""
============================================================
🚀 FINAL SYSTEM TEST - ALL FIXES VALIDATION
============================================================

This test validates:
1. ✅ Qwen2-VL chunking prevents repetitions
2. ✅ Data normalization fixes field names
3. ✅ Object/Product detection structure fixed
4. ✅ Performance optimizations
5. ✅ All 22 analyzers functional

""")

# Check if API is running
import requests

try:
    response = requests.get("http://localhost:8003/health")
    if response.status_code == 200:
        print("✅ API is running on port 8003")
    else:
        print("❌ API not responding properly")
        print("Please start the API with:")
        print("  source fix_ffmpeg_env.sh")
        print("  python3 api/stable_production_api_multiprocess.py")
        exit(1)
except:
    print("❌ API is not running\!")
    print("Please start the API with:")
    print("  source fix_ffmpeg_env.sh")
    print("  python3 api/stable_production_api_multiprocess.py")
    exit(1)

# Test video
video_path = "/home/user/tiktok_videos/videos/7522589683939921165.mp4"
if not Path(video_path).exists():
    print(f"❌ Test video not found: {video_path}")
    exit(1)

print(f"\n📹 Testing with Chase Ridgeway video (68.4s)")
print("🔄 Running analysis with ALL fixes applied...\n")

# Run analysis
start_time = time.time()
response = requests.post(
    "http://localhost:8003/analyze",
    json={"video_path": video_path}
)

if response.status_code \!= 200:
    print(f"❌ Analysis failed: {response.text}")
    exit(1)

result = response.json()
processing_time = time.time() - start_time

print(f"✅ Analysis complete in {processing_time:.1f}s")
print(f"   Result file: {result['results_file']}")

# Load and check results
with open(result['results_file'], 'r') as f:
    data = json.load(f)

print("\n" + "="*60)
print("📊 VALIDATION RESULTS")
print("="*60)

# 1. Check Qwen2-VL repetitions
print("\n1. QWEN2-VL CHUNKING FIX:")
qwen = data['analyzer_results'].get('qwen2_vl_temporal', {})
segments = qwen.get('segments', [])
repetition_rate = 0
if segments:
    repetitions = sum(1 for s in segments if 'possibly to pick something up' in s.get('description', '').lower())
    repetition_rate = repetitions / len(segments) * 100
    print(f"   Segments: {len(segments)}")
    print(f"   Repetitive: {repetitions} ({repetition_rate:.1f}%)")
    
    if repetition_rate < 10:
        print("   ✅ SUCCESS: Repetition bug fixed\!")
    else:
        print("   ❌ FAILED: Still has repetitions")

# 2. Check data normalization
print("\n2. DATA NORMALIZATION:")
issues = []

# Eye tracking
eye = data['analyzer_results'].get('eye_tracking', {})
if eye.get('segments'):
    seg = eye['segments'][0]
    if 'gaze_direction' in seg:
        print("   ✅ Eye tracking: gaze_direction available")
    else:
        print("   ❌ Eye tracking: gaze_direction missing")
        issues.append("eye_tracking")

# Speech rate
speech = data['analyzer_results'].get('speech_rate', {})
if speech.get('segments'):
    seg = speech['segments'][0]
    if 'pitch_hz' in seg:
        print(f"   ✅ Speech rate: pitch_hz available ({seg['pitch_hz']:.1f} Hz)")
    else:
        print("   ❌ Speech rate: pitch_hz missing")
        issues.append("speech_rate")

# Object detection
obj = data['analyzer_results'].get('object_detection', {})
if obj.get('segments'):
    seg = obj['segments'][0]
    if 'objects' in seg:
        print(f"   ✅ Object detection: objects array present ({len(seg['objects'])} objects)")
    else:
        print("   ❌ Object detection: objects array missing")
        issues.append("object_detection")

# Product detection
prod = data['analyzer_results'].get('product_detection', {})
if prod.get('segments'):
    seg = prod['segments'][0]
    if 'products' in seg:
        print(f"   ✅ Product detection: products array present")
    else:
        print("   ❌ Product detection: products array missing")
        issues.append("product_detection")

# 3. Check performance
print("\n3. PERFORMANCE:")
metadata = data['metadata']
realtime_factor = metadata.get('realtime_factor', 999)
print(f"   Processing time: {metadata['processing_time_seconds']:.1f}s")
print(f"   Realtime factor: {realtime_factor:.2f}x")

if realtime_factor < 3.0:
    print("   ✅ SUCCESS: Performance target achieved\!")
else:
    print(f"   ⚠️  WARNING: Performance {realtime_factor:.2f}x (target <3x)")

# 4. Check all analyzers
print("\n4. ANALYZER STATUS:")
total = metadata.get('total_analyzers', 0)
successful = metadata.get('successful_analyzers', 0)
print(f"   Total: {total}")
print(f"   Successful: {successful}")
print(f"   Failed: {total - successful}")

if successful == total:
    print("   ✅ All analyzers successful\!")
else:
    print(f"   ❌ {total - successful} analyzers failed")

# Final summary
print("\n" + "="*60)
print("🏁 FINAL VERDICT")
print("="*60)

all_good = True
if repetition_rate >= 10:
    print("❌ Qwen2-VL still has repetitions")
    all_good = False
if issues:
    print(f"❌ Normalization issues: {', '.join(issues)}")
    all_good = False
if realtime_factor >= 3.0:
    print(f"⚠️  Performance not optimal ({realtime_factor:.2f}x)")
if successful < total:
    print(f"❌ {total - successful} analyzers failed")
    all_good = False

if all_good and realtime_factor < 3.0:
    print("""
✅ ✅ ✅ ALL SYSTEMS GO\! ✅ ✅ ✅

All fixes are working correctly:
- Qwen2-VL produces unique descriptions
- Data normalization is functioning
- Performance is under 3x realtime
- All 22 analyzers are operational

The system is ready for production\!
""")
else:
    print("""
⚠️  Some issues remain. Check the details above.
""")

print(f"\nFull results saved to: {result['results_file']}")
EOF < /dev/null
