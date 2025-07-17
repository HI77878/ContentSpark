import time
import sys
sys.path.append('/home/user/tiktok_production')
from analyzers.qwen2_vl_ultra_detailed import Qwen2VLUltraDetailedAnalyzer

analyzer = Qwen2VLUltraDetailedAnalyzer()
video = "/home/user/tiktok_videos/videos/7446489995663117590.mp4"

print("Starting Qwen2-VL Ultra speed test...")
start = time.time()
result = analyzer.analyze(video)
duration = time.time() - start

segments = result.get('segments', [])
video_duration = 48.9  # seconds

print(f"\n=== RESULTS ===")
print(f"Zeit: {duration:.1f}s")
print(f"Segmente: {len(segments)}")
if len(segments) > 0:
    print(f"Durchschnitt pro Segment: {duration/len(segments):.1f}s")
print(f"Video Duration: {video_duration}s")
print(f"Realtime Factor: {duration/video_duration:.1f}x")

if duration < 150:
    print("\n✅ ZIEL ERREICHT! (<150s)")
else:
    print(f"\n❌ Noch {duration-150:.0f}s zu langsam! (Ziel: <150s)")
    
print(f"\nFür Gesamtsystem <5x Realtime brauchen wir Qwen <{video_duration*5 - 250:.0f}s")