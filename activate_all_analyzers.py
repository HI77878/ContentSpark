#!/usr/bin/env python3
"""
Aktiviere ALLE Analyzer für vollständige Analyse
"""
import sys
sys.path.append('/home/user/tiktok_production')

from configs.gpu_groups_config import DISABLED_ANALYZERS, GPU_ANALYZER_GROUPS

print("🔥 AKTIVIERE ALLE ANALYZER FÜR VOLLSTÄNDIGE ANALYSE")
print("=" * 60)

# Zeige was deaktiviert ist
print(f"\nAktuell deaktiviert: {len(DISABLED_ANALYZERS)} Analyzer")
print("\nWichtigste deaktivierte Analyzer:")
important_disabled = [
    'face_detection',
    'emotion_detection', 
    'body_pose',
    'body_language',
    'gesture_recognition',
    'hand_gesture',
    'facial_details'
]

for analyzer in important_disabled:
    if analyzer in DISABLED_ANALYZERS:
        print(f"  ❌ {analyzer}")

# Zeige was passieren würde
print("\n⚠️  WARNUNG:")
print("  - Processing Zeit wird von ~146s auf ~250s steigen")
print("  - Realtime Factor von 2.98x auf ~5x")
print("  - GPU Memory von 21GB auf ~35GB")
print("  - ABER: Vollständige Personen-Analyse!")

print("\n🎯 EMPFEHLUNG:")
print("1. Erstelle separate Konfiguration für Personen-Videos")
print("2. Aktiviere nur wichtigste Analyzer:")
print("   - face_detection (Basis für alles)")
print("   - emotion_detection (visuelle Emotionen)")
print("   - body_pose (Körpersprache)")
print("   - gesture_recognition (Gesten)")

print("\n📝 Code zum Aktivieren:")
print("""
# In configs/gpu_groups_config.py:
DISABLED_ANALYZERS = [
    # 'face_detection',  # AKTIVIERT!
    # 'emotion_detection',  # AKTIVIERT!
    # 'body_pose',  # AKTIVIERT!
    # 'gesture_recognition',  # AKTIVIERT!
    'depth_estimation',
    'temporal_consistency',
    # ... rest bleibt deaktiviert
]

# Füge zu GPU Groups hinzu:
GPU_ANALYZER_GROUPS['stage3_gpu_light'].extend([
    'face_detection',
    'emotion_detection',
])

GPU_ANALYZER_GROUPS['stage4_gpu_fast'].extend([
    'body_pose',
    'gesture_recognition',
])
""")