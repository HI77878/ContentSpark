#!/usr/bin/env python3
"""Test analyzer quality by checking their output structure"""
import sys
sys.path.append('/home/user/tiktok_production')
from ml_analyzer_registry_complete import ML_ANALYZERS
from configs.gpu_groups_config import DISABLED_ANALYZERS
import json
from pathlib import Path

print("=== ANALYZER QUALITÄTS-CHECK ===\n")

# Check active analyzers
active_analyzers = [name for name in ML_ANALYZERS.keys() if name not in DISABLED_ANALYZERS]
print(f"Aktive Analyzer: {len(active_analyzers)}")
print(f"Deaktivierte Analyzer: {len(DISABLED_ANALYZERS)}")

# Check for critical analyzers
critical_analyzers = {
    'qwen2_vl': ['qwen2_vl_temporal', 'qwen2_vl_optimized'],
    'speech': ['speech_transcription'],
    'object': ['object_detection'],
    'face': ['face_emotion'],
    'audio': ['audio_analysis'],
    'visual_effects': ['visual_effects']
}

print("\n=== KRITISCHE ANALYZER STATUS ===")
for category, analyzer_names in critical_analyzers.items():
    found = []
    for name in active_analyzers:
        for check in analyzer_names:
            if check in name:
                found.append(name)
    
    if found:
        print(f"✅ {category}: {', '.join(found)}")
    else:
        print(f"❌ {category}: FEHLT!")

# Check disabled analyzers
print("\n=== DEAKTIVIERTE ANALYZER ===")
for analyzer in DISABLED_ANALYZERS[:10]:  # First 10
    print(f"- {analyzer}")
if len(DISABLED_ANALYZERS) > 10:
    print(f"... und {len(DISABLED_ANALYZERS) - 10} weitere")

# Check for duplicate/redundant analyzers
print("\n=== POTENZIELLE DUPLIKATE ===")
analyzer_bases = {}
for name in active_analyzers:
    # Extract base name
    base = name
    for suffix in ['_detection', '_analysis', '_transcription', '_segmentation', '_emotion']:
        base = base.replace(suffix, '')
    
    if base not in analyzer_bases:
        analyzer_bases[base] = []
    analyzer_bases[base].append(name)

for base, names in analyzer_bases.items():
    if len(names) > 1:
        print(f"{base}: {', '.join(names)}")

# Check for which Qwen2-VL version is active
print("\n=== QWEN2-VL STATUS ===")
qwen_active = [name for name in active_analyzers if 'qwen' in name]
if qwen_active:
    for qwen in qwen_active:
        analyzer_class = ML_ANALYZERS[qwen]
        print(f"✅ {qwen}: {analyzer_class.__name__} aus {analyzer_class.__module__}")
else:
    print("❌ Kein Qwen2-VL Analyzer aktiv!")

# Show analyzer mapping
print("\n=== WICHTIGE ANALYZER MAPPINGS ===")
important = ['qwen2_vl_temporal', 'qwen2_vl_optimized', 'speech_transcription', 
             'face_emotion', 'visual_effects', 'object_detection']
for name in important:
    if name in ML_ANALYZERS:
        print(f"{name}: {ML_ANALYZERS[name].__name__}")
    else:
        print(f"{name}: NICHT IN REGISTRY")