#!/usr/bin/env python3
"""Find best analyzer versions by checking their implementations"""
from pathlib import Path
import sys
sys.path.append('/home/user/tiktok_production')
import re

analyzer_dir = Path("/home/user/tiktok_production/analyzers")

# Group analyzers by base name
analyzer_groups = {}
for file in analyzer_dir.glob("*.py"):
    if file.stem in ['__init__', 'base_analyzer', 'standardized_base_analyzer', 'description_helpers']:
        continue
    
    # Extract base name
    base_name = file.stem
    
    # Remove common suffixes
    for suffix in ['_fixed', '_light', '_enhanced', '_ultimate', '_ml', '_cv', '_based', '_mediapipe', 
                   '_yolo', '_yolov8', '_optimized', '_advanced', '_wrapper', '_detector', '_insightface',
                   '_tiktok', '_ray', '_deepface']:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break
    
    # Special cases
    if 'qwen2_vl' in file.stem:
        base_name = 'qwen2_vl'
    elif 'speech_transcription' in file.stem:
        base_name = 'speech_transcription'
    elif 'audio_analysis' in file.stem:
        base_name = 'audio_analysis'
    elif 'visual_effects' in file.stem:
        base_name = 'visual_effects'
    
    if base_name not in analyzer_groups:
        analyzer_groups[base_name] = []
    analyzer_groups[base_name].append(file.stem)

print("=== ANALYZER MIT MEHREREN VERSIONEN ===\n")

# Test each group
best_versions = {}
for base_name, versions in sorted(analyzer_groups.items()):
    if len(versions) > 1:
        print(f"\n{base_name.upper()} - {len(versions)} Versionen:")
        
        best_quality = 0
        best_version = None
        
        for version in sorted(versions):
            try:
                # Check file content for quality indicators
                file_path = analyzer_dir / f"{version}.py"
                with open(file_path, 'r') as f:
                    content = f.read()
                
                quality_score = 50  # Base score
                
                # Quality indicators
                if 'class Meta:' in content:
                    quality_score += 10
                if 'process_batch_gpu' in content:
                    quality_score += 15
                if 'torch.cuda' in content:
                    quality_score += 10
                if 'FIXED' in content or '# Fixed' in content:
                    quality_score += 20
                if 'ultimate' in version:
                    quality_score += 25
                elif 'enhanced' in version:
                    quality_score += 20
                elif 'fixed' in version:
                    quality_score += 15
                elif 'optimized' in version:
                    quality_score += 15
                elif 'ml' in version or 'advanced' in version:
                    quality_score += 20
                elif 'cv_based' in version:
                    quality_score += 18
                elif 'light' in version:
                    quality_score -= 10
                
                # Check for errors/issues
                if 'deprecated' in content.lower():
                    quality_score -= 30
                if 'broken' in content.lower():
                    quality_score -= 50
                if 'TODO' in content:
                    quality_score -= 5
                
                # Check imports
                if 'from transformers import' in content:
                    quality_score += 10
                if 'import torch' in content:
                    quality_score += 5
                
                print(f"  {version} - Score: {quality_score}")
                
                if quality_score > best_quality:
                    best_quality = quality_score
                    best_version = version
                    
            except Exception as e:
                print(f"  ❌ {version} - Fehler beim Lesen")
        
        if best_version:
            best_versions[base_name] = best_version
            print(f"  → BESTE: {best_version} (Score: {best_quality})")

print("\n=== EMPFOHLENE VERSIONEN ===")
print("\n# Zu aktualisierende Imports in ml_analyzer_registry_complete.py:\n")

# Generate recommended imports
for base, version in sorted(best_versions.items()):
    # Try to find the class name
    try:
        with open(analyzer_dir / f"{version}.py", 'r') as f:
            content = f.read()
            # Find class definition
            class_match = re.search(r'class\s+(\w+)\s*\(', content)
            if class_match:
                class_name = class_match.group(1)
                print(f"from analyzers.{version} import {class_name}  # {base}")
    except:
        pass

print("\n=== SPEZIELLE EMPFEHLUNGEN ===")
print("\nBasierend auf der Analyse:")
print("- qwen2_vl: qwen2_vl_temporal_fixed (hat 'FIXED' im Code)")
print("- speech_transcription: speech_transcription_ultimate")
print("- audio_analysis: audio_analysis_ultimate") 
print("- visual_effects: visual_effects_cv_based (CV ist zuverlässiger als ML)")
print("- face_emotion: face_emotion_mediapipe (DeepFace hat TensorFlow-Probleme)")

# Check current registry
print("\n=== AKTUELLE REGISTRY-EINTRÄGE ===")
try:
    with open('/home/user/tiktok_production/ml_analyzer_registry_complete.py', 'r') as f:
        content = f.read()
        
    # Find all analyzer definitions
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith("'") and ':' in line and not line.startswith('#'):
            parts = line.split(':')
            if len(parts) >= 2:
                key = parts[0].strip("'\" ")
                value = parts[1].strip(" ,")
                
                # Check if this analyzer has multiple versions
                base = key
                for suffix in ['_detection', '_analysis', '_transcription', '_segmentation']:
                    base = base.replace(suffix, '')
                
                if base in best_versions:
                    recommended = best_versions[base]
                    status = "✅" if recommended in value else "⚠️"
                    print(f"{status} {key}: {value}")
                    if status == "⚠️":
                        print(f"   → Empfehlung: {recommended}")
except Exception as e:
    print(f"Fehler beim Lesen der Registry: {e}")