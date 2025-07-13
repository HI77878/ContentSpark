#!/usr/bin/env python3
"""Test analyzer quality by checking their output structure"""
import sys
sys.path.append('/home/user/tiktok_production')
from ml_analyzer_registry_complete import ML_ANALYZERS, DISABLED_ANALYZERS
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

# Check latest analysis results
print("\n=== LETZTE ANALYSE-ERGEBNISSE ===")
results_dir = Path("/home/user/tiktok_production/results")
if results_dir.exists():
    results = list(results_dir.glob("*.json"))
    if results:
        latest = sorted(results)[-1]
        print(f"Neueste Analyse: {latest.name}")
        
        try:
            with open(latest) as f:
                data = json.load(f)
            
            if 'analyzer_results' in data:
                successful = 0
                failed = 0
                empty = 0
                
                for analyzer, result in data['analyzer_results'].items():
                    if 'error' in result:
                        failed += 1
                    elif 'segments' in result and result['segments']:
                        successful += 1
                    else:
                        empty += 1
                
                print(f"Erfolgreich: {successful}")
                print(f"Fehlgeschlagen: {failed}")
                print(f"Keine Daten: {empty}")
        except:
            print("Fehler beim Lesen der Ergebnisse")
    else:
        print("Keine Ergebnisse im results/ Verzeichnis")
else:
    print("results/ Verzeichnis nicht gefunden")

# Check GPU groups
print("\n=== GPU GRUPPEN ===")
try:
    from configs.gpu_groups_config import GPU_ANALYZER_GROUPS
    
    for group, analyzers in GPU_ANALYZER_GROUPS.items():
        active_in_group = [a for a in analyzers if a in active_analyzers]
        print(f"{group}: {len(active_in_group)}/{len(analyzers)} aktiv")
except:
    print("Fehler beim Laden der GPU-Gruppen")