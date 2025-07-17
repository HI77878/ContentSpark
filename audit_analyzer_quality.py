#!/usr/bin/env python3
"""Audit analyzer data quality"""
import json
import sys
from pathlib import Path

def audit_analyzer(name, data):
    """Prüft ob ein Analyzer sinnvolle Daten liefert"""
    issues = []
    
    # Prüfe auf leere/fehlende Daten
    if not data or data == {}:
        issues.append("LEER: Keine Daten")
        return issues
    
    # Prüfe auf Error
    if isinstance(data, dict) and 'error' in data:
        issues.append(f"FEHLER: {data['error']}")
        return issues
    
    # Prüfe auf Segmente
    if 'segments' in data:
        if not data['segments']:
            issues.append("KEINE SEGMENTE")
        else:
            # Prüfe erste 3 Segmente auf generische Daten
            for i, seg in enumerate(data['segments'][:3]):
                # Suche nach generischen Begriffen
                if isinstance(seg, dict):
                    seg_str = str(seg).lower()
                    generics = [
                        'balanced', 'natural', 'moderate', 'normal', 
                        'continues activity', 'person continues',
                        'unknown', 'neutral', 'general'
                    ]
                    if any(generic in seg_str for generic in generics):
                        issues.append(f"GENERISCH in Segment {i}")
                
                # Prüfe auf konkrete Werte
                if 'description' in seg and seg['description']:
                    if len(str(seg['description'])) < 20:
                        issues.append(f"ZU KURZ in Segment {i}: '{seg['description']}'")
                
                # Prüfe auf Timestamps
                if 'timestamp' not in seg and 'start_time' not in seg and 'time' not in seg:
                    issues.append(f"KEINE ZEITSTEMPEL in Segment {i}")
    
    # Analyzer-spezifische Checks
    if name == 'face_emotion':
        if 'faces' not in data and 'segments' not in data:
            issues.append("KEINE GESICHTER erkannt")
    
    elif name == 'body_pose':
        if 'poses' not in data and 'segments' not in data:
            issues.append("KEINE KÖRPERPOSEN erkannt")
    
    elif name == 'cross_analyzer_intelligence':
        if 'correlations' not in data and 'timeline' not in data:
            issues.append("KEINE VERKNÜPFUNGEN gefunden")
    
    elif name == 'visual_effects':
        # Check for specific effects
        if isinstance(data, dict):
            data_str = str(data).lower()
            if 'slow motion' not in data_str and 'filter' not in data_str and 'transition' not in data_str:
                issues.append("KEINE EFFEKTE erkannt")
    
    return issues

# Main
if len(sys.argv) < 2:
    print("Usage: python3 audit_analyzer_quality.py <results.json>")
    sys.exit(1)

results_file = Path(sys.argv[1])
if not results_file.exists():
    print(f"File not found: {results_file}")
    sys.exit(1)

# Load JSON
with open(results_file) as f:
    data = json.load(f)

print("\n=== ANALYZER QUALITÄTS-AUDIT ===")
print(f"File: {results_file.name}\n")

# Check metadata
metadata = data.get('metadata', {})
print(f"Video Duration: {metadata.get('duration', 'Unknown')}s")
print(f"Processing Time: {metadata.get('processing_time', 'Unknown')}s")
print(f"Successful: {metadata.get('successful_analyzers', 0)}/{metadata.get('total_analyzers', 0)}")
print("\n" + "="*50 + "\n")

# Audit each analyzer
good_analyzers = []
bad_analyzers = []

for analyzer, result in data.get('analyzer_results', {}).items():
    issues = audit_analyzer(analyzer, result)
    if issues:
        bad_analyzers.append((analyzer, issues))
        print(f"❌ {analyzer}: {', '.join(issues)}")
    else:
        good_analyzers.append(analyzer)
        print(f"✅ {analyzer}: Daten sehen gut aus")

# Summary
print("\n" + "="*50)
print(f"\n📊 ZUSAMMENFASSUNG:")
print(f"   ✅ Gute Analyzer: {len(good_analyzers)}")
print(f"   ❌ Problematische Analyzer: {len(bad_analyzers)}")

if bad_analyzers:
    print(f"\n⚠️  PROBLEME GEFUNDEN bei:")
    for analyzer, issues in bad_analyzers:
        print(f"   - {analyzer}: {issues[0]}")  # Show first issue

# Check key analyzers
print("\n🔑 NEUE ANALYZER STATUS:")
for key in ['face_emotion', 'body_pose', 'cross_analyzer_intelligence']:
    if key in data.get('analyzer_results', {}):
        result = data['analyzer_results'][key]
        if result and 'error' not in result:
            print(f"   ✅ {key}: Daten vorhanden")
        else:
            print(f"   ❌ {key}: Fehler oder keine Daten")
    else:
        print(f"   ⚠️  {key}: Nicht in Ergebnissen")