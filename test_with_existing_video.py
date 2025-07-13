#!/usr/bin/env python3
"""Test system with existing test video"""
import requests
import json
import time
from pathlib import Path

video_path = "/home/user/tiktok_production/test_video.mp4"

print("=== TESTE MIT EXISTIERENDEM VIDEO ===")
print(f"Video: {video_path}")

# Analyze video
print("\n1. Analysiere Video...")
start_time = time.time()

analyze_response = requests.post("http://localhost:8003/analyze",
    json={"video_path": str(video_path)})

if analyze_response.status_code == 200:
    result = analyze_response.json()
    duration = time.time() - start_time
    
    print(f"✅ Analyse abgeschlossen in {duration:.1f}s")
    print(f"   Erfolgreiche Analyzer: {result.get('successful_analyzers', 0)}")
    print(f"   Gesamt Analyzer: {result.get('total_analyzers', 0)}")
    
    if result.get('results_path'):
        print(f"   Ergebnisse: {result['results_path']}")
        
        # Load and check results
        results_path = Path(result['results_path'])
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
            
            print("\n2. Datenqualitäts-Check...")
            
            # Check each analyzer
            high_quality = []
            low_quality = []
            errors = []
            
            for analyzer_name, analyzer_result in data.get('analyzer_results', {}).items():
                if 'error' in analyzer_result:
                    errors.append((analyzer_name, analyzer_result['error'][:50]))
                elif 'segments' in analyzer_result and analyzer_result['segments']:
                    seg_count = len(analyzer_result['segments'])
                    first_seg = analyzer_result['segments'][0]
                    
                    # Quality check based on analyzer type
                    if 'qwen' in analyzer_name:
                        desc = first_seg.get('description', '')
                        if len(desc) > 100:
                            high_quality.append((analyzer_name, seg_count, desc[:100]))
                        else:
                            low_quality.append((analyzer_name, seg_count, 'Kurze Beschreibung'))
                    elif 'transcription' in analyzer_name:
                        text = first_seg.get('text', '')
                        if text and len(text) > 10:
                            high_quality.append((analyzer_name, seg_count, text[:50]))
                        else:
                            low_quality.append((analyzer_name, seg_count, 'Kein Text'))
                    else:
                        # Generic check
                        seg_str = str(first_seg)
                        if len(seg_str) > 50 and not all(term in seg_str.lower() for term in ['normal', 'moderate']):
                            high_quality.append((analyzer_name, seg_count, seg_str[:80]))
                        else:
                            low_quality.append((analyzer_name, seg_count, 'Generische Daten'))
                else:
                    low_quality.append((analyzer_name, 0, 'Keine Segmente'))
            
            # Print results
            print(f"\n✅ HOHE QUALITÄT ({len(high_quality)} Analyzer):")
            for name, count, sample in high_quality[:5]:  # Top 5
                print(f"   {name}: {count} Segmente")
                print(f"      → {sample}...")
            
            if len(high_quality) > 5:
                print(f"   ... und {len(high_quality) - 5} weitere")
            
            print(f"\n⚠️ NIEDRIGE QUALITÄT ({len(low_quality)} Analyzer):")
            for name, count, reason in low_quality[:3]:
                print(f"   {name}: {reason}")
            
            if errors:
                print(f"\n❌ FEHLER ({len(errors)} Analyzer):")
                for name, error in errors[:3]:
                    print(f"   {name}: {error}...")
            
            # Final summary
            total = len(data.get('analyzer_results', {}))
            print(f"\n=== ZUSAMMENFASSUNG ===")
            print(f"Gesamt: {total} Analyzer")
            print(f"Hohe Qualität: {len(high_quality)} ({len(high_quality)/total*100:.0f}%)")
            print(f"Niedrige Qualität: {len(low_quality)} ({len(low_quality)/total*100:.0f}%)")
            print(f"Fehler: {len(errors)} ({len(errors)/total*100:.0f}%)")
            
            # Critical analyzer check
            print("\n=== KRITISCHE ANALYZER ===")
            critical = ['qwen2_vl_temporal', 'speech_transcription', 'object_detection']
            for crit in critical:
                found = False
                for name, _, _ in high_quality:
                    if crit in name:
                        print(f"✅ {crit}: Hohe Qualität")
                        found = True
                        break
                if not found:
                    for name, _, _ in low_quality:
                        if crit in name:
                            print(f"⚠️ {crit}: Niedrige Qualität")
                            found = True
                            break
                if not found:
                    print(f"❌ {crit}: Fehlt oder Fehler")
            
else:
    print(f"❌ Analyse fehlgeschlagen: {analyze_response.text}")