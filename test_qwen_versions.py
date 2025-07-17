#!/usr/bin/env python3
"""Test all Qwen2-VL versions to find the best working one"""
import sys
sys.path.append('/home/user/tiktok_production')
import os
import traceback

print("=== TESTE ALLE QWEN2-VL VERSIONEN ===\n")

# Find all qwen versions in analyzers
qwen_versions = [
    'qwen2_vl_optimized_analyzer',
    'qwen2_vl_temporal_fixed', 
    'qwen2_vl_ultra_detailed',
    'qwen2_vl_2b_test'
]

working_versions = []

for version in qwen_versions:
    try:
        print(f"\nTeste {version}...")
        module = __import__(f'analyzers.{version}', fromlist=['*'])
        
        # Find the analyzer class
        analyzer_class = None
        for attr in dir(module):
            if 'Analyzer' in attr and attr != 'BaseAnalyzer':
                analyzer_class = getattr(module, attr)
                break
        
        if analyzer_class:
            print(f"  Gefundene Klasse: {analyzer_class.__name__}")
            
            # Try to instantiate
            analyzer = analyzer_class()
            print(f"  ✓ Instanziierung erfolgreich")
            
            # IMPORTANT: Test with real data!
            test_video = "/home/user/tiktok_production/test_video.mp4"
            if os.path.exists(test_video):
                print(f"  Analysiere {test_video}...")
                result = analyzer.analyze(test_video)
                
                if result and 'segments' in result and len(result['segments']) > 0:
                    seg = result['segments'][0]
                    desc = seg.get('description', '')
                    
                    # Check data quality
                    if len(desc) > 50:
                        print(f"  ✅ {version}: FUNKTIONIERT!")
                        print(f"     Beschreibung: {desc[:100]}...")
                        print(f"     Segmente: {len(result['segments'])}")
                        working_versions.append({
                            'version': version, 
                            'class': analyzer_class.__name__, 
                            'desc_len': len(desc),
                            'segments': len(result['segments']),
                            'sample': desc[:200]
                        })
                    else:
                        print(f"  ⚠️ {version}: Läuft aber schlechte Daten (nur {len(desc)} Zeichen)")
                else:
                    print(f"  ❌ {version}: Keine Segmente generiert")
            else:
                print(f"  ⚠️ Test-Video nicht gefunden, überspringe Analyse")
                # Still mark as working if it can be instantiated
                working_versions.append({
                    'version': version,
                    'class': analyzer_class.__name__,
                    'desc_len': 0,
                    'segments': 0,
                    'sample': 'Nicht getestet - kein Video'
                })
        else:
            print(f"  ❌ Keine Analyzer-Klasse gefunden")
                
    except Exception as e:
        print(f"  ❌ {version}: FEHLER - {str(e)}")
        traceback.print_exc()

print("\n=== ERGEBNIS ===")
if working_versions:
    # Sort by description length (better quality)
    working_versions.sort(key=lambda x: x['desc_len'], reverse=True)
    
    print("\nFunktionierende Versionen:")
    for i, v in enumerate(working_versions):
        print(f"\n{i+1}. {v['version']} ({v['class']})")
        print(f"   Beschreibungslänge: {v['desc_len']} Zeichen")
        print(f"   Segmente: {v['segments']}")
        if v['desc_len'] > 0:
            print(f"   Beispiel: {v['sample']}")
    
    if working_versions[0]['desc_len'] > 0:
        best = working_versions[0]
        print(f"\n🏆 BESTE VERSION: {best['version']} ({best['class']})")
        print(f"   Beschreibungslänge: {best['desc_len']} Zeichen")
    else:
        print(f"\n⚠️ Alle Versionen können instanziiert werden, aber kein Test-Video vorhanden")
else:
    print("\n❌ KEINE funktionierende Qwen2-VL Version gefunden!")

# Check which version is currently in registry
print("\n=== AKTUELLE REGISTRY ===")
try:
    with open('/home/user/tiktok_production/ml_analyzer_registry_complete.py', 'r') as f:
        content = f.read()
        for line in content.split('\n'):
            if 'qwen2_vl' in line and not line.strip().startswith('#'):
                print(f"Registry: {line.strip()}")
except:
    pass