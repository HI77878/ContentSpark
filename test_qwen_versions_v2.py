#!/usr/bin/env python3
"""Test all Qwen2-VL versions to find the best working one - V2"""
import sys
sys.path.append('/home/user/tiktok_production')
import os
import traceback

print("=== TESTE ALLE QWEN2-VL VERSIONEN V2 ===\n")

# Find all qwen versions in analyzers
qwen_versions = [
    ('qwen2_vl_optimized_analyzer', 'Qwen2VLOptimizedAnalyzer'),
    ('qwen2_vl_temporal_fixed', 'Qwen2VLTemporalFixed'),
    ('qwen2_vl_ultra_detailed', 'Qwen2VLUltraDetailed'),
    ('qwen2_vl_2b_test', 'Qwen2VL2BTest')
]

working_versions = []

for module_name, expected_class in qwen_versions:
    try:
        print(f"\nTeste {module_name}...")
        module = __import__(f'analyzers.{module_name}', fromlist=[expected_class])
        
        # Try to get the expected class
        if hasattr(module, expected_class):
            analyzer_class = getattr(module, expected_class)
            print(f"  Gefundene Klasse: {analyzer_class.__name__}")
        else:
            # Look for any class containing Qwen
            analyzer_class = None
            for attr in dir(module):
                obj = getattr(module, attr)
                if hasattr(obj, '__bases__') and 'Qwen' in attr and attr != 'Qwen2VLForConditionalGeneration':
                    analyzer_class = obj
                    print(f"  Gefundene alternative Klasse: {analyzer_class.__name__}")
                    break
        
        if analyzer_class:
            # Try to instantiate
            analyzer = analyzer_class()
            print(f"  ✓ Instanziierung erfolgreich")
            
            # Check if model loads
            if hasattr(analyzer, 'model_loaded'):
                print(f"  Model loaded: {analyzer.model_loaded}")
            
            # Test with real data if video exists
            test_video = "/home/user/tiktok_production/test_video.mp4"
            if os.path.exists(test_video):
                print(f"  Analysiere {test_video}...")
                result = analyzer.analyze(test_video)
                
                if result and 'segments' in result and len(result['segments']) > 0:
                    seg = result['segments'][0]
                    desc = seg.get('description', '')
                    
                    # Check data quality
                    if len(desc) > 50:
                        print(f"  ✅ {module_name}: FUNKTIONIERT!")
                        print(f"     Beschreibung: {desc[:100]}...")
                        print(f"     Segmente: {len(result['segments'])}")
                        working_versions.append({
                            'version': module_name, 
                            'class': analyzer_class.__name__, 
                            'desc_len': len(desc),
                            'segments': len(result['segments']),
                            'sample': desc[:200]
                        })
                    else:
                        print(f"  ⚠️ {module_name}: Läuft aber schlechte Daten (nur {len(desc)} Zeichen)")
                else:
                    print(f"  ❌ {module_name}: Keine Segmente generiert")
            else:
                print(f"  ⚠️ Test-Video nicht gefunden, überspringe Analyse")
                # Still mark as working if it can be instantiated
                working_versions.append({
                    'version': module_name,
                    'class': analyzer_class.__name__,
                    'desc_len': 0,
                    'segments': 0,
                    'sample': 'Nicht getestet - kein Video'
                })
        else:
            print(f"  ❌ Keine Analyzer-Klasse gefunden")
                
    except Exception as e:
        print(f"  ❌ {module_name}: FEHLER - {str(e)}")
        # Try to understand the error better
        if "abstract" in str(e):
            print(f"     → Abstrakte Klasse, suche konkrete Implementierung...")

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
    print("\n❌ KEINE funktionierende Qwen2-VL Version gefunden!")

# Let's also check which analyzers are actually imported in the registry
print("\n=== ÜBERPRÜFE REGISTRY IMPORTS ===")
try:
    from ml_analyzer_registry_complete import ML_ANALYZERS
    
    for name, analyzer_class in ML_ANALYZERS.items():
        if 'qwen' in name:
            print(f"{name}: {analyzer_class.__name__} aus {analyzer_class.__module__}")
except Exception as e:
    print(f"Fehler beim Import der Registry: {e}")