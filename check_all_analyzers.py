#!/usr/bin/env python3
"""Check which analyzers are actually broken"""

import json

# Load latest results
with open("/home/user/tiktok_production/results/7446489995663117590_multiprocess_20250712_153148.json", 'r') as f:
    data = json.load(f)

results = data['analyzer_results']

print("🔍 ANALYZER STATUS CHECK:")
print("="*60)

broken = []
partial = []
working = []

for analyzer, result in results.items():
    if not result or (isinstance(result, dict) and not result.get('segments') and not result.get('data')):
        broken.append(analyzer)
        print(f"❌ {analyzer}: KOMPLETT KAPUTT - Keine Daten")
    elif isinstance(result, dict):
        segments = result.get('segments', result.get('data', []))
        if not segments:
            broken.append(analyzer)
            print(f"❌ {analyzer}: KEINE SEGMENTS")
        elif len(segments) < 5:
            partial.append(analyzer)
            print(f"⚠️  {analyzer}: NUR {len(segments)} segments")
        else:
            # Check if segments have actual content
            sample = segments[0] if segments else {}
            if not sample or all(v is None or v == "" or v == [] for k, v in sample.items() if k != 'timestamp'):
                partial.append(analyzer)
                print(f"⚠️  {analyzer}: Segments ohne Inhalt")
            else:
                working.append(analyzer)
                print(f"✅ {analyzer}: {len(segments)} segments mit Daten")

print(f"\n📊 ZUSAMMENFASSUNG:")
print(f"✅ Funktioniert: {len(working)}/18")
print(f"⚠️  Teilweise: {len(partial)}/18")  
print(f"❌ Kaputt: {len(broken)}/18")

print(f"\n🔧 MUSS REPARIERT WERDEN:")
for a in broken + partial:
    print(f"  - {a}")