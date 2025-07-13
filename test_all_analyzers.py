#!/usr/bin/env python3
"""Test all analyzers for functionality"""
import sys
import importlib.util
from pathlib import Path

sys.path.insert(0, '/home/user/tiktok_production')

# Find all analyzers
analyzer_dir = Path("/home/user/tiktok_production/analyzers")
working = []
broken = []

for analyzer_file in analyzer_dir.glob("*.py"):
    if analyzer_file.name in ["__init__.py", "base_analyzer.py"]:
        continue
    
    module_name = analyzer_file.stem
    try:
        # Try to import
        spec = importlib.util.spec_from_file_location(module_name, analyzer_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find analyzer class
        analyzer_found = False
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and attr_name not in ["BaseAnalyzer", "GPUBatchAnalyzer"]:
                try:
                    # Try to instantiate
                    instance = attr()
                    working.append((module_name, attr_name))
                    analyzer_found = True
                    break
                except Exception as e:
                    if not analyzer_found:
                        broken.append((module_name, attr_name, str(e)[:100]))
                    analyzer_found = True
                    break
        
        if not analyzer_found:
            broken.append((module_name, "NO_CLASS", "No analyzer class found"))
                    
    except Exception as e:
        broken.append((module_name, "IMPORT_FAILED", str(e)[:100]))

print(f"\n✅ WORKING ANALYZERS ({len(working)}):")
for module, cls in sorted(working):
    print(f"  - {module} ({cls})")

print(f"\n❌ BROKEN ANALYZERS ({len(broken)}):")
for module, cls, error in sorted(broken):
    print(f"  - {module} ({cls}): {error}")

print(f"\nTOTAL: {len(working)} working, {len(broken)} broken out of {len(working) + len(broken)} analyzers")