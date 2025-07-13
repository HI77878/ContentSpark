#!/usr/bin/env python3
"""Debug why analyzers aren't loading"""

import sys
sys.path.append('/home/user/tiktok_production')

from configs.gpu_groups_config import GPU_ANALYZER_GROUPS, DISABLED_ANALYZERS
from ml_analyzer_registry_complete import ML_ANALYZERS

print("=== DEBUGGING ANALYZER LOADING ===\n")

# Simulate exactly what ProductionEngine does
active_analyzers = []
for group_name, analyzer_list in GPU_ANALYZER_GROUPS.items():
    print(f"\n{group_name}:")
    for analyzer in analyzer_list:
        in_disabled = analyzer in DISABLED_ANALYZERS
        in_registry = analyzer in ML_ANALYZERS
        
        print(f"  {analyzer}:")
        print(f"    - In DISABLED_ANALYZERS: {in_disabled}")
        print(f"    - In ML_ANALYZERS: {in_registry}")
        
        if analyzer not in DISABLED_ANALYZERS and analyzer in ML_ANALYZERS:
            active_analyzers.append(analyzer)
            print(f"    ✓ ADDED to active")
        else:
            print(f"    ✗ SKIPPED")

# Remove duplicates
seen = set()
active_analyzers = [x for x in active_analyzers if not (x in seen or seen.add(x))]

print(f"\n\n=== FINAL RESULTS ===")
print(f"Total active analyzers: {len(active_analyzers)}")
print("\nActive analyzer list:")
for a in sorted(active_analyzers):
    print(f"  - {a}")

print("\n\nChecking specific analyzers:")
for check in ['face_emotion', 'body_pose', 'cross_analyzer_intelligence']:
    if check in active_analyzers:
        print(f"  ✓ {check} is ACTIVE")
    else:
        print(f"  ✗ {check} is MISSING")