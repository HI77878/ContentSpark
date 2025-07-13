#!/usr/bin/env python3
import json
import glob
import os
from datetime import datetime

# Finde alle Results
results = glob.glob('/home/user/tiktok_production/results/*.json')
performance_data = []

for r in sorted(results)[-20:]:  # Last 20 results
    try:
        with open(r, 'r') as f:
            data = json.load(f)
        
        meta = data.get('metadata', {})
        analyzer_times = {}
        
        # Sammle Processing Times pro Analyzer
        for name, result in data.get('analyzer_results', {}).items():
            if isinstance(result, dict):
                if 'processing_time' in result:
                    analyzer_times[name] = result['processing_time']
                elif 'summary' in result and 'processing_time_seconds' in result['summary']:
                    analyzer_times[name] = result['summary']['processing_time_seconds']
        
        performance_data.append({
            'file': os.path.basename(r),
            'duration': meta.get('duration', 0),
            'total_time': meta.get('processing_time_seconds', 0),
            'realtime': meta.get('realtime_factor', 0),
            'analyzer_times': analyzer_times,
            'timestamp': r.split('_')[-2] + '_' + r.split('_')[-1].replace('.json', '')
        })
    except Exception as e:
        print(f"Error reading {r}: {e}")

# Zeige Trends
print("PERFORMANCE TREND (last 20 runs):")
print("-" * 80)
for p in sorted(performance_data, key=lambda x: x['timestamp']):
    print(f"{p['timestamp']} - {p['file'][:30]}... - {p['realtime']:.2f}x realtime ({p['total_time']:.1f}s for {p['duration']:.1f}s video)")

# Finde die langsamsten Analyzer
all_times = {}
for p in performance_data:
    for analyzer, time in p['analyzer_times'].items():
        if analyzer not in all_times:
            all_times[analyzer] = []
        all_times[analyzer].append(time)

print("\nLANGSAMSTE ANALYZER (Durchschnitt der letzten Runs):")
print("-" * 80)
avg_times = [(sum(times)/len(times), name, len(times)) for name, times in all_times.items() if times]
for time, name, count in sorted(avg_times, reverse=True)[:15]:
    print(f"{name:30} : {time:6.1f}s (from {count} runs)")

# Find best performing run
best_run = min(performance_data, key=lambda x: x['realtime'] if x['realtime'] > 0 else float('inf'))
print(f"\nBEST PERFORMING RUN: {best_run['file']} - {best_run['realtime']:.2f}x realtime")
print("Analyzer times for best run:")
for name, time in sorted(best_run['analyzer_times'].items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {name:30} : {time:6.1f}s")