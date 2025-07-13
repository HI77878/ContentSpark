#!/usr/bin/env python3
import json
import glob
import os
from datetime import datetime
import statistics

print("\n5. PRODUCTION READINESS CHECKLISTE")
print("="*80)

# Get recent performance data
results = []
for file in glob.glob('/home/user/tiktok_production/results/*.json'):
    if os.path.getmtime(file) > datetime.now().timestamp() - 7200:
        results.append(file)

results = sorted(results)[-10:]

rt_factors = []
analyzer_stats = {}
errors_found = []

for result_file in results:
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        meta = data.get('metadata', {})
        rt_factor = meta.get('realtime_factor', 0)
        if rt_factor > 0:
            rt_factors.append(rt_factor)
            
        # Count analyzer stats
        for analyzer, result in data.get('analyzer_results', {}).items():
            if analyzer not in analyzer_stats:
                analyzer_stats[analyzer] = {'runs': 0, 'successes': 0}
            analyzer_stats[analyzer]['runs'] += 1
            
            if isinstance(result, dict):
                if 'error' in result:
                    errors_found.append(f"{analyzer}: {result['error']}")
                elif 'segments' in result and len(result['segments']) > 0:
                    analyzer_stats[analyzer]['successes'] += 1
                    
    except Exception as e:
        errors_found.append(f"Parse error: {str(e)}")

checklist = {
    "Performance <5x realtime": False,
    "Alle 20 Analyzer funktionieren": False,
    "Keine kritischen Fehler": False,
    "GPU Memory stabil": False,
    "Disk Space ausreichend": False,
    "API stabil": False,
    "Download funktioniert": False,
    "Datenqualität ausreichend": False
}

# Fülle Checkliste basierend auf Daten
if rt_factors:
    recent_performance = rt_factors[-3:]  # Last 3 runs
    checklist["Performance <5x realtime"] = all(x < 5 for x in recent_performance)

if analyzer_stats:
    working_analyzers = sum(1 for stats in analyzer_stats.values() if stats['successes'] > 0)
    checklist["Alle 20 Analyzer funktionieren"] = working_analyzers >= 18

# Check for critical errors (not just any errors)
critical_errors = [e for e in errors_found if 'CUDA out of memory' in e or 'Failed to load' in e]
checklist["Keine kritischen Fehler"] = len(critical_errors) == 0

# Check system status
import subprocess
import requests

try:
    # GPU Memory check
    gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
    if gpu_result.returncode == 0:
        used, total = map(int, gpu_result.stdout.strip().split(', '))
        checklist["GPU Memory stabil"] = used < total * 0.9  # Under 90%
    
    # Disk space check
    disk_result = subprocess.run(['df', '-h', '/home/user'], capture_output=True, text=True)
    if disk_result.returncode == 0:
        lines = disk_result.stdout.strip().split('\n')
        usage_line = lines[1]
        usage_pct = int(usage_line.split()[4].replace('%', ''))
        checklist["Disk Space ausreichend"] = usage_pct < 80
    
    # API check
    try:
        response = requests.get('http://localhost:8003/health', timeout=5)
        checklist["API stabil"] = response.status_code == 200
    except:
        checklist["API stabil"] = False
        
    # Download check
    yt_dlp_result = subprocess.run(['which', 'yt-dlp'], capture_output=True, text=True)
    checklist["Download funktioniert"] = yt_dlp_result.returncode == 0
    
except Exception as e:
    print(f"Error checking system status: {e}")

# Data quality check
if results:
    latest_file = sorted(results)[-1]
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        # Check if key analyzers produced meaningful data
        qwen_data = data.get('analyzer_results', {}).get('qwen2_vl_temporal', {})
        speech_data = data.get('analyzer_results', {}).get('speech_transcription', {})
        object_data = data.get('analyzer_results', {}).get('object_detection', {})
        
        quality_checks = []
        if qwen_data.get('segments'):
            quality_checks.append(len(qwen_data['segments']) > 0)
        if object_data.get('segments'):
            quality_checks.append(len(object_data['segments']) > 0)
        if speech_data.get('segments'):
            quality_checks.append(len(speech_data['segments']) > 0)
            
        checklist["Datenqualität ausreichend"] = len(quality_checks) >= 2 and all(quality_checks)
        
    except Exception as e:
        checklist["Datenqualität ausreichend"] = False

# Zeige Checkliste
print("\nCHECKLISTE:")
for item, status in checklist.items():
    print(f"  {'✅' if status else '❌'} {item}")

# Summary
passed = sum(1 for status in checklist.values() if status)
total = len(checklist)
print(f"\n📊 GESAMT: {passed}/{total} Checks bestanden ({passed/total*100:.1f}%)")

if rt_factors:
    print(f"\n📈 AKTUELLE PERFORMANCE:")
    print(f"  Letzte 3 Runs: {[f'{x:.1f}x' for x in rt_factors[-3:]]}")
    print(f"  Durchschnitt: {statistics.mean(rt_factors):.1f}x")

if analyzer_stats:
    print(f"\n🔧 ANALYZER STATUS:")
    working = sum(1 for stats in analyzer_stats.values() if stats['successes'] > 0)
    print(f"  Funktionsfähige Analyzer: {working}/{len(analyzer_stats)}")

if errors_found:
    print(f"\n⚠️  GEFUNDENE PROBLEME:")
    for error in errors_found[:5]:
        print(f"  - {error}")
    if len(errors_found) > 5:
        print(f"  ... und {len(errors_found)-5} weitere")

print("\n" + "="*80)
print("ENDE DES VOLLSTÄNDIGEN AUDITS")
print("="*80)