#!/usr/bin/env python3
"""Detailed quality validation of analysis results"""
import json
from pathlib import Path
from collections import defaultdict
import sys

# Load latest analysis result
results_dir = Path("/home/user/tiktok_production/results")
analysis_files = list(results_dir.glob("test_video_*.json"))

if not analysis_files:
    print("❌ Keine Analyse-Ergebnisse gefunden!")
    sys.exit(1)

latest_result = sorted(analysis_files)[-1]
print(f"Lade Analyse: {latest_result.name}")

with open(latest_result) as f:
    data = json.load(f)

print("\n=== DETAILLIERTE QUALITÄTSPRÜFUNG ===\n")

# Video Info
if 'metadata' in data:
    meta = data['metadata']
    print("VIDEO INFORMATION:")
    print(f"Datei: {meta.get('video_filename', 'unknown')}")
    print(f"Dauer: {meta.get('duration', 0):.1f}s")
    print(f"Analyzer: {meta.get('successful_analyzers', 0)}/{meta.get('total_analyzers', 0)}")
    print(f"Processing: {meta.get('processing_time_seconds', 0):.1f}s")
    video_duration = meta.get('duration', 10)
else:
    print("⚠️ Keine Metadaten gefunden")
    video_duration = 10

print("\n" + "="*80)

# Analyzer Quality Check
quality_report = {}
analyzer_results = data.get('analyzer_results', {})

print(f"\nANALYZER ERGEBNISSE: {len(analyzer_results)} Analyzer")
print("="*80)

for analyzer_name, result in analyzer_results.items():
    report = {
        'status': 'unknown',
        'segments': 0,
        'coverage': 0,
        'quality_issues': [],
        'data_sample': None,
        'time_gaps': []
    }
    
    # Check for errors
    if 'error' in result:
        report['status'] = 'error'
        report['quality_issues'].append(f"ERROR: {result['error'][:100]}")
    
    # Check for segments
    elif 'segments' in result and isinstance(result['segments'], list):
        segments = result['segments']
        report['segments'] = len(segments)
        
        if segments:
            # Time coverage check
            timestamps = []
            for seg in segments:
                ts = seg.get('timestamp', seg.get('start_time', seg.get('time', None)))
                if ts is not None:
                    timestamps.append(float(ts))
            
            if timestamps:
                timestamps.sort()
                
                # Calculate coverage
                time_range = max(timestamps) - min(timestamps)
                coverage = (time_range / video_duration * 100) if video_duration > 0 else 0
                report['coverage'] = coverage
                
                # Check for time gaps
                for i in range(1, len(timestamps)):
                    gap = timestamps[i] - timestamps[i-1]
                    if gap > 2.0:  # More than 2 seconds gap
                        report['time_gaps'].append({
                            'from': timestamps[i-1],
                            'to': timestamps[i],
                            'gap': gap
                        })
                
                if report['time_gaps']:
                    report['quality_issues'].append(f"Zeitlücken: {len(report['time_gaps'])}")
            
            # Data quality check
            sample_seg = segments[0]
            
            # Check segment structure
            if isinstance(sample_seg, dict):
                # Check for generic terms
                seg_str = json.dumps(sample_seg).lower()
                generic_terms = ['balanced', 'natural', 'moderate', 'normal', 'continues activity', 
                               'placeholder', 'unknown', 'default', 'standard', 'neutral']
                found_generic = [term for term in generic_terms if term in seg_str]
                if found_generic:
                    report['quality_issues'].append(f"Generisch: {', '.join(found_generic[:3])}")
                
                # Check data richness
                field_count = len(sample_seg)
                if field_count < 3:
                    report['quality_issues'].append(f"Wenig Felder: {field_count}")
                
                # Extract sample
                if 'description' in sample_seg:
                    report['data_sample'] = sample_seg['description'][:150]
                elif 'text' in sample_seg:
                    report['data_sample'] = sample_seg['text'][:150]
                elif 'caption' in sample_seg:
                    report['data_sample'] = sample_seg['caption'][:150]
                else:
                    # Get first string value
                    for key, value in sample_seg.items():
                        if isinstance(value, str) and len(value) > 10:
                            report['data_sample'] = f"{key}: {value[:100]}"
                            break
                    
                    if not report['data_sample']:
                        report['data_sample'] = str(sample_seg)[:150]
            
            # Check for repetitions
            if len(segments) > 3:
                # Compare first 3 segments
                seg1_str = json.dumps(segments[0], sort_keys=True)
                seg2_str = json.dumps(segments[1], sort_keys=True)
                seg3_str = json.dumps(segments[2], sort_keys=True)
                
                if seg1_str == seg2_str or seg2_str == seg3_str:
                    report['quality_issues'].append("Duplizierte Segmente")
            
            # Determine status
            if not report['quality_issues'] and report['coverage'] > 80:
                report['status'] = 'excellent'
            elif len(report['quality_issues']) <= 1 and report['coverage'] > 50:
                report['status'] = 'good'
            elif report['segments'] > 0:
                report['status'] = 'poor'
            else:
                report['status'] = 'no_data'
    else:
        report['status'] = 'no_data'
        report['quality_issues'].append("Keine Segmente")
    
    quality_report[analyzer_name] = report

# Output sorted by status
print("\nQUALITÄTSBERICHT NACH STATUS:")
print("="*80)

status_counts = defaultdict(int)
for status in ['excellent', 'good', 'poor', 'no_data', 'error']:
    analyzers = [(name, rep) for name, rep in quality_report.items() if rep['status'] == status]
    status_counts[status] = len(analyzers)
    
    if analyzers:
        print(f"\n{status.upper()} ({len(analyzers)} Analyzer):")
        print("-"*60)
        
        for analyzer_name, rep in sorted(analyzers)[:5]:  # Show max 5 per category
            print(f"\n{analyzer_name}:")
            print(f"  Segmente: {rep['segments']}")
            print(f"  Abdeckung: {rep['coverage']:.0f}%")
            
            if rep['quality_issues']:
                print(f"  Issues: {'; '.join(rep['quality_issues'])}")
            
            if rep['data_sample']:
                print(f"  Daten: {rep['data_sample'][:80]}...")
            
            if rep['time_gaps']:
                print(f"  Lücken: {len(rep['time_gaps'])} (größte: {max(g['gap'] for g in rep['time_gaps']):.1f}s)")
        
        if len(analyzers) > 5:
            print(f"\n  ... und {len(analyzers) - 5} weitere")

# Critical analyzers check
print("\n" + "="*80)
print("KRITISCHE ANALYZER CHECK:")
print("-"*80)

critical = {
    'qwen2_vl_temporal': 'Temporal Video Understanding',
    'qwen2_vl_optimized': 'Optimized Video Understanding',
    'speech_transcription': 'Sprach-Transkription',
    'object_detection': 'Objekt-Erkennung',
    'face_emotion': 'Gesichts-Emotion',
    'body_pose': 'Körperhaltung',
    'audio_analysis': 'Audio-Analyse'
}

critical_scores = []
for key, desc in critical.items():
    if key in analyzer_results:
        rep = quality_report[key]
        status_icon = {
            'excellent': '✅',
            'good': '🟡',
            'poor': '⚠️',
            'error': '❌',
            'no_data': '❌'
        }.get(rep['status'], '❓')
        
        score = {
            'excellent': 100,
            'good': 70,
            'poor': 40,
            'error': 0,
            'no_data': 0
        }.get(rep['status'], 0)
        
        critical_scores.append(score)
        
        print(f"\n{status_icon} {desc}:")
        print(f"   Status: {rep['status']}")
        print(f"   Segmente: {rep['segments']}")
        print(f"   Abdeckung: {rep['coverage']:.0f}%")
        
        if rep['data_sample']:
            print(f"   Beispiel: {rep['data_sample'][:100]}...")
    else:
        print(f"\n❌ {desc}: NICHT GEFUNDEN")
        critical_scores.append(0)

# Overall assessment
print("\n" + "="*80)
print("GESAMTBEWERTUNG:")
print("-"*80)

# Calculate scores
total_analyzers = len(quality_report)
excellent_count = status_counts['excellent']
good_count = status_counts['good']
poor_count = status_counts['poor']
error_count = status_counts['error']

overall_score = (excellent_count * 100 + good_count * 70 + poor_count * 40) / total_analyzers if total_analyzers > 0 else 0
critical_score = sum(critical_scores) / len(critical_scores) if critical_scores else 0

print(f"\nStatus-Verteilung:")
print(f"  Excellent: {excellent_count} ({excellent_count/total_analyzers*100:.0f}%)")
print(f"  Good: {good_count} ({good_count/total_analyzers*100:.0f}%)")
print(f"  Poor: {poor_count} ({poor_count/total_analyzers*100:.0f}%)")
print(f"  No Data: {status_counts['no_data']} ({status_counts['no_data']/total_analyzers*100:.0f}%)")
print(f"  Error: {error_count} ({error_count/total_analyzers*100:.0f}%)")

print(f"\nGESAMT-QUALITÄTSSCORE: {overall_score:.0f}/100")
print(f"KRITISCHE ANALYZER SCORE: {critical_score:.0f}/100")

if overall_score >= 80 and critical_score >= 80:
    print("\n✅ EXZELLENTE DATENQUALITÄT - Video-Rekonstruktion vollständig möglich!")
elif overall_score >= 60 and critical_score >= 60:
    print("\n🟡 GUTE DATENQUALITÄT - Video-Rekonstruktion größtenteils möglich")
elif overall_score >= 40:
    print("\n⚠️ AUSREICHENDE DATENQUALITÄT - Teilweise Rekonstruktion möglich")
else:
    print("\n❌ UNZUREICHENDE DATENQUALITÄT - Erhebliche Verbesserungen notwendig")

# Save quality report
quality_report_file = f"results/quality_report_{latest_result.stem}_{int(Path(latest_result).stat().st_mtime)}.json"
with open(quality_report_file, 'w') as f:
    json.dump({
        'source_file': str(latest_result),
        'timestamp': data.get('metadata', {}).get('analysis_timestamp', 'unknown'),
        'video_duration': video_duration,
        'total_analyzers': total_analyzers,
        'status_counts': dict(status_counts),
        'overall_score': overall_score,
        'critical_score': critical_score,
        'quality_details': quality_report
    }, f, indent=2)

print(f"\n📊 Qualitätsbericht gespeichert: {quality_report_file}")