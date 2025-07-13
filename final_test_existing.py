#!/usr/bin/env python3
"""Finaler Test mit existierendem TikTok Video"""
import requests
import json
import time
from datetime import datetime

# Existierendes TikTok Video
TIKTOK_URL = "https://www.tiktok.com/@marcgebauer/video/7525171065367104790"
video_path = "/home/user/tiktok_videos/videos/7525171065367104790.mp4"
video_duration = 9.31

print("=== FINALER TEST MIT OPTIMIERTEN ANALYZERN ===")
print(f"TikTok URL: {TIKTOK_URL}")
print(f"Video: {video_path}")
print(f"Dauer: {video_duration:.1f}s")
print(f"Start: {datetime.now()}\n")

# Analyse mit TikTok URL
print("ANALYSIERE VIDEO MIT ALLEN OPTIMIERUNGEN...")
analysis_start = time.time()

response = requests.post(
    "http://localhost:8003/analyze",
    json={
        "video_path": video_path,
        "tiktok_url": TIKTOK_URL,
        "creator_username": "marcgebauer"
    },
    timeout=300
)

analysis_time = time.time() - analysis_start

if response.status_code == 200:
    result = response.json()
    print(f"✅ Analyse erfolgreich!")
    print(f"   Zeit: {analysis_time:.1f}s ({analysis_time/video_duration:.1f}x realtime)")
    print(f"   Erfolgreiche Analyzer: {result['successful_analyzers']}/{result['total_analyzers']}")
    print(f"   Ergebnis: {result['results_file']}\n")
    
    # Lade Ergebnisse
    with open(result['results_file']) as f:
        data = json.load(f)
    
    # TikTok Integration
    print("TIKTOK INTEGRATION:")
    print(f"  URL gespeichert: {'✅' if data['metadata'].get('tiktok_url') == TIKTOK_URL else '❌'}")
    print(f"  Creator: {data['metadata'].get('creator_username', 'N/A')}")
    print()
    
    # Analyzer Performance Tabelle
    print("ANALYZER PERFORMANCE:")
    print("="*80)
    print(f"{'Analyzer':<25} {'Segmente':<10} {'/Sekunde':<10} {'Status':<30}")
    print("-"*80)
    
    stats = {}
    for analyzer, result_data in sorted(data['analyzer_results'].items()):
        if 'error' in result_data:
            segments = 0
            status = f"❌ ERROR"
        elif 'segments' in result_data:
            segments = len(result_data['segments'])
            status = "✅ OK" if segments > 0 else "⚠️ Keine Daten"
        else:
            segments = 0
            status = "❌ Keine Segmente"
        
        rate = segments / video_duration
        stats[analyzer] = {'segments': segments, 'rate': rate}
        
        # Farbcodierung
        if rate >= 1.0:
            icon = "🟢"
        elif rate >= 0.5:
            icon = "🟡"
        else:
            icon = "🔴"
            
        print(f"{icon} {analyzer:<23} {segments:<10} {rate:<10.2f} {status}")
    
    # Kritische Analyzer Details
    print("\n" + "="*80)
    print("KRITISCHE ANALYZER IM DETAIL:\n")
    
    # 1. qwen2_vl_temporal
    qwen = data['analyzer_results'].get('qwen2_vl_temporal', {})
    print("QWEN2_VL_TEMPORAL (Video Understanding):")
    if 'segments' in qwen and qwen['segments']:
        segments = qwen['segments']
        print(f"  ✅ {len(segments)} Segmente ({len(segments)/video_duration:.1f}/s)")
        for i, seg in enumerate(segments[:3]):
            print(f"  [{seg.get('timestamp', 0):.1f}s] {seg.get('description', 'N/A')[:80]}...")
    else:
        print(f"  ❌ Keine Segmente")
    print()
    
    # 2. object_detection
    obj = data['analyzer_results'].get('object_detection', {})
    print("OBJECT_DETECTION:")
    if 'segments' in obj and obj['segments']:
        segments = obj['segments']
        print(f"  ✅ {len(segments)} Detektionen")
        # Zähle Objekt-Typen
        obj_counts = {}
        for seg in segments:
            obj_type = seg.get('object', 'unknown')
            obj_counts[obj_type] = obj_counts.get(obj_type, 0) + 1
        print(f"  Objekte: {dict(sorted(obj_counts.items(), key=lambda x: x[1], reverse=True)[:5])}")
    else:
        print(f"  ❌ Keine Objekte erkannt")
    print()
    
    # 3. speech_transcription
    speech = data['analyzer_results'].get('speech_transcription', {})
    print("SPEECH_TRANSCRIPTION:")
    if 'segments' in speech and speech['segments']:
        segments = speech['segments']
        print(f"  ✅ {len(segments)} Segmente ({len(segments)/video_duration:.1f}/s)")
        total_words = sum(seg.get('word_count', 0) for seg in segments)
        print(f"  Wörter gesamt: {total_words}")
        # Zeige erste Segmente
        for seg in segments[:3]:
            if seg.get('text'):
                print(f"  [{seg.get('start_time', 0):.1f}-{seg.get('end_time', 0):.1f}s] \"{seg['text']}\"")
    else:
        print(f"  ❌ Keine Sprache erkannt")
    print()
    
    # 4. visual_effects
    effects = data['analyzer_results'].get('visual_effects', {})
    print("VISUAL_EFFECTS:")
    if 'segments' in effects and effects['segments']:
        segments = effects['segments']
        print(f"  ✅ {len(segments)} Segmente ({len(segments)/video_duration:.1f}/s)")
        # Zähle Effekt-Typen
        effect_counts = {}
        for seg in segments:
            if 'effects' in seg:
                for effect_type, value in seg['effects'].items():
                    if effect_type != 'timestamp':
                        effect_counts[effect_type] = effect_counts.get(effect_type, 0) + 1
        print(f"  Top Effekte: {dict(sorted(effect_counts.items(), key=lambda x: x[1], reverse=True)[:5])}")
    print()
    
    # Zusammenfassung
    print("="*80)
    print("OPTIMIERUNGS-ERFOLG:\n")
    
    # Metriken
    successful = len([a for a, s in stats.items() if s['segments'] > 0])
    high_quality = len([a for a, s in stats.items() if s['rate'] >= 1.0])
    
    print(f"📊 Erfolgreiche Analyzer: {successful}/{len(stats)} ({successful/len(stats)*100:.0f}%)")
    print(f"🌟 High-Quality (≥1/s): {high_quality} Analyzer")
    print(f"⚡ Performance: {analysis_time/video_duration:.1f}x realtime")
    
    # Top 5 Performer
    print("\n🏆 TOP 5 PERFORMER:")
    top5 = sorted(stats.items(), key=lambda x: x[1]['rate'], reverse=True)[:5]
    for i, (analyzer, s) in enumerate(top5, 1):
        print(f"  {i}. {analyzer}: {s['rate']:.2f}/s ({s['segments']} Segmente)")
    
    # Speichere finalen Report
    final_report = {
        'test_timestamp': datetime.now().isoformat(),
        'video_duration': video_duration,
        'tiktok_url': TIKTOK_URL,
        'analysis_time': analysis_time,
        'realtime_factor': analysis_time/video_duration,
        'successful_analyzers': successful,
        'total_analyzers': len(stats),
        'high_quality_analyzers': high_quality,
        'analyzer_stats': stats
    }
    
    with open('final_optimization_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n✅ FINALER REPORT GESPEICHERT: final_optimization_report.json")
    
else:
    print(f"❌ Analyse fehlgeschlagen: {response.status_code}")
    print(response.text)