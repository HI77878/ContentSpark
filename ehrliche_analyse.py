#!/usr/bin/env python3
"""
EHRLICHE ANALYSE - Zeigt ALLE Probleme
"""

import json
import glob
import os
from datetime import datetime

def analyze_all_results():
    # Finde alle Results der letzten Stunde
    results = []
    for file in glob.glob('/home/user/tiktok_production/results/*.json'):
        if os.path.getmtime(file) > datetime.now().timestamp() - 3600:
            results.append(file)
    
    print("="*80)
    print("VOLLSTÄNDIGE EHRLICHE ANALYSE - KEINE LÜGEN")
    print("="*80)
    
    performance_summary = []
    
    for result_file in sorted(results)[-5:]:  # Letzte 5 Analysen
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        meta = data.get('metadata', {})
        video_id = meta.get('tiktok_video_id', 'unknown')
        creator = meta.get('creator_username', 'unknown')
        
        print(f"\n{'='*60}")
        print(f"VIDEO: {creator} - {video_id}")
        print(f"Datei: {os.path.basename(result_file)}")
        print(f"{'='*60}")
        
        # Performance
        duration = meta.get('duration', 0)
        processing = meta.get('processing_time_seconds', 0)
        realtime = meta.get('realtime_factor', 0)
        
        print(f"\n📊 PERFORMANCE:")
        print(f"  Video-Dauer: {duration:.1f}s")
        print(f"  Verarbeitung: {processing:.1f}s ({processing/60:.1f} Minuten)")
        print(f"  Realtime-Faktor: {realtime:.2f}x {'❌ ÜBER 3x!' if realtime > 3 else '✅ unter 3x'}")
        print(f"  Erfolgreiche Analyzer: {meta.get('successful_analyzers', 0)}/{meta.get('total_analyzers', 0)}")
        
        performance_summary.append({
            'video': video_id,
            'duration': duration,
            'processing': processing,
            'realtime': realtime
        })
        
        # Analyzer Status
        analyzer_results = data.get('analyzer_results', {})
        errors = []
        no_data = []
        success = []
        
        for name, result in analyzer_results.items():
            if isinstance(result, dict):
                if 'error' in result:
                    errors.append(f"{name}: {result['error'][:100]}...")
                elif 'segments' in result:
                    seg_count = len(result['segments'])
                    error_segs = sum(1 for s in result['segments'] if 'error' in s)
                    if error_segs > 0:
                        errors.append(f"{name}: {error_segs}/{seg_count} Segment-Fehler")
                    elif seg_count == 0:
                        no_data.append(name)
                    else:
                        success.append(f"{name}: {seg_count} segments")
                else:
                    # Check for other data structures
                    if any(k in result for k in ['text', 'objects', 'emotions', 'movements']):
                        success.append(f"{name}: data vorhanden")
                    else:
                        no_data.append(f"{name}: unbekannte Struktur")
        
        print(f"\n✅ FUNKTIONIERT: {len(success)}")
        for s in success[:5]:
            print(f"  - {s}")
        if len(success) > 5:
            print(f"  ... und {len(success)-5} weitere")
        
        if errors:
            print(f"\n❌ FEHLER: {len(errors)}")
            for e in errors:
                print(f"  - {e}")
        
        if no_data:
            print(f"\n⚠️ KEINE DATEN: {len(no_data)}")
            for n in no_data[:5]:
                print(f"  - {n}")
        
        # Spezielle Checks
        print(f"\n🔍 SPEZIELLE CHECKS:")
        
        # Speech
        speech = analyzer_results.get('speech_transcription_ultimate', {})
        if isinstance(speech, dict) and 'segments' in speech:
            transcribed = [s for s in speech.get('segments', []) if s.get('text')]
            if transcribed:
                print(f"  ✅ Speech: {len(transcribed)} Segmente transkribiert")
                print(f"     Sprache: {speech.get('language', 'unknown')}")
                print(f"     Beispiel: '{transcribed[0]['text'][:80]}...'")
            else:
                print(f"  ❌ Speech: KEINE Transkription trotz Audio!")
        
        # CTA Detection
        cta = analyzer_results.get('comment_cta_detection', {})
        if isinstance(cta, dict):
            cta_segs = cta.get('segments', [])
            if video_id == "7525171065367104790":  # Marc Gebauer
                if any('noch mal bestellen' in str(s).lower() for s in cta_segs):
                    print(f"  ✅ CTA: Marc Gebauer 'Noch mal bestellen' ERKANNT!")
                else:
                    print(f"  ❌ CTA: Marc Gebauer Pattern NICHT erkannt!")
            elif cta_segs:
                print(f"  ✅ CTA: {len(cta_segs)} CTAs gefunden")
            else:
                print(f"  ⚠️ CTA: Keine CTAs gefunden")
        
        # Qwen2-VL
        for qwen_name in ['qwen2_vl_temporal', 'qwen2_vl_ultra', 'qwen2_vl_optimized']:
            qwen = analyzer_results.get(qwen_name, {})
            if isinstance(qwen, dict) and 'segments' in qwen:
                qwen_success = [s for s in qwen['segments'] if 'description' in s and 'error' not in s]
                if qwen_success:
                    print(f"  ✅ {qwen_name}: {len(qwen_success)}/{len(qwen['segments'])} Segmente")
                    desc = qwen_success[0].get('description', '')[:100]
                    print(f"     Beispiel: '{desc}...'")
                else:
                    print(f"  ❌ {qwen_name}: KEINE erfolgreichen Segmente!")
    
    # Performance Zusammenfassung
    print(f"\n{'='*80}")
    print("PERFORMANCE ZUSAMMENFASSUNG")
    print(f"{'='*80}")
    
    for p in performance_summary:
        print(f"\n{p['video']}:")
        print(f"  Dauer: {p['duration']:.1f}s")
        print(f"  Verarbeitung: {p['processing']:.1f}s")
        print(f"  Realtime: {p['realtime']:.2f}x {'❌' if p['realtime'] > 3 else '✅'}")
    
    avg_realtime = sum(p['realtime'] for p in performance_summary) / len(performance_summary)
    print(f"\n📊 DURCHSCHNITT: {avg_realtime:.2f}x realtime")
    print(f"{'❌ ZIEL VERFEHLT!' if avg_realtime > 3 else '✅ Ziel erreicht!'}")
    
    # GPU Memory Check
    print(f"\n💾 GPU MEMORY:")
    os.system("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader")

if __name__ == "__main__":
    analyze_all_results()