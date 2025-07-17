#!/usr/bin/env python3
"""
Analyze and summarize the video analysis results
"""
import json
import sys

def analyze_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("="*80)
    print("VOLLSTÄNDIGE ANALYSE-ZUSAMMENFASSUNG")
    print("="*80)
    
    # Metadata
    meta = data.get('metadata', {})
    print("\nVIDEO METADATA:")
    print(f"  - Video: {meta.get('video_path', 'N/A')}")
    print(f"  - Dauer: {meta.get('duration', 'N/A')}s")
    print(f"  - Analysezeit: {meta.get('analysis_time', 'N/A')}s")
    print(f"  - Realtime-Faktor: {meta.get('realtime_factor', 'N/A')}x")
    print(f"  - Analyzer verwendet: {meta.get('analyzers_completed', 'N/A')}/{len(meta.get('analyzers_used', []))}")
    
    # Analyzer Results
    results = data.get('analyzer_results', {})
    print(f"\nANALYZER ERGEBNISSE ({len(results)} Analyzer):")
    
    # 1. Video-LLaVA
    if 'video_llava' in results:
        llava = results['video_llava']
        print("\n1. VIDEO-LLAVA (Primärer Video-Analyzer):")
        if 'error' not in llava:
            desc = llava.get('description', '')
            print(f"   Status: ✅ Erfolg")
            print(f"   Beschreibung ({len(desc)} Zeichen):")
            print(f"   {desc}")
            if 'metadata' in llava:
                print(f"   Modell: {llava['metadata'].get('model', 'N/A')}")
                print(f"   Frames analysiert: {llava['metadata'].get('frames_analyzed', 'N/A')}")
        else:
            print(f"   Status: ❌ Fehler - {llava['error']}")
    
    # 2. Speech Transcription
    if 'speech_transcription' in results:
        speech = results['speech_transcription']
        print("\n2. SPEECH TRANSCRIPTION:")
        if 'error' not in speech:
            segments = speech.get('segments', [])
            print(f"   Status: ✅ {len(segments)} Segmente")
            print(f"   Sprache: {speech.get('language', 'N/A')} (Konfidenz: {speech.get('language_confidence', 0):.2f})")
            if segments:
                print("   Transkript-Auszug:")
                for i, seg in enumerate(segments[:3]):  # Erste 3 Segmente
                    start = seg.get('start', seg.get('start_time', 0))
                    end = seg.get('end', seg.get('end_time', 0))
                    text = seg.get('text', seg.get('content', ''))
                    print(f"   [{start:.1f}s - {end:.1f}s]: {text}")
                if len(segments) > 3:
                    print(f"   ... und {len(segments)-3} weitere Segmente")
        else:
            print(f"   Status: ❌ Fehler")
    
    # 3. Object Detection
    if 'object_detection' in results:
        objects = results['object_detection']
        print("\n3. OBJECT DETECTION:")
        if 'error' not in objects:
            segments = objects.get('segments', [])
            total_objects = objects.get('total_objects_detected', 0)
            print(f"   Status: ✅ {total_objects} Objekte erkannt")
            print(f"   Frames verarbeitet: {objects.get('total_frames_processed', 0)}")
            
            # Zähle Objekt-Typen
            object_counts = {}
            for seg in segments:
                # Try multiple fields for object name
                obj_class = seg.get('object') or seg.get('class') or seg.get('label') or seg.get('object_class', 'unknown')
                object_counts[obj_class] = object_counts.get(obj_class, 0) + 1
            
            print("   Top Objekte:")
            for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   - {obj}: {count}x")
        else:
            print(f"   Status: ❌ Fehler")
    
    # 4. Text Overlay
    if 'text_overlay' in results:
        text = results['text_overlay']
        print("\n4. TEXT OVERLAY (Untertitel/Text im Video):")
        if 'error' not in text:
            segments = text.get('segments', [])
            print(f"   Status: ✅ {len(segments)} Text-Segmente")
            if segments:
                print("   Text-Beispiele:")
                for i, seg in enumerate(segments[:3]):
                    start = seg.get('start_time', seg.get('timestamp', 0))
                    text = seg.get('text', seg.get('content', ''))
                    print(f"   [{start:.1f}s]: {text}")
                if len(segments) > 3:
                    print(f"   ... und {len(segments)-3} weitere")
        else:
            print(f"   Status: ❌ Fehler")
    
    # 5. Audio Analysis
    if 'audio_analysis' in results:
        audio = results['audio_analysis']
        print("\n5. AUDIO ANALYSIS:")
        if 'error' not in audio:
            print(f"   Status: ✅ Erfolg")
            print(f"   Audio-Typ: {audio.get('audio_type', 'N/A')}")
            if 'segments' in audio:
                print(f"   Segmente: {len(audio['segments'])}")
        else:
            print(f"   Status: ❌ Fehler")
    
    # 6. Visual Effects
    if 'visual_effects' in results:
        effects = results['visual_effects']
        print("\n6. VISUAL EFFECTS:")
        if 'error' not in effects:
            segments = effects.get('segments', [])
            print(f"   Status: ✅ {len(segments)} Effekte erkannt")
            effect_types = set()
            for seg in segments:
                effect_types.update(seg.get('effects', []))
            if effect_types:
                print(f"   Effekt-Typen: {', '.join(effect_types)}")
        else:
            print(f"   Status: ❌ Fehler")
    
    # 7. Camera Analysis
    if 'camera_analysis' in results:
        camera = results['camera_analysis']
        print("\n7. CAMERA ANALYSIS:")
        if 'error' not in camera:
            segments = camera.get('segments', [])
            print(f"   Status: ✅ {len(segments)} Bewegungen")
            movements = set()
            for seg in segments:
                movements.add(seg.get('movement', 'unknown'))
            if movements:
                print(f"   Bewegungstypen: {', '.join(movements)}")
        else:
            print(f"   Status: ❌ Fehler")
    
    # Zusammenfassung
    print("\n" + "="*80)
    print("ZUSAMMENFASSUNG:")
    
    # Zähle erfolgreiche vs. fehlerhafte Analyzer
    successful = sum(1 for r in results.values() if 'error' not in r)
    failed = len(results) - successful
    
    print(f"✅ Erfolgreiche Analyzer: {successful}/{len(results)}")
    if failed > 0:
        print(f"❌ Fehlgeschlagene Analyzer: {failed}")
        for name, result in results.items():
            if 'error' in result:
                print(f"   - {name}: {result['error']}")
    
    # Check for placeholder data
    print("\nDATENQUALITÄT:")
    for name, result in results.items():
        if isinstance(result, dict) and 'segments' in result:
            segments = result['segments']
            if len(segments) == 0:
                print(f"   ⚠️  {name}: Keine Segmente (möglicherweise Platzhalter)")
            elif len(segments) == 1 and all(len(str(v)) < 20 for v in segments[0].values() if isinstance(v, str)):
                print(f"   ⚠️  {name}: Verdächtig kurze Daten (möglicherweise Platzhalter)")
    
    print("="*80)

if __name__ == "__main__":
    json_path = "/home/user/tiktok_production/results/leon_schliebach_7446489995663117590_multiprocess_20250707_071506.json"
    analyze_results(json_path)