#!/usr/bin/env python3
"""
Erstellt eine DETAILLIERTE Video-Beschreibung aus ALLEN Analyzer-Daten
Kombiniert die echten Daten statt sich auf halluzinierende LLaVA zu verlassen
"""
import json
import sys

def create_detailed_description(json_file):
    """Erstellt detaillierte Beschreibung aus allen Analyzer-Daten"""
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=== DETAILLIERTE VIDEO-BESCHREIBUNG AUS ALLEN ANALYZERN ===\n")
    
    # Metadata
    if 'metadata' in data:
        meta = data['metadata']
        print(f"Video-Datei: {meta.get('file_path', 'unbekannt')}")
        print(f"Dauer: {meta.get('duration', 0):.1f} Sekunden")
        print(f"Auflösung: {meta.get('width', 0)}x{meta.get('height', 0)}")
        print(f"FPS: {meta.get('fps', 0):.1f}")
        print(f"Analyzers: {meta.get('successful_analyzers', 0)} erfolgreich\n")
    
    # Speech Transcription - Was wird GESAGT
    print("=== GESPROCHENER TEXT ===")
    if 'speech_transcription' in data:
        segments = data['speech_transcription'].get('segments', [])
        for seg in segments:
            print(f"[{seg['start_time']:.1f}-{seg['end_time']:.1f}s] \"{seg['text']}\"")
    else:
        print("Keine Sprache erkannt")
    
    # Text Overlays - Was wird EINGEBLENDET
    print("\n=== TEXT-OVERLAYS ===")
    if 'text_overlay' in data:
        segments = data['text_overlay'].get('segments', [])
        unique_texts = {}
        for seg in segments:
            text = seg.get('text', '')
            if text and text not in unique_texts:
                unique_texts[text] = seg['timestamp']
        
        for text, time in sorted(unique_texts.items(), key=lambda x: x[1]):
            print(f"[{time:.1f}s] \"{text}\"")
    
    # Object Detection - Was ist zu SEHEN
    print("\n=== ERKANNTE OBJEKTE (Häufigkeit) ===")
    if 'object_detection' in data:
        segments = data['object_detection'].get('segments', [])
        object_counts = {}
        for seg in segments:
            obj = seg.get('object', seg.get('label', seg.get('class', 'unknown')))
            object_counts[obj] = object_counts.get(obj, 0) + 1
        
        for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"- {obj}: {count}x")
    
    # Camera Analysis - Kamerabewegungen
    print("\n=== KAMERA-BEWEGUNGEN ===")
    if 'camera_analysis' in data:
        segments = data['camera_analysis'].get('segments', [])
        movements = {}
        for seg in segments:
            movement = seg.get('movement_type', 'unknown')
            if movement not in movements:
                movements[movement] = []
            movements[movement].append(seg['timestamp'])
        
        for movement, times in movements.items():
            print(f"- {movement}: {len(times)}x (z.B. bei {times[0]:.1f}s)")
    
    # Visual Effects
    print("\n=== VISUELLE EFFEKTE ===")
    if 'visual_effects' in data:
        segments = data['visual_effects'].get('segments', [])
        effects = {}
        for seg in segments:
            for effect in seg.get('effects', []):
                if effect not in effects:
                    effects[effect] = []
                effects[effect].append(seg['timestamp'])
        
        for effect, times in sorted(effects.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            print(f"- {effect}: {len(times)}x")
    
    # Audio Analysis
    print("\n=== AUDIO-EIGENSCHAFTEN ===")
    if 'audio_analysis' in data:
        segments = data['audio_analysis'].get('segments', [])
        sources = {}
        for seg in segments:
            source = seg.get('audio_source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        for source, count in sources.items():
            print(f"- {source}: {count} Segmente")
    
    # Scene Segmentation
    print("\n=== SZENEN ===")
    if 'scene_segmentation' in data:
        segments = data['scene_segmentation'].get('segments', [])
        for i, seg in enumerate(segments):
            print(f"Szene {i+1}: [{seg['start_time']:.1f}-{seg['end_time']:.1f}s] {seg.get('scene_type', 'unknown')} ({seg['duration']:.1f}s)")
    
    # Erstelle Timeline
    print("\n=== ZEITLICHE ÜBERSICHT ===")
    
    # Sammle alle Events
    events = []
    
    # Text overlays
    if 'text_overlay' in data:
        for seg in data['text_overlay'].get('segments', []):
            events.append({
                'time': seg['timestamp'],
                'type': 'text',
                'description': f"Text: \"{seg.get('text', '')}\""
            })
    
    # Camera movements
    if 'camera_analysis' in data:
        for seg in data['camera_analysis'].get('segments', []):
            if seg.get('movement_type') != 'static':
                events.append({
                    'time': seg['timestamp'],
                    'type': 'camera',
                    'description': f"Kamera: {seg.get('movement_type', 'movement')}"
                })
    
    # Visual effects
    if 'visual_effects' in data:
        for seg in data['visual_effects'].get('segments', []):
            if seg.get('effects'):
                events.append({
                    'time': seg['timestamp'],
                    'type': 'effect',
                    'description': f"Effekt: {', '.join(seg['effects'])}"
                })
    
    # Sort by time and print
    events.sort(key=lambda x: x['time'])
    
    current_second = 0
    for event in events:
        second = int(event['time'])
        if second != current_second:
            print(f"\n[{second}s]")
            current_second = second
        print(f"  - {event['description']}")

if __name__ == "__main__":
    # Neueste Analyse-Datei
    result_file = "/home/user/tiktok_production/results/multiprocess_all_1751878133.json"
    create_detailed_description(result_file)