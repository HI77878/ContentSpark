#!/usr/bin/env python3
"""
Diagnose ALLER 22 aktiven Analyzer - Was funktioniert wirklich?
"""
import sys
sys.path.append('/home/user/tiktok_production')
import json

# Lade die letzte Analyse
with open('/home/user/tiktok_production/results/7446489995663117590_multiprocess_20250708_125345.json') as f:
    data = json.load(f)

print("🔍 DIAGNOSE DER 22 AKTIVEN ANALYZER")
print("="*80)

analyzer_results = data['analyzer_results']
problems = []
working = []
suspicious = []

for name, result in sorted(analyzer_results.items()):
    print(f"\n{name}:")
    
    # Check für leere oder nutzlose Daten
    if not result or result == {}:
        print(f"  ❌ KOMPLETT LEER!")
        problems.append((name, "Keine Daten"))
        continue
    
    segments = result.get('segments', [])
    
    # Spezifische Checks pro Analyzer
    if name == "eye_tracking":
        if segments:
            # Check ob alle "unbekannt" sind oder sinnlose Daten
            sample_data = segments[0] if segments else {}
            gaze_dirs = [seg.get('gaze_direction', seg.get('direction', 'none')) for seg in segments]
            unique_dirs = set(gaze_dirs)
            
            print(f"  Segmente: {len(segments)}")
            print(f"  Unique Blickrichtungen: {unique_dirs}")
            
            if unique_dirs == {'unbekannt'} or unique_dirs == {'none'} or not any(gaze_dirs):
                print(f"  ❌ KAPUTT! Alle {len(segments)} Messungen sind nutzlos")
                problems.append((name, f"Liefert nur {unique_dirs}"))
            else:
                working.append((name, f"{len(segments)} Messungen"))
    
    elif name == "speech_emotion":
        if segments:
            emotions = [seg.get('emotion', 'unknown') for seg in segments]
            unique_emotions = set(emotions)
            emotion_counts = {e: emotions.count(e) for e in unique_emotions}
            
            print(f"  Segmente: {len(segments)}")
            print(f"  Emotionen gefunden: {emotion_counts}")
            
            if unique_emotions == {'unknown'} or unique_emotions == {'neutral'}:
                print(f"  ❌ KAPUTT! Erkennt keine echten Emotionen")
                problems.append((name, f"Nur {unique_emotions} erkannt"))
            else:
                working.append((name, f"{len(unique_emotions)} Emotionen"))
    
    elif name == "age_estimation":
        if not segments:
            print(f"  ❌ KAPUTT! Keine Segmente trotz Video mit Person")
            problems.append((name, "Keine Segmente"))
        else:
            ages = []
            for seg in segments:
                age = seg.get('age', seg.get('estimated_age', None))
                if age and isinstance(age, (int, float)) and age > 0:
                    ages.append(age)
            
            print(f"  Segmente: {len(segments)}")
            print(f"  Altersschätzungen: {len(ages)}")
            
            if not ages:
                print(f"  ❌ KAPUTT! Keine gültigen Altersangaben")
                problems.append((name, "Keine Altersangaben"))
            else:
                avg_age = sum(ages) / len(ages)
                print(f"  ✅ Durchschnittsalter: {avg_age:.1f} Jahre")
                working.append((name, f"Avg: {avg_age:.0f} Jahre"))
    
    elif name == "speech_rate":
        if segments:
            # Check verschiedene Felder für Speech Rate
            wpm_values = []
            for seg in segments:
                wpm = seg.get('words_per_minute', seg.get('wpm', seg.get('speech_rate', 0)))
                if wpm and wpm > 0:
                    wpm_values.append(wpm)
            
            print(f"  Segmente: {len(segments)}")
            print(f"  WPM Werte gefunden: {len(wpm_values)}")
            
            if not wpm_values:
                print(f"  ❌ KAPUTT! Keine WPM-Daten")
                problems.append((name, "Keine Sprechgeschwindigkeit"))
            else:
                print(f"  ✅ WPM Range: {min(wpm_values):.0f}-{max(wpm_values):.0f}")
                working.append((name, f"WPM: {min(wpm_values):.0f}-{max(wpm_values):.0f}"))
    
    elif name == "comment_cta_detection":
        if len(segments) == 0:
            print(f"  ⚠️  Keine CTAs gefunden (könnte korrekt sein)")
            suspicious.append((name, "Keine Segmente - könnte OK sein"))
        else:
            working.append((name, f"{len(segments)} CTAs"))
    
    elif name == "visual_effects":
        if segments:
            effects = [seg.get('effect', seg.get('type', 'none')) for seg in segments]
            unique_effects = set(effects)
            
            print(f"  Segmente: {len(segments)}")
            print(f"  Effekte: {unique_effects}")
            
            if unique_effects == {'none'} or not any(effects):
                print(f"  ❌ VERDÄCHTIG! Keine echten Effekte erkannt")
                suspicious.append((name, "Möglicherweise Platzhalter"))
            else:
                working.append((name, f"{len(unique_effects)} Effekttypen"))
    
    else:
        # Generelle Prüfung für andere Analyzer
        if segments:
            # Prüfe ob Segmente sinnvolle Daten haben
            sample_seg = segments[0] if segments else {}
            
            # Check für verdächtige Patterns
            has_real_data = False
            for seg in segments[:3]:  # Check erste 3 Segmente
                for key, value in seg.items():
                    if key not in ['timestamp', 'start_time', 'end_time'] and value not in [None, '', 'unknown', 'unbekannt', 0, []]:
                        has_real_data = True
                        break
            
            if has_real_data:
                print(f"  ✅ OK: {len(segments)} Segmente mit Daten")
                working.append((name, f"{len(segments)} Segmente"))
            else:
                print(f"  ⚠️  VERDÄCHTIG! Segmente könnten Platzhalter sein")
                suspicious.append((name, "Möglicherweise leere Daten"))

print("\n" + "="*80)
print("\n📊 ZUSAMMENFASSUNG:")
print(f"\n❌ KAPUTTE ANALYZER ({len(problems)}):")
for analyzer, issue in problems:
    print(f"  - {analyzer}: {issue}")

print(f"\n⚠️  VERDÄCHTIGE ANALYZER ({len(suspicious)}):")
for analyzer, issue in suspicious:
    print(f"  - {analyzer}: {issue}")

print(f"\n✅ FUNKTIONIERENDE ANALYZER ({len(working)}):")
for analyzer, info in working:
    print(f"  - {analyzer}: {info}")

print(f"\n📈 STATISTIK:")
print(f"  Total Analyzer: {len(analyzer_results)}")
print(f"  Kaputt: {len(problems)} ({len(problems)/len(analyzer_results)*100:.0f}%)")
print(f"  Verdächtig: {len(suspicious)} ({len(suspicious)/len(analyzer_results)*100:.0f}%)")
print(f"  Funktionierend: {len(working)} ({len(working)/len(analyzer_results)*100:.0f}%)")