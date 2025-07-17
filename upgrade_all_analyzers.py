#\!/usr/bin/env python3
"""
Liste aller Analyzer die noch ein Upgrade brauchen
"""

ANALYZERS_TO_UPGRADE = [
    # HIGH PRIORITY (bereits 3 von 7 gemacht)
    {'name': 'product_detection', 'priority': 'HIGH', 'issue': 'Keine Marken/Details'},
    {'name': 'scene_segmentation', 'priority': 'HIGH', 'issue': 'Unpräzise Übergänge'},
    {'name': 'cut_analysis', 'priority': 'HIGH', 'issue': 'Verpasst Schnitte'},
    {'name': 'eye_tracking', 'priority': 'HIGH', 'issue': 'Keine echte Blickrichtung'},
    {'name': 'speech_emotion', 'priority': 'HIGH', 'issue': 'Keine Zeitstempel'},
    
    # MEDIUM PRIORITY
    {'name': 'composition_analysis', 'priority': 'MEDIUM', 'issue': 'Zu oberflächlich'},
    {'name': 'audio_environment', 'priority': 'MEDIUM', 'issue': 'Zu grobe Kategorien'},
    {'name': 'sound_effects', 'priority': 'MEDIUM', 'issue': 'Verpasst viele Sounds'},
    
    # LOW PRIORITY
    {'name': 'color_analysis', 'priority': 'LOW', 'issue': 'Nur basic Farben'},
    {'name': 'content_quality', 'priority': 'LOW', 'issue': 'Nur technische Metriken'},
    {'name': 'age_estimation', 'priority': 'LOW', 'issue': 'Ungenau'},
    {'name': 'temporal_flow', 'priority': 'LOW', 'issue': 'Zu abstrakt'},
]

print(f"Noch zu upgraden: {len(ANALYZERS_TO_UPGRADE)} Analyzer")
print("\nReihenfolge:")
for i, analyzer in enumerate(ANALYZERS_TO_UPGRADE, 1):
    print(f"{i}. {analyzer['name']} ({analyzer['priority']}) - {analyzer['issue']}")
