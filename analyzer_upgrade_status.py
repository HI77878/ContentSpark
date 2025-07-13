#!/usr/bin/env python3
"""
Status der Analyzer-Upgrades
"""

ANALYZER_STATUS = {
    # BEREITS UPGRADED (5/21)
    'text_overlay': {'status': 'ULTIMATE', 'file': 'text_overlay_ultimate_v2.py'},
    'gesture_body': {'status': 'ULTIMATE', 'file': 'gesture_body_ultimate.py'},
    'background_analysis': {'status': 'ULTIMATE', 'file': 'background_ultra_detailed.py'},
    'object_detection': {'status': 'ULTIMATE', 'file': 'object_detection_ultimate.py'},
    'speech_transcription': {'status': 'ULTIMATE', 'file': 'speech_transcription_ultimate.py'},
    'audio_analysis': {'status': 'ULTIMATE', 'file': 'audio_analysis_ultimate.py'},
    
    # NOCH ZU UPGRADEN (15/21)
    'video_llava': {'status': 'NEEDS_UPGRADE', 'priority': 'HIGH', 'issue': 'Halluzinationen, braucht bessere Prompts'},
    'camera_analysis': {'status': 'NEEDS_UPGRADE', 'priority': 'HIGH', 'issue': 'Nur basic movements'},
    'visual_effects': {'status': 'NEEDS_UPGRADE', 'priority': 'MEDIUM', 'issue': 'Erkennt nicht alle Effekte'},
    'composition_analysis': {'status': 'NEEDS_UPGRADE', 'priority': 'MEDIUM', 'issue': 'Zu oberflächlich'},
    'product_detection': {'status': 'NEEDS_UPGRADE', 'priority': 'HIGH', 'issue': 'Keine Marken/Details'},
    'color_analysis': {'status': 'NEEDS_UPGRADE', 'priority': 'LOW', 'issue': 'Nur basic Farben'},
    'scene_segmentation': {'status': 'NEEDS_UPGRADE', 'priority': 'HIGH', 'issue': 'Unpräzise Übergänge'},
    'content_quality': {'status': 'NEEDS_UPGRADE', 'priority': 'LOW', 'issue': 'Nur technische Metriken'},
    'cut_analysis': {'status': 'NEEDS_UPGRADE', 'priority': 'HIGH', 'issue': 'Verpasst Schnitte'},
    'eye_tracking': {'status': 'NEEDS_UPGRADE', 'priority': 'HIGH', 'issue': 'Keine echte Blickrichtung'},
    'age_estimation': {'status': 'NEEDS_UPGRADE', 'priority': 'LOW', 'issue': 'Ungenau'},
    'audio_environment': {'status': 'NEEDS_UPGRADE', 'priority': 'MEDIUM', 'issue': 'Zu grobe Kategorien'},
    'speech_emotion': {'status': 'NEEDS_UPGRADE', 'priority': 'HIGH', 'issue': 'Keine Zeitstempel'},
    'sound_effects': {'status': 'NEEDS_UPGRADE', 'priority': 'MEDIUM', 'issue': 'Verpasst viele Sounds'},
    'temporal_flow': {'status': 'NEEDS_UPGRADE', 'priority': 'LOW', 'issue': 'Zu abstrakt'},
    
    # Redundant/Merged
    'face_detection': {'status': 'MERGED', 'merged_into': 'gesture_body'},
    'emotion_detection': {'status': 'MERGED', 'merged_into': 'gesture_body'},
    'body_pose': {'status': 'MERGED', 'merged_into': 'gesture_body'},
    'hand_gesture': {'status': 'MERGED', 'merged_into': 'gesture_body'},
    'speech_rate': {'status': 'MERGED', 'merged_into': 'speech_transcription'},
}

def print_status():
    print("=== ANALYZER UPGRADE STATUS ===\n")
    
    ultimate = [k for k,v in ANALYZER_STATUS.items() if v['status'] == 'ULTIMATE']
    needs_upgrade = [k for k,v in ANALYZER_STATUS.items() if v['status'] == 'NEEDS_UPGRADE']
    merged = [k for k,v in ANALYZER_STATUS.items() if v['status'] == 'MERGED']
    
    print(f"✅ ULTIMATE VERSION: {len(ultimate)}/21")
    for analyzer in ultimate:
        print(f"   - {analyzer}: {ANALYZER_STATUS[analyzer]['file']}")
    
    print(f"\n❌ NEEDS UPGRADE: {len(needs_upgrade)}/21")
    high_priority = [k for k in needs_upgrade if ANALYZER_STATUS[k]['priority'] == 'HIGH']
    print(f"\n   HIGH PRIORITY ({len(high_priority)}):")
    for analyzer in high_priority:
        print(f"   - {analyzer}: {ANALYZER_STATUS[analyzer]['issue']}")
    
    print(f"\n🔄 MERGED INTO OTHER ANALYZERS: {len(merged)}/21")
    for analyzer in merged:
        print(f"   - {analyzer} → {ANALYZER_STATUS[analyzer]['merged_into']}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Ultimate: {len(ultimate)}")
    print(f"   Need Upgrade: {len(needs_upgrade)}")
    print(f"   Merged: {len(merged)}")
    print(f"   TOTAL EFFECTIVE: {len(ultimate) + len(needs_upgrade)} analyzers")

if __name__ == "__main__":
    print_status()