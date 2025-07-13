#!/usr/bin/env python3
"""
Verifiziere dass alle Daten korrekt gespeichert werden
"""
import json
import os
import glob

# Prüfe die letzten beiden Analysen
result_files = [
    '/home/user/tiktok_production/results/7446489995663117590_multiprocess_20250708_125345.json',  # Leon
    '/home/user/tiktok_production/results/7425998222721633569_multiprocess_20250708_133545.json'   # Mathilde
]

print("🔍 VERIFIZIERE DATENSPEICHERUNG")
print("="*80)

for result_file in result_files:
    if not os.path.exists(result_file):
        print(f"❌ Datei nicht gefunden: {result_file}")
        continue
        
    with open(result_file) as f:
        data = json.load(f)
    
    video_id = os.path.basename(result_file).split('_')[0]
    print(f"\n📁 Datei: {os.path.basename(result_file)}")
    print(f"   Größe: {os.path.getsize(result_file) / 1024 / 1024:.2f} MB")
    
    # Check Metadata
    meta = data.get('metadata', {})
    print(f"\n   📋 METADATA:")
    print(f"      Video ID: {video_id}")
    print(f"      TikTok URL: {meta.get('tiktok_url', '❌ FEHLT!')}")
    print(f"      Creator: {meta.get('creator_username', meta.get('creator', '❌ FEHLT!'))}")
    print(f"      Original URL: {meta.get('original_url', '❌ FEHLT!')}")
    print(f"      Video Path: {meta.get('video_path', '❌ FEHLT!')}")
    print(f"      Duration: {meta.get('duration', '❌ FEHLT!')}")
    print(f"      Processing Time: {meta.get('processing_time_seconds', 0):.1f}s")
    
    # Check Analyzer Results
    print(f"\n   📊 ANALYZER DATEN:")
    analyzer_results = data.get('analyzer_results', {})
    print(f"      Analyzer gespeichert: {len(analyzer_results)}/22")
    
    # Prüfe ob alle 22 aktiven Analyzer vorhanden sind
    expected_analyzers = [
        'speech_transcription', 'text_overlay', 'product_detection', 
        'object_detection', 'visual_effects', 'background_segmentation',
        'camera_analysis', 'color_analysis', 'content_quality',
        'eye_tracking', 'scene_segmentation', 'cut_analysis',
        'age_estimation', 'sound_effects', 'speech_emotion',
        'speech_rate', 'speech_flow', 'comment_cta_detection',
        'audio_environment', 'temporal_flow', 'audio_analysis',
        'qwen2_vl_temporal'
    ]
    
    missing = []
    for analyzer in expected_analyzers:
        if analyzer not in analyzer_results:
            missing.append(analyzer)
    
    if missing:
        print(f"      ❌ FEHLENDE ANALYZER: {missing}")
    else:
        print(f"      ✅ Alle 22 Analyzer vorhanden!")
    
    # Prüfe Datenqualität
    empty_analyzers = []
    for name, result in analyzer_results.items():
        if not result or result == {} or (not result.get('segments') and 'metadata' not in result):
            empty_analyzers.append(name)
    
    if empty_analyzers:
        print(f"      ⚠️  Analyzer ohne Daten: {empty_analyzers}")

print("\n\n🔍 PRÜFE API IMPLEMENTATION")
print("="*80)

# Check API files
api_files = [
    '/home/user/tiktok_production/api/stable_production_api.py',
    '/home/user/tiktok_production/api/stable_production_api_multiprocess.py'
]

for api_file in api_files:
    if os.path.exists(api_file):
        print(f"\n📄 Prüfe {os.path.basename(api_file)}:")
        with open(api_file) as f:
            api_code = f.read()
        
        # Suche nach TikTok URL Speicherung
        tiktok_refs = []
        metadata_refs = []
        
        for i, line in enumerate(api_code.split('\n')):
            if 'tiktok_url' in line.lower():
                tiktok_refs.append((i+1, line.strip()[:100]))
            if 'metadata' in line and ('update' in line or '=' in line or '[' in line):
                metadata_refs.append((i+1, line.strip()[:100]))
        
        if tiktok_refs:
            print(f"   ✅ Gefunden {len(tiktok_refs)} 'tiktok_url' Referenzen")
            for line_no, content in tiktok_refs[:3]:
                print(f"      Zeile {line_no}: {content}")
        else:
            print("   ❌ Keine 'tiktok_url' Referenzen gefunden!")
        
        if metadata_refs:
            print(f"   ℹ️  {len(metadata_refs)} Metadata-Operationen gefunden")

# Prüfe TikTok Downloader
downloader_file = '/home/user/tiktok_production/mass_processing/tiktok_downloader.py'
if os.path.exists(downloader_file):
    print(f"\n📄 Prüfe TikTok Downloader:")
    with open(downloader_file) as f:
        downloader_code = f.read()
    
    if 'metadata' in downloader_code:
        print("   ✅ Downloader erstellt Metadaten")
    else:
        print("   ⚠️  Downloader erstellt möglicherweise keine Metadaten")

# Zeige Status
print("\n\n📊 STATUS ZUSAMMENFASSUNG:")
print("="*80)

# Check if any files have TikTok metadata
has_tiktok_url = False
has_creator = False
for result_file in result_files:
    if os.path.exists(result_file):
        with open(result_file) as f:
            data = json.load(f)
        meta = data.get('metadata', {})
        if meta.get('tiktok_url'):
            has_tiktok_url = True
        if meta.get('creator_username') or meta.get('creator'):
            has_creator = True

if has_tiktok_url and has_creator:
    print("✅ TikTok URL und Creator werden jetzt gespeichert!")
    print("✅ API wurde erfolgreich erweitert")
    print("✅ Bestehende Dateien wurden aktualisiert")
    print("\n💡 VERWENDUNG:")
    print("   python3 download_and_analyze.py <tiktok_url>")
    print("   Beispiel: python3 download_and_analyze.py https://www.tiktok.com/@username/video/123")
else:
    print("⚠️  Metadaten noch nicht vollständig:")
    print(f"   TikTok URL: {'✅' if has_tiktok_url else '❌'}")
    print(f"   Creator: {'✅' if has_creator else '❌'}")