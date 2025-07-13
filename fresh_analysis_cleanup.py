#!/usr/bin/env python3
"""
Cleanup aller alten Analyse-Ergebnisse und Videos für frischen Start
"""
import os
import glob
import shutil
from datetime import datetime

print("🧹 CLEANUP ALLER ALTEN DATEN")
print("="*80)

# 1. Backup wichtiger Dateien (optional)
backup_dir = f"/home/user/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(backup_dir, exist_ok=True)

# Sichere Dokumentation
important_files = [
    '/home/user/tiktok_production/README.md',
    '/home/user/tiktok_production/OPTIMIZATION_SUMMARY.md',
    '/home/user/tiktok_production/METADATA_STORAGE_REPORT.md',
    '/home/user/tiktok_production/CLAUDE.md'
]

for file in important_files:
    if os.path.exists(file):
        shutil.copy2(file, backup_dir)
        print(f"✅ Backup: {os.path.basename(file)}")

# 2. Lösche alle Analyse-Ergebnisse
results_dir = '/home/user/tiktok_production/results/'
result_files = glob.glob(f"{results_dir}*.json")

print(f"\n📊 Lösche {len(result_files)} Analyse-Ergebnisse...")
for file in result_files[:10]:  # Zeige nur erste 10
    print(f"   ❌ Gelöscht: {os.path.basename(file)}")
    os.remove(file)
for file in result_files[10:]:  # Rest ohne Ausgabe
    os.remove(file)
if len(result_files) > 10:
    print(f"   ... und {len(result_files)-10} weitere Dateien")

# 3. Lösche alle heruntergeladenen Videos
video_dirs = [
    '/home/user/tiktok_videos/videos/',
    '/home/user/tiktok_production/test_videos/',
    '/home/user/tiktok_production/videos/',
    '/home/user/Downloads/',
    '/home/user/tiktok_production/downloads/videos/'
]

total_videos_deleted = 0
for video_dir in video_dirs:
    if os.path.exists(video_dir):
        videos = glob.glob(f"{video_dir}*.mp4") + glob.glob(f"{video_dir}*.MP4")
        if videos:
            print(f"\n📹 Lösche {len(videos)} Videos aus {video_dir}")
            for video in videos[:3]:  # Zeige nur erste 3
                print(f"   ❌ Gelöscht: {os.path.basename(video)}")
                os.remove(video)
            for video in videos[3:]:  # Rest ohne Ausgabe
                os.remove(video)
            if len(videos) > 3:
                print(f"   ... und {len(videos)-3} weitere Videos")
            total_videos_deleted += len(videos)

# 4. Lösche temporäre Dateien
temp_patterns = [
    '/tmp/*tiktok*',
    '/tmp/*analyzer*',
    '/tmp/*test*',
    '/home/user/tiktok_production/structured_analysis_*.py',
    '/home/user/tiktok_production/test_*.py',
    '/home/user/tiktok_production/*_analysis.py',
    '/home/user/tiktok_production/*_report.md',
    '/home/user/tiktok_production/*_report.json'
]

temp_deleted = 0
for pattern in temp_patterns:
    files = glob.glob(pattern)
    for file in files:
        try:
            if os.path.isfile(file) and 'fresh_analysis' not in file:
                os.remove(file)
                temp_deleted += 1
        except:
            pass

# 5. Leere Cache
print("\n🗑️ Leere Python Cache...")
cache_dirs = [
    '/home/user/tiktok_production/__pycache__',
    '/home/user/tiktok_production/analyzers/__pycache__',
    '/home/user/tiktok_production/api/__pycache__',
    '/home/user/tiktok_production/utils/__pycache__',
    '/home/user/tiktok_production/configs/__pycache__'
]

cache_deleted = 0
for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        cache_deleted += 1

print(f"\n✅ CLEANUP ABGESCHLOSSEN:")
print(f"   - {len(result_files)} Analyse-Ergebnisse gelöscht")
print(f"   - {total_videos_deleted} Videos gelöscht")
print(f"   - {temp_deleted} temporäre Dateien gelöscht")
print(f"   - {cache_deleted} Cache-Verzeichnisse bereinigt")
print(f"   - Backup erstellt in: {backup_dir}")
print("\n🎯 System ist bereit für frische Analyse!")