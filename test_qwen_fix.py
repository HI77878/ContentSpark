import sys
sys.path.append('/home/user/tiktok_production')

try:
    from analyzers.qwen2_vl_ultra_detailed import Qwen2VLUltraDetailedAnalyzer
    print("✅ Import erfolgreich")
    
    # Teste Initialisierung
    analyzer = Qwen2VLUltraDetailedAnalyzer()
    print("✅ Analyzer initialisiert")
    
    # Teste mit kurzem Video
    import os
    test_videos = [f for f in os.listdir('/home/user/tiktok_videos/videos/') if f.endswith('.mp4')]
    if test_videos:
        test_video = f'/home/user/tiktok_videos/videos/{test_videos[0]}'
        print(f"Teste mit: {test_video}")
        
        result = analyzer.analyze(test_video)
        segments = result.get('segments', [])
        print(f"✅ Analyse erfolgreich\! {len(segments)} Segmente")
        if segments:
            print(f"Erste Beschreibung: {segments[0]['description'][:100]}...")
except Exception as e:
    print(f"❌ FEHLER: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
EOF < /dev/null
