#!/usr/bin/env python3
"""
EINFACHER VIDEO ANALYSE WORKFLOW
- TikTok URL eingeben
- Video downloaden
- Analysieren
- Ergebnisse anzeigen
"""

import requests
import json
import time
import sys
from datetime import datetime

def analyze_tiktok_video(tiktok_url, creator_username=None):
    """Analysiere TikTok Video mit der neuen API"""
    
    print(f"🎬 STARTE VIDEO ANALYSE")
    print(f"━" * 60)
    print(f"TikTok URL: {tiktok_url}")
    print(f"Creator: {creator_username or 'Auto-detect'}")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # API Request
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8003/analyze",
            json={
                "tiktok_url": tiktok_url,
                "creator_username": creator_username
            },
            timeout=600  # 10 Minuten timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            elapsed = time.time() - start_time
            
            print(f"✅ ANALYSE ERFOLGREICH!")
            print(f"━" * 60)
            print(f"⏱️  Zeit: {elapsed:.1f}s")
            print(f"📁 Ergebnisse: {result['results_file']}")
            print(f"📊 Erfolgreiche Analyzer: {result['successful_analyzers']}/{result['total_analyzers']}")
            print()
            
            # Lade und zeige Ergebnisse
            with open(result['results_file'], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Qwen2-VL Temporal Ergebnisse prüfen
            qwen = data['analyzer_results'].get('qwen2_vl_temporal', {})
            
            if 'segments' in qwen and qwen['segments']:
                print(f"🎯 QWEN2-VL TEMPORAL BESCHREIBUNGEN:")
                print(f"━" * 60)
                for seg in qwen['segments'][:5]:  # Zeige erste 5
                    print(f"⏱️  [{seg['start_time']:.1f}s - {seg['end_time']:.1f}s]")
                    print(f"📝 {seg['description']}")
                    print()
                
                if len(qwen['segments']) > 5:
                    print(f"... und {len(qwen['segments']) - 5} weitere Segmente")
                
                print(f"✅ QWEN2-VL FUNKTIONIERT PERFEKT!")
                print(f"   Segmente: {len(qwen['segments'])}")
                print(f"   Video-Dauer: {data['metadata'].get('processing_time_seconds', 'N/A')}s")
                
            else:
                error = qwen.get('error', 'Unbekannter Fehler')
                print(f"❌ QWEN2-VL FEHLER: {error}")
            
            print()
            print(f"📈 ALLE ANALYZER ERGEBNISSE:")
            print(f"━" * 60)
            
            for analyzer, result_data in data['analyzer_results'].items():
                if 'segments' in result_data and result_data['segments']:
                    segments = len(result_data['segments'])
                    print(f"✅ {analyzer:<25} {segments:>4} Segmente")
                elif 'error' in result_data:
                    print(f"❌ {analyzer:<25} FEHLER: {result_data['error'][:40]}...")
                else:
                    print(f"⚠️ {analyzer:<25} Keine Daten")
            
            return True
            
        else:
            print(f"❌ API FEHLER: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ FEHLER: {e}")
        return False

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_video.py <tiktok_url> [creator_username]")
        print("Example: python analyze_video.py https://www.tiktok.com/@user/video/123 user")
        sys.exit(1)
    
    tiktok_url = sys.argv[1]
    creator_username = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Extrahiere Creator aus URL falls nicht angegeben
    if not creator_username and "tiktok.com/@" in tiktok_url:
        try:
            creator_username = tiktok_url.split("/@")[1].split("/")[0]
        except:
            pass
    
    analyze_tiktok_video(tiktok_url, creator_username)

if __name__ == "__main__":
    main()