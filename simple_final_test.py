#!/usr/bin/env python3
"""
Simple final test using existing infrastructure
"""
import os
import sys
import subprocess
import time
import json
from datetime import datetime

def main():
    print("=== FINALE SYSTEM-TESTUNG ===")
    print(f"Zeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Video details
    video_path = "/home/user/tiktok_production/downloads/videos/7522589683939921165.mp4"
    video_duration = 68  # seconds
    
    if not os.path.exists(video_path):
        print(f"❌ Video nicht gefunden: {video_path}")
        return
        
    print(f"\n📹 Video-Details:")
    print(f"  Pfad: {video_path}")
    print(f"  Größe: {os.path.getsize(video_path) / 1024**2:.1f} MB")
    print(f"  Dauer: {video_duration} Sekunden")
    
    # Check if we can use the simpler scripts
    analyze_script = None
    for script in ["analyze_video.py", "run_analysis.py", "process_video.py"]:
        if os.path.exists(script):
            analyze_script = script
            break
    
    if not analyze_script:
        print("\n⚠️  Kein einfaches Analyse-Skript gefunden")
        print("Versuche direkte Komponenten zu nutzen...")
        
        # Try to list what's available
        print("\nVerfügbare Python-Dateien im Root:")
        py_files = [f for f in os.listdir('.') if f.endswith('.py')]
        for f in sorted(py_files)[:10]:
            print(f"  - {f}")
            
        # Check for batch processor
        if os.path.exists("batch_processor.py"):
            print("\n✅ batch_processor.py gefunden - verwende diesen")
            
            # Create a simple batch file
            batch_file = "test_batch.txt"
            with open(batch_file, 'w') as f:
                f.write(f"{video_path}\n")
            
            print(f"Erstelle Batch-Datei: {batch_file}")
            
            # Run batch processor
            print("\nStarte Batch-Verarbeitung...")
            start_time = time.time()
            
            try:
                # First check what batch_processor expects
                result = subprocess.run(
                    ["python3", "batch_processor.py", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                print("Batch processor help:")
                print(result.stdout or result.stderr)
                
                # Try running it
                result = subprocess.run(
                    ["python3", "batch_processor.py", batch_file],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                print(f"\n⏱️  Verarbeitungszeit: {processing_time:.1f} Sekunden")
                print(f"📊 Realtime-Faktor: {processing_time/video_duration:.2f}x")
                
                if result.returncode == 0:
                    print("✅ Batch-Verarbeitung erfolgreich!")
                    # Look for results
                    results_files = sorted([f for f in os.listdir('results') if f.endswith('.json')])
                    if results_files:
                        latest_result = results_files[-1]
                        print(f"\nNeueste Ergebnisdatei: results/{latest_result}")
                        
                        # Load and check results
                        with open(f'results/{latest_result}', 'r') as f:
                            data = json.load(f)
                            
                        if 'analyzer_results' in data:
                            analyzer_count = len(data['analyzer_results'])
                            print(f"Analyzer-Ergebnisse: {analyzer_count}")
                            
                            # Count successful
                            successful = 0
                            for name, result in data['analyzer_results'].items():
                                if isinstance(result, dict) and 'error' not in result:
                                    successful += 1
                                    
                            print(f"Erfolgreiche Analyzer: {successful}/{analyzer_count}")
                            print(f"Rekonstruktions-Score: {(successful/analyzer_count*100):.1f}%")
                else:
                    print(f"❌ Fehler: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print("❌ Timeout nach 5 Minuten")
            except Exception as e:
                print(f"❌ Fehler: {e}")
                
            finally:
                # Cleanup
                if os.path.exists(batch_file):
                    os.remove(batch_file)
        else:
            print("\n❌ Keine geeignete Analyse-Methode gefunden")
            
    else:
        print(f"\n✅ Verwende {analyze_script}")
        # Run the analysis script
        subprocess.run(["python3", analyze_script, video_path])

if __name__ == "__main__":
    main()