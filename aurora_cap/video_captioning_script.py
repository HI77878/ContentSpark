# video_captioning_script.py (für AuroraCap)
import torch
import cv2
import numpy as np
import sys
import os
import json
from typing import List, Dict, Tuple
import traceback
from pathlib import Path

# Stelle sicher, dass das AuroraCap-Paket gefunden wird
try:
    from aurora import AuroraCap
except ImportError:
    print("Fehler: AuroraCap Paket nicht gefunden. Stelle sicher, dass es korrekt installiert ist.")
    sys.exit(1)

# --- Konfiguration ---
# Standardpfad für das Video im Container, wenn kein Argument übergeben wird
# Dieses wird durch das Mounten von Volumes überschrieben.
DEFAULT_VIDEO_PATH_IN_CONTAINER = "/app/videos/input.mp4"
OUTPUT_DIR_IN_CONTAINER = "/app/output/"
DEFAULT_FRAME_RATE = 1 # Beschreiben pro Sekunde

# Detaillierter Prompt für sekundengenaue Beschreibungen mit AuroraCap
# AuroraCap wurde auf Benchmarks mit detaillierten Prompts trainiert.
AURORA_DETAILED_PROMPT = """
Describe this video frame by frame in extreme detail. For each frame:
- Identify all visible objects and their precise locations (e.g., 'a red cup on the left side of the table').
- Describe all actions occurring, including movements and interactions.
- Capture any recognizable emotions or moods of people.
- Note down visual effects, filters, or text overlays.
- Provide context and background information about the scene.
Be factual, specific, and avoid speculation.
"""

# --- Hilfsfunktionen ---
def load_video_frames(video_path: str, frame_rate: int = 1) -> Tuple[List[np.ndarray], List[float]]:
    """
    Lädt ein Video und extrahiert Frames mit einer bestimmten Rate (Frames pro Sekunde).
    Gibt eine Liste von Frames (als NumPy-Arrays, RGB) und zugehörige Zeitstempel zurück.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Fehler: Konnte Video nicht öffnen unter {video_path}")
        return None, None

    frames = []
    timestamps = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Warnung: FPS konnte nicht ermittelt werden, verwende Standard 30.")
        fps = 30

    frame_interval = int(fps / frame_rate)
    if frame_interval == 0: frame_interval = 1 # Stelle sicher, dass mindestens ein Frame pro Sekunde genommen wird

    frame_count = 0

    print(f"Verarbeite Video: {video_path} mit FPS {fps:.2f}. Extrahiere jeden {frame_interval}-ten Frame.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Konvertiere von BGR zu RGB und von NumPy zu Tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            timestamps.append(frame_count / fps)
        frame_count += 1

    cap.release()
    print(f"Video geladen: {len(frames)} Frames extrahiert.")
    return frames, timestamps

# --- Hauptlogik ---
def main():
    # Ermittle den Pfad zum zu analysierenden Video
    # Nimmt den Pfad als Kommandozeilenargument, ansonsten nutze den Standardpfad
    video_path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    video_path = video_path_arg if video_path_arg else DEFAULT_VIDEO_PATH_IN_CONTAINER

    if not os.path.exists(video_path):
        print(f"Fehler: Video-Datei nicht gefunden unter '{video_path}'.")
        print(f"Verwendung: python {os.path.basename(__file__)} [/app/videos/your_video.mp4]")
        sys.exit(1)

    # Lade das AuroraCap Modell
    print("Lade AuroraCap Modell ('rese1f/aurora-7b')...")
    try:
        # Überprüfe, ob CUDA verfügbar ist und verwende die GPU.
        # Wenn nicht, falle auf CPU zurück, was aber extrem langsam wäre.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Verwende Gerät: {device}")

        # Lade das AuroraCap Modell.
        # 'device_map="auto"' ist gut, wenn Sie mehrere GPUs haben oder das Modell nicht komplett in den VRAM passt.
        # Für eine einzelne RTX 8000 sollte das Laden auf 'cuda' oder 'auto' gut funktionieren.
        model = AuroraCap.from_pretrained("rese1f/aurora-7b", device_map=device)
        print("AuroraCap Modell erfolgreich geladen.")
    except Exception as e:
        print(f"Fehler beim Laden des AuroraCap Modells: {e}")
        print(traceback.format_exc())
        sys.exit(1)

    # Lade die Frames des Videos
    frames, timestamps = load_video_frames(video_path, frame_rate=DEFAULT_FRAME_RATE)
    if frames is None or not frames:
        print("Fehler beim Laden des Videos oder keine Frames extrahiert.")
        sys.exit(1)

    # Generiere Beschreibungen für jeden Frame
    print(f"Generiere Beschreibungen für {len(frames)} Frames mit AuroraCap...")
    descriptions_data = []
    total_frames_processed = 0
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record() # Start timing

    for i, frame_np in enumerate(frames):
        try:
            # Konvertiere NumPy-Frame zu PyTorch-Tensor und verschiebe auf das GPU-Gerät
            # AuroraCap erwartet RGB Bilder.
            frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).float().to(device)

            # Generiere die Caption mit dem detaillierten Prompt
            # Siehe AuroraCap GitHub für empfohlene Generation-Parameter wie max_new_tokens, num_beams etc.
            caption = model.generate(frame_tensor, prompt=AURORA_DETAILED_PROMPT, max_new_tokens=250, num_beams=3, temperature=0.7, do_sample=False)
            description = caption.strip()

            # Stelle sicher, dass wir eine sinnvolle Beschreibung erhalten
            if not description or description == AURORA_DETAILED_PROMPT.split('\n')[0]: # Einfache Prüfung auf leere oder Prompt-ähnliche Antworten
                description = f"Frame at {timestamps[i]:.1f}s showing video content." # Fallback-Beschreibung

            descriptions_data.append({
                "timestamp": timestamps[i],
                "description": description
            })
            total_frames_processed += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(frames): # Log alle 10 Frames oder am Ende
                print(f"  - Verarbeitet: {i+1}/{len(frames)} Frames (Zeit: {timestamps[i]:.2f}s)")

        except Exception as e:
            print(f"Fehler bei der Verarbeitung von Frame {i+1} (Zeit: {timestamps[i]:.2f}s): {e}")
            print(traceback.format_exc())
            descriptions_data.append({
                "timestamp": timestamps[i],
                "description": f"ERROR: {e}"
            })

    end_time.record() # End timing
    torch.cuda.synchronize() # Warte auf den Abschluss aller CUDA Operationen

    analysis_time = start_time.elapsed_time(end_time) / 1000.0 # Zeit in Sekunden

    # Erstelle den Ausgabepfad und speichere die Ergebnisse
    base_video_name = Path(video_path).stem # Dateiname ohne Endung
    output_filename = os.path.join(OUTPUT_DIR_IN_CONTAINER, f"{base_video_name}_aurora_descriptions.json")

    result_data = {
        "video_path_analyzed": video_path,
        "model_used": "AuroraCap (rese1f/aurora-7b)",
        "total_frames_analyzed": total_frames_processed,
        "analysis_time_seconds": round(analysis_time, 2),
        "segments": descriptions_data
    }

    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ AuroraCap Analyse abgeschlossen. Ergebnisse gespeichert in: {output_filename}")
        print(f"   Gesamtanalysezeit: {analysis_time:.2f} Sekunden für {total_frames_processed} Frames.")
    except Exception as e:
        print(f"Fehler beim Speichern der Ergebnisse in '{output_filename}': {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()