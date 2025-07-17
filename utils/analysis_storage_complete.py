#!/usr/bin/env python3
"""
Complete Analysis Storage System - Speichert ALLE Analysedaten mit Video URL
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import shutil
from pathlib import Path


class CompleteAnalysisStorage:
    """Speichert und verwaltet vollständige Analysen mit allen Daten"""
    
    def __init__(self, base_dir: str = "/home/user/tiktok_production/complete_analyses"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Unterordner für verschiedene Formate
        self.json_dir = self.base_dir / "json"
        self.formatted_dir = self.base_dir / "formatted"
        self.video_dir = self.base_dir / "videos"
        
        for dir in [self.json_dir, self.formatted_dir, self.video_dir]:
            dir.mkdir(exist_ok=True)
    
    def save_complete_analysis(self, 
                             video_path: str,
                             tiktok_url: str,
                             analysis_data: Dict[str, Any],
                             copy_video: bool = True) -> Dict[str, str]:
        """
        Speichert komplette Analyse mit allen Daten
        
        Returns:
            Dict mit Pfaden zu gespeicherten Dateien
        """
        # Generiere eindeutige ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_id = Path(video_path).stem
        analysis_id = f"{video_id}_{timestamp}"
        
        # Erstelle Hauptdatenstruktur
        complete_data = {
            "analysis_id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "video_info": {
                "original_path": video_path,
                "tiktok_url": tiktok_url,
                "video_id": video_id,
                "filename": Path(video_path).name
            },
            "analysis_summary": self._create_summary(analysis_data),
            "all_analyzer_data": analysis_data,
            "metadata": {
                "total_analyzers": len(analysis_data),
                "successful_analyzers": self._count_successful(analysis_data),
                "processing_time": analysis_data.get("processing_time", "unknown"),
                "video_duration": self._get_video_duration(analysis_data),
                "total_data_points": self._count_data_points(analysis_data)
            }
        }
        
        # Konvertiere numpy/torch Typen zu Python Typen
        complete_data = self._convert_to_serializable(complete_data)
        
        # Speichere JSON
        json_path = self.json_dir / f"{analysis_id}_complete.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(complete_data, f, indent=2, ensure_ascii=False)
        
        # Erstelle formatierte Übersicht
        formatted_path = self._create_formatted_overview(complete_data, analysis_id)
        
        # Kopiere Video wenn gewünscht
        video_copy_path = None
        if copy_video and os.path.exists(video_path):
            video_copy_path = self.video_dir / f"{analysis_id}{Path(video_path).suffix}"
            shutil.copy2(video_path, video_copy_path)
        
        # Erstelle Index-Eintrag
        self._update_index(analysis_id, tiktok_url, complete_data)
        
        return {
            "analysis_id": analysis_id,
            "json_path": str(json_path),
            "formatted_path": str(formatted_path),
            "video_copy_path": str(video_copy_path) if video_copy_path else None,
            "index_updated": True
        }
    
    def _create_summary(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Erstellt Zusammenfassung aller wichtigen Erkenntnisse"""
        summary = {
            "content_description": [],
            "detected_objects": {},
            "people_count": 0,
            "emotions": [],
            "text_overlays": [],
            "speech_content": [],
            "key_moments": [],
            "technical_info": {}
        }
        
        # BLIP2 Beschreibungen
        if "blip2_video_analyzer" in analysis_data:
            blip2 = analysis_data["blip2_video_analyzer"]
            if "segments" in blip2:
                # Erste und wichtigste Beschreibungen
                for seg in blip2["segments"][:5]:
                    summary["content_description"].append({
                        "time": seg.get("timestamp", 0),
                        "description": seg.get("caption", "")
                    })
        
        # Objekt-Erkennung
        if "object_detection" in analysis_data:
            objects = analysis_data["object_detection"]
            if "segments" in objects:
                object_counts = {}
                for seg in objects["segments"]:
                    obj_type = seg.get("object", "unknown")
                    if obj_type not in object_counts:
                        object_counts[obj_type] = 0
                    object_counts[obj_type] += 1
                    if obj_type == "person":
                        summary["people_count"] = max(summary["people_count"], 
                                                     object_counts["person"])
                summary["detected_objects"] = object_counts
        
        # Emotionen
        if "emotion_detection" in analysis_data:
            emotions = analysis_data["emotion_detection"]
            if "segments" in emotions:
                emotion_summary = {}
                for seg in emotions["segments"]:
                    emotion = seg.get("dominant_emotion", "unknown")
                    if emotion not in emotion_summary:
                        emotion_summary[emotion] = 0
                    emotion_summary[emotion] += 1
                summary["emotions"] = [
                    {"emotion": k, "count": v} 
                    for k, v in sorted(emotion_summary.items(), 
                                     key=lambda x: x[1], reverse=True)
                ]
        
        # Speech/Transcription
        if "speech_transcription" in analysis_data:
            speech = analysis_data["speech_transcription"]
            if "segments" in speech:
                for seg in speech["segments"][:10]:  # Erste 10 Segmente
                    if seg.get("text"):
                        summary["speech_content"].append({
                            "time": seg.get("start_time", 0),
                            "text": seg.get("text", ""),
                            "language": seg.get("language", "unknown")
                        })
        
        # Text Overlays
        if "text_overlay_detection" in analysis_data:
            text_overlays = analysis_data["text_overlay_detection"]
            if "segments" in text_overlays:
                for seg in text_overlays["segments"][:5]:
                    if seg.get("text"):
                        summary["text_overlays"].append({
                            "time": seg.get("timestamp", 0),
                            "text": seg.get("text", "")
                        })
        
        return summary
    
    def _create_formatted_overview(self, data: Dict[str, Any], analysis_id: str) -> str:
        """Erstellt formatierte Übersichtsdatei"""
        formatted_path = self.formatted_dir / f"{analysis_id}_overview.txt"
        
        with open(formatted_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"VOLLSTÄNDIGE TIKTOK VIDEO ANALYSE\n")
            f.write("="*80 + "\n\n")
            
            # Video Info
            f.write("VIDEO INFORMATIONEN:\n")
            f.write("-"*40 + "\n")
            f.write(f"TikTok URL: {data['video_info']['tiktok_url']}\n")
            f.write(f"Video ID: {data['video_info']['video_id']}\n")
            f.write(f"Analyse ID: {data['analysis_id']}\n")
            f.write(f"Zeitstempel: {data['timestamp']}\n")
            f.write(f"Dateiname: {data['video_info']['filename']}\n\n")
            
            # Metadata
            f.write("ANALYSE ÜBERSICHT:\n")
            f.write("-"*40 + "\n")
            meta = data['metadata']
            f.write(f"Analyzer verwendet: {meta['successful_analyzers']}/{meta['total_analyzers']}\n")
            f.write(f"Verarbeitungszeit: {meta['processing_time']}\n")
            f.write(f"Video Dauer: {meta.get('video_duration', 'unbekannt')}\n")
            f.write(f"Datenpunkte gesamt: {meta['total_data_points']:,}\n\n")
            
            # Zusammenfassung
            summary = data['analysis_summary']
            
            f.write("INHALTSBESCHREIBUNG:\n")
            f.write("-"*40 + "\n")
            for desc in summary['content_description']:
                f.write(f"[{desc['time']:.1f}s] {desc['description']}\n")
            f.write("\n")
            
            # Erkannte Objekte
            if summary['detected_objects']:
                f.write("ERKANNTE OBJEKTE:\n")
                f.write("-"*40 + "\n")
                for obj, count in sorted(summary['detected_objects'].items(), 
                                       key=lambda x: x[1], reverse=True):
                    f.write(f"- {obj}: {count}x erkannt\n")
                f.write(f"\nPersonen im Video: {summary['people_count']}\n\n")
            
            # Emotionen
            if summary['emotions']:
                f.write("ERKANNTE EMOTIONEN:\n")
                f.write("-"*40 + "\n")
                for emotion_data in summary['emotions']:
                    f.write(f"- {emotion_data['emotion']}: {emotion_data['count']}x\n")
                f.write("\n")
            
            # Gesprochener Text
            if summary['speech_content']:
                f.write("GESPROCHENER INHALT:\n")
                f.write("-"*40 + "\n")
                for speech in summary['speech_content']:
                    f.write(f"[{speech['time']:.1f}s] \"{speech['text']}\" ({speech['language']})\n")
                f.write("\n")
            
            # Text Overlays
            if summary['text_overlays']:
                f.write("TEXT OVERLAYS:\n")
                f.write("-"*40 + "\n")
                for text in summary['text_overlays']:
                    f.write(f"[{text['time']:.1f}s] \"{text['text']}\"\n")
                f.write("\n")
            
            # Detaillierte Analyzer Ergebnisse
            f.write("\n" + "="*80 + "\n")
            f.write("DETAILLIERTE ANALYZER ERGEBNISSE:\n")
            f.write("="*80 + "\n\n")
            
            # Für jeden Analyzer
            for analyzer_name, analyzer_data in data['all_analyzer_data'].items():
                if analyzer_name in ['processing_time', 'timestamp', 'video_path']:
                    continue
                    
                f.write(f"\n{analyzer_name.upper()}:\n")
                f.write("-"*60 + "\n")
                
                if isinstance(analyzer_data, dict):
                    if "segments" in analyzer_data:
                        f.write(f"Segmente gefunden: {len(analyzer_data['segments'])}\n")
                        # Zeige erste 3 Segmente als Beispiel
                        for i, seg in enumerate(analyzer_data['segments'][:3]):
                            f.write(f"\nSegment {i+1}:\n")
                            self._write_segment_details(f, seg, analyzer_name)
                        if len(analyzer_data['segments']) > 3:
                            f.write(f"\n... und {len(analyzer_data['segments']) - 3} weitere Segmente\n")
                    else:
                        # Zeige andere Daten
                        for key, value in analyzer_data.items():
                            if key != "metadata":
                                f.write(f"  {key}: {value}\n")
                
                f.write("\n")
        
        return str(formatted_path)
    
    def _write_segment_details(self, f, segment: Dict, analyzer_name: str):
        """Schreibt Segment-Details basierend auf Analyzer-Typ"""
        if analyzer_name == "object_detection":
            f.write(f"  - Objekt: {segment.get('object', 'unknown')}\n")
            f.write(f"  - Zeit: {segment.get('timestamp', 0):.2f}s\n")
            f.write(f"  - Konfidenz: {segment.get('confidence', 0):.2%}\n")
            if 'bbox' in segment:
                f.write(f"  - Position: {segment['bbox']}\n")
        
        elif analyzer_name == "speech_transcription":
            f.write(f"  - Text: \"{segment.get('text', '')}\"\n")
            f.write(f"  - Zeit: {segment.get('start_time', 0):.2f}s - {segment.get('end_time', 0):.2f}s\n")
            f.write(f"  - Sprache: {segment.get('language', 'unknown')}\n")
        
        elif analyzer_name == "emotion_detection":
            f.write(f"  - Emotion: {segment.get('dominant_emotion', 'unknown')}\n")
            f.write(f"  - Zeit: {segment.get('timestamp', 0):.2f}s\n")
            f.write(f"  - Konfidenz: {segment.get('confidence', 0):.2%}\n")
        
        elif "blip2" in analyzer_name or "vid2seq" in analyzer_name:
            f.write(f"  - Beschreibung: {segment.get('caption', '')}\n")
            f.write(f"  - Zeit: {segment.get('timestamp', 0):.2f}s\n")
        
        else:
            # Generisches Format für andere Analyzer
            for key, value in segment.items():
                if key not in ['metadata', 'batch_processed', 'gpu_optimized']:
                    f.write(f"  - {key}: {value}\n")
    
    def _update_index(self, analysis_id: str, tiktok_url: str, data: Dict):
        """Aktualisiert den Hauptindex aller Analysen"""
        index_path = self.base_dir / "index.json"
        
        # Lade existierenden Index
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = {"analyses": []}
        
        # Füge neue Analyse hinzu
        index["analyses"].insert(0, {
            "analysis_id": analysis_id,
            "tiktok_url": tiktok_url,
            "timestamp": data['timestamp'],
            "video_id": data['video_info']['video_id'],
            "summary": {
                "people_count": data['analysis_summary']['people_count'],
                "object_types": len(data['analysis_summary']['detected_objects']),
                "has_speech": len(data['analysis_summary']['speech_content']) > 0,
                "emotion_count": len(data['analysis_summary']['emotions']),
                "analyzers_used": data['metadata']['successful_analyzers']
            }
        })
        
        # Behalte nur die letzten 1000 Einträge
        index["analyses"] = index["analyses"][:1000]
        
        # Speichere aktualisierten Index
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
    
    def _count_successful(self, analysis_data: Dict) -> int:
        """Zählt erfolgreiche Analyzer"""
        count = 0
        for key, value in analysis_data.items():
            if key not in ['processing_time', 'timestamp', 'video_path']:
                if isinstance(value, dict) and (
                    'segments' in value or 
                    'results' in value or 
                    len(value) > 0
                ):
                    count += 1
        return count
    
    def _get_video_duration(self, analysis_data: Dict) -> str:
        """Ermittelt Video-Dauer aus Analyse-Daten"""
        max_time = 0
        
        # Suche maximale Zeit in allen Segmenten
        for analyzer_name, analyzer_data in analysis_data.items():
            if isinstance(analyzer_data, dict) and 'segments' in analyzer_data:
                for segment in analyzer_data['segments']:
                    # Verschiedene Zeit-Felder prüfen
                    for time_field in ['timestamp', 'end_time', 'time']:
                        if time_field in segment:
                            try:
                                time_val = float(segment[time_field])
                                max_time = max(max_time, time_val)
                            except:
                                pass
        
        if max_time > 0:
            minutes = int(max_time // 60)
            seconds = int(max_time % 60)
            return f"{minutes}:{seconds:02d}"
        return "unbekannt"
    
    def _count_data_points(self, analysis_data: Dict) -> int:
        """Zählt alle Datenpunkte in der Analyse"""
        count = 0
        
        for analyzer_name, analyzer_data in analysis_data.items():
            if isinstance(analyzer_data, dict):
                if 'segments' in analyzer_data:
                    count += len(analyzer_data['segments'])
                else:
                    # Zähle andere Datenpunkte
                    count += len([k for k in analyzer_data.keys() 
                                 if k not in ['metadata', 'processing_time']])
        
        return count
    
    def load_analysis(self, analysis_id: str) -> Optional[Dict]:
        """Lädt eine gespeicherte Analyse"""
        json_path = self.json_dir / f"{analysis_id}_complete.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                return json.load(f)
        return None
    
    def search_by_url(self, tiktok_url: str) -> list:
        """Sucht Analysen nach TikTok URL"""
        index_path = self.base_dir / "index.json"
        if not index_path.exists():
            return []
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        results = []
        for entry in index['analyses']:
            if entry['tiktok_url'] == tiktok_url:
                results.append(entry)
        
        return results
    
    def _convert_to_serializable(self, obj):
        """Konvertiert numpy/torch Typen zu serialisierbaren Python-Typen"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(v) for v in obj)
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Für torch tensors
            return obj.item()
        elif hasattr(obj, 'tolist'):  # Für andere array-ähnliche Objekte
            return obj.tolist()
        else:
            return obj


# CLI Interface
if __name__ == "__main__":
    import sys
    
    storage = CompleteAnalysisStorage()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python analysis_storage_complete.py save <video_path> <tiktok_url> <analysis_json>")
        print("  python analysis_storage_complete.py load <analysis_id>")
        print("  python analysis_storage_complete.py search <tiktok_url>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "save" and len(sys.argv) >= 5:
        video_path = sys.argv[2]
        tiktok_url = sys.argv[3]
        analysis_json_path = sys.argv[4]
        
        # Lade Analyse-Daten
        with open(analysis_json_path, 'r') as f:
            analysis_data = json.load(f)
        
        # Speichere vollständige Analyse
        result = storage.save_complete_analysis(
            video_path=video_path,
            tiktok_url=tiktok_url,
            analysis_data=analysis_data
        )
        
        print(f"Analyse gespeichert!")
        print(f"ID: {result['analysis_id']}")
        print(f"JSON: {result['json_path']}")
        print(f"Übersicht: {result['formatted_path']}")
        
    elif command == "load" and len(sys.argv) >= 3:
        analysis_id = sys.argv[2]
        data = storage.load_analysis(analysis_id)
        if data:
            print(json.dumps(data, indent=2))
        else:
            print(f"Analyse {analysis_id} nicht gefunden")
    
    elif command == "search" and len(sys.argv) >= 3:
        tiktok_url = sys.argv[2]
        results = storage.search_by_url(tiktok_url)
        print(f"Gefundene Analysen für {tiktok_url}:")
        for r in results:
            print(f"- {r['analysis_id']} ({r['timestamp']})")