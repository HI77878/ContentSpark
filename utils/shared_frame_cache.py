#!/usr/bin/env python3
"""
Zentraler Frame Cache für alle Analyzer
Löst das Hauptproblem: 14x Frame-Extraction
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import threading
from collections import defaultdict
import time

class SharedFrameCache:
    """
    Zentrale Frame-Extraction für ALLE Analyzer.
    Einmal extrahieren, alle nutzen!
    """
    
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()
        self.extraction_configs = {
            'dense': {'fps': 1.0, 'max_frames': 300},      # video_llava, object_detection
            'medium': {'fps': 0.5, 'max_frames': 150},     # face, body, emotion
            'sparse': {'fps': 0.2, 'max_frames': 60},      # scene changes, cuts
            'text': {'fps': 3.0, 'max_frames': 300},       # text_overlay, speech
            'all': {'fps': None, 'max_frames': None}       # all frames (für special cases)
        }
        self.stats = defaultdict(int)
    
    def get_frames(self, video_path: str, config_name: str = 'dense') -> Tuple[List[np.ndarray], List[float]]:
        """
        Hole Frames aus Cache oder extrahiere sie einmal.
        
        Args:
            video_path: Pfad zum Video
            config_name: Welche Frame-Konfiguration ('dense', 'medium', 'sparse', 'text')
            
        Returns:
            frames: Liste von numpy arrays
            timestamps: Liste von Zeitstempeln
        """
        cache_key = f"{video_path}_{config_name}"
        
        # Prüfe Cache
        with self.lock:
            if cache_key in self.cache:
                self.stats['cache_hits'] += 1
                print(f"✅ Frame Cache HIT für {config_name}")
                return self.cache[cache_key]
        
        # Extrahiere Frames EINMAL
        self.stats['cache_misses'] += 1
        print(f"🎬 Extrahiere Frames für {config_name} (EINMALIG)")
        
        frames, timestamps = self._extract_frames(video_path, self.extraction_configs[config_name])
        
        # Cache speichern
        with self.lock:
            self.cache[cache_key] = (frames, timestamps)
        
        return frames, timestamps
    
    def _extract_frames(self, video_path: str, config: dict) -> Tuple[List[np.ndarray], List[float]]:
        """
        Eigentliche Frame-Extraction - nur EINMAL pro Video/Config!
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Kann Video nicht öffnen: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Berechne Frame-Intervalle
        if config['fps'] is None:
            # Alle Frames
            frame_indices = list(range(total_frames))
        else:
            # Sample mit gewünschter FPS
            target_fps = config['fps']
            frame_interval = int(fps / target_fps)
            frame_indices = list(range(0, total_frames, frame_interval))
            
            # Limitiere auf max_frames
            if config['max_frames'] and len(frame_indices) > config['max_frames']:
                frame_indices = frame_indices[:config['max_frames']]
        
        frames = []
        timestamps = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frames.append(frame)
                timestamps.append(idx / fps)
        
        cap.release()
        
        print(f"📊 Extrahiert: {len(frames)} frames")
        return frames, timestamps
    
    def preload_all_configs(self, video_path: str):
        """
        Lade alle Frame-Konfigurationen vorab (optional für Performance).
        """
        print(f"🔄 Preloading alle Frame-Sets für {video_path}")
        
        for config_name in ['dense', 'medium', 'sparse', 'text']:
            self.get_frames(video_path, config_name)
    
    def clear_cache(self, video_path: Optional[str] = None):
        """
        Cache leeren - entweder für ein Video oder komplett.
        """
        with self.lock:
            if video_path:
                # Lösche alle Configs für dieses Video
                keys_to_delete = [k for k in self.cache.keys() if k.startswith(video_path)]
                for key in keys_to_delete:
                    del self.cache[key]
                print(f"🧹 Cache für {video_path} geleert")
            else:
                # Kompletten Cache leeren
                self.cache.clear()
                print("🧹 Kompletter Cache geleert")
    
    def get_stats(self) -> dict:
        """
        Cache-Statistiken für Debugging.
        """
        total = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total * 100 if total > 0 else 0
        
        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': f"{hit_rate:.1f}%",
            'cached_videos': len(set(k.split('_')[0] for k in self.cache.keys()))
        }

# Globale Instanz für alle Analyzer
FRAME_CACHE = SharedFrameCache()

# Singleton pattern für konsistente Instanz
@staticmethod
def get_instance():
    """Get singleton instance of SharedFrameCache"""
    global FRAME_CACHE
    if FRAME_CACHE is None:
        FRAME_CACHE = SharedFrameCache()
    return FRAME_CACHE

# Add get_instance method to class
SharedFrameCache.get_instance = get_instance

# Erweiterte get_frames Methode für Kompatibilität
def get_frames_compat(self, video_path: str, sample_rate: int = 30, max_frames: int = 100) -> Tuple[List[np.ndarray], List[float]]:
    """
    Kompatibilitäts-Methode für bestehende Analyzer.
    
    Args:
        video_path: Pfad zum Video
        sample_rate: Frame-Intervall (30 = jedes 30. Frame)
        max_frames: Maximale Anzahl Frames
        
    Returns:
        frames, timestamps
    """
    # Konvertiere sample_rate zu fps
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    target_fps = video_fps / sample_rate if sample_rate > 0 else 1.0
    
    # Wähle passende Config
    if target_fps >= 3.0:
        config_name = 'text'
    elif target_fps >= 1.0:
        config_name = 'dense'
    elif target_fps >= 0.5:
        config_name = 'medium'
    else:
        config_name = 'sparse'
    
    # Hole Frames mit Cache
    frames, timestamps = self.get_frames(video_path, config_name)
    
    # Limitiere auf max_frames
    if len(frames) > max_frames:
        step = len(frames) // max_frames
        frames = frames[::step][:max_frames]
        timestamps = timestamps[::step][:max_frames]
    
    return frames, timestamps

# Füge Methode zur Klasse hinzu
SharedFrameCache.get_frames_compat = get_frames_compat

# Beispiel-Verwendung in Analyzern:
"""
from shared_frame_cache import FRAME_CACHE

class ImprovedAnalyzer:
    def analyze(self, video_path):
        # Statt eigenes cv2.VideoCapture:
        frames, timestamps = FRAME_CACHE.get_frames(video_path, 'dense')
        
        # Oder für Kompatibilität:
        frames, timestamps = FRAME_CACHE.get_frames_compat(video_path, sample_rate=30, max_frames=100)
        
        # Jetzt normal weiterarbeiten mit frames
        results = self.process_frames(frames, timestamps)
        return results
"""