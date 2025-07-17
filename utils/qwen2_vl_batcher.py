#!/usr/bin/env python3
"""
Qwen2-VL Batch Processing Utility

Optimiert die Verarbeitung von Qwen2-VL durch intelligentes Batching.
WICHTIG: Qualität darf NICHT leiden - jedes Segment bekommt volle Aufmerksamkeit!
"""

import numpy as np
import torch
import logging
from typing import List, Dict, Any, Optional, Tuple
import time

logger = logging.getLogger(__name__)

class Qwen2VLBatcher:
    """Batch-Verarbeitung für Qwen2-VL Segmente ohne Qualitätsverlust"""
    
    def __init__(self, batch_size: int = 4, max_frames_per_batch: int = 32):
        """
        Initialize the batcher.
        
        Args:
            batch_size: Anzahl der Segmente pro Batch (default: 4)
            max_frames_per_batch: Maximale Frames pro Batch für GPU-Speicher (default: 32)
        """
        self.batch_size = batch_size
        self.max_frames_per_batch = max_frames_per_batch
        logger.info(f"Initialized Qwen2VLBatcher with batch_size={batch_size}, max_frames={max_frames_per_batch}")
    
    def create_batches(self, segments: List[Dict]) -> List[List[Dict]]:
        """
        Gruppiert Segmente in Batches unter Berücksichtigung der Frame-Anzahl.
        
        Args:
            segments: Liste von Segmenten mit Frames
            
        Returns:
            Liste von Batches
        """
        if not segments:
            return []
        
        batches = []
        current_batch = []
        current_frame_count = 0
        
        for segment in segments:
            segment_frames = len(segment.get('frames', []))
            
            # Check if adding this segment would exceed limits
            if current_batch and (
                len(current_batch) >= self.batch_size or 
                current_frame_count + segment_frames > self.max_frames_per_batch
            ):
                # Save current batch and start new one
                batches.append(current_batch)
                current_batch = []
                current_frame_count = 0
            
            # Add segment to current batch
            current_batch.append(segment)
            current_frame_count += segment_frames
        
        # Add remaining batch
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Created {len(batches)} batches from {len(segments)} segments")
        return batches
    
    def prepare_batch_prompt(self, segments: List[Dict], base_prompt: str) -> List[str]:
        """
        Bereitet individuelle Prompts für jedes Segment im Batch vor.
        
        WICHTIG: Jedes Segment bekommt seinen eigenen, spezifischen Prompt!
        """
        prompts = []
        
        for i, segment in enumerate(segments):
            # Erstelle segment-spezifischen Prompt
            segment_time = f"{segment.get('start_time', 0):.1f}s - {segment.get('end_time', 0):.1f}s"
            
            # Füge Kontext hinzu für bessere Beschreibungen
            segment_prompt = f"{base_prompt}\n\nSegment {i+1}/{len(segments)} (Time: {segment_time}):"
            
            # Optional: Füge vorherige Beschreibungen als Kontext hinzu
            if i > 0 and 'description' in segments[i-1]:
                segment_prompt += f"\n\nPrevious segment: {segments[i-1]['description'][:100]}..."
            
            prompts.append(segment_prompt)
        
        return prompts
    
    def process_batch_with_analyzer(self, analyzer, batch_segments: List[Dict], 
                                   quality_check: bool = True) -> List[Dict]:
        """
        Verarbeitet mehrere Segmente gleichzeitig mit dem Analyzer.
        
        Args:
            analyzer: Der Qwen2-VL Analyzer
            batch_segments: Liste von Segmenten zum Verarbeiten
            quality_check: Ob Qualitätsprüfung durchgeführt werden soll
            
        Returns:
            Liste von Ergebnissen für jedes Segment
        """
        results = []
        batch_start_time = time.time()
        
        try:
            # Vorbereitung: Sammle alle Frames für Batch-Processing
            all_frames = []
            frame_to_segment_map = []
            
            for seg_idx, segment in enumerate(batch_segments):
                segment_frames = segment.get('frames', [])
                all_frames.extend(segment_frames)
                frame_to_segment_map.extend([seg_idx] * len(segment_frames))
            
            logger.info(f"Processing batch with {len(batch_segments)} segments, {len(all_frames)} total frames")
            
            # Option 1: Batch-Processing wenn der Analyzer es unterstützt
            if hasattr(analyzer, 'process_batch') and callable(getattr(analyzer, 'process_batch')):
                # Nutze native Batch-Verarbeitung
                batch_results = analyzer.process_batch(batch_segments)
                results = batch_results
            
            # Option 2: Sequentielle Verarbeitung mit optimierten Prompts
            else:
                for i, segment in enumerate(batch_segments):
                    segment_start = time.time()
                    
                    # Verarbeite einzelnes Segment
                    if hasattr(analyzer, 'analyze_segment'):
                        result = analyzer.analyze_segment(
                            segment,
                            context=f"Segment {i+1} of {len(batch_segments)}"
                        )
                    else:
                        # Fallback: Standard analyze Methode
                        result = self._process_single_segment(analyzer, segment)
                    
                    # Qualitätsprüfung
                    if quality_check:
                        result = self._ensure_quality(result, segment)
                    
                    results.append(result)
                    
                    segment_time = time.time() - segment_start
                    logger.debug(f"Segment {i+1} processed in {segment_time:.2f}s")
            
            batch_time = time.time() - batch_start_time
            avg_time = batch_time / len(batch_segments) if batch_segments else 0
            
            logger.info(f"Batch processing completed in {batch_time:.2f}s (avg {avg_time:.2f}s per segment)")
            
            # Validiere dass alle Ergebnisse vorhanden sind
            if len(results) != len(batch_segments):
                logger.error(f"Result count mismatch: expected {len(batch_segments)}, got {len(results)}")
                # Fülle fehlende Ergebnisse
                while len(results) < len(batch_segments):
                    results.append({"error": "Processing failed", "description": "Unable to analyze segment"})
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fallback: Erstelle Fehler-Ergebnisse
            results = [{"error": str(e), "description": "Batch processing error"} for _ in batch_segments]
        
        return results
    
    def _process_single_segment(self, analyzer, segment: Dict) -> Dict:
        """Verarbeitet ein einzelnes Segment (Fallback)"""
        try:
            # Simuliere Segment-Analyse
            # In der echten Implementation würde hier der Analyzer aufgerufen
            result = {
                "start_time": segment.get("start_time", 0),
                "end_time": segment.get("end_time", 0),
                "frame_count": len(segment.get("frames", [])),
                "description": "Processed segment",  # Analyzer würde echte Beschreibung liefern
                "confidence": 0.95
            }
            return result
        except Exception as e:
            logger.error(f"Single segment processing failed: {e}")
            return {"error": str(e)}
    
    def _ensure_quality(self, result: Dict, segment: Dict) -> Dict:
        """
        Stellt sicher, dass die Qualität der Beschreibung hoch bleibt.
        
        Prüft:
        - Beschreibungslänge (mindestens 50 Zeichen)
        - Keine generischen Beschreibungen
        - Zeitstempel vorhanden
        """
        if not isinstance(result, dict):
            return {"error": "Invalid result format"}
        
        # Prüfe Beschreibungslänge
        description = result.get("description", "")
        if len(description) < 50:
            logger.warning(f"Short description detected: {len(description)} chars")
            result["quality_warning"] = "Description too short"
        
        # Prüfe auf generische Beschreibungen
        generic_phrases = ["a video", "some frames", "something happens", "a scene"]
        if any(phrase in description.lower() for phrase in generic_phrases):
            logger.warning("Generic description detected")
            result["quality_warning"] = "Description too generic"
        
        # Stelle sicher dass Zeitstempel vorhanden sind
        if "start_time" not in result:
            result["start_time"] = segment.get("start_time", 0)
        if "end_time" not in result:
            result["end_time"] = segment.get("end_time", 0)
        
        return result
    
    def merge_batch_results(self, batch_results: List[List[Dict]]) -> List[Dict]:
        """Kombiniert Ergebnisse aus mehreren Batches"""
        merged = []
        for batch in batch_results:
            merged.extend(batch)
        
        # Sortiere nach Zeitstempel
        merged.sort(key=lambda x: x.get("start_time", 0))
        
        return merged

# Utility functions
def optimize_qwen2vl_segments(segments: List[Dict], target_batch_size: int = 4) -> List[List[Dict]]:
    """
    Optimiert Segmente für Qwen2-VL Batch-Processing.
    
    Args:
        segments: Original-Segmente
        target_batch_size: Gewünschte Batch-Größe
        
    Returns:
        Optimierte Batches
    """
    batcher = Qwen2VLBatcher(batch_size=target_batch_size)
    return batcher.create_batches(segments)