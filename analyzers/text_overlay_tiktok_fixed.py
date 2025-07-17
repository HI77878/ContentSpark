#!/usr/bin/env python3
"""
Optimized Text Overlay Analyzer specifically for TikTok subtitles
"""

import cv2
import easyocr
import numpy as np
import torch
from typing import Dict, List, Any
import logging
from pathlib import Path

from analyzers.base_analyzer import GPUBatchAnalyzer

from shared_frame_cache import FRAME_CACHE
# GPU Forcing
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)

class TikTokTextOverlayAnalyzer(GPUBatchAnalyzer):
    """
    Specialized text detection for TikTok voice-to-text overlays
    """
    
    def __init__(self):
        super().__init__(batch_size=1)
        self.name = "text_overlay"
        self.reader = None
        # Languages for TikTok - German and English
        self.languages = ['de', 'en']
        # Sample every 0.17 seconds for maximum text coverage
        self.sample_rate = 5  # ERHÖHT: Every 0.17s at 30fps für schnelle Textänderungen
        
    def _load_model_impl(self):
        """Load EasyOCR model optimized for TikTok subtitles"""
        logger.info("[TikTok Text Overlay] Loading EasyOCR with German + English")
        self.reader = easyocr.Reader(self.languages, gpu=True)
        logger.info("✅ EasyOCR loaded on GPU for TikTok text detection")
        
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> Dict[str, Any]:
        """Process frames to extract TikTok subtitles"""
        # Ensure model is loaded
        if not self.model_loaded:
            self.load_model()
            
        segments = []
        
        for frame, timestamp in zip(frames, frame_times):
            try:
                height, width = frame.shape[:2]
                texts_found = []
                
                # Method 1: Scan FULL frame for any text overlays
                # TikTok can have text anywhere - not just bottom!
                results = self.reader.readtext(
                    frame,  # FULL FRAME scan
                    detail=True,
                    paragraph=False,  # Don't merge paragraphs
                    width_ths=0.5,    # Lower threshold to catch more text
                    height_ths=0.5,   # Lower threshold to catch more text
                    text_threshold=0.5,  # Text confidence threshold
                    low_text=0.3     # Minimum text confidence
                )
                
                for (bbox, text, conf) in results:
                    if conf > 0.3 and len(text.strip()) > 1:  # Lower confidence threshold
                        # Calculate position relative to full frame
                        x_min = min(point[0] for point in bbox)
                        y_min = min(point[1] for point in bbox)  # No offset - full frame
                        x_max = max(point[0] for point in bbox)
                        y_max = max(point[1] for point in bbox)  # No offset - full frame
                        
                        texts_found.append({
                            'text': text.strip(),
                            'position': {
                                'x': int(x_min),
                                'y': int(y_min),
                                'width': int(x_max - x_min),
                                'height': int(y_max - y_min)
                            },
                            'confidence': float(conf),
                            'method': 'direct'
                        })
                
                # Method 2: Try with multiple preprocessing approaches
                if not texts_found:
                    # Try different preprocessing methods for better text detection
                    preprocessed_frames = []
                    
                    # 1. High contrast for white text
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    _, binary1 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
                    preprocessed_frames.append(binary1)
                    
                    # 2. Inverse for black text
                    _, binary2 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
                    preprocessed_frames.append(binary2)
                    
                    # 3. Adaptive threshold for varying backgrounds
                    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY, 11, 2)
                    preprocessed_frames.append(adaptive)
                    
                    for processed in preprocessed_frames:
                        # Convert back to BGR for EasyOCR
                        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                        
                        results = self.reader.readtext(
                            processed_bgr,
                            detail=True,
                            paragraph=False,
                            width_ths=0.4,
                            height_ths=0.4
                        )
                        
                        for (bbox, text, conf) in results:
                            if conf > 0.2 and len(text.strip()) > 1:
                                x_min = min(point[0] for point in bbox)
                                y_min = min(point[1] for point in bbox)
                                x_max = max(point[0] for point in bbox)
                                y_max = max(point[1] for point in bbox)
                            
                                texts_found.append({
                                    'text': text.strip(),
                                    'position': {
                                        'x': int(x_min),
                                        'y': int(y_min),
                                        'width': int(x_max - x_min),
                                        'height': int(y_max - y_min)
                                    },
                                    'confidence': float(conf),
                                    'method': 'enhanced'
                                })
                        
                        # Stop if we found text
                        if texts_found:
                            break
                
                if texts_found:
                    # Merge nearby texts (TikTok often splits subtitles)
                    merged_text = self.merge_nearby_texts(texts_found)
                    
                    segments.append({
                        'timestamp': float(timestamp),
                        'text': merged_text,  # ADD THIS FOR COMPATIBILITY!
                        'texts': texts_found,
                        'combined_text': merged_text,
                        'confidence': max(t['confidence'] for t in texts_found)
                    })
                else:
                    # Still add segment but with empty text (not "KEIN TEXT")
                    segments.append({
                        'timestamp': float(timestamp),
                        'text': "",  # Empty string instead of "KEIN TEXT"
                        'texts': [],
                        'combined_text': "",
                        'confidence': 0.0
                    })
                    
            except Exception as e:
                logger.error(f"Error processing frame at {timestamp}s: {e}")
                
        return {'segments': segments}
    
    def merge_nearby_texts(self, texts: List[Dict]) -> str:
        """Merge texts that are on the same line (typical for TikTok subtitles)"""
        if not texts:
            return ""
        
        # Sort by y position then x position
        sorted_texts = sorted(texts, key=lambda t: (t['position']['y'], t['position']['x']))
        
        # Group texts by similar y position (same line)
        lines = []
        current_line = [sorted_texts[0]]
        
        for text in sorted_texts[1:]:
            # If y position is within 20 pixels, consider same line
            if abs(text['position']['y'] - current_line[0]['position']['y']) < 20:
                current_line.append(text)
            else:
                lines.append(current_line)
                current_line = [text]
        
        lines.append(current_line)
        
        # Merge texts in each line
        merged_lines = []
        for line in lines:
            line_texts = sorted(line, key=lambda t: t['position']['x'])
            merged_text = ' '.join(t['text'] for t in line_texts)
            merged_lines.append(merged_text)
        
        return ' '.join(merged_lines)
        
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Main analysis function"""
        logger.info(f"[TikTok Text Overlay] Analyzing {video_path}")
        
        # Extract frames at higher rate for better subtitle capture
        frames, frame_times = self.extract_frames(
            video_path,
            sample_rate = 15  # Every 0.5 seconds for TikTok subtitles
        )
        
        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        if not frames:
            return {'segments': [], 'error': 'No frames extracted'}
            
        # Process frames
        result = self.process_batch_gpu(frames, frame_times)
        
        # Add metadata
        result['metadata'] = {
            'total_frames_analyzed': len(frames),
            'fps': fps,
            'duration': duration,
            'analyzer': 'tiktok_text_overlay_optimized',
            'languages': self.languages
        }
        
        # Post-process to clean up text
        for segment in result.get('segments', []):
            segment['combined_text'] = self.clean_tiktok_text(segment.get('combined_text', ''))
        
        logger.info(f"[TikTok Text] Found {len(result['segments'])} segments with text")
        
        return result
    
    def clean_tiktok_text(self, text: str) -> str:
        """Clean common issues in TikTok subtitle OCR"""
        if not text:
            return ""
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Common OCR fixes for German text
        replacements = {
            'lch': 'Ich',  # Common "Ich" misread
            ' lch ': ' Ich ',
            'Ä': 'Ä',
            'Ö': 'Ö', 
            'Ü': 'Ü',
            'ä': 'ä',
            'ö': 'ö',
            'ü': 'ü',
            'ß': 'ß'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text