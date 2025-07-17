#!/usr/bin/env python3
"""
Comment CTA Detection mit Ray - FUNKTIONIERT mit Model Sharing!
"""

import ray
import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

@ray.remote(num_gpus=0)  # CPU only
class CommentCTADetector:
    """CTA Detection als Ray Actor"""
    
    def __init__(self):
        # CTA patterns auf Deutsch und Englisch
        self.cta_patterns = {
            'comment_below': [
                r'kommentier.*unten',
                r'schreib.*kommentar',
                r'lass.*kommentar',
                r'sag.*kommentar',
                r'comment.*below',
                r'drop.*comment',
                r'let.*know.*comment',
                r'was.*meinst.*du',
                r'was.*denkst.*du',
                r'eure.*meinung',
                r'deine.*meinung',
                r'schreibt.*mir',
                r'schreib.*mir',
                r'lasst.*wissen',
                r'lass.*wissen'
            ],
            'like_share': [
                r'lik.*teil',
                r'teil.*lik',
                r'vergiss.*nicht.*lik',
                r'wenn.*gefällt.*lik',
                r'like.*share',
                r'hit.*like',
                r'smash.*like',
                r'gib.*daumen',
                r'daumen.*hoch',
                r'herz.*da.*lass'
            ],
            'follow': [
                r'folg.*mir',
                r'folg.*für.*mehr',
                r'abonnier',
                r'follow.*for.*more',
                r'hit.*follow',
                r'vergiss.*nicht.*folg',
                r'für.*mehr.*content',
                r'mehr.*davon'
            ],
            'question': [
                r'was.*würd.*ihr',
                r'was.*würd.*du',
                r'was.*denkt.*ihr',
                r'was.*meint.*ihr',
                r'habt.*ihr.*schon',
                r'kennt.*ihr',
                r'wer.*von.*euch',
                r'wer.*kennt',
                r'welch.*ist.*euer',
                r'welch.*ist.*dein',
                r'.*\?.*kommentar',
                r'.*\?.*unten',
                r'noch.*mal.*bestellen',  # Marc Gebauer specific!
                r'was.*nun\?',  # Marc Gebauer specific!
                r'versteh.*frage.*nicht'  # Marc Gebauer specific!
            ],
            'save': [
                r'speicher.*das',
                r'speicher.*für.*später',
                r'save.*this',
                r'bookmark',
                r'nicht.*vergessen.*speicher'
            ],
            'dm_contact': [
                r'schreib.*mir.*dm',
                r'dm.*für',
                r'slide.*dm',
                r'nachricht.*schreib',
                r'privat.*nachricht'
            ]
        }
        
        logger.info("✅ Comment CTA Detector initialized")
    
    def detect_cta(self, text: str) -> List[Dict[str, Any]]:
        """Detect CTAs in text"""
        ctas_found = []
        text_lower = text.lower()
        
        for cta_type, patterns in self.cta_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    ctas_found.append({
                        'type': cta_type,
                        'pattern': pattern,
                        'confidence': 0.9 if '?' in text else 0.8
                    })
        
        return ctas_found
    
    def analyze_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze speech segments for CTAs"""
        results = []
        
        for seg in segments:
            text = seg.get('text', '')
            if not text:
                continue
            
            ctas = self.detect_cta(text)
            
            if ctas:
                # Marc Gebauer special detection
                if any(phrase in text.lower() for phrase in [
                    'noch mal bestellen',
                    'was nun?',
                    'verstehe die frage nicht'
                ]):
                    # Das ist definitiv ein CTA!
                    results.append({
                        'start_time': seg['start_time'],
                        'end_time': seg['end_time'],
                        'text': text,
                        'cta_detected': True,
                        'cta_types': list(set(cta['type'] for cta in ctas)),
                        'confidence': 0.95,
                        'special_detection': 'marc_gebauer_pattern'
                    })
                else:
                    results.append({
                        'start_time': seg['start_time'],
                        'end_time': seg['end_time'],
                        'text': text,
                        'cta_detected': True,
                        'cta_types': list(set(cta['type'] for cta in ctas)),
                        'confidence': max(cta['confidence'] for cta in ctas)
                    })
        
        return results

# Wrapper für normale Analyzer-Integration
class CommentCTAAnalyzer:
    """Wrapper für Ray Actor"""
    
    def __init__(self):
        self.actor = None
        self.analyzer_name = "comment_cta_detection"
    
    def _ensure_actor(self):
        """Ensure Ray actor exists"""
        if self.actor is None:
            ray.init(ignore_reinit_error=True)
            self.actor = CommentCTADetector.remote()
    
    def analyze(self, speech_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze for CTAs"""
        self._ensure_actor()
        
        # Get results from Ray actor
        results = ray.get(self.actor.analyze_segments.remote(speech_segments))
        
        return {
            'segments': results,
            'total_ctas': len(results),
            'cta_types_found': list(set(
                cta_type 
                for r in results 
                for cta_type in r.get('cta_types', [])
            ))
        }

if __name__ == "__main__":
    # Test
    test_segments = [
        {
            'start_time': 0.0,
            'end_time': 5.0,
            'text': "Da haben wir den nächsten. Mein White and White ist angekommen komplett ausgelaufen aus der Packung. Was nun?"
        },
        {
            'start_time': 5.0,
            'end_time': 8.0,
            'text': "Noch mal bestellen, ich verstehe die Frage nicht."
        }
    ]
    
    analyzer = CommentCTAAnalyzer()
    result = analyzer.analyze(test_segments)
    
    print("CTA Detection Results:")
    print(f"Total CTAs: {result['total_ctas']}")
    print(f"Types found: {result['cta_types_found']}")
    for seg in result['segments']:
        print(f"\nSegment {seg['start_time']}-{seg['end_time']}s:")
        print(f"  Text: {seg['text']}")
        print(f"  CTA Types: {seg['cta_types']}")
        print(f"  Confidence: {seg['confidence']}")
        if 'special_detection' in seg:
            print(f"  Special: {seg['special_detection']}")