#!/usr/bin/env python3
"""
Advanced Narrative Analysis
Combines outputs from speech transcription, scene descriptions, and emotion analysis
to create a comprehensive narrative understanding
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from analyzers.base_analyzer import GPUBatchAnalyzer
import logging
from collections import defaultdict, Counter
import re

logger = logging.getLogger(__name__)

class NarrativeAnalysisAdvanced(GPUBatchAnalyzer):
    """Advanced narrative analysis combining multiple data sources"""
    
    def __init__(self):
        super().__init__(batch_size=1)  # Process video as a whole
        self.narrative_segments = []
        self.story_arc = None
        self.themes = []
        self.emotional_journey = []
        
        logger.info("[NarrativeAnalysis] Initialized advanced narrative analyzer")
    
    def analyze(self, video_path: str, analyzer_outputs: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze narrative structure using data from other analyzers
        
        Args:
            video_path: Path to video file
            analyzer_outputs: Optional pre-computed outputs from other analyzers
        """
        logger.info(f"[NarrativeAnalysis] Starting narrative analysis of {video_path}")
        
        # If analyzer outputs not provided, we need to get them
        if not analyzer_outputs:
            logger.warning("[NarrativeAnalysis] No analyzer outputs provided, using limited analysis")
            # In production, this would fetch from results storage
            return self._create_limited_analysis(video_path)
        
        # Extract relevant data from other analyzers
        scene_data = analyzer_outputs.get('qwen2_vl_temporal', {}).get('segments', [])
        speech_data = analyzer_outputs.get('speech_transcription', {}).get('segments', [])
        emotion_data = analyzer_outputs.get('speech_emotion', {}).get('segments', [])
        visual_effects = analyzer_outputs.get('visual_effects', {}).get('segments', [])
        audio_analysis = analyzer_outputs.get('audio_analysis', {})
        
        # Combine all temporal data
        timeline = self._create_unified_timeline(
            scene_data, speech_data, emotion_data, visual_effects
        )
        
        # Analyze narrative elements
        narrative_segments = self._identify_narrative_segments(timeline)
        story_arc = self._analyze_story_arc(narrative_segments, emotion_data)
        themes = self._extract_themes(speech_data, scene_data)
        emotional_journey = self._analyze_emotional_journey(emotion_data, timeline)
        key_moments = self._identify_key_moments(timeline, visual_effects)
        pacing = self._analyze_narrative_pacing(narrative_segments, visual_effects)
        
        # Character/Speaker analysis
        character_analysis = self._analyze_characters(speech_data, emotion_data)
        
        # Generate narrative summary
        narrative_summary = self._generate_narrative_summary(
            story_arc, themes, emotional_journey, key_moments
        )
        
        result = {
            'segments': narrative_segments,
            'story_arc': story_arc,
            'themes': themes,
            'emotional_journey': emotional_journey,
            'key_moments': key_moments,
            'pacing': pacing,
            'character_analysis': character_analysis,
            'narrative_summary': narrative_summary,
            'metadata': {
                'total_segments': len(narrative_segments),
                'narrative_complexity': self._calculate_complexity(narrative_segments),
                'emotional_range': self._calculate_emotional_range(emotional_journey),
                'thematic_coherence': self._calculate_thematic_coherence(themes)
            }
        }
        
        logger.info(f"[NarrativeAnalysis] Found {len(narrative_segments)} narrative segments")
        return result
    
    def _create_unified_timeline(self, scene_data: List[Dict], speech_data: List[Dict],
                               emotion_data: List[Dict], visual_effects: List[Dict]) -> List[Dict]:
        """Create unified timeline combining all data sources"""
        timeline = []
        
        # Add scene data
        for scene in scene_data:
            timeline.append({
                'timestamp': scene.get('start_time', 0),
                'end_time': scene.get('end_time', 0),
                'type': 'scene',
                'description': scene.get('description', ''),
                'key_objects': scene.get('key_objects', []),
                'scene_type': scene.get('scene_type', 'unknown')
            })
        
        # Add speech data
        for speech in speech_data:
            timeline.append({
                'timestamp': speech.get('start', 0),
                'end_time': speech.get('end', 0),
                'type': 'speech',
                'text': speech.get('text', ''),
                'speaker': speech.get('speaker', 'unknown'),
                'language': speech.get('language', 'unknown')
            })
        
        # Add emotion data
        for emotion in emotion_data:
            timeline.append({
                'timestamp': emotion.get('start_time', 0),
                'end_time': emotion.get('end_time', 0),
                'type': 'emotion',
                'dominant_emotion': emotion.get('dominant_emotion', 'neutral'),
                'confidence': emotion.get('confidence', 0),
                'valence': emotion.get('emotion_valence', 'neutral')
            })
        
        # Add visual effects
        for effect in visual_effects:
            timeline.append({
                'timestamp': effect.get('timestamp', 0),
                'type': 'effect',
                'effects': effect.get('effects', {}),
                'description': effect.get('description', '')
            })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        
        return timeline
    
    def _identify_narrative_segments(self, timeline: List[Dict]) -> List[Dict]:
        """Identify distinct narrative segments"""
        segments = []
        current_segment = None
        segment_threshold = 3.0  # seconds between events to consider new segment
        
        for i, event in enumerate(timeline):
            timestamp = event['timestamp']
            
            # Start new segment if needed
            if (current_segment is None or 
                timestamp - current_segment['end_time'] > segment_threshold):
                
                if current_segment:
                    segments.append(self._finalize_segment(current_segment))
                
                current_segment = {
                    'start_time': timestamp,
                    'end_time': event.get('end_time', timestamp),
                    'events': [event],
                    'scene_descriptions': [],
                    'dialogue': [],
                    'emotions': [],
                    'effects': []
                }
            else:
                # Add to current segment
                current_segment['events'].append(event)
                current_segment['end_time'] = max(
                    current_segment['end_time'],
                    event.get('end_time', timestamp)
                )
            
            # Categorize event
            if event['type'] == 'scene':
                current_segment['scene_descriptions'].append(event['description'])
            elif event['type'] == 'speech':
                current_segment['dialogue'].append({
                    'text': event['text'],
                    'speaker': event['speaker']
                })
            elif event['type'] == 'emotion':
                current_segment['emotions'].append(event['dominant_emotion'])
            elif event['type'] == 'effect':
                current_segment['effects'].append(event['description'])
        
        # Add final segment
        if current_segment:
            segments.append(self._finalize_segment(current_segment))
        
        return segments
    
    def _finalize_segment(self, segment: Dict) -> Dict:
        """Finalize a narrative segment with analysis"""
        # Determine segment type
        segment_type = self._classify_segment_type(segment)
        
        # Create unified description
        description = self._create_segment_description(segment)
        
        # Determine dominant emotion
<<<<<<< HEAD
        try:
            if segment['emotions']:
                emotion_counts = Counter(segment['emotions'])
                dominant_emotion = emotion_counts.most_common(1)[0][0]
            else:
                dominant_emotion = 'neutral'
        except (KeyError, IndexError, TypeError):
=======
        if segment['emotions']:
            emotion_counts = Counter(segment['emotions'])
            dominant_emotion = emotion_counts.most_common(1)[0][0]
        else:
>>>>>>> 737fef1f5ce8d7eec45c5518784ebaf5218324cc
            dominant_emotion = 'neutral'
        
        # Analyze segment purpose
        purpose = self._analyze_segment_purpose(segment)
        
        return {
            'start_time': round(segment['start_time'], 2),
            'end_time': round(segment['end_time'], 2),
            'duration': round(segment['end_time'] - segment['start_time'], 2),
            'type': segment_type,
            'description': description,
            'dialogue_count': len(segment['dialogue']),
            'dominant_emotion': dominant_emotion,
            'purpose': purpose,
            'has_effects': len(segment['effects']) > 0,
            'narrative_weight': self._calculate_segment_weight(segment)
        }
    
    def _classify_segment_type(self, segment: Dict) -> str:
        """Classify the type of narrative segment"""
        has_dialogue = len(segment['dialogue']) > 0
        has_effects = len(segment['effects']) > 0
        scene_count = len(segment['scene_descriptions'])
        
        if has_dialogue and scene_count > 0:
            return 'dialogue_scene'
        elif has_dialogue and not scene_count:
            return 'voice_over'
        elif has_effects and scene_count > 1:
            return 'montage'
        elif scene_count > 2:
            return 'action_sequence'
        elif scene_count == 1 and not has_dialogue:
            return 'establishing_shot'
        else:
            return 'transitional'
    
    def _create_segment_description(self, segment: Dict) -> str:
        """Create a unified description for the segment"""
        parts = []
        
        # Add scene description
        if segment['scene_descriptions']:
            # Use the most detailed description
            scene_desc = max(segment['scene_descriptions'], key=len)
            parts.append(scene_desc)
        
        # Add dialogue summary
        if segment['dialogue']:
            dialogue_summary = self._summarize_dialogue(segment['dialogue'])
            parts.append(dialogue_summary)
        
        # Add emotional context
        if segment['emotions'] and segment['emotions'][0] != 'neutral':
            emotion_desc = f"Emotional tone: {segment['emotions'][0]}"
            parts.append(emotion_desc)
        
        # Add effects if significant
        if segment['effects']:
            effects_desc = f"Visual effects: {', '.join(segment['effects'][:2])}"
            parts.append(effects_desc)
        
        return " | ".join(parts) if parts else "Narrative segment"
    
    def _summarize_dialogue(self, dialogue: List[Dict]) -> str:
        """Summarize dialogue content"""
        if not dialogue:
            return ""
        
        # Extract key topics from dialogue
        all_text = " ".join([d['text'] for d in dialogue])
        
        # Simple keyword extraction (in production, use NLP)
        keywords = self._extract_keywords(all_text)
        
        if len(dialogue) == 1:
            return f"Speaker says: '{dialogue[0]['text'][:50]}...'"
        else:
            return f"Dialogue about: {', '.join(keywords[:3])}"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction"""
        # Remove common words (simplified - use NLTK in production)
        stop_words = {'ich', 'bin', 'ist', 'und', 'der', 'die', 'das', 'in', 'zu',
                     'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        
        # Count frequencies
        word_freq = Counter(keywords)
        return [word for word, _ in word_freq.most_common(5)]
    
    def _analyze_segment_purpose(self, segment: Dict) -> str:
        """Analyze the narrative purpose of a segment"""
        dialogue_ratio = len(segment['dialogue']) / max(len(segment['events']), 1)
        
        # Check for specific patterns
        if 'establishing' in " ".join(segment['scene_descriptions']).lower():
            return 'establishment'
        elif dialogue_ratio > 0.7:
            return 'exposition'
        elif any('action' in desc.lower() for desc in segment['scene_descriptions']):
            return 'action'
        elif segment['emotions'] and all(e in ['sad', 'disappointed'] for e in segment['emotions']):
            return 'emotional_moment'
        elif len(segment['effects']) > 2:
            return 'stylistic'
        else:
            return 'development'
    
    def _calculate_segment_weight(self, segment: Dict) -> float:
        """Calculate narrative importance of segment"""
        weight = 0.0
        
        # Dialogue adds weight
        weight += len(segment['dialogue']) * 0.3
        
        # Emotional moments add weight
        if segment['emotions']:
            non_neutral = [e for e in segment['emotions'] if e != 'neutral']
            weight += len(non_neutral) * 0.2
        
        # Scene changes add weight
        weight += len(segment['scene_descriptions']) * 0.1
        
        # Effects can add weight
        weight += min(len(segment['effects']) * 0.1, 0.3)
        
        # Duration factor
        duration = segment['end_time'] - segment['start_time']
        weight *= (1 + duration / 10)  # Longer segments generally more important
        
        return min(weight, 1.0)
    
    def _analyze_story_arc(self, segments: List[Dict], emotion_data: List[Dict]) -> Dict:
        """Analyze the overall story arc"""
        if not segments:
            return {'type': 'unknown', 'description': 'No narrative data'}
        
        # Divide into acts (simple 3-act structure)
        total_duration = segments[-1]['end_time'] - segments[0]['start_time']
        act_duration = total_duration / 3
        
        acts = {
            'setup': [],
            'confrontation': [],
            'resolution': []
        }
        
        for segment in segments:
            relative_time = segment['start_time'] - segments[0]['start_time']
            if relative_time < act_duration:
                acts['setup'].append(segment)
            elif relative_time < 2 * act_duration:
                acts['confrontation'].append(segment)
            else:
                acts['resolution'].append(segment)
        
        # Analyze each act
        act_analysis = {}
        for act_name, act_segments in acts.items():
            if act_segments:
                act_analysis[act_name] = {
                    'segment_count': len(act_segments),
                    'avg_weight': np.mean([s['narrative_weight'] for s in act_segments]),
                    'dominant_type': Counter([s['type'] for s in act_segments]).most_common(1)[0][0],
                    'emotional_tone': self._analyze_act_emotion(act_segments)
                }
        
        # Classify arc type
        arc_type = self._classify_arc_type(act_analysis)
        
        return {
            'type': arc_type,
            'acts': act_analysis,
            'turning_points': self._identify_turning_points(segments),
            'climax': self._identify_climax(segments),
            'description': self._describe_story_arc(arc_type, act_analysis)
        }
    
    def _analyze_act_emotion(self, segments: List[Dict]) -> str:
        """Analyze emotional tone of an act"""
        emotions = [s['dominant_emotion'] for s in segments if s['dominant_emotion']]
        if not emotions:
            return 'neutral'
        
        emotion_counts = Counter(emotions)
        return emotion_counts.most_common(1)[0][0]
    
    def _classify_arc_type(self, act_analysis: Dict) -> str:
        """Classify the type of story arc"""
        if not act_analysis:
            return 'linear'
        
        # Look at narrative weight progression
        weights = []
        for act in ['setup', 'confrontation', 'resolution']:
            if act in act_analysis:
                weights.append(act_analysis[act]['avg_weight'])
        
        if not weights:
            return 'minimal'
        
        # Classic arc: rising action
        if len(weights) >= 3 and weights[1] > weights[0] and weights[1] > weights[2]:
            return 'classic_arc'
        # Building to climax
        elif len(weights) >= 2 and weights[-1] > weights[0]:
            return 'rising_action'
        # Declining tension
        elif len(weights) >= 2 and weights[0] > weights[-1]:
            return 'declining_action'
        # Steady
        else:
            return 'steady_progression'
    
    def _identify_turning_points(self, segments: List[Dict]) -> List[Dict]:
        """Identify narrative turning points"""
        turning_points = []
        
        for i in range(1, len(segments) - 1):
            prev_segment = segments[i-1]
            curr_segment = segments[i]
            next_segment = segments[i+1]
            
            # Check for significant changes
            emotion_change = (prev_segment['dominant_emotion'] != next_segment['dominant_emotion'])
            type_change = (prev_segment['type'] != next_segment['type'])
            weight_spike = curr_segment['narrative_weight'] > max(
                prev_segment['narrative_weight'],
                next_segment['narrative_weight']
            ) * 1.5
            
            if (emotion_change and type_change) or weight_spike:
                turning_points.append({
                    'timestamp': curr_segment['start_time'],
                    'type': 'emotional_shift' if emotion_change else 'narrative_shift',
                    'description': curr_segment['description'][:100]
                })
        
        return turning_points
    
    def _identify_climax(self, segments: List[Dict]) -> Optional[Dict]:
        """Identify the narrative climax"""
        if not segments:
            return None
        
        # Find segment with highest narrative weight
        climax_segment = max(segments, key=lambda s: s['narrative_weight'])
        
        # Verify it's significant enough
        avg_weight = np.mean([s['narrative_weight'] for s in segments])
        if climax_segment['narrative_weight'] > avg_weight * 1.5:
            return {
                'timestamp': climax_segment['start_time'],
                'description': climax_segment['description'],
                'type': climax_segment['type']
            }
        
        return None
    
    def _describe_story_arc(self, arc_type: str, act_analysis: Dict) -> str:
        """Create description of story arc"""
        descriptions = {
            'classic_arc': "Classic three-act structure with setup, rising action, and resolution",
            'rising_action': "Building tension towards the end",
            'declining_action': "Front-loaded narrative with decreasing intensity",
            'steady_progression': "Consistent narrative flow throughout",
            'linear': "Simple linear progression",
            'minimal': "Minimal narrative structure"
        }
        
        base_desc = descriptions.get(arc_type, "Narrative structure")
        
        # Add act-specific details
        if 'setup' in act_analysis:
            base_desc += f". Setup establishes {act_analysis['setup']['dominant_type']}"
        
        return base_desc
    
    def _extract_themes(self, speech_data: List[Dict], scene_data: List[Dict]) -> List[Dict]:
        """Extract thematic elements from content"""
        themes = []
        
        # Analyze speech for themes
        all_dialogue = " ".join([s.get('text', '') for s in speech_data])
        dialogue_themes = self._extract_themes_from_text(all_dialogue)
        
        # Analyze scenes for visual themes
        all_scenes = " ".join([s.get('description', '') for s in scene_data])
        visual_themes = self._extract_themes_from_text(all_scenes)
        
        # Combine and rank themes
        all_theme_words = dialogue_themes + visual_themes
        theme_counts = Counter(all_theme_words)
        
        for theme, count in theme_counts.most_common(5):
            themes.append({
                'theme': theme,
                'occurrences': count,
                'type': 'dialogue' if theme in dialogue_themes else 'visual',
                'strength': min(count / len(all_theme_words), 1.0) if all_theme_words else 0
            })
        
        return themes
    
    def _extract_themes_from_text(self, text: str) -> List[str]:
        """Extract thematic keywords from text"""
        # Simplified theme extraction
        theme_patterns = {
            'growth': ['grow', 'develop', 'change', 'transform', 'evolution'],
            'conflict': ['fight', 'struggle', 'problem', 'challenge', 'difficult'],
            'discovery': ['find', 'discover', 'realize', 'understand', 'learn'],
            'relationship': ['friend', 'family', 'love', 'together', 'connect'],
            'journey': ['travel', 'go', 'move', 'path', 'way'],
            'identity': ['who', 'self', 'identity', 'become', 'am']
        }
        
        text_lower = text.lower()
        found_themes = []
        
        for theme, keywords in theme_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                found_themes.append(theme)
        
        return found_themes
    
    def _analyze_emotional_journey(self, emotion_data: List[Dict], 
                                  timeline: List[Dict]) -> List[Dict]:
        """Analyze emotional progression through the narrative"""
        if not emotion_data:
            return []
        
        journey = []
        
        # Group emotions by time windows
        window_size = 5.0  # 5 second windows
        current_window = []
        window_start = 0
        
        for emotion in emotion_data:
            if emotion['start_time'] - window_start > window_size:
                if current_window:
                    journey.append(self._summarize_emotion_window(
                        current_window, window_start
                    ))
                window_start = emotion['start_time']
                current_window = [emotion]
            else:
                current_window.append(emotion)
        
        # Add final window
        if current_window:
            journey.append(self._summarize_emotion_window(
                current_window, window_start
            ))
        
        return journey
    
    def _summarize_emotion_window(self, emotions: List[Dict], 
                                 start_time: float) -> Dict:
        """Summarize emotions in a time window"""
        emotion_types = [e['dominant_emotion'] for e in emotions]
        valences = [e.get('emotion_valence', 'neutral') for e in emotions]
        
        # Count emotions
        emotion_counts = Counter(emotion_types)
        dominant = emotion_counts.most_common(1)[0][0]
        
        # Determine overall valence
        valence_counts = Counter(valences)
        dominant_valence = valence_counts.most_common(1)[0][0]
        
        # Calculate intensity (based on confidence and variety)
        avg_confidence = np.mean([e.get('confidence', 0.5) for e in emotions])
        emotion_variety = len(set(emotion_types))
        intensity = avg_confidence * (1 + emotion_variety * 0.1)
        
        return {
            'timestamp': round(start_time, 2),
            'dominant_emotion': dominant,
            'valence': dominant_valence,
            'intensity': min(intensity, 1.0),
            'emotion_mix': dict(emotion_counts),
            'stability': 1.0 / max(emotion_variety, 1)  # More variety = less stable
        }
    
    def _identify_key_moments(self, timeline: List[Dict], 
                            visual_effects: List[Dict]) -> List[Dict]:
        """Identify key narrative moments"""
        key_moments = []
        
        # Look for effect-heavy moments
        for effect in visual_effects:
            if effect.get('effects', {}):
                effect_count = len(effect['effects'])
                if effect_count >= 3:  # Multiple effects = important moment
                    key_moments.append({
                        'timestamp': effect['timestamp'],
                        'type': 'visual_emphasis',
                        'description': effect.get('description', 'Multiple visual effects'),
                        'importance': min(effect_count / 5, 1.0)
                    })
        
        # Look for emotional peaks in timeline
        emotion_events = [e for e in timeline if e['type'] == 'emotion']
        for event in emotion_events:
            if (event.get('confidence', 0) > 0.8 and 
                event.get('dominant_emotion') not in ['neutral', 'calm']):
                key_moments.append({
                    'timestamp': event['timestamp'],
                    'type': 'emotional_peak',
                    'description': f"Strong {event['dominant_emotion']} emotion",
                    'importance': event.get('confidence', 0.5)
                })
        
        # Sort by timestamp and filter overlapping moments
        key_moments.sort(key=lambda x: x['timestamp'])
        filtered_moments = []
        last_timestamp = -5
        
        for moment in key_moments:
            if moment['timestamp'] - last_timestamp > 2:  # At least 2 seconds apart
                filtered_moments.append(moment)
                last_timestamp = moment['timestamp']
        
        return filtered_moments
    
    def _analyze_narrative_pacing(self, segments: List[Dict], 
                                visual_effects: List[Dict]) -> Dict:
        """Analyze pacing of the narrative"""
        if not segments:
            return {'type': 'unknown', 'description': 'No pacing data'}
        
        # Calculate segment durations
        durations = [s['duration'] for s in segments]
        avg_duration = np.mean(durations)
        duration_variance = np.var(durations)
        
        # Count effects per segment
        effect_density = len(visual_effects) / len(segments) if segments else 0
        
        # Analyze dialogue pacing
        dialogue_segments = [s for s in segments if s['dialogue_count'] > 0]
        dialogue_ratio = len(dialogue_segments) / len(segments) if segments else 0
        
        # Classify pacing
        if avg_duration < 3 and effect_density > 2:
            pacing_type = 'rapid'
        elif avg_duration < 5 and duration_variance < 2:
            pacing_type = 'steady'
        elif avg_duration > 8:
            pacing_type = 'slow'
        elif duration_variance > 5:
            pacing_type = 'variable'
        else:
            pacing_type = 'moderate'
        
        # Analyze rhythm
        if dialogue_ratio > 0.7:
            rhythm = 'dialogue_driven'
        elif effect_density > 1.5:
            rhythm = 'visually_driven'
        else:
            rhythm = 'balanced'
        
        return {
            'type': pacing_type,
            'rhythm': rhythm,
            'avg_segment_duration': round(avg_duration, 2),
            'duration_variance': round(duration_variance, 2),
            'effect_density': round(effect_density, 2),
            'dialogue_ratio': round(dialogue_ratio, 2),
            'description': self._describe_pacing(pacing_type, rhythm)
        }
    
    def _describe_pacing(self, pacing_type: str, rhythm: str) -> str:
        """Create pacing description"""
        pacing_desc = {
            'rapid': "Fast-paced with quick cuts",
            'steady': "Consistent, measured pacing",
            'slow': "Deliberate, contemplative pacing",
            'variable': "Dynamic pacing with varied rhythms",
            'moderate': "Balanced pacing"
        }
        
        rhythm_desc = {
            'dialogue_driven': "driven by conversation",
            'visually_driven': "driven by visual elements",
            'balanced': "with balanced audio-visual elements"
        }
        
        return f"{pacing_desc.get(pacing_type, 'Pacing')} {rhythm_desc.get(rhythm, '')}"
    
    def _analyze_characters(self, speech_data: List[Dict], 
                          emotion_data: List[Dict]) -> Dict:
        """Analyze characters/speakers in the narrative"""
        characters = defaultdict(lambda: {
            'lines': 0,
            'words': 0,
            'emotions': [],
            'first_appearance': float('inf'),
            'last_appearance': 0
        })
        
        # Analyze speech
        for speech in speech_data:
            speaker = speech.get('speaker', 'unknown')
            characters[speaker]['lines'] += 1
            characters[speaker]['words'] += len(speech.get('text', '').split())
            characters[speaker]['first_appearance'] = min(
                characters[speaker]['first_appearance'],
                speech.get('start', 0)
            )
            characters[speaker]['last_appearance'] = max(
                characters[speaker]['last_appearance'],
                speech.get('end', 0)
            )
        
        # Match emotions to speakers (simplified - assumes temporal proximity)
        for emotion in emotion_data:
            emotion_time = emotion.get('start_time', 0)
            # Find closest speech
            closest_speaker = None
            min_distance = float('inf')
            
            for speech in speech_data:
                speech_time = speech.get('start', 0)
                distance = abs(emotion_time - speech_time)
                if distance < min_distance and distance < 2.0:  # Within 2 seconds
                    min_distance = distance
                    closest_speaker = speech.get('speaker', 'unknown')
            
            if closest_speaker:
                characters[closest_speaker]['emotions'].append(
                    emotion.get('dominant_emotion', 'neutral')
                )
        
        # Summarize characters
        character_summary = {}
        for speaker, data in characters.items():
            if data['lines'] > 0:  # Only include speakers with dialogue
                emotion_counts = Counter(data['emotions'])
                dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else 'neutral'
                
                character_summary[speaker] = {
                    'line_count': data['lines'],
                    'word_count': data['words'],
                    'screen_time': round(
                        data['last_appearance'] - data['first_appearance'], 2
                    ),
                    'first_appearance': round(data['first_appearance'], 2),
                    'dominant_emotion': dominant_emotion,
                    'role': self._determine_character_role(data, len(speech_data))
                }
        
        return character_summary
    
    def _determine_character_role(self, character_data: Dict, 
                                 total_lines: int) -> str:
        """Determine character's narrative role"""
        line_ratio = character_data['lines'] / max(total_lines, 1)
        
        if line_ratio > 0.5:
            return 'protagonist'
        elif line_ratio > 0.2:
            return 'major'
        elif line_ratio > 0.1:
            return 'supporting'
        else:
            return 'minor'
    
    def _calculate_complexity(self, segments: List[Dict]) -> float:
        """Calculate narrative complexity score"""
        if not segments:
            return 0.0
        
        # Factors that increase complexity
        complexity = 0.0
        
        # Variety of segment types
        segment_types = set(s['type'] for s in segments)
        complexity += len(segment_types) * 0.1
        
        # Number of narrative shifts
        shifts = sum(1 for i in range(1, len(segments)) 
                    if segments[i]['type'] != segments[i-1]['type'])
        complexity += shifts * 0.05
        
        # Emotional variety
        emotions = set(s['dominant_emotion'] for s in segments)
        complexity += len(emotions) * 0.05
        
        # Dialogue density
        dialogue_segments = sum(1 for s in segments if s['dialogue_count'] > 0)
        complexity += (dialogue_segments / len(segments)) * 0.2
        
        return min(complexity, 1.0)
    
    def _calculate_emotional_range(self, emotional_journey: List[Dict]) -> float:
        """Calculate emotional range score"""
        if not emotional_journey:
            return 0.0
        
        # Count unique emotions
        all_emotions = set()
        for point in emotional_journey:
            all_emotions.update(point.get('emotion_mix', {}).keys())
        
        # More emotions = wider range
        return min(len(all_emotions) / 10, 1.0)
    
    def _calculate_thematic_coherence(self, themes: List[Dict]) -> float:
        """Calculate how coherent the themes are"""
        if not themes:
            return 0.0
        
        # If one theme dominates, high coherence
        if themes[0]['strength'] > 0.5:
            return 0.9
        # If themes are evenly distributed, lower coherence
        elif len(themes) > 3 and all(t['strength'] < 0.2 for t in themes):
            return 0.3
        else:
            return 0.6
    
    def _generate_narrative_summary(self, story_arc: Dict, themes: List[Dict],
                                  emotional_journey: List[Dict], 
                                  key_moments: List[Dict]) -> str:
        """Generate a narrative summary"""
        parts = []
        
        # Story structure
        arc_type = story_arc.get('type', 'unknown')
        parts.append(f"The narrative follows a {arc_type.replace('_', ' ')} structure")
        
        # Themes
        if themes:
            theme_names = [t['theme'] for t in themes[:2]]
            parts.append(f"exploring themes of {' and '.join(theme_names)}")
        
        # Emotional journey
        if emotional_journey:
            start_emotion = emotional_journey[0].get('dominant_emotion', 'neutral')
            end_emotion = emotional_journey[-1].get('dominant_emotion', 'neutral')
            if start_emotion != end_emotion:
                parts.append(f"with an emotional journey from {start_emotion} to {end_emotion}")
        
        # Key moments
        if key_moments:
            parts.append(f"featuring {len(key_moments)} key dramatic moments")
        
        return ". ".join(parts) + "."
    
    def _create_limited_analysis(self, video_path: str) -> Dict[str, Any]:
        """Create limited analysis when no analyzer outputs available"""
        return {
            'segments': [],
            'story_arc': {
                'type': 'unknown',
                'description': 'Unable to analyze without additional data'
            },
            'themes': [],
            'emotional_journey': [],
            'key_moments': [],
            'pacing': {'type': 'unknown'},
            'character_analysis': {},
            'narrative_summary': 'Narrative analysis requires data from other analyzers',
            'metadata': {
                'error': 'No analyzer outputs provided'
            }
        }
    
    def process_batch_gpu(self, frames: List[np.ndarray], 
                         frame_times: List[float]) -> Dict[str, Any]:
        """Not used - this analyzer works with other analyzer outputs"""
        return {'segments': []}