#!/usr/bin/env python3

# GPU Forcing
import torch
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

"""
Speech Emotion Analyzer with Real ML Model
Uses Wav2Vec2 pretrained on emotion recognition
"""

import os
# Set environment variable for tf-keras compatibility
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import torch
import librosa
import numpy as np
from typing import List, Dict, Any
from transformers import pipeline, Wav2Vec2FeatureExtractor
import warnings
warnings.filterwarnings('ignore')

from analyzers.base_analyzer import GPUBatchAnalyzer

class GPUBatchSpeechEmotion(GPUBatchAnalyzer):
    """Real ML-based speech emotion recognition using Wav2Vec2"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.classifier = None
        self.models_loaded = False
        print("[SpeechEmotionML] Initialized - will load model on first use")
        
    def _load_model_impl(self):
        """Load pretrained emotion recognition model"""
        if self.models_loaded:
            return
            
        print("[SpeechEmotionML] Loading pretrained emotion model...")
        
        try:
            # Use pretrained emotion model
            self.classifier = pipeline(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                device=0 if self.device == 'cuda' and torch.cuda.is_available() else -1
            )
            self.models_loaded = True
            print("[SpeechEmotionML] ✅ Loaded pretrained emotion model")
            
        except Exception as e:
            print(f"[SpeechEmotionML] Failed to load emotion model: {e}")
            # Kein Fallback - fail fast
            raise RuntimeError(f"Failed to load speech emotion model: {e}")
                
    def extract_audio_segments(self, audio_path, segment_duration=3.0):
        """Extract audio in segments for emotion analysis"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio) / sr
            
            segments = []
            for start in np.arange(0, duration, segment_duration):
                end = min(start + segment_duration, duration)
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                
                segment_audio = audio[start_sample:end_sample]
                
                if len(segment_audio) > sr * 0.5:  # At least 0.5 seconds
                    segments.append({
                        'audio': segment_audio,
                        'start': start,
                        'end': end
                    })
                    
            return segments
            
        except Exception as e:
            print(f"[SpeechEmotionML] Audio extraction error: {e}")
            return []
            
    def analyze(self, video_path):
        """Analyze speech emotions using real ML model"""
        print(f"[SpeechEmotionML] Analyzing: {video_path}")
        
        # Load model if needed
        if not self.models_loaded:
            self._load_model_impl()
            
        if not self.classifier:
            return {'data': [], 'error': 'Model not loaded'}
            
        try:
            # Extract audio segments
            segments = self.extract_audio_segments(video_path)
            
            if not segments:
                print("[SpeechEmotionML] No audio segments extracted")
                return {'segments': [], 'error': 'No audio segments found'}
                
            results = []
            for i, seg in enumerate(segments):
                try:
                    # Predict emotions
                    predictions = self.classifier(seg['audio'])
                    
                    # Convert to expected format with enhanced categories
                    emotion_scores = {}
                    raw_predictions = {}
                    
                    for pred in predictions[:8]:  # Get more predictions for nuanced analysis
                        label = pred['label'].lower()
                        score = pred['score']
                        raw_predictions[label] = score
                        
                        # Enhanced emotion mapping with subcategories
                        if 'ang' in label or 'mad' in label or 'rage' in label:
                            emotion_scores['angry'] = emotion_scores.get('angry', 0) + score
                        elif 'irrit' in label or 'annoy' in label:
                            emotion_scores['irritated'] = emotion_scores.get('irritated', 0) + score
                        elif 'hap' in label or 'joy' in label or 'cheer' in label:
                            emotion_scores['happy'] = emotion_scores.get('happy', 0) + score
                        elif 'excit' in label or 'enthus' in label:
                            emotion_scores['excited'] = emotion_scores.get('excited', 0) + score
                        elif 'sad' in label or 'unhap' in label or 'depress' in label:
                            emotion_scores['sad'] = emotion_scores.get('sad', 0) + score
                        elif 'disappoint' in label or 'dejected' in label:
                            emotion_scores['disappointed'] = emotion_scores.get('disappointed', 0) + score
                        elif 'neu' in label or 'calm' in label:
                            emotion_scores['neutral'] = emotion_scores.get('neutral', 0) + score
                        elif 'fear' in label or 'sca' in label or 'afraid' in label:
                            emotion_scores['fear'] = emotion_scores.get('fear', 0) + score
                        elif 'anx' in label or 'worry' in label or 'nervous' in label:
                            emotion_scores['anxious'] = emotion_scores.get('anxious', 0) + score
                        elif 'dis' in label or 'revolt' in label:
                            emotion_scores['disgust'] = emotion_scores.get('disgust', 0) + score
                        elif 'sur' in label or 'amaz' in label:
                            emotion_scores['surprise'] = emotion_scores.get('surprise', 0) + score
                        elif 'confus' in label or 'perplex' in label:
                            emotion_scores['confused'] = emotion_scores.get('confused', 0) + score
                        elif 'trust' in label or 'confident' in label:
                            emotion_scores['confident'] = emotion_scores.get('confident', 0) + score
                        elif 'love' in label or 'tender' in label:
                            emotion_scores['loving'] = emotion_scores.get('loving', 0) + score
                        elif 'bore' in label or 'indiff' in label:
                            emotion_scores['bored'] = emotion_scores.get('bored', 0) + score
                        else:
                            # Map common emotion labels more broadly
                            if any(x in label for x in ['neutral', 'calm', 'normal']):
                                emotion_scores['neutral'] = emotion_scores.get('neutral', 0) + score
                            elif any(x in label for x in ['positive', 'good']):
                                emotion_scores['happy'] = emotion_scores.get('happy', 0) + score
                            elif any(x in label for x in ['negative', 'bad']):
                                emotion_scores['sad'] = emotion_scores.get('sad', 0) + score
                            else:
                                # Fallback to basic categories
                                if score > 0.05:
                                    emotion_scores['other'] = emotion_scores.get('other', 0) + score
                    
                    # Normalize scores
                    total_score = sum(emotion_scores.values())
                    if total_score > 0:
                        emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
                    
                    # Get dominant emotion - ensure we always have something
                    if emotion_scores:
                        dominant = max(emotion_scores.items(), key=lambda x: x[1])
                    else:
                        # If no emotions detected, use raw predictions
                        if raw_predictions:
                            top_raw = max(raw_predictions.items(), key=lambda x: x[1])
                            dominant = (self._map_raw_to_emotion(top_raw[0]), top_raw[1])
                        else:
                            dominant = ('neutral', 0.5)
                    
                    # Enhanced German emotion mapping
                    emotion_map = {
                        'angry': 'wütend',
                        'irritated': 'gereizt',
                        'disgust': 'angewidert',
                        'fear': 'ängstlich',
                        'anxious': 'besorgt',
                        'happy': 'glücklich',
                        'excited': 'aufgeregt',
                        'neutral': 'neutral',
                        'sad': 'traurig',
                        'disappointed': 'enttäuscht',
                        'surprise': 'überrascht',
                        'confused': 'verwirrt',
                        'confident': 'selbstbewusst',
                        'loving': 'liebevoll',
                        'bored': 'gelangweilt',
                        'other': 'sonstige'
                    }
                    
                    # Categorize emotion intensity
                    emotion_intensity = self._classify_emotion_intensity(dominant[1])
                    
                    # Determine emotion valence (positive/negative/neutral)
                    valence = self._determine_valence(dominant[0])
                    
                    # Analyze emotional complexity
                    complexity = self._analyze_emotional_complexity(emotion_scores)
                    
                    result = {
                        'timestamp': round(seg['start'], 1),
                        'start_time': round(seg['start'], 1),
                        'end_time': round(seg['end'], 1),
                        'duration': round(seg['end'] - seg['start'], 1),
                        'emotions': {k: round(v, 3) for k, v in emotion_scores.items() if v > 0.05},
                        'raw_predictions': {k: round(v, 3) for k, v in raw_predictions.items()},
                        'dominant_emotion': dominant[0],
                        'dominant_emotion_de': emotion_map.get(dominant[0], dominant[0]),
                        'emotion_intensity': emotion_intensity,
                        'emotion_valence': valence,
                        'emotional_complexity': complexity,
                        'confidence': round(dominant[1], 3),
                        'secondary_emotions': self._get_secondary_emotions(emotion_scores, dominant[0]),
                        'emotion_transition': self._detect_emotion_transition(results, dominant[0]) if results else None,
                        'ml_model': 'wav2vec2-xlsr-emotion',
                        'speech_detected': True
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"[SpeechEmotionML] Error processing segment {i}: {e}")
                    continue
                    
            # Add overall emotion summary
            overall_summary = self._create_overall_emotion_summary(results)
            
            print(f"[SpeechEmotionML] ✅ Analyzed {len(results)} segments")
            return {
                'segments': results, 
                'status': 'success', 
                'ml_model': 'wav2vec2-xlsr-emotion',
                'summary': overall_summary
            }
            
        except Exception as e:
            print(f"[SpeechEmotionML] Analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {'segments': [], 'error': str(e), 'status': 'error'}
            
    def process_batch_gpu(self, frames, frame_times):
        """Required abstract method - not used in this analyzer"""
        # This analyzer works with audio, not frames
        return []
    
    def _classify_emotion_intensity(self, confidence: float) -> str:
        """Classify emotion intensity based on confidence score"""
        if confidence > 0.8:
            return 'sehr stark'
        elif confidence > 0.6:
            return 'stark'
        elif confidence > 0.4:
            return 'moderat'
        elif confidence > 0.25:
            return 'schwach'
        else:
            return 'sehr schwach'
    
    def _determine_valence(self, emotion: str) -> str:
        """Determine emotional valence (positive/negative/neutral)"""
        positive_emotions = {'happy', 'excited', 'confident', 'loving', 'surprise'}
        negative_emotions = {'angry', 'irritated', 'sad', 'disappointed', 'fear', 'anxious', 'disgust', 'bored'}
        
        if emotion in positive_emotions:
            return 'positiv'
        elif emotion in negative_emotions:
            return 'negativ'
        else:
            return 'neutral'
    
    def _analyze_emotional_complexity(self, emotion_scores: Dict[str, float]) -> str:
        """Analyze emotional complexity based on score distribution"""
        # Count emotions with significant scores
        significant_emotions = sum(1 for score in emotion_scores.values() if score > 0.15)
        
        # Calculate entropy of emotion distribution
        entropy = 0
        for score in emotion_scores.values():
            if score > 0:
                entropy -= score * np.log(score + 1e-10)
        
        if significant_emotions == 1:
            return 'einfach'
        elif significant_emotions == 2:
            return 'gemischt'
        elif entropy > 1.5:
            return 'sehr komplex'
        else:
            return 'komplex'
    
    def _get_secondary_emotions(self, emotion_scores: Dict[str, float], dominant: str) -> List[str]:
        """Get secondary emotions with significant scores"""
        secondary = []
        for emotion, score in emotion_scores.items():
            if emotion != dominant and score > 0.2:
                secondary.append(emotion)
        return secondary[:3]  # Return top 3 secondary emotions
    
    def _detect_emotion_transition(self, previous_results: List[Dict], current_emotion: str) -> str:
        """Detect emotion transitions between segments"""
        if not previous_results:
            return None
        
        prev_emotion = previous_results[-1].get('dominant_emotion')
        if not prev_emotion:
            return None
        
        if prev_emotion == current_emotion:
            return 'stabil'
        
        # Analyze transition type
        prev_valence = self._determine_valence(prev_emotion)
        curr_valence = self._determine_valence(current_emotion)
        
        if prev_valence == 'negativ' and curr_valence == 'positiv':
            return 'Verbesserung'
        elif prev_valence == 'positiv' and curr_valence == 'negativ':
            return 'Verschlechterung'
        elif prev_valence == curr_valence:
            return 'Wechsel innerhalb Valenz'
        else:
            return 'Neutralisierung'
        
    def _cleanup_models(self):
        """Clean up models from memory"""
        if self.classifier:
            del self.classifier
            self.classifier = None
        self.models_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _create_overall_emotion_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Create overall emotion summary from all segments"""
        if not results:
            return {}
        
        # Aggregate all emotions
        all_emotions = {}
        valence_counts = {'positiv': 0, 'negativ': 0, 'neutral': 0}
        intensity_counts = {}
        dominant_emotions = []
        
        for result in results:
            # Collect dominant emotions
            dominant = result.get('dominant_emotion')
            if dominant:
                dominant_emotions.append(dominant)
            
            # Aggregate emotion scores
            for emotion, score in result.get('emotions', {}).items():
                all_emotions[emotion] = all_emotions.get(emotion, 0) + score
            
            # Count valence
            valence = result.get('emotion_valence')
            if valence:
                valence_counts[valence] += 1
            
            # Count intensity
            intensity = result.get('emotion_intensity')
            if intensity:
                intensity_counts[intensity] = intensity_counts.get(intensity, 0) + 1
        
        # Calculate dominant emotion across all segments
        from collections import Counter
        emotion_counter = Counter(dominant_emotions)
        most_common_emotion = emotion_counter.most_common(1)[0] if emotion_counter else ('neutral', 0)
        
        # Calculate average confidence
        avg_confidence = np.mean([r.get('confidence', 0) for r in results])
        
        # Determine overall emotional tone
        total_valence = sum(valence_counts.values())
        if total_valence > 0:
            positive_ratio = valence_counts['positiv'] / total_valence
            negative_ratio = valence_counts['negativ'] / total_valence
            
            if positive_ratio > 0.6:
                overall_tone = 'überwiegend positiv'
            elif negative_ratio > 0.6:
                overall_tone = 'überwiegend negativ'
            elif valence_counts['neutral'] / total_valence > 0.5:
                overall_tone = 'überwiegend neutral'
            else:
                overall_tone = 'gemischte Gefühle'
        else:
            overall_tone = 'keine Bewertung'
        
        # Emotional stability
        emotion_changes = sum(1 for i in range(1, len(results)) 
                            if results[i].get('dominant_emotion') != results[i-1].get('dominant_emotion'))
        stability_score = 1.0 - (emotion_changes / len(results)) if len(results) > 1 else 1.0
        
        if stability_score > 0.8:
            stability = 'sehr stabil'
        elif stability_score > 0.6:
            stability = 'stabil'
        elif stability_score > 0.4:
            stability = 'wechselhaft'
        else:
            stability = 'sehr wechselhaft'
        
        return {
            'total_segments': len(results),
            'overall_dominant_emotion': most_common_emotion[0],
            'overall_dominant_emotion_de': self._get_german_emotion(most_common_emotion[0]),
            'emotion_distribution': dict(emotion_counter),
            'overall_tone': overall_tone,
            'emotional_stability': stability,
            'stability_score': round(stability_score, 2),
            'average_confidence': round(avg_confidence, 3),
            'valence_distribution': valence_counts,
            'intensity_distribution': dict(intensity_counts),
            'unique_emotions_detected': len(all_emotions),
            'emotional_range': 'breit' if len(all_emotions) > 5 else 'begrenzt'
        }
    
    def _map_raw_to_emotion(self, raw_label: str) -> str:
        """Map raw model labels to standard emotions"""
        label_lower = raw_label.lower()
        
        # Common mappings
        if any(x in label_lower for x in ['ang', 'mad', 'rage', 'angry']):
            return 'angry'
        elif any(x in label_lower for x in ['hap', 'joy', 'happy']):
            return 'happy'
        elif any(x in label_lower for x in ['sad', 'unhap']):
            return 'sad'
        elif any(x in label_lower for x in ['neu', 'calm', 'neutral']):
            return 'neutral'
        elif any(x in label_lower for x in ['fear', 'sca', 'afraid']):
            return 'fear'
        elif any(x in label_lower for x in ['sur', 'amaz']):
            return 'surprise'
        elif any(x in label_lower for x in ['dis', 'revolt']):
            return 'disgust'
        else:
            return 'neutral'  # Default fallback
    
    def _get_german_emotion(self, emotion: str) -> str:
        """Get German translation for emotion"""
        emotion_map = {
            'angry': 'wütend',
            'irritated': 'gereizt',
            'disgust': 'angewidert',
            'fear': 'ängstlich',
            'anxious': 'besorgt',
            'happy': 'glücklich',
            'excited': 'aufgeregt',
            'neutral': 'neutral',
            'sad': 'traurig',
            'disappointed': 'enttäuscht',
            'surprise': 'überrascht',
            'confused': 'verwirrt',
            'confident': 'selbstbewusst',
            'loving': 'liebevoll',
            'bored': 'gelangweilt',
            'other': 'sonstige'
        }
        return emotion_map.get(emotion, emotion)