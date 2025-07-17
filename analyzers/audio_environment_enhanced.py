#!/usr/bin/env python3
"""
Enhanced Audio Environment Analysis with temporal segmentation
"""
import os
os.environ["OPENCV_FFMPEG_THREAD_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"

import numpy as np
import logging
from typing import List, Dict, Any
from analyzers.base_analyzer import GPUBatchAnalyzer
import subprocess
import tempfile
import json

logger = logging.getLogger(__name__)

class AudioEnvironmentEnhanced(GPUBatchAnalyzer):
    def __init__(self):
        super().__init__(batch_size=1)
        self.device = 'cpu'
        self.window_duration = 3.0  # Analyze in 3-second windows
        print("[AudioEnvironment-Enhanced] Initializing enhanced audio environment analysis")
        
        # Check for required libraries
        try:
            import librosa
            self.librosa = librosa
            self.librosa_available = True
            print("✅ Librosa available for advanced audio analysis")
        except ImportError:
            self.librosa_available = False
            print("⚠️ Librosa not available, using basic audio analysis")
    
    def extract_and_segment_audio(self, video_path: str) -> Dict[str, Any]:
        """Extract audio and analyze in temporal segments"""
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        try:
            # Extract audio using ffmpeg
            ffmpeg_cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-ar', '22050',  # Sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite
                temp_audio_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.warning(f"FFmpeg extraction failed: {result.stderr}")
                return self.create_basic_segments(video_path)
            
            # Analyze audio in segments
            if self.librosa_available:
                return self.analyze_audio_segments(temp_audio_path)
            else:
                return self.create_basic_segments_from_audio(temp_audio_path)
                
        except Exception as e:
            logger.error(f"Audio extraction error: {e}")
            return self.create_basic_segments(video_path)
        
        finally:
            # Clean up
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    
    def analyze_audio_segments(self, audio_path: str) -> Dict[str, Any]:
        """Analyze audio in temporal segments using librosa"""
        
        # Load audio
        y, sr = self.librosa.load(audio_path, sr=22050)
        duration = len(y) / sr
        
        # Calculate number of segments
        num_segments = int(np.ceil(duration / self.window_duration))
        
        segments = []
        
        for i in range(num_segments):
            # Calculate segment boundaries
            start_time = i * self.window_duration
            end_time = min((i + 1) * self.window_duration, duration)
            
            # Extract segment audio
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = y[start_sample:end_sample]
            
            if len(segment_audio) < 1024:  # Too short
                continue
            
            # Analyze segment
            segment_analysis = self.analyze_audio_segment(
                segment_audio, sr, start_time, end_time
            )
            segments.append(segment_analysis)
        
        # Add overall summary
        overall_analysis = self.analyze_full_audio(y, sr)
        
        return {
            'segments': segments,
            'overall_analysis': overall_analysis,
            'duration': duration
        }
    
    def analyze_audio_segment(self, audio: np.ndarray, sr: int, 
                            start_time: float, end_time: float) -> Dict[str, Any]:
        """Analyze a single audio segment"""
        
        # Basic features
        rms = float(np.sqrt(np.mean(audio**2)))
        
        # Spectral features
        spectral_centroid = float(np.mean(
            self.librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        ))
        
        # Zero crossing rate (indicates speech vs music)
        zcr = float(np.mean(
            self.librosa.feature.zero_crossing_rate(audio)[0]
        ))
        
        # MFCC for acoustic characteristics
        mfccs = self.librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1).tolist()
        
        # Spectral rolloff
        rolloff = float(np.mean(
            self.librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        ))
        
        # Classify environment for this segment
        environment = self.classify_segment_environment(
            rms, spectral_centroid, zcr, rolloff
        )
        
        # Detect prominent sounds
        sounds = self.detect_segment_sounds(
            audio, sr, spectral_centroid, zcr
        )
        
        # Create comprehensive description for this segment
        sounds_desc = ', '.join([s['description'] for s in sounds[:3]])  # Top 3 sounds
        description = f"{environment['description']}. Audio: {sounds_desc}. {self.describe_audio_characteristics(rms, spectral_centroid, zcr)}"
        
        return {
            'timestamp': start_time,
            'start_time': start_time,
            'end_time': end_time,
            'segment_id': f'audio_env_{int(start_time * 10)}',
            'description': description,
            'duration': end_time - start_time,
            'environment_type': environment['type'],
            'environment_description': environment['description'],
            'environment_subcategory': environment.get('subcategory', 'undefined'),
            'confidence': environment['confidence'],
            'acoustic_features': {
                'energy': rms,
                'spectral_centroid': spectral_centroid,
                'zero_crossing_rate': zcr,
                'spectral_rolloff': rolloff
            },
            'detected_sounds': sounds,
            'sound_categories': self._categorize_sounds(sounds),
            'audio_characteristics': self.describe_audio_characteristics(
                rms, spectral_centroid, zcr
            ),
            'noise_level': self._classify_noise_level(rms),
            'clarity_score': self._calculate_clarity_score(spectral_centroid, zcr, rolloff)
        }
    
    def classify_segment_environment(self, rms: float, centroid: float, 
                                   zcr: float, rolloff: float) -> Dict[str, Any]:
        """Classify environment based on acoustic features with detailed categories"""
        
        # Bedroom/sleep room (very quiet, low frequencies)
        if rms < 0.02 and centroid < 1500 and zcr < 0.05:
            return {
                'type': 'bedroom_quiet',
                'description': 'Schlafzimmer oder sehr ruhiger Raum',
                'confidence': 0.85,
                'subcategory': 'private_space'
            }
        
        # Living room/home (moderate activity)
        elif rms < 0.08 and 1500 < centroid < 2500 and zcr < 0.15:
            return {
                'type': 'living_room',
                'description': 'Wohnzimmer oder häusliche Umgebung',
                'confidence': 0.8,
                'subcategory': 'home_environment'
            }
        
        # Kitchen (clanking, water sounds)
        elif 0.05 < rms < 0.12 and 2000 < centroid < 4000 and zcr > 0.1:
            return {
                'type': 'kitchen',
                'description': 'Küche mit typischen Kochgeräuschen',
                'confidence': 0.75,
                'subcategory': 'home_activity'
            }
        
        # Bathroom (echo, water)
        elif rms < 0.1 and 1000 < centroid < 3000 and rolloff > 3500:
            return {
                'type': 'bathroom',
                'description': 'Badezimmer mit Hall und Wassergeräuschen',
                'confidence': 0.7,
                'subcategory': 'private_space'
            }
        
        # Office/workspace (keyboard, moderate noise)
        elif 0.03 < rms < 0.08 and 1800 < centroid < 3000:
            return {
                'type': 'office',
                'description': 'Büro oder Arbeitsumgebung',
                'confidence': 0.75,
                'subcategory': 'work_environment'
            }
        
        # Cafe/restaurant (voices, dishes)
        elif 0.08 < rms < 0.15 and 2000 < centroid < 3500 and zcr > 0.12:
            return {
                'type': 'cafe_restaurant',
                'description': 'Café oder Restaurant mit Stimmengewirr',
                'confidence': 0.8,
                'subcategory': 'public_indoor'
            }
        
        # Shopping mall/store
        elif 0.06 < rms < 0.14 and centroid > 2500 and rolloff > 4000:
            return {
                'type': 'shopping_mall',
                'description': 'Einkaufszentrum oder Geschäft',
                'confidence': 0.75,
                'subcategory': 'public_indoor'
            }
        
        # Street/traffic
        elif rms > 0.1 and centroid > 3000 and rolloff > 4500:
            return {
                'type': 'street_traffic',
                'description': 'Straße mit Verkehrsgeräuschen',
                'confidence': 0.85,
                'subcategory': 'urban_outdoor'
            }
        
        # Park/nature
        elif rms < 0.08 and 1500 < centroid < 3000 and zcr < 0.1:
            return {
                'type': 'park_nature',
                'description': 'Park oder natürliche Umgebung',
                'confidence': 0.7,
                'subcategory': 'natural_outdoor'
            }
        
        # Beach/water
        elif centroid < 2000 and rolloff > 3000 and 0.05 < rms < 0.12:
            return {
                'type': 'beach_water',
                'description': 'Strand oder Wasserumgebung',
                'confidence': 0.7,
                'subcategory': 'natural_outdoor'
            }
        
        # Gym/sports facility
        elif rms > 0.12 and 2500 < centroid < 4000:
            return {
                'type': 'gym_sports',
                'description': 'Fitnessstudio oder Sportanlage',
                'confidence': 0.75,
                'subcategory': 'activity_space'
            }
        
        # Studio/professional recording
        elif 0.04 < rms < 0.10 and 1800 < centroid < 3500 and zcr < 0.08:
            return {
                'type': 'studio_professional',
                'description': 'professionelles Aufnahmestudio',
                'confidence': 0.85,
                'subcategory': 'controlled_environment'
            }
        
        # Car interior
        elif 0.06 < rms < 0.11 and 1000 < centroid < 2500 and rolloff < 3500:
            return {
                'type': 'car_interior',
                'description': 'Fahrzeuginnenraum',
                'confidence': 0.7,
                'subcategory': 'vehicle'
            }
        
        # Concert/club (very loud, wide spectrum)
        elif rms > 0.18 and centroid > 2000:
            return {
                'type': 'concert_club',
                'description': 'Konzert oder Club mit lauter Musik',
                'confidence': 0.85,
                'subcategory': 'entertainment_venue'
            }
        
        # Default/mixed
        else:
            return {
                'type': 'mixed_environment',
                'description': 'gemischte oder nicht eindeutige Umgebung',
                'confidence': 0.5,
                'subcategory': 'undefined'
            }
    
    def detect_segment_sounds(self, audio: np.ndarray, sr: int,
                            centroid: float, zcr: float) -> List[Dict[str, Any]]:
        """Detect prominent sounds in segment with detailed categorization"""
        sounds = []
        rms = float(np.sqrt(np.mean(audio**2)))
        
        # Enhanced speech detection
        if 300 < centroid < 2200 and 0.03 < zcr < 0.18:
            speech_type = 'männliche Stimme' if centroid < 1200 else 'weibliche Stimme'
            sounds.append({
                'type': 'speech',
                'description': speech_type,
                'confidence': 0.8 if 0.05 < zcr < 0.15 else 0.6
            })
        elif 2200 < centroid < 3000 and zcr > 0.15:
            sounds.append({
                'type': 'speech', 
                'description': 'Kinderstimme',
                'confidence': 0.7
            })
        
        # Detailed music detection
        if 400 < centroid < 1500 and zcr < 0.08 and rms > 0.05:
            sounds.append({
                'type': 'music',
                'description': 'Bass/tiefe Instrumente',
                'confidence': 0.75
            })
        elif 1500 < centroid < 4000 and zcr < 0.1 and rms > 0.04:
            sounds.append({
                'type': 'music',
                'description': 'melodische Musik',
                'confidence': 0.8
            })
        elif centroid > 4000 and zcr < 0.12:
            sounds.append({
                'type': 'music',
                'description': 'hohe Instrumente/Synthesizer',
                'confidence': 0.7
            })
        
        # Environmental sounds
        if centroid > 5500:
            sounds.append({
                'type': 'environmental',
                'description': 'metallische/gläserne Geräusche',
                'confidence': 0.7
            })
        
        if centroid < 400 and rms > 0.08:
            sounds.append({
                'type': 'environmental', 
                'description': 'dumpfe Schlaggeräusche',
                'confidence': 0.75
            })
        
        # Water sounds
        if 800 < centroid < 2000 and zcr > 0.2:
            sounds.append({
                'type': 'water',
                'description': 'Wassergeräusche',
                'confidence': 0.7
            })
        
        # Traffic/machinery
        if rms > 0.1 and 1000 < centroid < 3000:
            sounds.append({
                'type': 'machinery',
                'description': 'Verkehr oder Maschinen',
                'confidence': 0.75
            })
        
        # Birds/nature
        if 2500 < centroid < 4500 and zcr > 0.25 and rms < 0.08:
            sounds.append({
                'type': 'nature',
                'description': 'Vogelgezwitscher',
                'confidence': 0.7
            })
        
        # Wind
        if centroid < 1000 and 0.02 < rms < 0.06 and zcr > 0.1:
            sounds.append({
                'type': 'nature',
                'description': 'Windgeräusche', 
                'confidence': 0.65
            })
        
        # Applause/clapping
        if zcr > 0.3 and 1500 < centroid < 3500 and rms > 0.06:
            sounds.append({
                'type': 'human_activity',
                'description': 'Applaus oder Klatschen',
                'confidence': 0.75
            })
        
        # Footsteps
        if 0.04 < rms < 0.09 and zcr > 0.15 and 500 < centroid < 2000:
            sounds.append({
                'type': 'human_activity',
                'description': 'Schritte',
                'confidence': 0.6
            })
        
        # Silence/very quiet
        if rms < 0.01:
            sounds.append({
                'type': 'silence',
                'description': 'Stille',
                'confidence': 0.9
            })
        
        # Background noise
        if not sounds and rms > 0.02:
            sounds.append({
                'type': 'ambient',
                'description': 'Umgebungsgeräusche',
                'confidence': 0.5
            })
        
        return sounds if sounds else [{'type': 'unknown', 'description': 'nicht identifiziert', 'confidence': 0.3}]
    
    def describe_audio_characteristics(self, rms: float, 
                                     centroid: float, zcr: float) -> str:
        """Create descriptive text of audio characteristics"""
        parts = []
        
        # Detailed energy level
        if rms < 0.02:
            parts.append('sehr leise')
        elif rms < 0.05:
            parts.append('leise')
        elif rms < 0.08:
            parts.append('normale Lautstärke')
        elif rms < 0.12:
            parts.append('erhöhte Lautstärke')
        elif rms < 0.16:
            parts.append('laut')
        else:
            parts.append('sehr laut')
        
        # Detailed frequency content
        if centroid < 800:
            parts.append('sehr tiefe Frequenzen')
        elif centroid < 1500:
            parts.append('tiefe Frequenzen')
        elif centroid < 2500:
            parts.append('mittlere Frequenzen')
        elif centroid < 4000:
            parts.append('helle Klangfarbe')
        elif centroid < 6000:
            parts.append('sehr helle Frequenzen')
        else:
            parts.append('extrem hohe Frequenzen')
        
        # Activity level with more detail
        if zcr < 0.03:
            parts.append('sehr ruhig')
        elif zcr < 0.08:
            parts.append('ruhige Atmosphäre')
        elif zcr < 0.15:
            parts.append('moderate Aktivität')
        elif zcr < 0.25:
            parts.append('lebhafte Umgebung')
        else:
            parts.append('sehr viel Bewegung')
        
        return ', '.join(parts)
    
    def _categorize_sounds(self, sounds: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize detected sounds"""
        categories = {}
        for sound in sounds:
            cat = sound['type']
            categories[cat] = categories.get(cat, 0) + 1
        return categories
    
    def _classify_noise_level(self, rms: float) -> Dict[str, Any]:
        """Classify noise level in decibels approximation"""
        # Rough dB approximation from RMS
        db_approx = 20 * np.log10(rms + 1e-6) + 60  # Rough calibration
        
        if db_approx < 30:
            return {'level': 'sehr leise', 'db_estimate': db_approx, 'description': 'Flüsterlautstärke'}
        elif db_approx < 50:
            return {'level': 'leise', 'db_estimate': db_approx, 'description': 'ruhige Unterhaltung'}
        elif db_approx < 70:
            return {'level': 'normal', 'db_estimate': db_approx, 'description': 'normale Sprachlautstärke'}
        elif db_approx < 85:
            return {'level': 'laut', 'db_estimate': db_approx, 'description': 'laute Umgebung'}
        else:
            return {'level': 'sehr laut', 'db_estimate': db_approx, 'description': 'potentiell schädlich'}
    
    def _calculate_clarity_score(self, centroid: float, zcr: float, rolloff: float) -> float:
        """Calculate audio clarity score (0-1)"""
        # Higher score = clearer audio
        clarity = 0.0
        
        # Good frequency distribution
        if 1000 < centroid < 4000:
            clarity += 0.4
        elif 500 < centroid < 5000:
            clarity += 0.2
        
        # Moderate activity (not too quiet, not too chaotic)
        if 0.05 < zcr < 0.2:
            clarity += 0.3
        elif 0.02 < zcr < 0.25:
            clarity += 0.15
        
        # Good spectral rolloff
        if 2000 < rolloff < 6000:
            clarity += 0.3
        elif 1000 < rolloff < 8000:
            clarity += 0.15
        
        return min(1.0, clarity)
    
    def analyze_full_audio(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze full audio for overall characteristics"""
        
        # Overall features
        overall_rms = float(np.sqrt(np.mean(y**2)))
        overall_centroid = float(np.mean(
            self.librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        ))
        
        # Dynamic range
        dynamic_range = float(np.max(np.abs(y)) - np.min(np.abs(y)))
        
        # Tempo detection
        tempo, _ = self.librosa.beat.beat_track(y=y, sr=sr)
        
        return {
            'average_energy': overall_rms,
            'average_spectral_centroid': overall_centroid,
            'dynamic_range': dynamic_range,
            'tempo': float(tempo),
            'overall_description': self.create_overall_description(
                overall_rms, overall_centroid, tempo
            )
        }
    
    def create_overall_description(self, rms: float, centroid: float, tempo: float) -> str:
        """Create overall audio description"""
        parts = []
        
        # Overall loudness
        if rms < 0.05:
            parts.append('insgesamt leise Aufnahme')
        elif rms > 0.1:
            parts.append('insgesamt laute Aufnahme')
        
        # Frequency characteristics
        if centroid < 2000:
            parts.append('tendenziell tiefe Frequenzen')
        elif centroid > 4000:
            parts.append('tendenziell hohe Frequenzen')
        
        # Rhythm
        if tempo > 0:
            if tempo < 80:
                parts.append('langsamer Rhythmus')
            elif tempo > 120:
                parts.append('schneller Rhythmus')
        
        return ', '.join(parts) if parts else 'ausgeglichene Audiocharakteristik'
    
    def create_basic_segments(self, video_path: str) -> Dict[str, Any]:
        """Create basic segments when advanced analysis not available"""
        # Get video duration
        try:
            ffprobe_cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'json', video_path
            ]
            
            result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                duration = float(info.get('format', {}).get('duration', 30))
            else:
                duration = 30  # Default
                
        except:
            duration = 30
        
        # Create segments every 3 seconds
        num_segments = int(np.ceil(duration / self.window_duration))
        segments = []
        
        for i in range(num_segments):
            start_time = i * self.window_duration
            end_time = min((i + 1) * self.window_duration, duration)
            
            segments.append({
                'timestamp': start_time,
                'start_time': start_time,
                'end_time': end_time,
                'segment_id': f'audio_env_{int(start_time * 10)}',
                'description': f'Audio-Umgebung im Segment {start_time:.1f}s-{end_time:.1f}s. Basis-Analyse ohne erweiterte Features.',
                'duration': end_time - start_time,
                'environment_type': 'unknown',
                'environment_description': 'Audio-Umgebung',
                'confidence': 0.3,
                'detected_sounds': [{'type': 'unknown', 'description': 'nicht analysiert', 'confidence': 0.3}],
                'sound_categories': {'unknown': 1},
                'audio_characteristics': 'Basis-Analyse',
                'noise_level': {'level': 'unbekannt', 'db_estimate': 0, 'description': 'nicht gemessen'},
                'clarity_score': 0.5
            })
        
        return {
            'segments': segments,
            'overall_analysis': {
                'overall_description': 'Basis-Audioanalyse durchgeführt'
            },
            'duration': duration
        }
    
    def create_basic_segments_from_audio(self, audio_path: str) -> Dict[str, Any]:
        """Create basic segments from audio file without librosa"""
        # Similar to create_basic_segments but tries to get more info
        return self.create_basic_segments(audio_path)
    
    def process_batch_gpu(self, frames: List[np.ndarray], frame_times: List[float]) -> Dict[str, Any]:
        """Not used for audio analysis"""
        return {'segments': []}
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """Main analysis entry point"""
        print(f"[AudioEnvironment-Enhanced] Analyzing {video_path}")
        
        try:
            # Extract and analyze audio in segments
            analysis = self.extract_and_segment_audio(video_path)
            
            segments = analysis.get('segments', [])
            
            if segments:
                print(f"[AudioEnvironment-Enhanced] Created {len(segments)} temporal segments")
                # Sample first segment
                first = segments[0]
                print(f"   First segment: {first['environment_description']} at {first['timestamp']:.1f}s")
            
            return {'segments': segments}
            
        except Exception as e:
            logger.error(f"Audio environment analysis error: {e}")
            # Return at least one segment
            return {
                'segments': [{
                    'timestamp': 0.0,
                    'start_time': 0.0,
                    'end_time': 3.0,
                    'segment_id': 'audio_env_error',
                    'description': f'Audio-Umgebungsanalyse fehlgeschlagen: {str(e)}',
                    'environment_type': 'error',
                    'environment_description': 'Analyse fehlgeschlagen',
                    'error': str(e)
                }]
            }