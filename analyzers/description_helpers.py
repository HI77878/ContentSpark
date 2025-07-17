#!/usr/bin/env python3
"""
Helper Functions for Natural Language Descriptions across all Analyzers
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

class DescriptionHelpers:
    """Centralized helper functions for generating natural language descriptions"""
    
    @staticmethod
    def face_to_description(face_data: Dict[str, Any]) -> str:
        """Convert face detection data to natural language description"""
        parts = []
        
        # Gender and age
        gender = face_data.get('gender', 'unbestimmt')
        age_range = face_data.get('age_range', 'unbestimmt')
        
        if gender != 'unbestimmt':
            parts.append(f"{gender[0].upper()}{gender[1:]}e Person")
        else:
            parts.append("Person")
            
        if age_range != 'unbestimmt':
            parts.append(age_range)
            
        # Facial features
        if face_data.get('has_beard'):
            parts.append("mit Bart")
        if face_data.get('has_glasses'):
            parts.append("mit Brille")
            
        # Position
        position = face_data.get('position', '')
        if position:
            parts.append(position)
            
        # Size/framing
        framing = face_data.get('framing', '')
        if framing:
            parts.append(f"({framing})")
            
        return ", ".join(parts) if parts else "Gesicht erkannt"
    
    @staticmethod
    def emotion_to_context(emotion: str, scene_type: str = None, intensity: float = 0.5) -> str:
        """Convert emotion to contextual description"""
        base_emotions = {
            'happy': 'zeigt Freude',
            'sad': 'wirkt traurig',
            'angry': 'zeigt Verärgerung',
            'surprise': 'ist überrascht',
            'fear': 'wirkt ängstlich',
            'disgust': 'zeigt Ekel',
            'neutral': 'bleibt neutral'
        }
        
        base_desc = base_emotions.get(emotion, f"zeigt {emotion}")
        
        # Add intensity
        if intensity > 0.8:
            intensity_desc = "deutlich"
        elif intensity > 0.5:
            intensity_desc = "sichtlich"
        else:
            intensity_desc = "leicht"
            
        # Add scene context
        if scene_type == 'product_presentation' and emotion == 'happy':
            return f"{intensity_desc} begeistert vom Produkt"
        elif scene_type == 'tutorial' and emotion == 'neutral':
            return f"erklärt {intensity_desc} konzentriert"
        elif scene_type == 'review' and emotion == 'surprise':
            return f"{intensity_desc} überrascht von der Qualität"
        else:
            return f"{intensity_desc} {base_desc}"
    
    @staticmethod
    def movement_to_purpose(movement_type: str, detected_objects: List[str], 
                          movement_direction: str = None) -> str:
        """Convert camera movement to purpose description"""
        # Static camera
        if movement_type == 'static':
            if 'person' in detected_objects:
                return "hält Fokus auf sprechender Person"
            elif any(obj in detected_objects for obj in ['bottle', 'product']):
                return "zeigt Produkt in statischer Einstellung"
            else:
                return "bietet ruhigen Blick auf die Szene"
                
        # Pan movements
        if movement_type in ['pan', 'pan_left', 'pan_right']:
            if 'person' in detected_objects:
                return "folgt der Bewegung der Person"
            elif movement_direction:
                return f"schwenkt {movement_direction} über die Szene"
            else:
                return "erkundet den Raum horizontal"
                
        # Zoom movements
        if movement_type == 'zoom_in':
            if any(obj in detected_objects for obj in ['bottle', 'product']):
                return "zoomt für Produktdetails heran"
            elif 'face' in detected_objects:
                return "kommt für emotionale Nähe näher"
            else:
                return "betont wichtige Details"
                
        if movement_type == 'zoom_out':
            return "gibt mehr Kontext zur Gesamtszene"
            
        # Tilt movements
        if movement_type in ['tilt', 'tilt_up', 'tilt_down']:
            if 'person' in detected_objects:
                return "zeigt Person von oben bis unten"
            else:
                return "erkundet vertikale Elemente"
                
        # Dynamic movements
        if movement_type in ['tracking', 'follow']:
            return "verfolgt die Aktion im Bild"
            
        return "bewegt sich für visuelle Dynamik"
    
    @staticmethod
    def objects_to_scene_description(objects: List[Dict[str, Any]]) -> str:
        """Convert detected objects to scene description"""
        if not objects:
            return "Szene ohne erkennbare Objekte"
            
        # Count object types
        object_counts = {}
        for obj in objects:
            obj_class = obj.get('object_class', 'unknown')
            if obj_class not in object_counts:
                object_counts[obj_class] = 0
            object_counts[obj_class] += 1
            
        # Determine scene type
        has_person = 'person' in object_counts
        has_bottle = 'bottle' in object_counts
        has_furniture = any(obj in object_counts for obj in ['chair', 'couch', 'bed', 'table'])
        has_tech = any(obj in object_counts for obj in ['laptop', 'tv', 'cell phone', 'keyboard'])
        
        parts = []
        
        # Primary subject
        if has_person:
            person_count = object_counts['person']
            if person_count == 1:
                parts.append("Eine Person")
            else:
                parts.append(f"{person_count} Personen")
                
        # Setting
        if has_furniture:
            if 'couch' in object_counts or 'chair' in object_counts:
                parts.append("in Wohnumgebung")
            elif 'bed' in object_counts:
                parts.append("im Schlafzimmer")
            elif 'dining table' in object_counts:
                parts.append("am Esstisch")
                
        # Objects of interest
        if has_bottle:
            bottle_count = object_counts['bottle']
            if bottle_count == 1:
                parts.append("mit einer Flasche")
            else:
                parts.append(f"mit {bottle_count} Flaschen")
                
        if has_tech:
            tech_items = [obj for obj in ['laptop', 'tv', 'cell phone'] if obj in object_counts]
            if len(tech_items) == 1:
                parts.append(f"und {DescriptionHelpers.translate_object(tech_items[0])}")
            elif len(tech_items) > 1:
                parts.append("und technischen Geräten")
                
        return " ".join(parts) if parts else "Szene mit verschiedenen Objekten"
    
    @staticmethod
    def translate_object(object_name: str) -> str:
        """Translate object names to German"""
        translations = {
            'person': 'Person',
            'bottle': 'Flasche',
            'cup': 'Tasse',
            'chair': 'Stuhl',
            'couch': 'Sofa',
            'bed': 'Bett',
            'tv': 'Fernseher',
            'laptop': 'Laptop',
            'cell phone': 'Smartphone',
            'potted plant': 'Topfpflanze',
            'dining table': 'Esstisch',
            'book': 'Buch',
            'clock': 'Uhr',
            'keyboard': 'Tastatur',
            'mouse': 'Maus',
            'backpack': 'Rucksack',
            'handbag': 'Handtasche',
            'car': 'Auto',
            'bicycle': 'Fahrrad',
            'skateboard': 'Skateboard'
        }
        return translations.get(object_name, object_name)
    
    @staticmethod
    def color_to_description(dominant_colors: List[Tuple[str, float]]) -> str:
        """Convert color analysis to description"""
        if not dominant_colors:
            return "mit neutralen Farben"
            
        color_names = {
            'red': 'rot',
            'green': 'grün',
            'blue': 'blau',
            'yellow': 'gelb',
            'orange': 'orange',
            'purple': 'lila',
            'pink': 'rosa',
            'brown': 'braun',
            'black': 'schwarz',
            'white': 'weiß',
            'gray': 'grau'
        }
        
        # Get top colors
        top_colors = []
        for color, percentage in dominant_colors[:2]:
            if percentage > 0.2:  # At least 20% of image
                german_color = color_names.get(color, color)
                if percentage > 0.5:
                    top_colors.append(f"dominantes {german_color}")
                else:
                    top_colors.append(german_color)
                    
        if not top_colors:
            return "mit gemischten Farben"
        elif len(top_colors) == 1:
            return f"in {top_colors[0]}en Tönen"
        else:
            return f"in {' und '.join(top_colors)}en Tönen"
    
    @staticmethod
    def quality_to_description(quality_score: float, quality_aspects: Dict[str, float]) -> str:
        """Convert quality metrics to description"""
        if quality_score > 0.9:
            base = "Exzellente Videoqualität"
        elif quality_score > 0.7:
            base = "Gute Videoqualität"
        elif quality_score > 0.5:
            base = "Durchschnittliche Qualität"
        else:
            base = "Verbesserungswürdige Qualität"
            
        # Add specific aspects
        issues = []
        if quality_aspects.get('sharpness', 1.0) < 0.5:
            issues.append("leicht unscharf")
        if quality_aspects.get('brightness', 0.5) < 0.3:
            issues.append("zu dunkel")
        elif quality_aspects.get('brightness', 0.5) > 0.8:
            issues.append("überbelichtet")
        if quality_aspects.get('stability', 1.0) < 0.7:
            issues.append("verwackelt")
            
        if issues:
            return f"{base} ({', '.join(issues)})"
        else:
            return base
    
    @staticmethod
    def gesture_to_description(gesture_type: str, hand_position: str = None) -> str:
        """Convert gesture detection to description"""
        gestures = {
            'pointing': 'zeigt auf etwas',
            'thumbs_up': 'Daumen hoch',
            'thumbs_down': 'Daumen runter',
            'ok_sign': 'OK-Zeichen',
            'peace_sign': 'Peace-Zeichen',
            'wave': 'winkt',
            'open_palm': 'offene Handfläche',
            'fist': 'geballte Faust',
            'clapping': 'klatscht'
        }
        
        base = gestures.get(gesture_type, f"macht {gesture_type}-Geste")
        
        if hand_position:
            position_map = {
                'left': 'mit linker Hand',
                'right': 'mit rechter Hand',
                'both': 'mit beiden Händen'
            }
            position = position_map.get(hand_position, '')
            if position:
                return f"{base} {position}"
                
        return base
    
    @staticmethod
    def combine_descriptions(descriptions: List[str], connector: str = "und") -> str:
        """Combine multiple descriptions naturally"""
        if not descriptions:
            return ""
        elif len(descriptions) == 1:
            return descriptions[0]
        elif len(descriptions) == 2:
            return f"{descriptions[0]} {connector} {descriptions[1]}"
        else:
            return f"{', '.join(descriptions[:-1])} {connector} {descriptions[-1]}"
    
    @staticmethod
    def timestamp_to_phase(timestamp: float, total_duration: float) -> str:
        """Convert timestamp to video phase description"""
        if total_duration <= 0:
            return "unbekannte Phase"
            
        progress = timestamp / total_duration
        
        if progress < 0.1:
            return "ganz am Anfang"
        elif progress < 0.25:
            return "in der Einleitung"
        elif progress < 0.75:
            return "im Hauptteil"
        elif progress < 0.9:
            return "gegen Ende"
        else:
            return "im Abschluss"
    
    @staticmethod
    def confidence_to_qualifier(confidence: float) -> str:
        """Convert confidence score to German qualifier"""
        if confidence > 0.95:
            return "eindeutig"
        elif confidence > 0.85:
            return "sehr wahrscheinlich"
        elif confidence > 0.7:
            return "wahrscheinlich"
        elif confidence > 0.5:
            return "möglicherweise"
        else:
            return "unsicher"