# Analyse der fehlenden Daten - Leon Schliebach Video

## Zusammenfassung
Trotz 22/22 erfolgreicher Analyzer fehlen wichtige Personen-bezogene Daten, weil die entsprechenden Analyzer in der Produktion **deaktiviert** sind.

## 1. Fehlende Daten und verantwortliche Analyzer

### Gesichtserkennung
- **Benötigter Analyzer**: `face_detection` (GPUBatchFaceDetectionOptimized)
- **Status**: ❌ DEAKTIVIERT in `DISABLED_ANALYZERS`
- **Würde liefern**: Gesichtspositionen, Bounding Boxes, Anzahl Gesichter

### Emotionserkennung
- **Benötigte Analyzer**: 
  - `emotion_detection` (GPUBatchEmotionDetectionReal) 
  - `facial_details` (GPUBatchFacialDetails)
- **Status**: ❌ BEIDE DEAKTIVIERT
- **Würde liefern**: Gesichtsausdrücke, Emotionen (Freude, Trauer, etc.)

### Körpersprache & Pose
- **Benötigte Analyzer**:
  - `body_pose` (GPUBatchBodyPoseYOLO)
  - `body_language` (GPUBatchBodyLanguage)
- **Status**: ❌ BEIDE DEAKTIVIERT
- **Würde liefern**: Körperhaltung, Skelett-Keypoints, Bewegungsmuster

### Handgesten
- **Benötigte Analyzer**:
  - `hand_gesture` (GPUBatchHandGesture)
  - `gesture_recognition` (GPUBatchGestureRecognitionFixed)
- **Status**: ❌ BEIDE DEAKTIVIERT
- **Würde liefern**: Handpositionen, Gesten (Peace, Daumen hoch, etc.)

## 2. Vorhandene Daten als Ersatz

### Voice Pitch ✅ VORHANDEN
- **Analyzer**: `speech_rate` 
- **Daten**: 
  - `average_pitch`: 109.9 Hz
  - `pitch_range`: [77.3, 136.2] Hz
  - `pitch_std`: 9.7 (Variation)

### Sprach-Emotionen ✅ TEILWEISE VORHANDEN
- **Analyzer**: `speech_emotion`
- **Daten**: Emotionen aus Sprachanalyse (nicht visuell)
  - Beispiel: "angry" (13.5%), "disgust" (13.4%), etc.

### Eye Tracking ✅ VORHANDEN
- **Analyzer**: `eye_tracking`
- **Daten**: 
  - Blickrichtung
  - Pupillendistanz
  - Augensymmetrie

### Personen-Beschreibung ✅ TEILWEISE
- **Analyzer**: `qwen2_vl_temporal`
- **Daten**: Textuelle Beschreibungen der Person
  - Beispiel: "The person is shirtless and appears to be in a bathroom"

## 3. Grund für fehlende Daten

Die Analyzer wurden aus **Performance-Gründen** deaktiviert:
- Kommentar in gpu_groups_config.py: "DISABLED for performance"
- Ziel: <3x Realtime Processing erreichen
- Trade-off: Geschwindigkeit vs. Detailtiefe

## 4. Aktivierungsoptionen

Um die fehlenden Daten zu erhalten, müssten folgende Analyzer reaktiviert werden:

```python
# In configs/gpu_groups_config.py aus DISABLED_ANALYZERS entfernen:
- 'face_detection'
- 'emotion_detection' 
- 'body_pose'
- 'hand_gesture'
- 'facial_details'

# Dann zu GPU_ANALYZER_GROUPS hinzufügen, z.B.:
'stage2_gpu_medium': [
    'face_detection',  # NEU
    'emotion_detection',  # NEU
    # ... existing analyzers
]
```

## 5. Performance-Impact

Bei Aktivierung aller Personen-Analyzer:
- Geschätzte zusätzliche Zeit: +60-80 Sekunden
- Realtime-Faktor würde von 2.97x auf ~4.5x steigen
- GPU-Auslastung würde länger bei 95% bleiben

## Empfehlung

Für vollständige Personen-Analyse sollten selektiv Analyzer aktiviert werden:
1. `face_detection` - Basis für alle Gesichtsanalysen
2. `emotion_detection` - Visuelle Emotionserkennung
3. `body_pose` - Körpersprache und Bewegung

Die Audio-basierten Alternativen (speech_emotion, speech_rate) liefern bereits wichtige Daten, aber keine visuellen Informationen über Mimik und Gestik.