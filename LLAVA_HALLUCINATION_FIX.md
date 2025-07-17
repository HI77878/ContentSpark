# LLaVA Halluzination Fix - Dokumentation

**Datum:** 7. Juli 2025  
**Problem:** LLaVA-NeXT-Video halluzinierte stark (beschrieb Fahrräder, Autos, Fitnessstudios statt Mann beim Zähneputzen)  
**Status:** ✅ GELÖST

## 1. Ursachen der Halluzinationen

1. **Falsches Model**: Standard-Model statt DPO-Version
2. **Falsche Generation-Parameter**: `do_sample=True` mit `temperature=0.7`
3. **Tokenizer-Problem**: `padding_side` nicht auf "left" gesetzt
4. **Quantisierung**: 8-bit statt empfohlener 4-bit für DPO

## 2. Lösung

### 2.1 Model-Wechsel
```python
# ALT (halluziniert):
self.model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"

# NEU (akkurat):
self.model_id = "llava-hf/LLaVA-NeXT-Video-7B-DPO-hf"
```

### 2.2 Anti-Halluzination Settings
```python
# Tokenizer Fix
self.processor.tokenizer.padding_side = "left"

# Generation Parameters
output = self.model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False,      # KRITISCH: Muss False sein!
    temperature=0.0,      # KRITISCH: 0.0 für Determinismus
    top_p=1.0,
    repetition_penalty=1.0,
    num_beams=1          # Kein beam search
)
```

### 2.3 Bessere Quantisierung
```python
# 4-bit statt 8-bit
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

### 2.4 Frame Sampling
```python
# 32 frames uniform sampling (empfohlen)
num_frames = min(32, total_frames)
```

## 3. Ergebnisse

### Vorher (Halluzinationen):
```
"Person auf dem Fahrrad sitzt auf einem Fahrrad..."
"Person im Fitnessstudio trainiert..."
"Person im Auto..."
```

### Nachher (Akkurat):
```
"Shirtless man with tattooed chest and arms, wearing black beanie and jacket, 
holding toothbrush and brushing teeth in modern room with desk, plant, window.
Text overlay: 'WIR SIND DANN MANN' and 'WIR SIND DANN WERKEN'"
```

## 4. Performance

- **GPU Speicher**: Reduziert von 6.92GB auf 3.81GB
- **Analysezeit**: ~93 Sekunden für 49-Sekunden Video
- **Genauigkeit**: Keine Halluzinationen mehr

## 5. Wichtige Erkenntnisse

1. **DPO (Direct Preference Optimization)** Models sind speziell gegen Halluzinationen trainiert
2. **Deterministische Generation** (`do_sample=False`, `temperature=0.0`) ist kritisch
3. **Padding-Side** muss "left" sein für Video-Models
4. **4-bit Quantisierung** funktioniert besser mit DPO als 8-bit

## 6. Integration

Die Fixes wurden in folgende Dateien integriert:
- `/analyzers/llava_next_video_analyzer.py` - Hauptdatei aktualisiert
- `/analyzers/llava_next_video_analyzer_fixed.py` - Vollständig überarbeitete Version

## 7. Empfehlungen

1. **Immer DPO-Version verwenden** wenn verfügbar
2. **Generation-Settings nicht ändern** - sie sind optimal konfiguriert
3. **Bei längeren Videos** ggf. in Segmente aufteilen
4. **Prompts einfach halten** - zu komplexe Prompts können wieder zu Halluzinationen führen

## 8. Quellen

- GitHub Issue: hasanar1f/llava-hallunication-fix
- HuggingFace: llava-hf/LLaVA-NeXT-Video-7B-DPO-hf
- LLaVA Blog: DPO training reduces hallucinations significantly