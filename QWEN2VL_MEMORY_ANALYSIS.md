# 🔍 Qwen2-VL Memory Problem Analysis

## 📊 Aktueller Code-Status

### Version und Konfiguration
- **Version**: 2.0_ULTRA_FAST
- **Model**: Qwen/Qwen2-VL-7B-Instruct (7B Parameter)
- **Dtype**: torch.float16
- **Device Map**: "cuda:0" (direkt auf GPU)
- **Attention**: Standard (Flash Attention nicht verfügbar)

### Segment-Parameter
- **segment_duration**: 2.0 Sekunden
- **frames_per_segment**: 3 Frames
- **overlap_ratio**: 0.0 (kein Overlap)
- **target_resolution**: (336, 336)
- **min_pixels**: 256 * 28 * 28 = ~200k
- **max_pixels**: 1280 * 28 * 28 = ~1M

## 🚨 Memory-Probleme Identifiziert

### 1. **Model Loading Phase**
```python
# Zeile 107-110
self.model = Qwen2VLForConditionalGeneration.from_pretrained(
    self.model_name,
    **model_kwargs
)
```
**Problem**: Das Model wird während dem Loading der Checkpoint Shards geladen und verursacht OOM beim 3. Shard (60%). Das bedeutet, dass allein das Model-Loading schon zu viel Memory braucht.

### 2. **Keine Memory-Map Optimierung**
```python
# Zeile 93
"device_map": "cuda:0",  # Direct GPU placement for speed
```
**Problem**: Das gesamte Model wird direkt auf GPU geladen statt intelligent über device_map="auto" verteilt zu werden.

### 3. **Frame Grid Creation**
```python
# Zeile 168-208: create_frame_grid()
```
**Problem**: Erstellt ein horizontales Grid aus 3 Frames. Bei (336, 336) sind das:
- Single Frame: 336 * 336 * 3 = 338KB
- 3-Frame Grid: 1008 * 336 * 3 = 1MB pro Grid
- Bei vielen Segmenten können sich diese akkumulieren

### 4. **Batch Processing nicht implementiert**
```python
# Zeile 399-425: Segmente werden sequenziell verarbeitet
for i, seg_start in enumerate(segment_starts):
    # Jedes Segment einzeln
```
**Problem**: Jedes Segment wird einzeln verarbeitet, keine Batch-Optimierung.

### 5. **Memory Cleanup nur alle 5 Segmente**
```python
# Zeile 422-424
if i % 5 == 0 and i > 0:
    torch.cuda.empty_cache()
    gc.collect()
```
**Problem**: GPU Memory wird nur alle 5 Segmente geleert, nicht nach jedem Segment.

### 6. **Warmup Model verbraucht Memory**
```python
# Zeile 132-167: _warmup_model()
```
**Problem**: Warmup erstellt Test-Frames und generiert Text, aber Memory wird danach nicht vollständig freigegeben.

### 7. **Processor und Tokenizer zusätzlich**
```python
# Zeile 79-84: Processor
# Zeile 115-118: Tokenizer
```
**Problem**: Beide werden zusätzlich zum Model geladen und verbrauchen Memory.

## 📈 Memory-Verbrauch Schätzung

### Model Loading
- **Qwen2-VL-7B in FP16**: ~14GB
- **Processor**: ~500MB
- **Tokenizer**: ~100MB
- **Total beim Loading**: ~15GB

### Runtime Memory
- **Model Inference**: +2-4GB
- **Frame Grids**: ~1MB pro Segment
- **Activation Memory**: +1-2GB
- **Total Runtime**: ~18-20GB

## 🎯 Fehler-Details

Der OOM tritt beim **Loading der Checkpoint Shards** auf:
```
Loading checkpoint shards: 60%|██████| 3/5 [00:14<00:09, 4.72s/it]
CUDA error: out of memory
```

Das bedeutet: Das Model passt nicht mal komplett in den GPU Memory (44.5GB verfügbar).

## 🔧 Konkrete Fix-Vorschläge

### 1. **Use device_map="auto"**
```python
model_kwargs = {
    "torch_dtype": torch.float16,
    "device_map": "auto",  # Statt "cuda:0"
    "max_memory": {0: "20GB"},  # Limit GPU usage
    "offload_folder": "/tmp/offload",
}
```

### 2. **Load in 8-bit oder 4-bit**
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Oder load_in_4bit=True
    bnb_8bit_compute_dtype=torch.float16,
)

model_kwargs["quantization_config"] = quantization_config
```

### 3. **Aggressive Memory Cleanup**
```python
# Nach JEDEM Segment:
torch.cuda.empty_cache()
torch.cuda.synchronize()
gc.collect()
```

### 4. **Reduce Frame Resolution**
```python
self.target_resolution = (224, 224)  # Statt (336, 336)
self.max_pixels = 512 * 28 * 28  # Statt 1280 * 28 * 28
```

### 5. **Process Single Frames**
```python
self.frames_per_segment = 1  # Statt 3
# Kein Frame Grid, nur einzelne Frames
```

### 6. **Enable gradient_checkpointing**
```python
self.model.gradient_checkpointing_enable()
```

### 7. **Use CPU Offloading**
```python
model_kwargs = {
    "torch_dtype": torch.float16,
    "device_map": "auto",
    "offload_state_dict": True,
    "offload_buffers": True,
}
```

## 🚀 Empfohlene Sofortmaßnahmen

1. **8-bit Quantization** aktivieren (50% Memory-Ersparnis)
2. **device_map="auto"** verwenden
3. **target_resolution** auf (224, 224) reduzieren
4. **frames_per_segment** auf 1 setzen
5. **Memory Cleanup** nach jedem Segment

Diese Änderungen sollten den Memory-Verbrauch von ~20GB auf ~10GB reduzieren.