# NOTFALL-WIEDERHERSTELLUNGSANLEITUNG
## TikTok Analyzer System - Recovery Guide

### ⚡ SCHNELL-WIEDERHERSTELLUNG (wenn System nicht mehr funktioniert)

```bash
# 1. SOFORT: Prozesse stoppen
pkill -f stable_production_api
pkill -f python3

# 2. GPU bereinigen
nvidia-smi
# Falls Prozesse hängen:
sudo fuser -v /dev/nvidia*
sudo kill -9 [PID]

# 3. Aus letztem Backup wiederherstellen
cd /home/user
LATEST_BACKUP=$(ls -dt WORKING_BACKUP_* | head -1)
cp -r $LATEST_BACKUP/* /home/user/tiktok_production/

# 4. System neu starten
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh
python3 api/stable_production_api_multiprocess.py
```

### 🔍 DIAGNOSE - Was ist kaputt?

#### Test 1: API startet nicht
```bash
cd /home/user/tiktok_production
source fix_ffmpeg_env.sh
python3 -c "from configs.ml_analyzer_registry_complete import ML_ANALYZERS; print(f'Analyzer: {len(ML_ANALYZERS)}')"
# Sollte zeigen: Analyzer: 24
```

#### Test 2: Qwen2-VL zu langsam
```bash
# Direkt testen
python3 -c "
import time
from analyzers.qwen2_vl_temporal_analyzer import Qwen2VLTemporalAnalyzer
analyzer = Qwen2VLTemporalAnalyzer()
start = time.time()
result = analyzer.analyze('/home/user/tiktok_production/test_videos/test1.mp4')
print(f'Zeit: {time.time()-start:.1f}s, Segmente: {len(result.get(\"segments\", []))}')"
# Sollte <30s sein, NICHT 128s!
```

#### Test 3: Audio-Analyzer crashen
```bash
# Test audio analyzer
python3 -c "
from analyzers.audio_analysis_ultimate import UltimateAudioAnalysis
analyzer = UltimateAudioAnalysis()
result = analyzer.analyze('/home/user/tiktok_production/test_videos/test1.mp4')
print(f'Audio Segmente: {len(result.get(\"segments\", []))}')"
# Sollte KEINE ProcessPool Errors zeigen
```

### 🛠️ SPEZIFISCHE FIXES

#### Problem: "Process pool terminated abruptly"
```bash
# 1. Prüfe staged_gpu_executor.py
md5sum utils/staged_gpu_executor.py
# MUSS sein: 5be18e2a470a22eae3d2382db08c75fc

# 2. Falls falsch, wiederherstellen:
cp /home/user/WORKING_BACKUP_*/utils/staged_gpu_executor.py /home/user/tiktok_production/utils/

# 3. Verify Audio-Fix ist drin:
grep -A5 "Stage 4 - Audio Analyzer direkt" utils/staged_gpu_executor.py
```

#### Problem: Qwen2-VL braucht 128s statt 11s
```bash
# 1. Prüfe qwen2_vl_temporal_analyzer.py
md5sum analyzers/qwen2_vl_temporal_analyzer.py
# MUSS sein: de2a03703ca1618c48f80a038cb4e5f5

# 2. Prüfe ob Global Model Loading aktiv:
grep -n "Global model loading" analyzers/qwen2_vl_temporal_analyzer.py
grep -n "process_all_segments" analyzers/qwen2_vl_temporal_analyzer.py

# 3. Falls fehlt, wiederherstellen:
cp /home/user/WORKING_BACKUP_*/analyzers/qwen2_vl_temporal_analyzer.py /home/user/tiktok_production/analyzers/
```

#### Problem: Cross-Analyzer "str has no attribute get"
```bash
# 1. Prüfe cross_analyzer_intelligence_safe.py existiert
ls -la analyzers/cross_analyzer_intelligence_safe.py

# 2. MD5 Check:
md5sum analyzers/cross_analyzer_intelligence_safe.py
# MUSS sein: 16e4419c98c2430834ceeac93ef3c3c9

# 3. Registry prüfen:
grep cross_analyzer_intelligence configs/ml_analyzer_registry_complete.py
# MUSS zeigen: CrossAnalyzerIntelligence (mit _safe)
```

### 📦 VOLLSTÄNDIGE NEUINSTALLATION

Falls nichts mehr geht:

```bash
# 1. Backup aktuellen Zustand
mv /home/user/tiktok_production /home/user/tiktok_production_broken_$(date +%Y%m%d_%H%M%S)

# 2. Aus Git-Backup wiederherstellen
cd /home/user
git clone https://github.com/HI77878/ContentSpark.git tiktok_production

# 3. Kritische Fixes anwenden
cd /home/user/tiktok_production

# Fix 1: Qwen2-VL Optimierung
# Stelle sicher dass qwen2_vl_temporal_analyzer.py die optimierte Version ist

# Fix 2: Audio ProcessPool Fix
# Stelle sicher dass staged_gpu_executor.py den Audio-Fix hat

# Fix 3: Cross-Analyzer Safe Wrapper
# Stelle sicher dass cross_analyzer_intelligence_safe.py existiert

# 4. Environment setup
source fix_ffmpeg_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 5. Test
python3 test_final_100_percent.py
```

### 🔧 ENVIRONMENT VARIABLEN

IMMER setzen vor Start:
```bash
# FFmpeg Fix (KRITISCH!)
source fix_ffmpeg_env.sh

# GPU Optimierungen
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TOKENIZERS_PARALLELISM=false
```

### 🚨 NOTFALL-KONTAKTE

Bei schwerwiegenden Problemen:

1. **Logs prüfen**:
```bash
tail -100 /home/user/tiktok_production/logs/stable_multiprocess_api.log
tail -100 /home/user/tiktok_production/logs/stable_api.log
```

2. **GPU Status**:
```bash
nvidia-smi
# Bei hängenden Prozessen:
sudo nvidia-smi -r
```

3. **System Neustart** (letzter Ausweg):
```bash
sudo reboot
# Nach Neustart:
cd /home/user/tiktok_production
./start_clean_server.sh
```

### ✅ ERFOLGS-VALIDIERUNG

System funktioniert wenn:
```bash
# 1. Health Check OK
curl http://localhost:8003/health | grep "healthy"

# 2. Analyzer Count = 24
curl http://localhost:8003/health | grep "active_analyzers.*24"

# 3. Test-Analyse < 3 Minuten
time curl -X POST "http://localhost:8003/analyze" \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/home/user/tiktok_production/test_videos/test1.mp4"}'

# 4. Keine Errors in Logs
grep -i error /home/user/tiktok_production/logs/stable_multiprocess_api.log | tail -10
```

### 📋 CHECKLISTE FÜR VOLLSTÄNDIGE WIEDERHERSTELLUNG

- [ ] FFmpeg environment gesetzt (`source fix_ffmpeg_env.sh`)
- [ ] GPU-Optimierungen exportiert
- [ ] API startet ohne Fehler
- [ ] Health Check zeigt 24 Analyzer
- [ ] Test-Video wird in <3 Minuten analysiert
- [ ] Qwen2-VL braucht <30s (nicht 128s)
- [ ] Keine ProcessPool Errors
- [ ] GPU nutzt >10GB während Analyse
- [ ] Results JSON wird korrekt generiert

---
Recovery Guide erstellt: 17. Juli 2025
Für 100% Success Rate System