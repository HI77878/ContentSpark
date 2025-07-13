#!/bin/bash
# Complete Server Audit Script

echo "======================================"
echo "🔍 COMPLETE SERVER AUDIT - $(date)"
echo "======================================"

# Create audit directory
AUDIT_DIR="server_audit_$(date +%Y%m%d_%H%M%S)"
mkdir -p $AUDIT_DIR

# Function to write to both console and file
log() {
    echo -e "$1" | tee -a "$AUDIT_DIR/audit_log.txt"
}

# TEIL 1: LAUFENDE PROZESSE & PORTS
log "\n📊 PYTHON PROZESSE:"
ps aux | grep python | grep -v grep | awk '{print $2, $11, $12, $13, $14}' | sort | tee -a "$AUDIT_DIR/python_processes.txt"

log "\n🌐 BELEGTE PORTS:"
netstat -tlnp 2>/dev/null | grep LISTEN | tee -a "$AUDIT_DIR/ports.txt" || ss -tlnp | grep LISTEN | tee -a "$AUDIT_DIR/ports.txt"

log "\n🐳 DOCKER STATUS:"
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Size}}" 2>/dev/null | tee -a "$AUDIT_DIR/docker_status.txt"

log "\n💾 RESSOURCEN:"
free -h | tee -a "$AUDIT_DIR/resources.txt"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null | tee -a "$AUDIT_DIR/gpu_status.txt"

# TEIL 2: API & SERVICE DISCOVERY
log "\n🚀 API & SERVICES:"

log "\n📁 API Files:"
find /home/user/tiktok_production -name "*api*.py" -type f 2>/dev/null | grep -E "(api|service)" | sort | tee -a "$AUDIT_DIR/api_files.txt"

log "\n📡 API ENDPOINTS CHECK:"
for port in 8000 8001 8002 8003 8004 8005 8024; do
    result=$(echo -n "Port $port: ")
    if curl -s http://localhost:$port/health >/dev/null 2>&1; then
        status=$(curl -s http://localhost:$port/health | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('status', data.get('orchestrator', 'active')))" 2>/dev/null || echo "active")
        result="${result}✅ ACTIVE - $status"
    else
        result="${result}❌ INACTIVE"
    fi
    log "$result"
done | tee -a "$AUDIT_DIR/api_endpoints.txt"

log "\n🔧 API DEFINITIONEN:"
grep -r "FastAPI\|Flask\|app = " --include="*.py" /home/user/tiktok_production 2>/dev/null | grep -v "__pycache__" | cut -d: -f1 | sort -u | tee -a "$AUDIT_DIR/api_definitions.txt"

# TEIL 3: ANALYZER INVENTORY
log "\n🎯 ANALYZER INVENTORY:"

log "\n📁 Analyzer Dateien:"
find /home/user/tiktok_production \( -name "*analyzer*.py" -o -name "*analysis*.py" \) -not -path "*/__pycache__/*" 2>/dev/null | sort | while read file; do
    echo "  - $file ($(wc -l < "$file") lines)"
done | tee -a "$AUDIT_DIR/analyzer_files.txt"

log "\n📋 Registrierte Analyzer in Registry:"
if [ -f "ml_analyzer_registry_complete.py" ]; then
    grep -E "'[a-z_]+': " ml_analyzer_registry_complete.py | grep -v "#" | sort | tee -a "$AUDIT_DIR/registered_analyzers.txt"
fi

# TEIL 4: MODEL LOCATIONS & DUPLICATES
log "\n🤖 ML MODELS & CACHE:"

log "\n🤗 HuggingFace Models:"
du -sh ~/.cache/huggingface/* 2>/dev/null | sort -h | tail -20 | tee -a "$AUDIT_DIR/huggingface_cache.txt"

log "\n🔥 PyTorch Models:"
du -sh ~/.cache/torch/* 2>/dev/null | sort -h | tee -a "$AUDIT_DIR/torch_cache.txt"

log "\n📦 Lokale Model Files:"
find /home/user/tiktok_production \( -name "*.pth" -o -name "*.pt" -o -name "*.onnx" -o -name "*.pb" \) 2>/dev/null | head -20 | tee -a "$AUDIT_DIR/model_files.txt"

log "\n🔍 Potentielle Duplikate:"
find /home/user/tiktok_production -name "*.py" -not -path "*/__pycache__/*" -exec basename {} \; 2>/dev/null | sort | uniq -d | tee -a "$AUDIT_DIR/duplicate_files.txt"

# TEIL 5: DOCKER DEEP DIVE
log "\n🐳 DOCKER DEEP ANALYSIS:"

log "\n📦 Docker Images:"
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" 2>/dev/null | tee -a "$AUDIT_DIR/docker_images.txt"

log "\n💾 Docker Volumes:"
docker volume ls 2>/dev/null | tee -a "$AUDIT_DIR/docker_volumes.txt"

log "\n🌐 Docker Networks:"
docker network ls 2>/dev/null | tee -a "$AUDIT_DIR/docker_networks.txt"

# TEIL 6: CONFIGURATION & WORKFLOWS
log "\n⚙️ CONFIGURATIONS & WORKFLOWS:"

log "\n📝 Config Files:"
find /home/user/tiktok_production \( -name "*config*.py" -o -name "*settings*.py" \) -not -path "*/__pycache__/*" 2>/dev/null | sort | tee -a "$AUDIT_DIR/config_files.txt"

log "\n🔄 Workflows & Pipelines:"
find /home/user/tiktok_production \( -name "*pipeline*.py" -o -name "*workflow*.py" \) -not -path "*/__pycache__/*" 2>/dev/null | sort | tee -a "$AUDIT_DIR/workflow_files.txt"

# TEIL 7: DISK USAGE & OLD FILES
log "\n💾 DISK USAGE ANALYSIS:"

log "\n📁 Größte Verzeichnisse:"
du -sh /home/user/tiktok_production/*/ 2>/dev/null | sort -h | tail -15 | tee -a "$AUDIT_DIR/disk_usage.txt"

log "\n📄 Log Files:"
find /home/user/tiktok_production -name "*.log" -type f -exec ls -lh {} \; 2>/dev/null | sort -k5 -h | tail -10 | tee -a "$AUDIT_DIR/log_files.txt"

log "\n🗑️ Cache Directories:"
find /home/user/tiktok_production -type d \( -name "*cache*" -o -name "*tmp*" -o -name "__pycache__" \) 2>/dev/null | xargs du -sh 2>/dev/null | sort -h | tee -a "$AUDIT_DIR/cache_dirs.txt"

# TEIL 8: SYSTEM HEALTH CHECK
log "\n🏥 SYSTEM HEALTH:"

log "\n💾 Memory Usage by Process:"
ps aux --sort=-%mem | head -10 | tee -a "$AUDIT_DIR/memory_usage.txt"

log "\n💿 Disk Space:"
df -h | grep -E "/$|/home" | tee -a "$AUDIT_DIR/disk_space.txt"

log "\n📊 System Load:"
uptime | tee -a "$AUDIT_DIR/system_load.txt"

# Generate summary
log "\n✅ AUDIT COMPLETE!"
log "📁 Results saved to: $AUDIT_DIR/"

# Create summary report
cat > "$AUDIT_DIR/SUMMARY.md" << EOF
# Server Audit Summary - $(date)

## Overview
- Audit Directory: $AUDIT_DIR
- Server: $(hostname)
- Directory: /home/user/tiktok_production

## Quick Stats
- Python Processes: $(ps aux | grep python | grep -v grep | wc -l)
- Open Ports: $(netstat -tlnp 2>/dev/null | grep LISTEN | wc -l || ss -tlnp | grep LISTEN | wc -l)
- Docker Containers: $(docker ps -a 2>/dev/null | wc -l)
- Analyzer Files: $(find /home/user/tiktok_production -name "*analyzer*.py" -not -path "*/__pycache__/*" 2>/dev/null | wc -l)

## Key Findings
See individual files in this directory for details.

## Next Steps
1. Review duplicate files
2. Check active ports and services
3. Clean up unused Docker resources
4. Remove old logs and cache
EOF

echo "📊 Summary saved to: $AUDIT_DIR/SUMMARY.md"