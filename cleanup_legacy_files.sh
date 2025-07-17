#!/bin/bash
# Cleanup script for TikTok Production - Remove duplicates and legacy files
# IMPORTANT: Review before running! This will delete files.

echo "TikTok Production Cleanup Script"
echo "================================"
echo "This script will remove duplicate and legacy files."
echo "Please review the cleanup_report.md first!"
echo ""
read -p "Are you sure you want to proceed? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

# Create backup directory
BACKUP_DIR="/home/user/tiktok_production_backup_$(date +%Y%m%d_%H%M%S)"
echo "Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# 1. Archive directories
echo ""
echo "Moving archive directories to backup..."
mv _archive_tests_20250704 "$BACKUP_DIR/" 2>/dev/null && echo "✓ Moved _archive_tests_20250704"
mv _archive_utils_20250704 "$BACKUP_DIR/" 2>/dev/null && echo "✓ Moved _archive_utils_20250704"

# 2. Legacy API implementations
echo ""
echo "Moving legacy API implementations..."
mkdir -p "$BACKUP_DIR/api"
mv api/stable_production_api_blip2_fix.py "$BACKUP_DIR/api/" 2>/dev/null && echo "✓ Moved stable_production_api_blip2_fix.py"
mv api/stable_production_api_multiprocess.py "$BACKUP_DIR/api/" 2>/dev/null && echo "✓ Moved stable_production_api_multiprocess.py"
mv api/ultimate_production_api.py "$BACKUP_DIR/api/" 2>/dev/null && echo "✓ Moved ultimate_production_api.py"

# 3. BLIP2 test and legacy files
echo ""
echo "Moving BLIP2 legacy files..."
mkdir -p "$BACKUP_DIR/blip2_legacy"
# Test files
mv test_blip2_*.py "$BACKUP_DIR/blip2_legacy/" 2>/dev/null && echo "✓ Moved BLIP2 test files"
mv quick_blip2_test.py "$BACKUP_DIR/blip2_legacy/" 2>/dev/null
mv test_fixed_blip2_complete.py "$BACKUP_DIR/blip2_legacy/" 2>/dev/null
mv fix_blip2_*.py "$BACKUP_DIR/blip2_legacy/" 2>/dev/null
mv debug_blip2_worker.py "$BACKUP_DIR/blip2_legacy/" 2>/dev/null

# BLIP2 documentation
mv BLIP2_*.md "$BACKUP_DIR/blip2_legacy/" 2>/dev/null && echo "✓ Moved BLIP2 documentation"
mv blip2_vs_auroracap_comparison.md "$BACKUP_DIR/blip2_legacy/" 2>/dev/null

# Legacy BLIP2 analyzers (keeping only the ones in ML_ANALYZERS)
mkdir -p "$BACKUP_DIR/analyzers_blip2"
mv analyzers/blip2_detailed.py "$BACKUP_DIR/analyzers_blip2/" 2>/dev/null
mv analyzers/blip2_docker_analyzer.py "$BACKUP_DIR/analyzers_blip2/" 2>/dev/null
mv analyzers/blip2_final_fixed.py "$BACKUP_DIR/analyzers_blip2/" 2>/dev/null
mv analyzers/blip2_ultimate_fixed.py "$BACKUP_DIR/analyzers_blip2/" 2>/dev/null
mv analyzers/blip2_video_analyzer_fixed.py "$BACKUP_DIR/analyzers_blip2/" 2>/dev/null
mv analyzers/blip2_video_analyzer_optimized_fixed.py "$BACKUP_DIR/analyzers_blip2/" 2>/dev/null

# 4. Ultimate analyzer duplicates
echo ""
echo "Moving ultimate analyzer duplicates..."
mkdir -p "$BACKUP_DIR/analyzers_ultimate"
# These are not in ML_ANALYZERS, so they're unused
mv analyzers/*_ultimate.py "$BACKUP_DIR/analyzers_ultimate/" 2>/dev/null && echo "✓ Moved ultimate analyzers"
mv analyzers/*_ultimate_v2.py "$BACKUP_DIR/analyzers_ultimate/" 2>/dev/null
mv analyzers/*_ultimate_fixed.py "$BACKUP_DIR/analyzers_ultimate/" 2>/dev/null

# 5. Test files in root
echo ""
echo "Moving test files from root..."
mkdir -p "$BACKUP_DIR/root_tests"
mv final_*_test.py "$BACKUP_DIR/root_tests/" 2>/dev/null && echo "✓ Moved final test files"
mv final_test_*.py "$BACKUP_DIR/root_tests/" 2>/dev/null
mv direct_analysis_test.py "$BACKUP_DIR/root_tests/" 2>/dev/null
mv simple_final_test.py "$BACKUP_DIR/root_tests/" 2>/dev/null
mv quick_final_test.py "$BACKUP_DIR/root_tests/" 2>/dev/null
mv run_final_analysis.py "$BACKUP_DIR/root_tests/" 2>/dev/null
mv final_performance_monitor.py "$BACKUP_DIR/root_tests/" 2>/dev/null
mv final_production_validation.py "$BACKUP_DIR/root_tests/" 2>/dev/null
mv final_system_test.py "$BACKUP_DIR/root_tests/" 2>/dev/null

# 6. Utility duplicates
echo ""
echo "Moving utility duplicates..."
mkdir -p "$BACKUP_DIR/utils_duplicates"
mv utils/multiprocess_gpu_executor_blip2_fix.py "$BACKUP_DIR/utils_duplicates/" 2>/dev/null
mv utils/multiprocess_gpu_executor_final.py "$BACKUP_DIR/utils_duplicates/" 2>/dev/null
mv utils/multiprocess_gpu_executor_ultimate.py "$BACKUP_DIR/utils_duplicates/" 2>/dev/null
mv utils/ultimate_gpu_executor.py "$BACKUP_DIR/utils_duplicates/" 2>/dev/null

# 7. Old fix scripts
echo ""
echo "Moving old fix scripts..."
mkdir -p "$BACKUP_DIR/fix_scripts"
mv fix_missing_modules.py "$BACKUP_DIR/fix_scripts/" 2>/dev/null
mv fix_model_loaded_attribute.py "$BACKUP_DIR/fix_scripts/" 2>/dev/null
mv fix_ultimate_analyzers.py "$BACKUP_DIR/fix_scripts/" 2>/dev/null
mv fix_ultimate_analyzers_complete.py "$BACKUP_DIR/fix_scripts/" 2>/dev/null
mv fix_video_llava.py "$BACKUP_DIR/fix_scripts/" 2>/dev/null

# 8. Clean up log files (keep only recent)
echo ""
echo "Cleaning old log files..."
mkdir -p "$BACKUP_DIR/old_logs"
# Move logs older than 7 days
find logs -name "*.log" -mtime +7 -exec mv {} "$BACKUP_DIR/old_logs/" \; 2>/dev/null
echo "✓ Moved old log files"

# 9. Clean up PID files
echo ""
echo "Removing PID files..."
rm -f *.pid logs/*.pid 2>/dev/null && echo "✓ Removed PID files"

# 10. AuroraCap cleanup (optional - large directory)
echo ""
echo "AuroraCap cleanup (optional)..."
echo "The aurora_cap directory contains ~2200 files and multiple virtual environments."
read -p "Do you want to clean up AuroraCap duplicates? (yes/no): " aurora_confirm

if [ "$aurora_confirm" = "yes" ]; then
    mkdir -p "$BACKUP_DIR/aurora_cap_cleanup"
    # Remove duplicate virtual environments
    mv aurora_cap/aurora_venv "$BACKUP_DIR/aurora_cap_cleanup/" 2>/dev/null && echo "✓ Moved aurora_venv"
    mv aurora_cap/temp_aurora "$BACKUP_DIR/aurora_cap_cleanup/" 2>/dev/null && echo "✓ Moved temp_aurora"
    # Remove duplicate inference scripts
    mv aurora_cap/auroracap_*inference*.py "$BACKUP_DIR/aurora_cap_cleanup/" 2>/dev/null
    mv aurora_cap/auroracap_debug*.py "$BACKUP_DIR/aurora_cap_cleanup/" 2>/dev/null
    mv aurora_cap/auroracap_fix*.py "$BACKUP_DIR/aurora_cap_cleanup/" 2>/dev/null
    echo "✓ Moved AuroraCap duplicates"
fi

# Summary
echo ""
echo "================================"
echo "Cleanup Complete!"
echo "================================"
echo "Backup created at: $BACKUP_DIR"
echo ""
echo "Space saved:"
du -sh "$BACKUP_DIR" 2>/dev/null

echo ""
echo "If everything works correctly, you can permanently delete the backup with:"
echo "rm -rf $BACKUP_DIR"
echo ""
echo "Active production files remain in place:"
echo "- api/stable_production_api.py (Port 8003)"
echo "- ml_analyzer_registry_complete.py"
echo "- 29 active analyzers in analyzers/"
echo "- Core utilities in utils/"