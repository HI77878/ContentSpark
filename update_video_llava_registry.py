#!/usr/bin/env python3
"""
Update the ML analyzer registry to use the Docker-enabled Video-LLaVA
This script updates the import and mapping without breaking other analyzers
"""
import re
from pathlib import Path

def update_registry():
    """Update ml_analyzer_registry_complete.py to use Docker-enabled LLaVA"""
    registry_path = Path("/home/user/tiktok_production/ml_analyzer_registry_complete.py")
    
    # Read the current content
    with open(registry_path, 'r') as f:
        content = f.read()
    
    # Check if already updated
    if "llava_video_optimized_docker" in content:
        print("✅ Registry already uses Docker-enabled Video-LLaVA")
        return True
    
    # Add import for Docker-enabled version
    import_section = content.find("from analyzers.llava_video_optimized import LLaVAVideoOptimized")
    if import_section != -1:
        # Replace the import
        old_import = "from analyzers.llava_video_optimized import LLaVAVideoOptimized"
        new_import = "from analyzers.llava_video_optimized_docker import LLaVAVideoOptimizedDocker as LLaVAVideoOptimized"
        content = content.replace(old_import, new_import)
        print("✅ Updated import to use Docker-enabled version")
    else:
        print("❌ Could not find LLaVA import to update")
        return False
    
    # Write back the updated content
    with open(registry_path, 'w') as f:
        f.write(content)
    
    print("✅ Registry updated successfully")
    return True

def verify_update():
    """Verify the update was successful"""
    try:
        import sys
        sys.path.append('/home/user/tiktok_production')
        from ml_analyzer_registry_complete import ML_ANALYZERS
        
        # Check if video_llava is present
        if 'video_llava' in ML_ANALYZERS:
            analyzer_class = ML_ANALYZERS['video_llava']
            print(f"✅ video_llava mapped to: {analyzer_class.__module__}.{analyzer_class.__name__}")
            
            # Check if it's the Docker version
            if "docker" in analyzer_class.__module__.lower():
                print("✅ Using Docker-enabled Video-LLaVA")
                return True
            else:
                print("⚠️  Still using non-Docker version")
                return False
        else:
            print("❌ video_llava not found in registry")
            return False
            
    except Exception as e:
        print(f"❌ Error verifying update: {e}")
        return False

def main():
    """Main update process"""
    print("="*80)
    print("Updating ML Analyzer Registry for Docker-enabled Video-LLaVA")
    print("="*80)
    
    # Create backup
    registry_path = Path("/home/user/tiktok_production/ml_analyzer_registry_complete.py")
    backup_path = registry_path.with_suffix('.py.backup')
    
    print(f"\n1. Creating backup: {backup_path}")
    import shutil
    shutil.copy2(registry_path, backup_path)
    print("✅ Backup created")
    
    # Update registry
    print("\n2. Updating registry...")
    if update_registry():
        # Verify update
        print("\n3. Verifying update...")
        if verify_update():
            print("\n✅ Registry successfully updated to use Docker-enabled Video-LLaVA")
            print("\nNext steps:")
            print("1. Ensure Docker service is running:")
            print("   cd /home/user/tiktok_production/docker/video_llava")
            print("   ./build_and_run.sh")
            print("\n2. Restart the main API to use updated registry:")
            print("   # Find and kill current API")
            print("   ps aux | grep stable_production_api")
            print("   kill <PID>")
            print("   # Restart API")
            print("   cd /home/user/tiktok_production")
            print("   source fix_ffmpeg_env.sh")
            print("   python3 api/stable_production_api_multiprocess.py &")
        else:
            print("\n❌ Update verification failed")
            print("Restoring backup...")
            shutil.copy2(backup_path, registry_path)
            print("✅ Backup restored")
    else:
        print("\n❌ Update failed")

if __name__ == "__main__":
    main()