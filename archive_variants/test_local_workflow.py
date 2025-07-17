#!/usr/bin/env python3
"""
Test workflow with local video file
"""

import sys
sys.path.insert(0, '/home/user/tiktok_production')

from single_workflow import TikTokProductionWorkflow
import time
from pathlib import Path

def test_local_video():
    """Test with local video file instead of download"""
    workflow = TikTokProductionWorkflow()
    
    # Local video path
    video_path = "/home/user/tiktok_production/test_local_video.mp4"
    video_id = "test_local_video"
    
    print(f"\n🚀 TESTING WITH LOCAL VIDEO")
    print(f"📍 Video: {video_path}")
    print(f"🎯 Analyzers: {len(workflow.analyzers)}")
    print("="*60)
    
    workflow_start = time.time()
    
    try:
        # Skip download phase, go directly to analysis
        print(f"\n🔍 PHASE 1: ANALYSIS")
        results = workflow.analyze_video(video_path, video_id)
        
        # Save results
        print(f"\n💾 PHASE 2: SAVE RESULTS")
        output_path = workflow.save_results(results)
        
        total_time = time.time() - workflow_start
        
        print(f"\n" + "="*60)
        print(f"✅ WORKFLOW COMPLETE!")
        print(f"📊 Analyzers: {results['summary']['successful']}/{len(workflow.analyzers)}")
        print(f"⏱️ Total Time: {total_time:.1f}s")
        print(f"🎯 Success Rate: {results['summary']['successful']/len(workflow.analyzers)*100:.1f}%")
        print(f"📁 Results: {output_path}")
        
        if total_time < 180:  # 3 minutes
            print(f"🏆 TARGET ACHIEVED: <3 minutes!")
        else:
            print(f"⚠️ Target missed: {total_time/60:.1f} minutes")
        
        # Show some results
        print(f"\n📊 Quick Results Summary:")
        for analyzer_name, result in list(results['analyzers'].items())[:5]:
            if result['status'] == 'success':
                print(f"  ✅ {analyzer_name}: {result['duration']:.1f}s")
            else:
                print(f"  ❌ {analyzer_name}: {result.get('error', 'Unknown error')}")
        
        return output_path
        
    except Exception as e:
        total_time = time.time() - workflow_start
        print(f"\n❌ WORKFLOW FAILED after {total_time:.1f}s")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_local_video()