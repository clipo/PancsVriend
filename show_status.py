#!/usr/bin/env python3
"""
Quick status check for running experiments
"""

import json
import glob
from pathlib import Path
from datetime import datetime

def show_all_experiments():
    """Show status of all experiments"""
    
    print("🔬 EXPERIMENT STATUS REPORT")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Find all experiment directories
    all_dirs = glob.glob("comprehensive_study_*/llm_results") + glob.glob("design_space_exploration")
    
    if not all_dirs:
        print("❌ No experiments found!")
        return
    
    for exp_dir in sorted(all_dirs, reverse=True):
        print(f"\n📁 {exp_dir}")
        print("-" * 40)
        
        # Check for progress file
        progress_files = glob.glob(f"{exp_dir}/progress_*.json")
        
        if progress_files:
            latest_progress = max(progress_files, key=lambda x: Path(x).stat().st_mtime)
            
            try:
                with open(latest_progress) as f:
                    progress = json.load(f)
                
                total = progress.get('total_planned', 0)
                completed = progress.get('completed', 0)
                successful = progress.get('successful', 0)
                failed = progress.get('failed', 0)
                percent = progress.get('progress_percent', 0)
                
                # Status emoji
                if completed >= total:
                    status = "✅ COMPLETE"
                elif percent > 0:
                    status = "🔄 RUNNING"
                else:
                    status = "⏳ STARTING"
                
                print(f"Status: {status}")
                print(f"Progress: {completed}/{total} ({percent:.1f}%)")
                print(f"Success: {successful} | Failed: {failed}")
                
                # Progress bar
                bar_length = 30
                filled = int((percent / 100) * bar_length)
                bar = "█" * filled + "░" * (bar_length - filled)
                print(f"[{bar}]")
                
                # Last update
                mtime = Path(latest_progress).stat().st_mtime
                last_update = datetime.fromtimestamp(mtime)
                time_ago = (datetime.now() - last_update).total_seconds()
                
                if time_ago < 60:
                    print(f"Last update: {int(time_ago)} seconds ago")
                elif time_ago < 3600:
                    print(f"Last update: {int(time_ago/60)} minutes ago")
                else:
                    print(f"Last update: {int(time_ago/3600)} hours ago")
                    
            except Exception as e:
                print(f"❌ Error reading progress: {e}")
        else:
            print("❓ No progress data found")

if __name__ == "__main__":
    show_all_experiments()
    
    print("\n" + "=" * 60)
    print("💡 To monitor the latest experiment in real-time:")
    print("   python monitor_latest.py")
    print("\n💡 To launch the dashboard:")
    print("   streamlit run dashboard_latest.py")