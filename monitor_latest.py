#!/usr/bin/env python3
"""
Monitor the latest running experiment automatically
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

def find_latest_experiment():
    """Find the most recent experiment directory"""
    
    # Look for comprehensive study directories
    comp_dirs = glob.glob("comprehensive_study_*/llm_results")
    
    # Look for design space directories  
    design_dirs = glob.glob("design_space_exploration")
    
    # Combine all possibilities
    all_dirs = comp_dirs + design_dirs
    
    if not all_dirs:
        print("❌ No experiment directories found!")
        print("Make sure you're in the project root directory.")
        return None
    
    # Find the most recently modified directory with progress files
    latest_dir = None
    latest_time = 0
    
    for dir_path in all_dirs:
        progress_files = glob.glob(f"{dir_path}/progress_*.json")
        if progress_files:
            # Get the most recent progress file
            for pf in progress_files:
                mtime = os.path.getmtime(pf)
                if mtime > latest_time:
                    latest_time = mtime
                    latest_dir = dir_path
    
    return latest_dir

def main():
    """Main entry point"""
    
    print("🔍 Searching for latest experiment...")
    
    latest_dir = find_latest_experiment()
    
    if not latest_dir:
        print("❌ No active experiments found!")
        print("\nTo start an experiment, run one of:")
        print("  python comprehensive_comparison_study.py --quick-test")
        print("  python run_experiments.py")
        print("  python run_design_space_exploration.py --all")
        sys.exit(1)
    
    print(f"✅ Found latest experiment: {latest_dir}")
    print(f"\nMonitoring progress...")
    print("-" * 50)
    
    # Launch monitor with the found directory
    cmd = [sys.executable, "monitor_progress.py", "--output-dir", latest_dir]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped")

if __name__ == "__main__":
    main()