#!/usr/bin/env python3
"""
Real-time Progress Monitor for Design Space Exploration
Run this in a separate terminal to monitor experiment progress
"""

import json
import time
import glob
import sys
from pathlib import Path
from datetime import datetime

def monitor_experiment_progress(output_dir="design_space_exploration"):
    """Monitor and display experiment progress in real-time"""
    
    print("🔍 EXPERIMENT PROGRESS MONITOR")
    print("=" * 50)
    print(f"Monitoring: {output_dir}/")
    print("Press Ctrl+C to stop monitoring")
    print("-" * 50)
    
    progress_pattern = f"{output_dir}/progress_*.json"
    last_progress = None
    last_update_time = None
    
    try:
        while True:
            try:
                # Find latest progress file
                progress_files = glob.glob(progress_pattern)
                
                if not progress_files:
                    print("\r⏳ Waiting for experiments to start...", end="", flush=True)
                    time.sleep(5)
                    continue
                
                latest_file = max(progress_files, key=lambda x: Path(x).stat().st_mtime)
                
                with open(latest_file) as f:
                    progress = json.load(f)
                
                # Check if progress file was updated
                file_time = Path(latest_file).stat().st_mtime
                if file_time != last_update_time:
                    last_update_time = file_time
                    
                    completed = progress.get("completed", 0)
                    total = progress.get("total_planned", 0)
                    successful = progress.get("successful", 0)
                    failed = progress.get("failed", 0)
                    percent = progress.get("progress_percent", 0)
                    session_id = progress.get("session_id", "unknown")
                    
                    current_time = datetime.now().strftime("%H:%M:%S")
                    
                    if total > 0:
                        # Clear line and show detailed progress
                        print(f"\r🕐 {current_time} | Session: {session_id[:8]}... | Progress: {completed}/{total} ({percent:.1f}%) | ✅ {successful} | ❌ {failed}     ", end="")
                        
                        # Show progress bar
                        if percent > 0:
                            bar_length = 30
                            filled = int((percent / 100) * bar_length)
                            bar = "█" * filled + "▒" * (bar_length - filled)
                            print(f"\n[{bar}] {percent:.1f}%", end="")
                            
                        if completed == total:
                            print(f"\n\n🎉 EXPERIMENT BATCH COMPLETE!")
                            print(f"   Total: {total} | Successful: {successful} | Failed: {failed}")
                            if failed > 0:
                                print(f"   ⚠️  {failed} experiments failed - check logs for details")
                            break
                    
                    last_progress = progress
                    
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                print(f"\r❌ Error reading progress: {e}", end="", flush=True)
            
            time.sleep(3)  # Check every 3 seconds
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Monitoring stopped by user")
        
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")

def show_latest_progress(output_dir="design_space_exploration"):
    """Show the latest progress without continuous monitoring"""
    
    progress_pattern = f"{output_dir}/progress_*.json"
    progress_files = glob.glob(progress_pattern)
    
    if not progress_files:
        print("❌ No progress files found. Experiments may not have started yet.")
        return
        
    latest_file = max(progress_files, key=lambda x: Path(x).stat().st_mtime)
    
    try:
        with open(latest_file) as f:
            progress = json.load(f)
        
        print("📊 LATEST EXPERIMENT PROGRESS")
        print("=" * 40)
        print(f"Session ID: {progress.get('session_id', 'unknown')}")
        print(f"Total Planned: {progress.get('total_planned', 0)}")
        print(f"Completed: {progress.get('completed', 0)}")
        print(f"Successful: {progress.get('successful', 0)}")
        print(f"Failed: {progress.get('failed', 0)}")
        print(f"Progress: {progress.get('progress_percent', 0):.1f}%")
        
        # Show estimated time remaining
        completed = progress.get('completed', 0)
        total = progress.get('total_planned', 0)
        if completed > 0 and total > completed:
            # Rough estimate based on current progress
            remaining = total - completed
            print(f"Remaining: {remaining} experiments")
            
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ Error reading progress file: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor experiment progress")
    parser.add_argument("--output-dir", default="design_space_exploration",
                       help="Output directory to monitor")
    parser.add_argument("--once", action="store_true",
                       help="Show progress once instead of continuous monitoring")
    
    args = parser.parse_args()
    
    if args.once:
        show_latest_progress(args.output_dir)
    else:
        monitor_experiment_progress(args.output_dir)

if __name__ == "__main__":
    main()