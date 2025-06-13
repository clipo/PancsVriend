#!/usr/bin/env python3
"""
Interactive menu to launch dashboard for any experiment
"""

import glob
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def find_all_experiments():
    """Find all experiment directories with details"""
    experiments = []
    
    # Find different types of experiment directories
    patterns = [
        "comprehensive_study_*/llm_results",
        "design_space_exploration",
        "experiments/llm_*",
        "experiments/baseline_*"
    ]
    
    for pattern in patterns:
        dirs = glob.glob(pattern)
        for dir_path in dirs:
            if Path(dir_path).exists():
                # Get experiment info
                info = {
                    'path': dir_path,
                    'type': 'Unknown',
                    'status': 'Unknown',
                    'progress': 0,
                    'last_update': None
                }
                
                # Determine experiment type
                if 'comprehensive_study' in dir_path:
                    info['type'] = 'Comprehensive Study'
                elif 'design_space' in dir_path:
                    info['type'] = 'Design Space Exploration'
                elif 'llm_' in dir_path:
                    info['type'] = 'LLM Experiment'
                elif 'baseline_' in dir_path:
                    info['type'] = 'Baseline Experiment'
                
                # Check for progress
                progress_files = glob.glob(f"{dir_path}/progress_*.json")
                if progress_files:
                    latest_progress = max(progress_files, key=lambda x: Path(x).stat().st_mtime)
                    try:
                        import json
                        with open(latest_progress) as f:
                            progress_data = json.load(f)
                        
                        completed = progress_data.get('completed', 0)
                        total = progress_data.get('total_planned', 0)
                        info['progress'] = (completed / total * 100) if total > 0 else 0
                        
                        if completed >= total:
                            info['status'] = '✅ Complete'
                        elif info['progress'] > 0:
                            info['status'] = '🔄 Running'
                        else:
                            info['status'] = '⏳ Starting'
                        
                        # Get last update time
                        mtime = Path(latest_progress).stat().st_mtime
                        info['last_update'] = datetime.fromtimestamp(mtime)
                    except:
                        pass
                
                experiments.append(info)
    
    # Sort by last update time (most recent first)
    experiments.sort(key=lambda x: x['last_update'] or datetime(1900, 1, 1), reverse=True)
    
    return experiments

def show_menu(experiments):
    """Display interactive menu"""
    print("\n" + "="*70)
    print("🏘️  SCHELLING SEGREGATION EXPERIMENT DASHBOARD LAUNCHER")
    print("="*70)
    
    if not experiments:
        print("\n❌ No experiments found!")
        print("\nTo start an experiment, run one of:")
        print("  python comprehensive_comparison_study.py --quick-test")
        print("  python run_experiments.py")
        return None
    
    print(f"\nFound {len(experiments)} experiment(s):\n")
    
    # Display experiments
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['status']} {exp['type']}")
        print(f"   Path: {exp['path']}")
        print(f"   Progress: {exp['progress']:.0f}%")
        
        if exp['last_update']:
            time_ago = (datetime.now() - exp['last_update']).total_seconds()
            if time_ago < 60:
                print(f"   Last update: {int(time_ago)} seconds ago")
            elif time_ago < 3600:
                print(f"   Last update: {int(time_ago/60)} minutes ago")
            else:
                print(f"   Last update: {int(time_ago/3600)} hours ago")
        print()
    
    # Get user choice
    while True:
        try:
            choice = input("Select experiment number (or 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(experiments):
                return experiments[idx]
            else:
                print(f"Please enter a number between 1 and {len(experiments)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")

def launch_dashboard(experiment):
    """Launch dashboard for selected experiment"""
    print(f"\n🚀 Launching dashboard for: {experiment['path']}")
    print("="*70)
    
    # Create temporary launcher script
    launcher_code = f'''
import streamlit as st
import sys
sys.path.append(".")
from dashboard import *

# Override default directory
DEFAULT_DIR = "{experiment['path']}"

if __name__ == "__main__":
    st.set_page_config(
        page_title="Schelling Experiments - {experiment['type']}",
        page_icon="🏘️",
        layout="wide"
    )
    
    # Run main with modified sidebar
    main()
'''
    
    # Save temporary launcher
    temp_file = "temp_dashboard_launcher.py"
    with open(temp_file, 'w') as f:
        f.write(launcher_code)
    
    try:
        # Launch streamlit
        print("\n💡 Dashboard will open in your browser automatically")
        print("   If not, navigate to: http://localhost:8501")
        print("\n   Press Ctrl+C to stop the dashboard\n")
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", temp_file])
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)

def main():
    """Main entry point"""
    experiments = find_all_experiments()
    selected = show_menu(experiments)
    
    if selected:
        launch_dashboard(selected)
    else:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()