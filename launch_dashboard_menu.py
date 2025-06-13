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

def find_active_experiments():
    """Find experiments with real-time progress files"""
    active_experiments = []
    
    # Look for comprehensive study directories with progress files
    for study_dir in glob.glob("comprehensive_study_*/"):
        if Path(f"{study_dir}/progress_realtime.json").exists():
            active_experiments.append(study_dir.rstrip('/'))
    
    # Look for individual progress files
    for progress_file in glob.glob("*/progress_realtime.json"):
        exp_path = progress_file.replace('/progress_realtime.json', '')
        if not exp_path.startswith('comprehensive_study_'):
            active_experiments.append(exp_path)
    
    return active_experiments

def launch_dashboard(experiment, dashboard_type="main"):
    """Launch dashboard for selected experiment"""
    print(f"\n🚀 Launching dashboard for: {experiment['path']}")
    print("="*70)
    
    # Choose dashboard script based on type
    if dashboard_type == "progress":
        dashboard_script = "dashboard_with_progress.py"
        print("📊 Launching Real-Time Progress Dashboard...")
        print("   This dashboard shows live progress for active experiments!")
    else:
        dashboard_script = "dashboard.py"
        # Set environment variable for the dashboard to pick up
        os.environ["DASHBOARD_DEFAULT_DIR"] = experiment['path']
    
    # Launch streamlit directly
    print("\n💡 Dashboard will open in your browser automatically")
    print("   If not, navigate to: http://localhost:8501")
    
    if dashboard_type != "progress":
        print(f"\n   The dashboard will default to: {experiment['path']}")
    
    print("   Press Ctrl+C to stop the dashboard\n")
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_script])

def main():
    """Main entry point"""
    experiments = find_all_experiments()
    active_experiments = find_active_experiments()
    
    # Check if we have active experiments for progress dashboard
    if active_experiments:
        print(f"\n⚡ Found {len(active_experiments)} active experiment(s) with real-time progress!")
        print("💡 You can use the Real-Time Progress Dashboard to monitor them.")
        
        # Offer quick launch for progress dashboard
        quick_launch = input("\n🚀 Launch Real-Time Progress Dashboard now? (y/n): ").strip().lower()
        if quick_launch in ['y', 'yes']:
            print("\n📊 Launching Real-Time Progress Dashboard...")
            print("   This dashboard shows live progress for active experiments!")
            print("\n💡 Dashboard will open in your browser automatically")
            print("   If not, navigate to: http://localhost:8501")
            print("   Press Ctrl+C to stop the dashboard\n")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard_with_progress.py"])
            return
    
    # Show regular experiment selection menu
    selected = show_menu(experiments)
    
    if selected:
        # Ask which dashboard to use
        print("\nDashboard Options:")
        print("1. 📊 Standard Dashboard (detailed analysis)")
        if active_experiments:
            print("2. ⏱️  Real-Time Progress Dashboard (live monitoring)")
        
        dashboard_choice = input(f"\nSelect dashboard type (1{'-2' if active_experiments else ''}): ").strip()
        
        if dashboard_choice == "2" and active_experiments:
            launch_dashboard(selected, "progress")
        else:
            launch_dashboard(selected)
    else:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()