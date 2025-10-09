#!/usr/bin/env python3
"""
Standalone experiment status checker
"""

import os
import glob
import json

def check_experiment_status(experiment_name):
    """Check status of a specific experiment"""
    output_dir = f"experiments/{experiment_name}"
    
    if not os.path.exists(output_dir):
        return False, 0, 0, "Directory not found"
    
    # Look for different possible result file patterns
    patterns_to_check = [
        os.path.join(output_dir, "run_*.json.gz"),  # Original pattern
        os.path.join(output_dir, "states", "states_run_*.npz"),  # Actual pattern used
        os.path.join(output_dir, "states_run_*.npz"),  # Alternative pattern
        os.path.join(output_dir, "*.npz"),  # Direct npz files
    ]
    
    completed_runs = 0
    pattern_used = "none"
    
    for pattern in patterns_to_check:
        existing_files = glob.glob(pattern)
        if existing_files:
            completed_runs = len(existing_files)
            pattern_used = pattern
            break
    
    # Get total runs from config
    config_file = os.path.join(output_dir, "config.json")
    total_runs = "unknown"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                total_runs = config.get('n_runs', 'unknown')
        except Exception as e:
            total_runs = f"error: {e}"
    
    return True, completed_runs, total_runs, pattern_used

def list_all_experiments():
    """List all experiments with their status"""
    exp_dir = "experiments"
    if not os.path.exists(exp_dir):
        print("No experiments directory found.")
        return
    
    print("Experiment Status Report")
    print("=" * 100)
    print(f"{'Experiment Name':<50} {'Status':<20} {'Pattern Used':<30}")
    print("-" * 100)
    
    for exp_name in sorted(os.listdir(exp_dir)):
        exp_path = os.path.join(exp_dir, exp_name)
        if os.path.isdir(exp_path):
            exists, completed, total, pattern = check_experiment_status(exp_name)
            
            if exists:
                if total != "unknown" and isinstance(total, int):
                    if completed == total:
                        status = f"{completed}/{total} (complete)"
                    elif completed > 0:
                        status = f"{completed}/{total} (incomplete)"
                    else:
                        status = f"{completed}/{total} (not started)"
                else:
                    status = f"{completed}/{total}"
                
                # Shorten pattern for display
                short_pattern = pattern.replace(f"experiments/{exp_name}/", "")
                print(f"{exp_name:<50} {status:<20} {short_pattern:<30}")
            else:
                print(f"{exp_name:<50} {'ERROR':<20} {pattern:<30}")

if __name__ == "__main__":
    list_all_experiments()
