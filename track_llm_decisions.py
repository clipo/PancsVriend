#!/usr/bin/env python3
"""
Real-time tracking of LLM agent decisions
Shows each agent decision as it happens
"""

import json
import time
from pathlib import Path
from datetime import datetime
import sys
import os

def find_latest_experiment():
    """Find the most recently modified experiment"""
    exp_dirs = []
    
    # Look for memory LLM experiments
    for pattern in ["memory_llm_experiments/experiments/exp_*", 
                    "comprehensive_study_*/llm_results/experiments/exp_*",
                    "experiments/llm_*"]:
        for exp_dir in Path(".").glob(pattern):
            if exp_dir.is_dir():
                exp_dirs.append(exp_dir)
    
    if not exp_dirs:
        return None
    
    # Find most recently modified
    return max(exp_dirs, key=lambda x: x.stat().st_mtime)

def track_experiment_progress(exp_dir):
    """Track progress of a specific experiment"""
    
    print(f"📊 Tracking experiment: {exp_dir}")
    print("=" * 60)
    
    # Look for metrics file to track decisions
    metrics_file = exp_dir / "metrics_history.csv"
    convergence_file = exp_dir / "convergence_summary.csv"
    config_file = exp_dir / "experiment_config.json"
    
    # Load experiment config
    experiment_info = {}
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
                experiment_info = {
                    'name': config.get('name', 'Unknown'),
                    'agent_type': config.get('agent_type', 'unknown'),
                    'scenario': config.get('scenario', 'unknown'),
                    'grid_size': config.get('grid_size', 'unknown'),
                    'total_agents': config.get('num_type_a', 0) + config.get('num_type_b', 0),
                    'n_runs': config.get('n_runs', 'unknown'),
                    'max_steps': config.get('max_steps', 'unknown')
                }
        except:
            pass
    
    print(f"🧪 Experiment: {experiment_info.get('name', 'Unknown')}")
    print(f"🤖 Agent Type: {experiment_info.get('agent_type', 'Unknown')}")
    print(f"🎭 Scenario: {experiment_info.get('scenario', 'Unknown')}")
    print(f"📏 Grid: {experiment_info.get('grid_size')}x{experiment_info.get('grid_size')}")
    print(f"👥 Agents: {experiment_info.get('total_agents')}")
    print(f"🔄 Runs: {experiment_info.get('n_runs')}")
    print(f"📊 Max Steps: {experiment_info.get('max_steps')}")
    print("-" * 60)
    
    last_size = 0
    last_run = -1
    last_step = -1
    decision_count = 0
    start_time = time.time()
    
    print("\n⏳ Waiting for agent decisions...")
    print("(Press Ctrl+C to stop tracking)\n")
    
    try:
        while True:
            if metrics_file.exists():
                current_size = metrics_file.stat().st_size
                
                if current_size > last_size:
                    # File has grown, new data available
                    try:
                        # Read the last few lines efficiently
                        with open(metrics_file, 'rb') as f:
                            f.seek(max(0, current_size - 4096))  # Read last 4KB
                            tail = f.read().decode('utf-8', errors='ignore')
                            lines = tail.strip().split('\n')
                            
                            # Parse CSV lines
                            for line in lines[-10:]:  # Look at last 10 lines
                                if ',' in line and not line.startswith('run_id'):
                                    try:
                                        parts = line.split(',')
                                        if len(parts) >= 3:
                                            run_id = int(parts[0])
                                            step = int(parts[1])
                                            
                                            # Check if this is a new step
                                            if run_id > last_run or (run_id == last_run and step > last_step):
                                                last_run = run_id
                                                last_step = step
                                                
                                                # Calculate decisions per step
                                                agents_per_step = experiment_info.get('total_agents', 0)
                                                decision_count += agents_per_step
                                                
                                                # Calculate timing
                                                elapsed = time.time() - start_time
                                                decisions_per_second = decision_count / elapsed if elapsed > 0 else 0
                                                
                                                # Estimate time remaining
                                                total_decisions = (experiment_info.get('n_runs', 1) * 
                                                                 experiment_info.get('max_steps', 1000) * 
                                                                 agents_per_step)
                                                remaining_decisions = total_decisions - decision_count
                                                eta_seconds = remaining_decisions / decisions_per_second if decisions_per_second > 0 else 0
                                                eta_str = format_time(eta_seconds)
                                                
                                                # Print update
                                                print(f"\r🏃 Run {run_id+1}/{experiment_info.get('n_runs')} | "
                                                      f"Step {step}/{experiment_info.get('max_steps')} | "
                                                      f"💭 {decision_count:,} decisions | "
                                                      f"⚡ {decisions_per_second:.1f} dec/s | "
                                                      f"⏱️ ETA: {eta_str}", end='', flush=True)
                                    except:
                                        pass
                        
                    except Exception as e:
                        pass
                    
                    last_size = current_size
            
            # Also check for convergence
            if convergence_file.exists() and convergence_file.stat().st_size > 0:
                try:
                    with open(convergence_file) as f:
                        # Skip header
                        f.readline()
                        # Check if we have any complete runs
                        for line in f:
                            if ',' in line:
                                parts = line.strip().split(',')
                                if len(parts) >= 4 and parts[2] == 'True':  # converged column
                                    conv_step = parts[3]
                                    print(f"\n\n✅ Run {parts[0]} converged at step {conv_step}!")
                except:
                    pass
            
            time.sleep(1)  # Check every second
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Tracking stopped")
        print(f"📊 Total decisions tracked: {decision_count:,}")
        print(f"⏱️ Total time: {format_time(time.time() - start_time)}")
        print(f"⚡ Average speed: {decision_count/(time.time()-start_time):.1f} decisions/second")

def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def main():
    print("🔍 LLM DECISION TRACKER")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        # Track specific experiment
        exp_dir = Path(sys.argv[1])
        if not exp_dir.exists():
            print(f"❌ Experiment directory not found: {exp_dir}")
            return 1
    else:
        # Find latest experiment
        exp_dir = find_latest_experiment()
        if not exp_dir:
            print("❌ No active experiments found")
            print("\n💡 Usage:")
            print("  python track_llm_decisions.py                    # Track latest experiment")
            print("  python track_llm_decisions.py path/to/exp_dir   # Track specific experiment")
            return 1
    
    try:
        track_experiment_progress(exp_dir)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())