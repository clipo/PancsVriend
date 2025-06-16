#!/usr/bin/env python3
"""
Live monitoring of LLM experiments with detailed status
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

def find_experiment_process():
    """Check if experiment is running"""
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'llm_runner_with_memory' in line or 'LLMSimulationWithMemory' in line:
                parts = line.split()
                if len(parts) > 1:
                    return {'pid': parts[1], 'running': True, 'command': ' '.join(parts[10:])}
    except:
        pass
    return {'running': False}

def monitor_experiment(exp_dir):
    """Monitor experiment with multiple data sources"""
    
    exp_path = Path(exp_dir)
    print(f"📊 LIVE EXPERIMENT MONITOR")
    print(f"📁 Monitoring: {exp_path}")
    print("=" * 60)
    
    # Load experiment config
    config_file = exp_path / "experiment_config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print(f"🧪 Experiment: {config.get('name', 'Unknown')}")
        print(f"🤖 Type: {config.get('agent_type', 'Unknown')} agents")
        print(f"📏 Grid: {config.get('grid_size', '?')}x{config.get('grid_size', '?')}")
        print(f"👥 Agents: {config.get('num_type_a', 0) + config.get('num_type_b', 0)}")
        print(f"🔄 Total Runs: {config.get('n_runs', '?')}")
    
    print("\n" + "=" * 60)
    
    last_update = time.time()
    iteration = 0
    
    while True:
        iteration += 1
        status_parts = []
        
        # Check process status
        proc_info = find_experiment_process()
        if proc_info['running']:
            status_parts.append(f"🟢 Process running (PID: {proc_info['pid']})")
        else:
            status_parts.append("🔴 No experiment process detected")
        
        # Check for various output files
        possible_files = {
            'metrics_history.csv': '📈 Metrics',
            'convergence_summary.csv': '🎯 Convergence',
            'results.json': '📊 Results',
            'progress.json': '📍 Progress',
            'run_0_metrics.csv': '📊 Run 0 metrics'
        }
        
        files_found = []
        for filename, label in possible_files.items():
            filepath = exp_path / filename
            if filepath.exists():
                size = filepath.stat().st_size
                mtime = filepath.stat().st_mtime
                age = time.time() - mtime
                if age < 60:
                    files_found.append(f"{label}: {size} bytes (updated {age:.0f}s ago)")
                else:
                    files_found.append(f"{label}: {size} bytes")
        
        # Check parent directory for logs
        parent_logs = exp_path.parent.parent / "logs"
        if parent_logs.exists():
            log_files = list(parent_logs.glob("*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                log_age = time.time() - latest_log.stat().st_mtime
                if log_age < 300:
                    status_parts.append(f"📝 Log activity: {latest_log.name} ({log_age:.0f}s ago)")
        
        # Display status
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Clear previous line and show update
        if iteration > 1:
            # Move cursor up based on number of lines
            print(f"\033[{len(status_parts) + len(files_found) + 2}A", end='')
        
        print(f"\n[{timestamp}] Monitoring... (iteration {iteration})")
        
        for part in status_parts:
            print(f"  {part}")
        
        if files_found:
            print("  📁 Files detected:")
            for file_info in files_found:
                print(f"    - {file_info}")
        else:
            print("  ⏳ No output files yet...")
        
        # Check for console output
        if not proc_info['running'] and not files_found:
            print("\n⚠️  Experiment may not be running properly.")
            print("Check the terminal where you started the experiment for errors.")
            print("\n💡 Common issues:")
            print("  - LLM connection failed (check with: python check_llm.py)")
            print("  - Import errors (missing llm_runner_with_memory.py)")
            print("  - Configuration issues")
        
        time.sleep(2)  # Update every 2 seconds

def main():
    if len(sys.argv) > 1:
        exp_dir = sys.argv[1]
    else:
        # Default to first memory experiment
        exp_dir = "memory_llm_experiments/experiments/exp_0001"
    
    if not Path(exp_dir).exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        return 1
    
    try:
        monitor_experiment(exp_dir)
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())