#!/usr/bin/env python3
"""
Monitor individual LLM API calls in real-time
Shows each agent's decision request and response
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
import subprocess

def tail_file(filepath, n=10):
    """Get last n lines of a file efficiently"""
    try:
        result = subprocess.run(['tail', '-n', str(n), str(filepath)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
    except:
        pass
    return []

def monitor_llm_activity():
    """Monitor LLM activity across all active experiments"""
    
    print("🔍 REAL-TIME LLM CALL MONITOR")
    print("=" * 60)
    print("Monitoring for individual LLM agent decisions...")
    print("(Press Ctrl+C to stop)\n")
    
    # Track what we've already seen
    seen_calls = set()
    call_count = 0
    start_time = time.time()
    
    # Look for active experiments
    while True:
        found_activity = False
        
        # Check all possible log locations
        log_patterns = [
            "memory_llm_experiments/logs/*.log",
            "comprehensive_study_*/llm_results/logs/*.log",
            "experiments/llm_*/logs/*.log",
            "*.log"
        ]
        
        for pattern in log_patterns:
            for log_file in Path(".").glob(pattern):
                if log_file.exists():
                    # Check file modification time
                    mtime = log_file.stat().st_mtime
                    if time.time() - mtime < 300:  # Active in last 5 minutes
                        
                        # Read recent lines
                        lines = tail_file(log_file, 50)
                        
                        for line in lines:
                            # Look for LLM call indicators
                            if any(indicator in line.lower() for indicator in 
                                   ['llm', 'agent', 'decision', 'move', 'api', 'request']):
                                
                                # Create unique identifier for this log entry
                                line_id = f"{log_file}:{line[:50]}"
                                
                                if line_id not in seen_calls:
                                    seen_calls.add(line_id)
                                    call_count += 1
                                    found_activity = True
                                    
                                    # Parse and display the activity
                                    timestamp = datetime.now().strftime("%H:%M:%S")
                                    
                                    # Try to extract agent info
                                    agent_info = "Unknown"
                                    if "agent" in line.lower():
                                        # Try to find coordinates
                                        import re
                                        coords = re.findall(r'\((\d+),\s*(\d+)\)', line)
                                        if coords:
                                            agent_info = f"Agent at ({coords[0][0]},{coords[0][1]})"
                                    
                                    # Determine action type
                                    if "timeout" in line.lower():
                                        action = "⏰ TIMEOUT"
                                        color = "\033[93m"  # Yellow
                                    elif "error" in line.lower():
                                        action = "❌ ERROR"
                                        color = "\033[91m"  # Red
                                    elif "decision" in line.lower() or "move" in line.lower():
                                        action = "💭 DECISION"
                                        color = "\033[92m"  # Green
                                    else:
                                        action = "📡 CALL"
                                        color = "\033[94m"  # Blue
                                    
                                    # Display the call
                                    print(f"{color}[{timestamp}] {action} | {agent_info} | Calls: {call_count}{'\033[0m'}")
                                    
                                    # Show abbreviated log content
                                    if len(line) > 80:
                                        print(f"  └─ {line[:77]}...")
                                    else:
                                        print(f"  └─ {line}")
        
        # Show periodic summary
        if call_count > 0 and call_count % 10 == 0:
            elapsed = time.time() - start_time
            rate = call_count / elapsed if elapsed > 0 else 0
            print(f"\n📊 Summary: {call_count} LLM calls | {rate:.1f} calls/sec | Runtime: {elapsed/60:.1f}m\n")
        
        time.sleep(0.5)  # Check every 500ms

def monitor_specific_experiment(exp_dir):
    """Monitor a specific experiment directory"""
    
    print(f"📁 Monitoring experiment: {exp_dir}")
    
    # Look for progress indicators
    progress_file = exp_dir / "progress_realtime.json"
    metrics_file = exp_dir / "metrics_history.csv"
    
    last_progress = None
    last_metrics_size = 0
    
    while True:
        updates = []
        
        # Check progress file
        if progress_file.exists():
            try:
                with open(progress_file) as f:
                    progress = json.load(f)
                
                if progress != last_progress:
                    last_progress = progress
                    
                    run = progress.get('current_run', 0)
                    total_runs = progress.get('total_runs', 0)
                    step = progress.get('current_step', 0)
                    max_steps = progress.get('max_steps', 0)
                    
                    # Estimate LLM calls
                    # Assuming each step processes all agents once
                    grid_size = 15  # Default, could parse from config
                    agents_per_step = 50  # Approximate
                    total_calls = run * max_steps * agents_per_step + step * agents_per_step
                    
                    updates.append(f"🏃 Run {run}/{total_runs} | Step {step}/{max_steps} | ~{total_calls:,} agent decisions")
                    
            except:
                pass
        
        # Check metrics file growth
        if metrics_file.exists():
            current_size = metrics_file.stat().st_size
            if current_size > last_metrics_size:
                growth = current_size - last_metrics_size
                last_metrics_size = current_size
                updates.append(f"📈 Metrics grew by {growth} bytes")
        
        # Display updates
        if updates:
            timestamp = datetime.now().strftime("%H:%M:%S")
            for update in updates:
                print(f"[{timestamp}] {update}")
        
        time.sleep(2)

def main():
    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        # Monitor specific experiment
        exp_dir = Path(sys.argv[1])
        if exp_dir.exists():
            try:
                monitor_specific_experiment(exp_dir)
            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped")
        else:
            print(f"❌ Directory not found: {exp_dir}")
            return 1
    else:
        # Monitor all LLM activity
        try:
            monitor_llm_activity()
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())