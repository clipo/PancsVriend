#!/usr/bin/env python3
"""
Kill stuck experiment processes and clean up
"""

import subprocess
import signal
import os
import time

def kill_stuck_experiments():
    """Kill any stuck experiment processes"""
    
    print("🔍 Finding stuck experiment processes...")
    
    # Find all Python processes related to experiments
    result = subprocess.run(
        ["ps", "aux"], 
        capture_output=True, 
        text=True
    )
    
    processes_to_kill = []
    
    for line in result.stdout.split('\n'):
        if any(keyword in line for keyword in ['llm_runner.py', 'experiment_explorer.py', 'comprehensive_comparison_study.py', 'run_design_space_exploration.py']):
            if 'grep' not in line:
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    cmd = ' '.join(parts[10:])
                    processes_to_kill.append((pid, cmd))
    
    if not processes_to_kill:
        print("✅ No stuck processes found")
        return
    
    print(f"🚨 Found {len(processes_to_kill)} processes to kill:")
    for pid, cmd in processes_to_kill:
        print(f"   PID {pid}: {cmd[:80]}...")
    
    print(f"\n⚠️  Killing processes...")
    
    for pid, cmd in processes_to_kill:
        try:
            os.kill(int(pid), signal.SIGTERM)
            print(f"   ✅ Killed PID {pid}")
        except ProcessLookupError:
            print(f"   ⚠️  PID {pid} already dead")
        except Exception as e:
            print(f"   ❌ Failed to kill PID {pid}: {e}")
    
    # Wait a moment for processes to die
    time.sleep(2)
    
    # Force kill any remaining processes
    for pid, cmd in processes_to_kill:
        try:
            os.kill(int(pid), signal.SIGKILL)
            print(f"   🔨 Force killed PID {pid}")
        except ProcessLookupError:
            pass  # Process already dead
        except Exception as e:
            print(f"   ❌ Failed to force kill PID {pid}: {e}")
    
    print(f"\n✅ Process cleanup complete")

if __name__ == "__main__":
    kill_stuck_experiments()