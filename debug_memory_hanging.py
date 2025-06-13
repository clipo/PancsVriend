#!/usr/bin/env python3
"""
Debug memory agent hanging issues with detailed logging
"""

import time
import threading
import signal
import sys
from datetime import datetime

def timeout_handler(signum, frame):
    print(f"\n⏰ TIMEOUT! Process took longer than expected")
    print(f"   This suggests a hanging issue in memory agent processing")
    sys.exit(1)

def debug_memory_agent_step_by_step():
    """Debug memory agent execution step by step"""
    
    print("🔍 DEBUGGING MEMORY AGENT EXECUTION")
    print("=" * 60)
    
    # Set timeout alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60 second timeout
    
    try:
        print("1️⃣ Testing imports...")
        start = time.time()
        
        from llm_runner_with_memory import LLMSimulationWithMemory
        from llm_agent_with_memory import LLMAgentWithMemory
        import config as cfg
        
        print(f"   ✅ Imports successful ({time.time() - start:.3f}s)")
        
        print("\n2️⃣ Testing memory agent creation...")
        start = time.time()
        
        agent = LLMAgentWithMemory(0, 'baseline')
        print(f"   ✅ Memory agent created ({time.time() - start:.3f}s)")
        print(f"      Agent ID: {agent.personal_id}")
        
        print("\n3️⃣ Testing simulation initialization...")
        start = time.time()
        
        # Use very small grid
        orig_grid = cfg.GRID_SIZE
        orig_a = cfg.NUM_TYPE_A
        orig_b = cfg.NUM_TYPE_B
        
        cfg.GRID_SIZE = 3
        cfg.NUM_TYPE_A = 2
        cfg.NUM_TYPE_B = 2
        
        sim = LLMSimulationWithMemory(
            run_id=999,
            scenario='baseline',
            use_llm_probability=1.0,
            enable_memory=True
        )
        
        print(f"   ✅ Simulation created ({time.time() - start:.3f}s)")
        
        # Restore config
        cfg.GRID_SIZE = orig_grid
        cfg.NUM_TYPE_A = orig_a
        cfg.NUM_TYPE_B = orig_b
        
        print("\n4️⃣ Testing first simulation step...")
        start = time.time()
        
        # This is where it might hang
        print("   🔄 Running step (this is where hanging usually occurs)...")
        
        # Add step-by-step monitoring
        step_start = time.time()
        
        # Run step with monitoring
        moved = sim.run_step()
        step_time = time.time() - step_start
        
        print(f"   ✅ First step completed ({step_time:.3f}s)")
        print(f"      Agents moved: {moved}")
        print(f"      LLM calls: {sim.llm_call_count}")
        print(f"      LLM failures: {sim.llm_failure_count}")
        
        if step_time > 10:
            print(f"   ⚠️  Step took {step_time:.1f}s - investigating slow components...")
            
            # Check threading state
            if hasattr(sim, 'llm_thread') and sim.llm_thread:
                print(f"      LLM thread alive: {sim.llm_thread.is_alive()}")
            
            # Check queue sizes
            if hasattr(sim, 'query_queue'):
                print(f"      Query queue size: {sim.query_queue.qsize()}")
            if hasattr(sim, 'result_queue'):
                print(f"      Result queue size: {sim.result_queue.qsize()}")
        
        print("\n5️⃣ Testing second step...")
        start = time.time()
        
        moved2 = sim.run_step()
        step_time2 = time.time() - start
        
        print(f"   ✅ Second step completed ({step_time2:.3f}s)")
        print(f"      Agents moved: {moved2}")
        print(f"      Total LLM calls: {sim.llm_call_count}")
        
        print(f"\n🎉 SUCCESS! Memory agents are working correctly")
        print(f"   First step: {step_time:.3f}s")
        print(f"   Second step: {step_time2:.3f}s")
        
        if step_time > 5 or step_time2 > 5:
            print(f"\n💡 PERFORMANCE NOTES:")
            print(f"   - Steps are taking longer than expected")
            print(f"   - This might explain why experiments appear to hang")
            print(f"   - Memory agents may just be very slow on larger grids")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during debugging: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        signal.alarm(0)  # Cancel timeout

def check_current_experiment_status():
    """Check what the current experiment is doing"""
    
    print("\n🔍 CHECKING CURRENT EXPERIMENT STATUS")
    print("=" * 60)
    
    import glob
    import json
    
    # Find most recent experiment directory
    exp_dirs = glob.glob("comprehensive_study_*/llm_results")
    if not exp_dirs:
        print("❌ No experiment directories found")
        return
    
    latest_dir = max(exp_dirs, key=lambda x: os.path.getmtime(x))
    print(f"📁 Checking: {latest_dir}")
    
    # Check progress
    progress_files = glob.glob(f"{latest_dir}/progress_*.json")
    if progress_files:
        latest_progress = max(progress_files, key=lambda x: os.path.getmtime(x))
        with open(latest_progress) as f:
            progress = json.load(f)
        
        print(f"📊 Progress: {progress.get('completed', 0)}/{progress.get('total_planned', 0)}")
        print(f"📊 Success rate: {progress.get('successful', 0)}/{progress.get('completed', 0)}")
    
    # Check which experiment is currently running
    exp_configs = glob.glob(f"{latest_dir}/experiments/exp_*/experiment_config.json")
    
    completed_exps = []
    for config_file in sorted(exp_configs):
        exp_dir = os.path.dirname(config_file)
        results_file = os.path.join(exp_dir, "results.json")
        
        with open(config_file) as f:
            config = json.load(f)
        
        if os.path.exists(results_file):
            completed_exps.append(config['experiment_id'])
        else:
            print(f"🔄 Currently running: {config['experiment_id']}")
            print(f"   Description: {config.get('description', 'N/A')}")
            print(f"   Agent type: {config.get('agent_type', 'N/A')}")
            print(f"   Started: {time.ctime(os.path.getmtime(config_file))}")
            break
    
    print(f"✅ Completed experiments: {len(completed_exps)}")
    if completed_exps:
        print(f"   Latest: {completed_exps[-1]}")

def main():
    import os
    
    print("🐛 MEMORY AGENT HANGING DIAGNOSTIC")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check current status first
    check_current_experiment_status()
    
    # Then run debugging
    print(f"\n" + "="*60)
    success = debug_memory_agent_step_by_step()
    
    if success:
        print(f"\n💭 DIAGNOSIS:")
        print(f"   Memory agents are technically working, but may be very slow")
        print(f"   on larger grids due to complex LLM prompts and processing.")
        print(f"   ")
        print(f"   The 'hanging' might actually be normal slow execution")
        print(f"   for memory agents with large numbers of agents.")
    else:
        print(f"\n🚨 DIAGNOSIS:")
        print(f"   Memory agents have a fundamental issue that needs fixing")

if __name__ == "__main__":
    main()