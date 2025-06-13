#!/usr/bin/env python3
"""
Simple memory agent test to check for hanging issues
"""

import sys
import time
import threading
from datetime import datetime

def test_llm_connectivity():
    """Quick LLM connectivity test"""
    print("🔌 Testing LLM connectivity...")
    
    try:
        from llm_runner import check_llm_connection
        result = check_llm_connection(timeout=5)
        if result:
            print("✅ LLM connection working")
            return True
        else:
            print("❌ LLM connection failed")
            return False
    except Exception as e:
        print(f"❌ LLM test error: {e}")
        return False

def test_memory_agent_creation():
    """Test creating a memory agent"""
    print("\n🧠 Testing memory agent creation...")
    
    try:
        from llm_agent_with_memory import LLMAgentWithMemory
        
        start_time = time.time()
        agent = LLMAgentWithMemory(0, 'baseline')
        creation_time = time.time() - start_time
        
        print(f"✅ Memory agent created in {creation_time:.3f}s")
        print(f"   Agent ID: {agent.personal_id}")
        print(f"   Type: {agent.type_id}")
        print(f"   Identity keys: {list(agent.identity.keys()) if hasattr(agent, 'identity') else 'None'}")
        
        return True
    except Exception as e:
        print(f"❌ Memory agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_simulation_creation():
    """Test creating a memory simulation"""
    print("\n⚙️ Testing memory simulation creation...")
    
    try:
        from llm_runner_with_memory import LLMSimulationWithMemory
        import config as cfg
        
        # Save original config
        orig_grid = cfg.GRID_SIZE
        orig_a = cfg.NUM_TYPE_A
        orig_b = cfg.NUM_TYPE_B
        
        # Use tiny config for testing
        cfg.GRID_SIZE = 3
        cfg.NUM_TYPE_A = 2
        cfg.NUM_TYPE_B = 2
        
        start_time = time.time()
        sim = LLMSimulationWithMemory(
            run_id=999,
            scenario='baseline',
            use_llm_probability=1.0,
            enable_memory=True
        )
        creation_time = time.time() - start_time
        
        print(f"✅ Memory simulation created in {creation_time:.3f}s")
        print(f"   Grid size: {cfg.GRID_SIZE}x{cfg.GRID_SIZE}")
        print(f"   Total agents: {cfg.NUM_TYPE_A + cfg.NUM_TYPE_B}")
        
        # Restore original config
        cfg.GRID_SIZE = orig_grid
        cfg.NUM_TYPE_A = orig_a
        cfg.NUM_TYPE_B = orig_b
        
        return True, sim
    except Exception as e:
        print(f"❌ Memory simulation creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def timeout_test(func, timeout_seconds=30):
    """Run a function with timeout"""
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        print(f"⏰ Function timed out after {timeout_seconds}s")
        return False, "timeout"
    elif exception[0]:
        print(f"❌ Function failed: {exception[0]}")
        return False, str(exception[0])
    else:
        return True, result[0]

def main():
    print("🔍 SIMPLE MEMORY AGENT DIAGNOSTIC")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Test 1: LLM connectivity
    llm_ok = test_llm_connectivity()
    
    # Test 2: Memory agent creation
    agent_ok = test_memory_agent_creation()
    
    # Test 3: Memory simulation creation with timeout
    print(f"\n⏱️ Testing simulation creation (30s timeout)...")
    sim_ok, sim_result = timeout_test(test_memory_simulation_creation, 30)
    
    print(f"\n📋 RESULTS:")
    print(f"LLM Connection: {'✅' if llm_ok else '❌'}")
    print(f"Agent Creation: {'✅' if agent_ok else '❌'}")
    print(f"Simulation Creation: {'✅' if sim_ok else '❌'}")
    
    if not sim_ok:
        print(f"\n🚨 MEMORY SIMULATION ISSUE DETECTED!")
        print(f"   This suggests memory agents are hanging during initialization")
        print(f"   or the first step of simulation.")
        
        if sim_result == "timeout":
            print(f"\n💡 LIKELY CAUSES:")
            print(f"   1. LLM calls taking too long or hanging")
            print(f"   2. Memory prompt generation is too complex")
            print(f"   3. Threading issues in memory simulation")
            print(f"   4. Infinite loop in memory agent logic")
    else:
        print(f"\n✅ Memory agents appear to be working!")
        print(f"   The issue may be with larger configurations or specific scenarios.")

if __name__ == "__main__":
    main()