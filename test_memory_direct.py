#!/usr/bin/env python3
"""
Direct test of memory LLM runner to diagnose issues
"""

import os
import sys

# Enable debugging
os.environ['DEBUG_LLM'] = 'true'

print("🧪 DIRECT MEMORY LLM TEST")
print("=" * 50)

# Test 1: Check imports
print("\n1️⃣ Testing imports...")
try:
    import config as cfg
    print("✅ config imported")
    
    from llm_runner_with_memory import LLMSimulationWithMemory
    print("✅ LLMSimulationWithMemory imported")
    
    from llm_agent_with_memory import LLMAgentWithMemory
    print("✅ LLMAgentWithMemory imported")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Check LLM connection
print("\n2️⃣ Testing LLM connection...")
try:
    from llm_runner import check_llm_connection
    if check_llm_connection():
        print("✅ LLM connection successful")
    else:
        print("❌ LLM connection failed")
        sys.exit(1)
except Exception as e:
    print(f"❌ Connection test error: {e}")

# Test 3: Create minimal simulation
print("\n3️⃣ Creating minimal memory simulation...")
try:
    # Use tiny grid
    cfg.GRID_SIZE = 3
    cfg.NUM_TYPE_A = 2
    cfg.NUM_TYPE_B = 2
    
    print(f"   Grid: {cfg.GRID_SIZE}x{cfg.GRID_SIZE}")
    print(f"   Agents: {cfg.NUM_TYPE_A + cfg.NUM_TYPE_B}")
    
    sim = LLMSimulationWithMemory(
        run_id=0,
        scenario='baseline',
        use_llm_probability=1.0,
        enable_memory=True
    )
    print("✅ Simulation created")
    
except Exception as e:
    print(f"❌ Simulation creation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Run one step
print("\n4️⃣ Running single step...")
try:
    print("   This should show DEBUG output for each agent decision...")
    sim.run_step()
    print(f"✅ Step completed! Converged: {sim.converged}")
    
except Exception as e:
    print(f"❌ Step execution error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check for output
print("\n5️⃣ Checking for data output...")
if hasattr(sim, 'metrics_history') and sim.metrics_history:
    print(f"✅ Metrics recorded: {len(sim.metrics_history)} entries")
else:
    print("❌ No metrics recorded")

print("\n✨ Test complete!")
print("\nIf you saw DEBUG output above, the system is working.")
print("If not, there may be an issue with the LLM agent implementation.")