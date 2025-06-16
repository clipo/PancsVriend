#!/usr/bin/env python3
"""
Direct memory experiment runner with visible output
"""

import os
import sys
import json
from pathlib import Path

# Enable debugging
os.environ['DEBUG_LLM'] = 'true'

print("🚀 DIRECT MEMORY EXPERIMENT RUNNER")
print("=" * 50)

# Load experiment config
exp_dir = Path("memory_llm_experiments/experiments/exp_0001")
config_file = exp_dir / "experiment_config.json"

if not config_file.exists():
    print(f"❌ Config not found: {config_file}")
    sys.exit(1)

with open(config_file) as f:
    exp_config = json.load(f)

print(f"📋 Experiment: {exp_config['name']}")
print(f"🤖 Type: {exp_config['agent_type']}")
print(f"🎭 Scenario: {exp_config['scenario']}")
print(f"📏 Grid: {exp_config['grid_size']}x{exp_config['grid_size']}")
print(f"👥 Agents: {exp_config['num_type_a']} + {exp_config['num_type_b']}")
print(f"🔄 Runs: {exp_config['n_runs']}")
print("-" * 50)

# Configure the simulation
import config as cfg
cfg.GRID_SIZE = exp_config['grid_size']
cfg.NUM_TYPE_A = exp_config['num_type_a']
cfg.NUM_TYPE_B = exp_config['num_type_b']

print(f"\n✅ Configuration set:")
print(f"   GRID_SIZE = {cfg.GRID_SIZE}")
print(f"   NUM_TYPE_A = {cfg.NUM_TYPE_A}")
print(f"   NUM_TYPE_B = {cfg.NUM_TYPE_B}")

# Import and run
print(f"\n📦 Importing LLM runner with memory...")
try:
    from llm_runner_with_memory import LLMSimulationWithMemory
    print("✅ Import successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test with just one run first
print(f"\n🧪 Running test run (1 of {exp_config['n_runs']})...")
print("You should see DEBUG output below:")
print("-" * 50)

try:
    sim = LLMSimulationWithMemory(
        run_id=0,
        scenario=exp_config['scenario'],
        use_llm_probability=exp_config['use_llm_probability'],
        llm_model=exp_config['llm_model'],
        llm_url=exp_config['llm_url'],
        llm_api_key=exp_config['llm_api_key'],
        enable_memory=True
    )
    
    print(f"\n✅ Simulation created")
    print(f"🏃 Starting run with max_steps={exp_config['max_steps']}...")
    
    # Run with limited steps for testing
    result = sim.run(max_steps=10)  # Just 10 steps for testing
    
    print(f"\n✅ Test run completed!")
    print(f"   Converged: {result['converged']}")
    print(f"   Final step: {result['final_step']}")
    print(f"   LLM calls: {result.get('llm_call_count', 'unknown')}")
    
    # Save test result
    test_output = exp_dir / "test_result.json"
    with open(test_output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"   Saved to: {test_output}")
    
except Exception as e:
    print(f"\n❌ Error during simulation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n✨ Test successful! Ready for full experiment.")
print(f"\nTo run the full experiment, modify this script to run all {exp_config['n_runs']} runs.")