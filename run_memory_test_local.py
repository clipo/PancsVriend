#!/usr/bin/env python3
"""
Test memory agents with local LLM or fallback configuration
"""

import sys
import os
from llm_runner_with_memory import LLMSimulationWithMemory
import config as cfg

def test_memory_agents_with_fallback():
    """Test memory agents with proper fallback configuration"""
    
    print("🧠 Testing Memory Agents with Local Configuration")
    print("=" * 60)
    
    # Try local Ollama first
    local_configs = [
        {
            'name': 'Local Ollama',
            'model': 'mixtral:8x22b-instruct',
            'url': 'http://localhost:11434/api/chat/completions',
            'key': None
        },
        {
            'name': 'Local Ollama (alternative)',
            'model': 'llama2:7b',
            'url': 'http://localhost:11434/api/chat/completions', 
            'key': None
        }
    ]
    
    print("🔍 Checking available LLM configurations...")
    
    working_config = None
    
    for config_test in local_configs:
        print(f"\nTesting {config_test['name']}...")
        try:
            from llm_runner import check_llm_connection
            if check_llm_connection(
                llm_model=config_test['model'],
                llm_url=config_test['url'],
                llm_api_key=config_test['key'],
                timeout=3
            ):
                print(f"✅ {config_test['name']} is working!")
                working_config = config_test
                break
            else:
                print(f"❌ {config_test['name']} not available")
        except Exception as e:
            print(f"❌ {config_test['name']} error: {e}")
    
    if not working_config:
        print("\n⚠️  No working LLM found. Testing with mechanical fallback...")
        working_config = {
            'name': 'Mechanical Fallback',
            'model': 'none',
            'url': 'none', 
            'key': 'none'
        }
    
    # Test memory simulation with working config
    print(f"\n🚀 Testing memory simulation with {working_config['name']}...")
    
    # Use small configuration
    original_grid = cfg.GRID_SIZE
    original_a = cfg.NUM_TYPE_A
    original_b = cfg.NUM_TYPE_B
    
    cfg.GRID_SIZE = 5
    cfg.NUM_TYPE_A = 3
    cfg.NUM_TYPE_B = 3
    
    try:
        sim = LLMSimulationWithMemory(
            run_id=999,
            scenario='baseline',
            use_llm_probability=0.5,  # 50% LLM, 50% mechanical fallback
            llm_model=working_config['model'],
            llm_url=working_config['url'],
            llm_api_key=working_config['key'],
            enable_memory=True
        )
        
        print("✅ Memory simulation created successfully!")
        
        # Run a few steps
        print("\n🎯 Running simulation steps...")
        for step in range(5):
            print(f"Step {step + 1}...", end=" ")
            moved = sim.run_step()
            print(f"✅ ({moved} agents moved)")
            
            if sim.converged:
                print("🎯 Converged!")
                break
        
        print(f"\n📊 Results:")
        print(f"   LLM calls: {sim.llm_call_count}")
        print(f"   LLM failures: {sim.llm_failure_count}")
        print(f"   Steps: {sim.step}")
        print(f"   Converged: {sim.converged}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restore original config
        cfg.GRID_SIZE = original_grid
        cfg.NUM_TYPE_A = original_a
        cfg.NUM_TYPE_B = original_b

def main():
    success = test_memory_agents_with_fallback()
    
    if success:
        print(f"\n🎉 Memory agents are working!")
        print(f"\n💡 To fix your hanging issue:")
        print(f"   1. Set up local Ollama: https://ollama.ai")
        print(f"   2. Or update config.py with a working LLM URL")
        print(f"   3. Or reduce use_llm_probability to add mechanical fallback")
    else:
        print(f"\n🚨 Memory agents have fundamental issues beyond LLM connectivity")

if __name__ == "__main__":
    main()