#!/usr/bin/env python3
"""
Test script to diagnose memory agent issues
"""

import time
import sys
from datetime import datetime
from llm_runner_with_memory import LLMSimulationWithMemory
import config as cfg

def test_memory_agent_simple():
    """Test a very simple memory agent simulation"""
    
    print("🧪 Testing Memory Agent - Simple Configuration")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Create a minimal simulation
        sim = LLMSimulationWithMemory(
            run_id=999,
            scenario='baseline',
            use_llm_probability=1.0,
            llm_model=cfg.OLLAMA_MODEL,
            llm_url=cfg.OLLAMA_URL,
            llm_api_key=cfg.OLLAMA_API_KEY,
            enable_memory=True
        )
        
        print(f"✅ Simulation created successfully")
        print(f"   Grid size: {cfg.GRID_SIZE}x{cfg.GRID_SIZE}")
        print(f"   Agents: {cfg.NUM_TYPE_A + cfg.NUM_TYPE_B}")
        print(f"   LLM: {cfg.OLLAMA_MODEL}")
        
        # Override with small grid for testing
        import config as cfg_temp
        original_grid = cfg_temp.GRID_SIZE
        original_a = cfg_temp.NUM_TYPE_A  
        original_b = cfg_temp.NUM_TYPE_B
        
        cfg_temp.GRID_SIZE = 5  # Very small for testing
        cfg_temp.NUM_TYPE_A = 5
        cfg_temp.NUM_TYPE_B = 5
        
        print(f"\n🔧 Using test configuration:")
        print(f"   Grid: 5x5, Agents: 10 total")
        
        # Create new sim with small config
        sim = LLMSimulationWithMemory(
            run_id=999,
            scenario='baseline',
            use_llm_probability=1.0,
            llm_model=cfg.OLLAMA_MODEL,
            llm_url=cfg.OLLAMA_URL,
            llm_api_key=cfg.OLLAMA_API_KEY,
            enable_memory=True
        )
        
        print(f"\n🚀 Starting simulation with max 10 steps...")
        
        # Run with timeout monitoring
        step_times = []
        
        for step in range(10):
            step_start = time.time()
            print(f"\n--- Step {step + 1} ---")
            
            # Run one step
            try:
                moved = sim.run_step()
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                print(f"   ✅ Step completed in {step_time:.2f}s")
                print(f"   Agents moved: {moved}")
                print(f"   LLM calls: {sim.llm_call_count}")
                print(f"   LLM failures: {sim.llm_failure_count}")
                
                if step_time > 30:
                    print(f"   ⚠️  Step took {step_time:.2f}s - this is slow!")
                
                if sim.converged:
                    print(f"   🎯 Converged at step {step + 1}")
                    break
                    
            except Exception as e:
                print(f"   ❌ Step failed: {e}")
                break
        
        # Restore original config
        cfg_temp.GRID_SIZE = original_grid
        cfg_temp.NUM_TYPE_A = original_a
        cfg_temp.NUM_TYPE_B = original_b
        
        total_time = time.time() - start_time
        
        print(f"\n📊 Test Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Steps completed: {len(step_times)}")
        if step_times:
            print(f"   Average step time: {sum(step_times)/len(step_times):.2f}s")
            print(f"   Slowest step: {max(step_times):.2f}s")
        print(f"   LLM calls made: {sim.llm_call_count}")
        print(f"   LLM failures: {sim.llm_failure_count}")
        
        # Check agent memory usage
        memory_agents = []
        for r in range(5):  # Use test grid size
            for c in range(5):
                agent = sim.grid[r][c]
                if agent and hasattr(agent, 'move_history'):
                    memory_agents.append(agent)
        
        if memory_agents:
            print(f"\n🧠 Memory Agent Analysis:")
            print(f"   Memory agents found: {len(memory_agents)}")
            
            for i, agent in enumerate(memory_agents[:3]):  # Show first 3
                print(f"   Agent {i+1}:")
                print(f"     Moves: {len(agent.move_history)}")
                print(f"     Experiences: {len(agent.neighborhood_experiences)}")
                print(f"     Relationships: {len(agent.neighbor_relationships)}")
                print(f"     Satisfaction records: {len(agent.satisfaction_history)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_agent_prompt():
    """Test memory agent prompt generation"""
    
    print("\n🧪 Testing Memory Agent Prompt Generation")
    print("=" * 60)
    
    try:
        from llm_agent_with_memory import LLMAgentWithMemory
        
        # Create a memory agent
        agent = LLMAgentWithMemory(0, 'baseline')
        
        # Add some fake memory
        agent.move_history = [
            {'step': 1, 'from': (0,0), 'to': (1,1), 'reason': 'test'},
            {'step': 5, 'from': (1,1), 'to': (2,2), 'reason': 'another test'}
        ]
        agent.satisfaction_history = [
            {'step': 1, 'satisfaction': 0.8},
            {'step': 2, 'satisfaction': 0.7}
        ]
        agent.neighborhood_experiences = [
            {'step': 3, 'event': 'friendly neighbor', 'impact': 'positive'}
        ]
        
        # Test prompt generation
        print("🔍 Testing prompt generation...")
        
        # Mock neighborhood context
        context = """S S E
S X O  
E O E"""
        
        start_time = time.time()
        prompt = agent.get_enhanced_prompt(context, 'blue team resident')
        prompt_time = time.time() - start_time
        
        print(f"✅ Prompt generated in {prompt_time:.3f}s")
        print(f"📏 Prompt length: {len(prompt)} characters")
        
        if len(prompt) > 4000:
            print(f"⚠️  Prompt is very long ({len(prompt)} chars) - might cause LLM issues")
        
        # Show first and last parts of prompt
        print(f"\n📝 Prompt preview (first 300 chars):")
        print(prompt[:300] + "...")
        
        print(f"\n📝 Prompt ending (last 200 chars):")
        print("..." + prompt[-200:])
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run memory agent diagnostics"""
    
    print("🔍 MEMORY AGENT DIAGNOSTIC TOOL")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {cfg.OLLAMA_MODEL} at {cfg.OLLAMA_URL}")
    
    # Test 1: Prompt generation
    test1_success = test_memory_agent_prompt()
    
    # Test 2: Simple simulation
    test2_success = test_memory_agent_simple()
    
    print(f"\n📋 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Prompt Generation: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"Simple Simulation: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print(f"\n🎉 Memory agents appear to be working correctly!")
        print(f"   The issue may be with larger grid sizes or specific scenarios.")
        print(f"   Try reducing grid size or checking LLM connectivity.")
    else:
        print(f"\n🚨 Memory agents have issues that need to be fixed!")
        print(f"   Check the error messages above for specific problems.")

if __name__ == "__main__":
    main()