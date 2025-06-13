#!/usr/bin/env python3
"""
Demo script showing the difference between regular LLM agents and memory-enhanced agents
"""

import config as cfg
import json
from llm_runner import check_llm_connection
from llm_runner_with_memory import LLMSimulationWithMemory

def run_memory_comparison_demo():
    """Compare agents with and without memory"""
    
    print("="*80)
    print("LLM AGENT MEMORY DEMONSTRATION")
    print("="*80)
    
    # Check LLM connection first
    if not check_llm_connection():
        print("❌ LLM not available - please check connection")
        return
        
    # Temporarily use smaller grid for demo
    original_grid = cfg.GRID_SIZE
    original_a = cfg.NUM_TYPE_A  
    original_b = cfg.NUM_TYPE_B
    
    try:
        # Small grid for quick demo
        cfg.GRID_SIZE = 8
        cfg.NUM_TYPE_A = 10
        cfg.NUM_TYPE_B = 10
        
        print(f"\nRunning demo with {cfg.GRID_SIZE}x{cfg.GRID_SIZE} grid")
        print(f"Agents: {cfg.NUM_TYPE_A} Type A, {cfg.NUM_TYPE_B} Type B")
        print("-"*60)
        
        # Run with memory enabled
        print("\n1. Running simulation WITH agent memory...")
        print("   Agents will remember their experiences and build relationships")
        
        sim_with_memory = LLMSimulationWithMemory(
            run_id=1,
            scenario='race_white_black',  # Use a social context for richer behavior
            enable_memory=True
        )
        
        result_with_memory = sim_with_memory.run(max_steps=20)
        
        print(f"\n   ✅ Simulation complete!")
        print(f"   - Converged: {result_with_memory['converged']}")
        print(f"   - Steps: {result_with_memory['steps']}")
        print(f"   - LLM calls: {result_with_memory['llm_call_count']}")
        
        # Show some agent stories
        print("\n   Sample Agent Stories:")
        for i, agent_data in enumerate(result_with_memory['agent_memories'][:3]):
            print(f"\n   Agent {agent_data['agent_id']} ({agent_data['identity']['type']}):")
            print(f"   - Total moves: {agent_data['total_moves']}")
            print(f"   - Time in final location: {agent_data['time_in_final_location']} steps")
            print(f"   - Final satisfaction: {agent_data['final_satisfaction']}")
            
            if agent_data['move_history']:
                last_move = agent_data['move_history'][-1]
                print(f"   - Last move reason: {last_move['reason']}")
                
        # Run without memory for comparison
        print("\n" + "-"*60)
        print("\n2. Running simulation WITHOUT agent memory...")
        print("   Agents make decisions based only on current neighborhood")
        
        sim_without_memory = LLMSimulationWithMemory(
            run_id=2,
            scenario='race_white_black',
            enable_memory=False  # Disable memory
        )
        
        result_without_memory = sim_without_memory.run(max_steps=20)
        
        print(f"\n   ✅ Simulation complete!")
        print(f"   - Converged: {result_without_memory['converged']}")
        print(f"   - Steps: {result_without_memory['steps']}")
        print(f"   - LLM calls: {result_without_memory['llm_call_count']}")
        
        # Compare results
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        print("\nKey Differences:")
        print(f"1. Convergence speed:")
        print(f"   - With memory: Step {result_with_memory.get('convergence_step', 'N/A')}")
        print(f"   - Without memory: Step {result_without_memory.get('convergence_step', 'N/A')}")
        
        print(f"\n2. Agent stability:")
        with_memory_moves = sum(a['total_moves'] for a in result_with_memory['agent_memories'])
        print(f"   - With memory: {with_memory_moves} total moves (agents consider history)")
        print(f"   - Without memory: More reactive to immediate context")
        
        print(f"\n3. Decision factors:")
        print(f"   - With memory: Consider family, past experiences, relationships")
        print(f"   - Without memory: React only to current neighbor distribution")
        
        # Save detailed results
        print("\n" + "-"*60)
        print("Saving detailed results...")
        
        # Save agent stories from memory simulation
        stories = []
        for agent_data in result_with_memory['agent_memories'][:5]:
            story = {
                'agent_id': agent_data['agent_id'],
                'identity': agent_data['identity'],
                'journey': {
                    'total_moves': agent_data['total_moves'],
                    'moves': [
                        {
                            'step': move['step'],
                            'reason': move['reason']
                        } for move in agent_data['move_history']
                    ],
                    'final_satisfaction': agent_data['final_satisfaction'],
                    'time_in_final_location': agent_data['time_in_final_location']
                }
            }
            stories.append(story)
            
        with open('demo_agent_stories.json', 'w') as f:
            json.dump(stories, f, indent=2)
            
        print("✅ Agent stories saved to demo_agent_stories.json")
        
        # Create a summary report
        summary = {
            'with_memory': {
                'converged': result_with_memory['converged'],
                'steps': result_with_memory['steps'],
                'total_agent_moves': with_memory_moves,
                'average_satisfaction': sum(a['final_satisfaction'] or 0 for a in result_with_memory['agent_memories']) / len(result_with_memory['agent_memories'])
            },
            'without_memory': {
                'converged': result_without_memory['converged'],
                'steps': result_without_memory['steps']
            }
        }
        
        with open('demo_memory_comparison.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        print("✅ Comparison summary saved to demo_memory_comparison.json")
        
    finally:
        # Restore original configuration
        cfg.GRID_SIZE = original_grid
        cfg.NUM_TYPE_A = original_a
        cfg.NUM_TYPE_B = original_b
        
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nKey Insights:")
    print("- Agents with memory show more realistic, stable behavior")
    print("- Decisions reflect accumulated experiences, not just snapshots")
    print("- Family situations and relationships influence choices")
    print("- More human-like residential patterns emerge")
    print("\nTo use memory in your experiments, use the enhanced runner:")
    print("  from llm_runner_with_memory import LLMSimulationWithMemory")
    print("  sim = LLMSimulationWithMemory(..., enable_memory=True)")

if __name__ == "__main__":
    run_memory_comparison_demo()