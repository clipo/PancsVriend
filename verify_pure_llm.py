#!/usr/bin/env python3
"""
Verify that LLM experiments are using 100% LLM decisions
"""

import json
from pathlib import Path

def check_llm_usage_in_experiments():
    """Check recent LLM experiments for pure LLM usage"""
    
    print("🔍 VERIFYING PURE LLM AGENT USAGE")
    print("=" * 50)
    
    # Check comprehensive study directories
    study_dirs = list(Path(".").glob("comprehensive_study_*"))
    
    if not study_dirs:
        print("❌ No comprehensive study directories found")
        return
    
    latest_study = max(study_dirs, key=lambda x: x.name)
    print(f"📁 Checking latest study: {latest_study.name}")
    
    llm_results_dir = latest_study / "llm_results"
    if not llm_results_dir.exists():
        print("❌ No LLM results directory found")
        return
    
    experiments_dir = llm_results_dir / "experiments"
    if not experiments_dir.exists():
        print("❌ No experiments directory found")
        return
    
    # Check individual experiments
    exp_dirs = list(experiments_dir.glob("exp_*"))
    if not exp_dirs:
        print("❌ No experiment directories found")
        return
    
    print(f"🧪 Found {len(exp_dirs)} experiments to check")
    
    llm_usage_data = []
    
    for exp_dir in exp_dirs:
        # Try both possible config file names
        config_file = exp_dir / "experiment_config.json"
        if not config_file.exists():
            config_file = exp_dir / "config.json"
        
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                
                agent_type = config.get('agent_type', 'unknown')
                scenario = config.get('scenario', 'unknown')
                use_llm_prob = config.get('use_llm_probability', 'not_set')
                
                llm_usage_data.append({
                    'experiment': exp_dir.name,
                    'agent_type': agent_type,
                    'scenario': scenario,
                    'llm_probability': use_llm_prob
                })
                
            except Exception as e:
                print(f"⚠️  Error reading {exp_dir}: {e}")
    
    if not llm_usage_data:
        print("❌ No valid experiment configurations found")
        return
    
    # Analyze LLM usage
    print(f"\n📊 LLM USAGE ANALYSIS")
    print("-" * 30)
    
    pure_llm_count = 0
    mixed_count = 0
    mechanical_count = 0
    
    for exp in llm_usage_data:
        prob = exp['llm_probability']
        if prob == 1.0:
            pure_llm_count += 1
            status = "✅ Pure LLM"
        elif prob == 0.0:
            mechanical_count += 1
            status = "⚪ Pure Mechanical"
        elif 0 < prob < 1.0:
            mixed_count += 1
            status = f"⚠️  Mixed ({prob*100:.0f}% LLM)"
        else:
            status = f"❓ Unknown ({prob})"
        
        print(f"{exp['experiment']}: {exp['agent_type']:8} | {exp['scenario']:20} | {status}")
    
    print(f"\n📈 SUMMARY")
    print("-" * 20)
    print(f"✅ Pure LLM (100%):     {pure_llm_count}")
    print(f"⚠️  Mixed usage:        {mixed_count}")
    print(f"⚪ Pure Mechanical (0%): {mechanical_count}")
    
    if mixed_count > 0:
        print(f"\n❌ WARNING: Found {mixed_count} experiments with mixed LLM usage!")
        print("   This means some agents are using LLM while others use mechanical decisions.")
        print("   For pure comparison, all experiments should be 100% LLM (except mechanical baseline).")
    elif pure_llm_count > 0:
        print(f"\n✅ PERFECT: All {pure_llm_count} LLM experiments are using 100% LLM decisions!")
        print("   This ensures pure agent type comparison: Mechanical vs LLM vs Memory-LLM")
    
    # Check agent types
    agent_types = set(exp['agent_type'] for exp in llm_usage_data)
    print(f"\n🤖 AGENT TYPES FOUND")
    print("-" * 20)
    for agent_type in sorted(agent_types):
        count = sum(1 for exp in llm_usage_data if exp['agent_type'] == agent_type)
        print(f"{agent_type}: {count} experiments")
    
    # Check scenarios
    scenarios = set(exp['scenario'] for exp in llm_usage_data)
    print(f"\n🎭 SCENARIOS TESTED")
    print("-" * 20)
    for scenario in sorted(scenarios):
        count = sum(1 for exp in llm_usage_data if exp['scenario'] == scenario)
        print(f"{scenario}: {count} experiments")

if __name__ == "__main__":
    check_llm_usage_in_experiments()