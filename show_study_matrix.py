#!/usr/bin/env python3
"""
Show the complete experimental design matrix for the comprehensive comparison study
"""

import pandas as pd
import yaml

def show_complete_study_matrix():
    """Display the full experimental design matrix"""
    
    print("🔬 COMPREHENSIVE COMPARISON STUDY MATRIX")
    print("=" * 80)
    
    # Load configuration
    with open("baseline_vs_llm_study.yaml") as f:
        config = yaml.safe_load(f)
    
    scenarios = config["scenarios"]
    grid_configs = list(config["grid_configurations"].keys())
    runs_per_config = config["experiment_parameters"]["runs_per_config"]
    
    # Create complete experimental matrix
    experiments = []
    
    # 1. Mechanical Baseline (traditional Schelling) - NO LLM
    for scenario in scenarios:
        for grid in grid_configs:
            experiments.append({
                "Agent Type": "🔧 Mechanical",
                "LLM Used": "None",
                "Scenario": scenario,
                "Grid Size": grid,
                "Runs": runs_per_config,
                "Description": "Traditional utility-maximizing agents"
            })
    
    # 2. Standard LLM Agents - WITH LLM but no memory
    for scenario in scenarios:
        for grid in grid_configs:
            experiments.append({
                "Agent Type": "🤖 Standard LLM", 
                "LLM Used": "qwen2.5-coder:32B",
                "Scenario": scenario,
                "Grid Size": grid,
                "Runs": runs_per_config,
                "Description": "LLM decisions based on current neighborhood only"
            })
    
    # 3. Memory LLM Agents - WITH LLM and memory/identity
    for scenario in scenarios:
        for grid in grid_configs:
            experiments.append({
                "Agent Type": "🧠 Memory LLM",
                "LLM Used": "qwen2.5-coder:32B", 
                "Scenario": scenario,
                "Grid Size": grid,
                "Runs": runs_per_config,
                "Description": "LLM with personal history, relationships, identity"
            })
    
    # Create DataFrame for display
    df = pd.DataFrame(experiments)
    
    print(f"Total Experiments: {len(experiments)}")
    print(f"Total Simulation Runs: {len(experiments) * runs_per_config}")
    print("\n" + "=" * 80)
    
    # Show matrix overview
    print("\nEXPERIMENT MATRIX OVERVIEW:")
    print("-" * 40)
    
    matrix_summary = df.groupby(["Agent Type", "Scenario"]).size().unstack(fill_value=0)
    print(matrix_summary)
    
    print(f"\nThis gives us a complete {len(config['scenarios'])} × 3 × {len(grid_configs)} design:")
    print(f"• {len(scenarios)} Social Contexts")
    print(f"• 3 Agent Types (Mechanical, Standard LLM, Memory LLM)")  
    print(f"• {len(grid_configs)} Grid Sizes")
    print(f"• {runs_per_config} runs each = {len(experiments) * runs_per_config} total simulations")
    
    # Show what questions this answers
    print(f"\n🎯 RESEARCH QUESTIONS THIS DESIGN ANSWERS:")
    print("-" * 50)
    print("1. 📐 vs 🤖: How do LLM agents differ from traditional mechanical agents?")
    print("2. 🤖 vs 🧠: How does memory change LLM agent behavior?") 
    print("3. 🌍 Context Effects: Do social contexts affect each agent type differently?")
    print("4. 📊 Scale Effects: How do population sizes influence each agent type?")
    print("5. 🔄 Interaction Effects: Which combinations produce unique patterns?")
    
    # Show expected outcomes
    print(f"\n🔮 EXPECTED FINDINGS:")
    print("-" * 30)
    print("📐 Mechanical Baseline:")
    print("   • Fast, predictable convergence")
    print("   • Same patterns across all social contexts")
    print("   • Pure utility-maximization")
    
    print("\n🤖 Standard LLM Agents:")
    print("   • Context-sensitive behavior") 
    print("   • More variation than mechanical")
    print("   • Reactive to immediate neighborhood")
    
    print("\n🧠 Memory LLM Agents:")
    print("   • Slower, more stable convergence")
    print("   • Family/relationship considerations")
    print("   • Human-like residential patterns")
    print("   • Individual agent stories")
    
    # Show deliverables
    print(f"\n📊 ANALYSIS DELIVERABLES:")
    print("-" * 35)
    print("• Statistical comparisons between all agent types")
    print("• Segregation pattern analysis across social contexts")
    print("• Convergence speed analysis") 
    print("• Individual agent journey analysis (memory agents)")
    print("• Publication-ready visualizations")
    print("• Comprehensive research report")
    
    return experiments

def estimate_study_resources():
    """Estimate time and cost for the complete study"""
    
    with open("baseline_vs_llm_study.yaml") as f:
        config = yaml.safe_load(f)
    
    scenarios = len(config["scenarios"])
    grids = len(config["grid_configurations"])
    runs = config["experiment_parameters"]["runs_per_config"]
    
    # Experiments breakdown
    mechanical_experiments = scenarios * grids
    llm_experiments = scenarios * grids * 2  # standard + memory
    total_experiments = mechanical_experiments + llm_experiments
    total_runs = total_experiments * runs
    
    # Time estimates (conservative)
    mechanical_time_per_run = 30  # seconds
    llm_standard_time_per_run = 180  # 3 minutes
    llm_memory_time_per_run = 270   # 4.5 minutes
    
    mechanical_total_time = mechanical_experiments * runs * mechanical_time_per_run
    llm_standard_total_time = (scenarios * grids) * runs * llm_standard_time_per_run  
    llm_memory_total_time = (scenarios * grids) * runs * llm_memory_time_per_run
    
    total_time_seconds = mechanical_total_time + llm_standard_total_time + llm_memory_total_time
    total_time_hours = total_time_seconds / 3600
    
    # Cost estimates (LLM only, mechanical is free)
    tokens_per_standard_run = 500000  # 500K tokens per run
    tokens_per_memory_run = 650000    # 650K tokens per run (more context)
    cost_per_1k_tokens = 0.0  # Free for local qwen model
    
    llm_standard_cost = (scenarios * grids * runs * tokens_per_standard_run / 1000) * cost_per_1k_tokens
    llm_memory_cost = (scenarios * grids * runs * tokens_per_memory_run / 1000) * cost_per_1k_tokens
    total_cost = llm_standard_cost + llm_memory_cost
    
    print(f"\n💰 RESOURCE ESTIMATES:")
    print("-" * 30)
    print(f"Total Experiments: {total_experiments}")
    print(f"Total Simulation Runs: {total_runs:,}")
    print(f"Estimated Runtime: {total_time_hours:.1f} hours")
    print(f"Estimated Cost: ${total_cost:.2f} (free with local qwen)")
    
    print(f"\nBreakdown:")
    print(f"• Mechanical: {mechanical_experiments} experiments, {mechanical_total_time/3600:.1f} hours")
    print(f"• Standard LLM: {scenarios * grids} experiments, {llm_standard_total_time/3600:.1f} hours")
    print(f"• Memory LLM: {scenarios * grids} experiments, {llm_memory_total_time/3600:.1f} hours")
    
    if total_time_hours > 24:
        print(f"\n⚠️  Large study! Consider:")
        print(f"   • Running overnight/weekend")
        print(f"   • Using quick test mode first") 
        print(f"   • Parallel execution on multiple cores")

if __name__ == "__main__":
    experiments = show_complete_study_matrix()
    estimate_study_resources()
    
    print(f"\n🚀 TO RUN THIS COMPLETE STUDY:")
    print("-" * 40)
    print("# Quick test first (recommended):")
    print("python comprehensive_comparison_study.py --quick-test")
    print()
    print("# Full study:")
    print("python comprehensive_comparison_study.py")
    print()
    print("# Or step by step:")
    print("python run_design_space_exploration.py --all --config baseline_vs_llm_study.yaml")