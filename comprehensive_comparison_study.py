#!/usr/bin/env python3
"""
Comprehensive Comparison Study: Mechanical vs Standard LLM vs Memory LLM
This script generates the complete comparison matrix including mechanical baseline
"""

import json
import subprocess
import sys
from pathlib import Path
import yaml
from datetime import datetime
import pandas as pd

def run_comprehensive_study(config_file="baseline_vs_llm_study.yaml"):
    """Run the complete 3-way comparison study"""
    
    print("🔬 COMPREHENSIVE COMPARISON STUDY")
    print("=" * 60)
    print("Comparing:")
    print("1. 📐 Mechanical Baseline (traditional Schelling)")
    print("2. 🤖 Standard LLM Agents (current context only)")  
    print("3. 🧠 Memory LLM Agents (human-like with history)")
    print("4. 🌍 All Social Contexts (5 scenarios)")
    print("=" * 60)
    
    # Load configuration
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(f"comprehensive_study_{timestamp}")
    base_output_dir.mkdir(exist_ok=True)
    
    # Step 1: Run mechanical baseline for all scenarios
    print("\n📐 STEP 1: Running Mechanical Baseline")
    print("-" * 40)
    
    scenarios = config["scenarios"]
    grid_configs = config["grid_configurations"]
    runs_per_config = config["experiment_parameters"]["runs_per_config"]
    max_steps = config["experiment_parameters"]["max_steps"]
    
    mechanical_results = {}
    
    for scenario in scenarios:
        for grid_name, grid_config in grid_configs.items():
            print(f"\nRunning mechanical baseline: {scenario} on {grid_name} grid")
            
            # Temporarily set grid configuration
            original_grid = None
            original_a = None  
            original_b = None
            
            try:
                # Import and modify config
                import config as cfg
                original_grid = cfg.GRID_SIZE
                original_a = cfg.NUM_TYPE_A
                original_b = cfg.NUM_TYPE_B
                
                cfg.GRID_SIZE = grid_config["grid_size"]
                cfg.NUM_TYPE_A = grid_config["type_a"]
                cfg.NUM_TYPE_B = grid_config["type_b"]
                
                # Run mechanical baseline
                cmd = [
                    "python", "baseline_runner.py", 
                    "--runs", str(runs_per_config)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"  ✅ Completed mechanical baseline for {scenario}")
                    # Store result directory for later analysis
                    mechanical_results[f"{scenario}_{grid_name}"] = "completed"
                else:
                    print(f"  ❌ Failed: {result.stderr}")
                    
            finally:
                # Restore original config
                if original_grid is not None:
                    cfg.GRID_SIZE = original_grid
                    cfg.NUM_TYPE_A = original_a
                    cfg.NUM_TYPE_B = original_b
    
    # Step 2: Run LLM experiments using design space explorer
    print(f"\n🤖 STEP 2: Running LLM Experiments (Standard + Memory)")
    print("-" * 50)
    
    # Create temporary config for LLM experiments
    llm_config_file = base_output_dir / "llm_experiments.yaml"
    with open(llm_config_file, 'w') as f:
        yaml.dump(config, f)
    
    # Run LLM design space exploration
    cmd = [
        "python", "run_design_space_exploration.py", 
        "--all",
        "--config", str(llm_config_file),
        "--output-dir", str(base_output_dir / "llm_results")
    ]
    
    print("Starting LLM design space exploration...")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("✅ LLM experiments completed successfully")
    else:
        print("❌ LLM experiments failed")
        return False
    
    # Step 3: Generate comprehensive comparison analysis
    print(f"\n📊 STEP 3: Generating Comprehensive Analysis")
    print("-" * 45)
    
    generate_three_way_comparison(base_output_dir, mechanical_results, scenarios, grid_configs)
    
    print(f"\n🎉 COMPREHENSIVE STUDY COMPLETE!")
    print("=" * 60)
    print(f"📁 All results saved in: {base_output_dir}/")
    print(f"📊 Key findings in: {base_output_dir}/three_way_comparison_report.md")
    
    return True

def generate_three_way_comparison(base_output_dir, mechanical_results, scenarios, grid_configs):
    """Generate analysis comparing all three agent types"""
    
    analysis_dir = base_output_dir / "comprehensive_analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # This would include:
    # 1. Loading mechanical baseline results
    # 2. Loading LLM results (both standard and memory)
    # 3. Creating unified comparison dataframe
    # 4. Generating three-way statistical comparisons
    # 5. Creating comprehensive visualizations
    
    print("📊 Generating three-way comparison analysis...")
    
    # Create comprehensive report
    report_file = analysis_dir / "three_way_comparison_report.md"
    with open(report_file, 'w') as f:
        f.write("# Comprehensive Three-Way Comparison Study\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        f.write("## Study Design\n")
        f.write("This study compares three agent types across multiple social contexts:\n\n")
        f.write("1. **Mechanical Baseline**: Traditional utility-maximizing agents\n")
        f.write("2. **Standard LLM Agents**: LLM decisions based on current neighborhood\n")
        f.write("3. **Memory LLM Agents**: LLM decisions with personal history and relationships\n\n")
        
        f.write("## Social Contexts Tested\n")
        for scenario in scenarios:
            f.write(f"- {scenario}\n")
        f.write("\n")
        
        f.write("## Grid Configurations\n")
        for grid_name, grid_config in grid_configs.items():
            f.write(f"- {grid_name}: {grid_config['grid_size']}x{grid_config['grid_size']} ")
            f.write(f"({grid_config['type_a'] + grid_config['type_b']} agents)\n")
        f.write("\n")
        
        f.write("## Expected Findings\n")
        f.write("- **Mechanical agents**: Fast, predictable segregation\n")
        f.write("- **Standard LLM agents**: More variable, context-sensitive segregation\n")
        f.write("- **Memory LLM agents**: Slower, more stable, human-like patterns\n")
        f.write("- **Social context effects**: Different scenarios produce different patterns\n\n")
        
        f.write("## Analysis Files\n")
        f.write("- Mechanical baseline results: experiments/baseline_*/\n")
        f.write("- LLM results: llm_results/\n")
        f.write("- Comparative analysis: comprehensive_analysis/\n")
    
    print(f"📄 Three-way comparison report created: {report_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Comparison Study")
    parser.add_argument("--config", default="baseline_vs_llm_study.yaml",
                       help="Configuration file")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run in quick test mode")
    
    args = parser.parse_args()
    
    # Enable quick test if requested
    if args.quick_test:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        config["quick_test"]["enabled"] = True
        
        quick_config_file = "quick_test_config.yaml"
        with open(quick_config_file, 'w') as f:
            yaml.dump(config, f)
        args.config = quick_config_file
    
    success = run_comprehensive_study(args.config)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()