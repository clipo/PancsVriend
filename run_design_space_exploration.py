#!/usr/bin/env python3
"""
Design Space Exploration Launcher
Easy-to-use script for running comprehensive experimental design space exploration
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml
import json

def load_config(config_file="experiment_configs.yaml"):
    """Load experiment configuration from YAML file"""
    if Path(config_file).exists():
        with open(config_file) as f:
            return yaml.safe_load(f)
    else:
        print(f"❌ Configuration file {config_file} not found")
        print("Please create experiment_configs.yaml or specify a different config file")
        sys.exit(1)

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🚀 {description}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error:", e.stderr)
        return False

def plan_experiments(config, args):
    """Plan the experimental design space"""
    
    # Build command arguments
    cmd_parts = ["python experiment_explorer.py --mode plan"]
    
    if args.llms:
        cmd_parts.append(f"--llms {' '.join(args.llms)}")
    
    if args.agents:
        cmd_parts.append(f"--agents {' '.join(args.agents)}")
    
    if args.scenarios:
        cmd_parts.append(f"--scenarios {' '.join(args.scenarios)}")
        
    if args.grids:
        cmd_parts.append(f"--grids {' '.join(args.grids)}")
        
    # Use quick test settings if enabled
    if config.get("quick_test", {}).get("enabled", False):
        quick_config = config["quick_test"]
        runs = quick_config.get("runs_per_config", 3)
        cmd_parts.append(f"--runs {runs}")
        print("⚡ Quick test mode enabled")
    else:
        runs = config["experiment_parameters"]["runs_per_config"]
        cmd_parts.append(f"--runs {runs}")
    
    if args.output_dir:
        cmd_parts.append(f"--output-dir {args.output_dir}")
        
    cmd = " ".join(cmd_parts)
    return run_command(cmd, "Planning experimental design space")

def run_experiments(args):
    """Run the planned experiments"""
    
    cmd_parts = ["python experiment_explorer.py --mode run"]
    
    if args.start_idx:
        cmd_parts.append(f"--start-idx {args.start_idx}")
        
    if args.max_experiments:
        cmd_parts.append(f"--max-experiments {args.max_experiments}")
        
    if args.output_dir:
        cmd_parts.append(f"--output-dir {args.output_dir}")
        
    cmd = " ".join(cmd_parts)
    return run_command(cmd, "Running experimental design space")

def analyze_results(args):
    """Analyze the experimental results"""
    
    cmd_parts = ["python design_space_analyzer.py"]
    
    if args.output_dir:
        cmd_parts.append(f"--input-dir {args.output_dir}")
        
    cmd = " ".join(cmd_parts)
    return run_command(cmd, "Analyzing experimental results")

def print_next_steps(output_dir):
    """Print instructions for next steps"""
    
    print(f"\n🎉 DESIGN SPACE EXPLORATION COMPLETE!")
    print("=" * 60)
    print(f"\n📁 All results saved in: {output_dir}/")
    print(f"\n📊 Key outputs:")
    print(f"   • Experiment plans: {output_dir}/experiment_plan_*.json")
    print(f"   • Individual results: {output_dir}/experiments/exp_*/")
    print(f"   • Comparative analysis: {output_dir}/analysis/")
    print(f"   • Statistical tests: {output_dir}/analysis/statistical_tests.json")
    print(f"   • Visualizations: {output_dir}/analysis/*.png")
    print(f"   • Summary report: {output_dir}/analysis/comprehensive_report.md")
    
    print(f"\n🔍 To explore your results:")
    print(f"   1. Read the summary: {output_dir}/analysis/comprehensive_report.md")
    print(f"   2. View visualizations: {output_dir}/analysis/*.png")
    print(f"   3. Check statistics: {output_dir}/analysis/statistical_tests.json")
    print(f"   4. Examine raw data: {output_dir}/analysis/summary_statistics.json")
    
    print(f"\n📊 To run additional analysis:")
    print(f"   python design_space_analyzer.py --input-dir {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Design Space Exploration Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full exploration with all configurations
  python run_design_space_exploration.py --all
  
  # Quick test with limited scope
  python run_design_space_exploration.py --plan --run --analyze --grids small --scenarios baseline
  
  # Plan experiments only
  python run_design_space_exploration.py --plan --llms qwen_local gpt4 --agents memory
  
  # Run previously planned experiments
  python run_design_space_exploration.py --run --max-experiments 10
  
  # Analyze existing results
  python run_design_space_exploration.py --analyze
        """
    )
    
    # Workflow control
    workflow = parser.add_argument_group("Workflow Control")
    workflow.add_argument("--all", action="store_true", 
                         help="Run complete pipeline: plan, run, and analyze")
    workflow.add_argument("--plan", action="store_true", 
                         help="Plan experiments")
    workflow.add_argument("--run", action="store_true", 
                         help="Run planned experiments")
    workflow.add_argument("--analyze", action="store_true", 
                         help="Analyze experimental results")
    
    # Configuration filters
    config_group = parser.add_argument_group("Configuration Filters")
    config_group.add_argument("--llms", nargs="+", 
                             help="LLM configurations to include (from experiment_configs.yaml)")
    config_group.add_argument("--agents", nargs="+", choices=["standard", "memory"],
                             help="Agent types to include")
    config_group.add_argument("--scenarios", nargs="+",
                             help="Scenarios to include")
    config_group.add_argument("--grids", nargs="+", choices=["small", "medium", "large", "xlarge"],
                             help="Grid sizes to include")
    
    # Execution control
    exec_group = parser.add_argument_group("Execution Control")
    exec_group.add_argument("--start-idx", type=int, 
                           help="Start index for batch runs")
    exec_group.add_argument("--max-experiments", type=int,
                           help="Maximum experiments to run")
    exec_group.add_argument("--output-dir", default="design_space_exploration",
                           help="Output directory")
    exec_group.add_argument("--config", default="experiment_configs.yaml",
                           help="Configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Default to all if no specific workflow specified
    if not any([args.all, args.plan, args.run, args.analyze]):
        args.all = True
    
    # Set workflow flags
    if args.all:
        args.plan = args.run = args.analyze = True
    
    print("🔬 DESIGN SPACE EXPLORATION LAUNCHER")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Output directory: {args.output_dir}")
    
    success = True
    
    # Execute workflow
    if args.plan and success:
        success = plan_experiments(config, args)
        
    if args.run and success:
        success = run_experiments(args)
        
    if args.analyze and success:
        success = analyze_results(args)
        
    if success:
        print_next_steps(args.output_dir)
    else:
        print("\n❌ Workflow failed. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()