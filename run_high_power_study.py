#!/usr/bin/env python3
"""
High Statistical Power Study Runner
Runs 20-replicate comparison study for robust statistical analysis
"""

import argparse
import sys
from pathlib import Path
import subprocess
import yaml
from datetime import datetime
import json

def load_study_config(config_file='high_power_study.yaml'):
    """Load the high power study configuration"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def estimate_runtime(config):
    """Estimate total runtime for the study based on observed convergence patterns"""
    runs = config['experiment_parameters']['runs_per_config']
    
    # Realistic estimates based on our previous observations
    mechanical_time = runs * 187 * 0.02  # 187 avg steps × 0.02 sec/step
    standard_llm_time = runs * 99 * 19   # 99 avg steps × 19 sec/step  
    memory_llm_time = runs * 84 * 19     # 84 avg steps × 19 sec/step
    
    print(f"📊 RUNTIME ESTIMATES ({runs} replicates, based on observed convergence)")
    print(f"{'='*60}")
    print(f"Mechanical baseline: {mechanical_time/60:.1f} minutes (187 avg steps)")
    print(f"Standard LLM: {standard_llm_time/3600:.1f} hours (99 avg steps)") 
    print(f"Memory LLM: {memory_llm_time/3600:.1f} hours (84 avg steps)")
    print(f"Total estimated: {(mechanical_time + standard_llm_time + memory_llm_time)/3600:.1f} hours")
    print(f"Note: Plateau detection will stop runs early, reducing actual time")
    print(f"{'='*60}")

def run_mechanical_baseline(config, study_dir):
    """Run mechanical baseline using config.py settings"""
    runs = config['experiment_parameters']['runs_per_config']
    print(f"\n🔧 RUNNING MECHANICAL BASELINE ({runs} replicates)")
    print(f"{'='*50}")
    
    params = config['experiment_parameters']
    
    # Baseline runner uses config.py settings, so just specify runs and max-steps
    cmd = [
        'python', 'baseline_runner.py',
        '--runs', str(params['runs_per_config']),
        '--max-steps', str(params['max_steps'])
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Note: Using config.py settings for grid size and agent counts")
    
    # Create study directory but run from current location
    study_path = Path(study_dir)
    study_path.mkdir(parents=True, exist_ok=True)
    
    # Run baseline experiment and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Mechanical baseline failed: {result.stderr}")
        return False
    
    print(f"✅ Mechanical baseline completed")
    print(f"Note: Output will be in experiments/ directory (standard location)")
    return True

def run_llm_experiments(config, study_dir):
    """Run both standard and memory LLM experiments using design space exploration"""
    runs = config['experiment_parameters']['runs_per_config']
    print(f"\n🤖 RUNNING LLM EXPERIMENTS ({runs} replicates each)")
    print(f"{'='*50}")
    
    # Create a temporary experiment config for the design space exploration
    temp_config = {
        'llm_configurations': config['llm_configurations'],
        'agent_types': config['agent_types'],  # ['standard', 'memory']
        'scenarios': config['scenarios'],      # ['baseline']
        'grid_configurations': config['grid_configurations'],
        'experiment_parameters': config['experiment_parameters']
    }
    
    # Write temporary config file
    temp_config_file = Path(study_dir) / "temp_experiment_config.yaml"
    with open(temp_config_file, 'w') as f:
        yaml.dump(temp_config, f)
    
    # Run design space exploration for LLM experiments
    cmd = [
        'python', 'run_design_space_exploration.py',
        '--all',  # Plan, run, and analyze
        '--agents', 'standard', 'memory',  # Both agent types
        '--scenarios', 'baseline',  # Just baseline scenario
        '--grids', 'small',  # 10x10 grid with 25+25 agents (matches our density)
        '--config', str(temp_config_file),
        '--output-dir', str(Path(study_dir) / "llm_results")
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Running design space exploration for standard and memory LLM agents...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ LLM experiments failed: {result.stderr}")
            return False
        
        print(f"✅ LLM experiments completed")
        print(f"Output saved to: {Path(study_dir) / 'llm_results'}/")
        return True
        
    finally:
        # Clean up temporary config file
        if temp_config_file.exists():
            temp_config_file.unlink()

def create_study_summary(config, study_dir):
    """Create a summary of the study configuration and results"""
    summary = {
        'study_type': 'High Statistical Power Comparison',
        'timestamp': datetime.now().isoformat(),
        'configuration': config,
        'experiments': {
            'mechanical_baseline': f"{study_dir}/mechanical_baseline",
            'standard_llm': f"{study_dir}/llm_standard", 
            'memory_llm': f"{study_dir}/llm_memory"
        },
        'replicates': config['experiment_parameters']['runs_per_config'],
        'statistical_power': 'High (n=20 enables detection of medium effect sizes)',
        'analysis_files': [
            'Run statistical_analysis.py for ANOVA and pairwise comparisons',
            'Run convergence_analysis.py for convergence metrics',
            'Run comprehensive_visualization_report.py for publication figures'
        ]
    }
    
    with open(f"{study_dir}/study_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📋 Study summary saved to: {study_dir}/study_summary.json")

def main():
    parser = argparse.ArgumentParser(description='Run high statistical power study with 20 replicates')
    parser.add_argument('--config', default='high_power_study.yaml', 
                       help='Configuration file (default: high_power_study.yaml)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show estimates and commands without running')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt and start immediately')
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_study_config(args.config)
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Create study directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = f"high_power_study_{timestamp}"
    
    print(f"🎯 HIGH STATISTICAL POWER STUDY")
    print(f"{'='*50}")
    print(f"Study directory: {study_dir}")
    print(f"Replicates per condition: {config['experiment_parameters']['runs_per_config']}")
    print(f"Max steps per run: {config['experiment_parameters']['max_steps']}")
    
    # Show runtime estimates
    estimate_runtime(config)
    
    if args.dry_run:
        runs = config['experiment_parameters']['runs_per_config']
        print("\n🔍 DRY RUN - Commands that would be executed:")
        print(f"1. mkdir -p {study_dir}")
        print(f"2. Run mechanical baseline ({runs} runs)")
        print(f"3. Run LLM experiments - standard and memory ({runs} runs each)")
        print(f"4. Generate study summary")
        return
    
    # Confirm before starting (unless --yes flag used)
    if not args.yes:
        runs = config['experiment_parameters']['runs_per_config']
        mechanical_time = runs * 187 * 0.02
        standard_llm_time = runs * 99 * 19
        memory_llm_time = runs * 84 * 19
        total_hours = (mechanical_time + standard_llm_time + memory_llm_time) / 3600
        
        response = input(f"\nThis study will take ~{total_hours:.1f} hours. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Study cancelled.")
            return
    else:
        print("\n🚀 Starting high-power study automatically (--yes flag used)")

    print(f"\n⏰ Study started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create study directory
    Path(study_dir).mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    start_time = datetime.now()
    
    # 1. Mechanical baseline
    if not run_mechanical_baseline(config, study_dir):
        print("❌ Study failed at mechanical baseline")
        sys.exit(1)
    
    # 2. LLM experiments (both standard and memory)
    if not run_llm_experiments(config, study_dir):
        print("❌ Study failed at LLM experiments")
        sys.exit(1)
    
    # Create summary
    create_study_summary(config, study_dir)
    
    # Final report
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n🎉 HIGH POWER STUDY COMPLETED!")
    print(f"{'='*50}")
    print(f"Duration: {duration}")
    print(f"Study directory: {study_dir}")
    print(f"Total replicates: {3 * config['experiment_parameters']['runs_per_config']}")
    print(f"\n📊 Next steps:")
    print(f"1. cd {study_dir}")
    print(f"2. python ../statistical_analysis.py mechanical_baseline llm_standard llm_memory")
    print(f"3. python ../convergence_analysis.py")
    print(f"4. python ../comprehensive_visualization_report.py")

if __name__ == "__main__":
    main()