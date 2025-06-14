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
    """Run mechanical baseline"""
    runs = config['experiment_parameters']['runs_per_config']
    print(f"\n🔧 RUNNING MECHANICAL BASELINE ({runs} replicates)")
    print(f"{'='*50}")
    
    grid_config = config['grid_configurations']['optimized']
    params = config['experiment_parameters']
    
    cmd = [
        'python', 'baseline_runner.py',
        '--runs', str(params['runs_per_config']),
        '--max-steps', str(params['max_steps']),
        '--grid-size', str(grid_config['grid_size']),
        '--type-a', str(grid_config['type_a']),
        '--type-b', str(grid_config['type_b']),
        '--output-dir', f"{study_dir}/mechanical_baseline"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Mechanical baseline failed: {result.stderr}")
        return False
    
    print(f"✅ Mechanical baseline completed")
    return True

def run_llm_experiment(agent_type, config, study_dir):
    """Run LLM experiment with specified agent type"""
    runs = config['experiment_parameters']['runs_per_config']
    print(f"\n🤖 RUNNING {agent_type.upper()} LLM AGENTS ({runs} replicates)")
    print(f"{'='*50}")
    
    grid_config = config['grid_configurations']['optimized']
    params = config['experiment_parameters']
    llm_config = config['llm_configurations']['default']
    scenario = config['scenarios'][0]  # baseline
    
    cmd = [
        'python', 'llm_runner.py',
        '--scenario', scenario,
        '--runs', str(params['runs_per_config']),
        '--max-steps', str(params['max_steps']),
        '--grid-size', str(grid_config['grid_size']),
        '--type-a', str(grid_config['type_a']),
        '--type-b', str(grid_config['type_b']),
        '--llm-probability', str(params['use_llm_probability']),
        '--agent-type', agent_type,
        '--llm-model', llm_config['model'],
        '--llm-url', llm_config['url'],
        '--llm-api-key', llm_config['api_key'],
        '--output-dir', f"{study_dir}/llm_{agent_type}"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ {agent_type} LLM failed: {result.stderr}")
        return False
    
    print(f"✅ {agent_type} LLM completed")
    return True

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
        print(f"3. Run standard LLM ({runs} runs)")
        print(f"4. Run memory LLM ({runs} runs)")
        print(f"5. Generate study summary")
        return
    
    # Confirm before starting
    runs = config['experiment_parameters']['runs_per_config']
    mechanical_time = runs * 187 * 0.02
    standard_llm_time = runs * 99 * 19
    memory_llm_time = runs * 84 * 19
    total_hours = (mechanical_time + standard_llm_time + memory_llm_time) / 3600
    
    response = input(f"\nThis study will take ~{total_hours:.1f} hours. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Study cancelled.")
        return
    
    # Create study directory
    Path(study_dir).mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    start_time = datetime.now()
    
    # 1. Mechanical baseline
    if not run_mechanical_baseline(config, study_dir):
        print("❌ Study failed at mechanical baseline")
        sys.exit(1)
    
    # 2. Standard LLM agents
    if not run_llm_experiment('standard', config, study_dir):
        print("❌ Study failed at standard LLM")
        sys.exit(1)
    
    # 3. Memory LLM agents  
    if not run_llm_experiment('memory', config, study_dir):
        print("❌ Study failed at memory LLM")
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