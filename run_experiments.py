#!/usr/bin/env python3
"""
Master script to run complete segregation experiments
comparing mechanical agents with LLM agents under different contexts
"""

import os
import argparse
import json
from datetime import datetime
import time
import config as cfg

from baseline_runner import run_baseline_experiment
from llm_runner import run_llm_experiment, CONTEXT_SCENARIOS, check_llm_connection
from visualization import create_comprehensive_report
from statistical_analysis import create_statistical_report
from plateau_detection import compare_convergence_across_runs

def ensure_directories():
    """Ensure required directories exist"""
    os.makedirs('experiments', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

def run_full_experiment_suite(baseline_runs=100, llm_runs=10, max_steps=1000, 
                            scenarios=None, llm_probability=1.0):
    """
    Run complete experiment suite
    
    Parameters:
    - baseline_runs: Number of baseline mechanical agent runs
    - llm_runs: Number of runs per LLM scenario
    - max_steps: Maximum steps per simulation
    - scenarios: List of LLM scenarios to run (None for all)
    - llm_probability: Probability of using LLM for decisions
    """
    
    ensure_directories()
    
    # Track experiment results
    experiment_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*80)
    print("SCHELLING SEGREGATION MODEL - COMPREHENSIVE EXPERIMENT")
    print(f"Timestamp: {timestamp}")
    print("="*80)
    
    # 1. Run baseline experiments
    print(f"\n1. Running baseline experiments ({baseline_runs} runs)...")
    print("-"*60)
    
    baseline_dir, baseline_results = run_baseline_experiment(
        n_runs=baseline_runs,
        max_steps=max_steps
    )
    experiment_results['baseline'] = baseline_dir
    
    # 2. Run LLM experiments
    if scenarios is None:
        scenarios = list(CONTEXT_SCENARIOS.keys())
    
    print(f"\n2. Running LLM experiments ({llm_runs} runs per scenario)...")
    print(f"   Scenarios: {', '.join(scenarios)}")
    print("-"*60)
    
    # Check LLM connection before starting LLM experiments
    print("\n   Checking LLM availability...")
    if not check_llm_connection():
        print("\n   ⚠️  WARNING: LLM connection check failed!")
        print("   Skipping all LLM experiments.")
        print("\n   To run LLM experiments:")
        print("   1. Ensure Ollama is running: ollama serve")
        print(f"   2. Pull the model: ollama pull {cfg.OLLAMA_MODEL}")
        print("   3. Verify the URL and API key in config.py")
        llm_results = {}
    else:
        llm_results = {}
        for scenario in scenarios:
            print(f"\n   Running scenario: {scenario}")
            
            # Add delay between scenarios to avoid overwhelming LLM
            if len(llm_results) > 0:
                print("   Waiting 10 seconds before next scenario...")
                time.sleep(10)
            
            llm_dir, llm_run_results = run_llm_experiment(
                scenario=scenario,
                n_runs=llm_runs,
                max_steps=max_steps,
                use_llm_probability=llm_probability
            )
            
            if llm_dir is None:
                print(f"   ⚠️  Scenario {scenario} failed - skipping")
                continue
                
            llm_results[scenario] = llm_dir
            experiment_results[f'llm_{scenario}'] = llm_dir
    
    # 3. Analyze convergence patterns
    print("\n3. Analyzing convergence patterns...")
    print("-"*60)
    
    for exp_name, exp_dir in experiment_results.items():
        try:
            comparison_df, summary_stats = compare_convergence_across_runs(exp_dir)
            comparison_df.to_csv(f"{exp_dir}/convergence_analysis.csv", index=False)
            
            with open(f"{exp_dir}/convergence_summary_stats.json", 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            
            print(f"   {exp_name}: Convergence analysis completed")
        except Exception as e:
            print(f"   {exp_name}: Error in convergence analysis - {e}")
    
    # 4. Create visualizations
    print("\n4. Creating comprehensive visualizations...")
    print("-"*60)
    
    report_path = f"reports/comprehensive_report_{timestamp}.pdf"
    try:
        create_comprehensive_report(
            baseline_dir=experiment_results['baseline'],
            llm_dirs={k.replace('llm_', ''): v for k, v in experiment_results.items() if k.startswith('llm_')},
            output_path=report_path
        )
        print(f"   Visual report created: {report_path}")
    except Exception as e:
        print(f"   Error creating visual report: {e}")
    
    # 5. Perform statistical analysis
    print("\n5. Performing statistical analysis...")
    print("-"*60)
    
    stats_report_path = f"reports/statistical_analysis_{timestamp}.txt"
    try:
        create_statistical_report(
            experiment_results,
            output_file=stats_report_path
        )
        print(f"   Statistical report created: {stats_report_path}")
    except Exception as e:
        print(f"   Error creating statistical report: {e}")
    
    # 6. Create master summary
    print("\n6. Creating master summary...")
    print("-"*60)
    
    summary = {
        'timestamp': timestamp,
        'configuration': {
            'baseline_runs': baseline_runs,
            'llm_runs': llm_runs,
            'max_steps': max_steps,
            'scenarios': scenarios,
            'llm_probability': llm_probability
        },
        'experiments': experiment_results,
        'reports': {
            'visual': report_path,
            'statistical': stats_report_path
        }
    }
    
    summary_path = f"reports/experiment_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   Master summary created: {summary_path}")
    
    # Print final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nTotal experiments run: {len(experiment_results)}")
    print(f"Total simulations: {baseline_runs + llm_runs * len(scenarios)}")
    print(f"\nResults saved in:")
    print(f"  - Experiments: experiments/")
    print(f"  - Reports: reports/")
    print(f"  - Master summary: {summary_path}")
    
    return summary

def run_quick_test():
    """Run a quick test with reduced parameters"""
    print("\nRunning quick test mode...")
    return run_full_experiment_suite(
        baseline_runs=5,
        llm_runs=2,
        max_steps=100,
        scenarios=['baseline', 'race_white_black']
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run comprehensive Schelling segregation experiments"
    )
    
    parser.add_argument('--baseline-runs', type=int, default=100,
                        help='Number of baseline mechanical agent runs (default: 100)')
    parser.add_argument('--llm-runs', type=int, default=10,
                        help='Number of runs per LLM scenario (default: 10)')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Maximum steps per simulation (default: 1000)')
    parser.add_argument('--scenarios', nargs='+', 
                        choices=list(CONTEXT_SCENARIOS.keys()),
                        help='LLM scenarios to run (default: all)')
    parser.add_argument('--llm-probability', type=float, default=1.0,
                        help='Probability of using LLM for agent decisions (default: 1.0)')
    parser.add_argument('--quick-test', action='store_true',
                        help='Run quick test with reduced parameters')
    
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()

    if args.quick_test:
        run_quick_test()
    else:
        run_full_experiment_suite(
            baseline_runs=args.baseline_runs,
            llm_runs=args.llm_runs,
            max_steps=args.max_steps,
            scenarios=args.scenarios,
            llm_probability=args.llm_probability
        )