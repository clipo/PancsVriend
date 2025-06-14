#!/usr/bin/env python3
"""
Pairwise Comparison Analysis for Pure Agent Type Study
Focuses on the key comparisons:
1. Mechanical Baseline vs Standard LLM
2. Mechanical Baseline vs Memory LLM  
3. Standard LLM vs Memory LLM
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_final_metrics(exp_dir):
    """Load final metrics from metrics history"""
    try:
        # Load metrics history and get final values for each run
        df = pd.read_csv(f"{exp_dir}/metrics_history.csv")
        
        # Group by run_id and get final step for each run
        final_metrics = []
        for run_id in df['run_id'].unique():
            run_data = df[df['run_id'] == run_id]
            final_step = run_data.iloc[-1]  # Last step of this run
            final_metrics.append(final_step.to_dict())
        
        return final_metrics
    except Exception as e:
        print(f"Error loading metrics from {exp_dir}: {e}")
        return []

def calculate_effect_size(group1, group2):
    """Calculate Cohen's d effect size"""
    if len(group1) <= 1 or len(group2) <= 1:
        return 0, "insufficient_data"
    
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0, "no_variance"
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    # Interpretation
    if abs(d) < 0.2:
        interpretation = "negligible"
    elif abs(d) < 0.5:
        interpretation = "small"
    elif abs(d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return d, interpretation

def compare_two_groups(data1, data2, name1, name2, metric):
    """Compare two groups for a specific metric"""
    
    # Get values
    values1 = []
    values2 = []
    
    for exp_data in data1:
        if metric in exp_data:
            values1.append(exp_data[metric])
    
    for exp_data in data2:
        if metric in exp_data:
            values2.append(exp_data[metric])
    
    if len(values1) == 0 or len(values2) == 0:
        return None
    
    # Basic statistics
    mean1, std1 = np.mean(values1), np.std(values1, ddof=1) if len(values1) > 1 else 0
    mean2, std2 = np.mean(values2), np.std(values2, ddof=1) if len(values2) > 1 else 0
    
    # Statistical test (use Mann-Whitney U for small samples)
    if len(values1) >= 2 and len(values2) >= 2:
        try:
            stat, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
            test_type = "Mann-Whitney U"
        except:
            stat, p_value = 0, 1.0
            test_type = "Failed"
    else:
        stat, p_value = 0, 1.0
        test_type = "Insufficient data"
    
    # Effect size
    effect_size, effect_interpretation = calculate_effect_size(values1, values2)
    
    # Practical difference (percentage change)
    if mean1 != 0:
        percent_change = ((mean2 - mean1) / abs(mean1)) * 100
    else:
        percent_change = 0
    
    return {
        'metric': metric,
        'group1': name1,
        'group2': name2,
        'n1': len(values1),
        'n2': len(values2),
        'mean1': mean1,
        'std1': std1,
        'mean2': mean2,
        'std2': std2,
        'test_type': test_type,
        'statistic': stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size': effect_size,
        'effect_interpretation': effect_interpretation,
        'percent_change': percent_change,
        'practical_interpretation': get_practical_interpretation(metric, percent_change, effect_interpretation)
    }

def get_practical_interpretation(metric, percent_change, effect_size):
    """Get practical interpretation of the difference"""
    direction = "higher" if percent_change > 0 else "lower"
    magnitude = abs(percent_change)
    
    if effect_size == "negligible":
        return f"Negligible difference ({magnitude:.1f}% {direction})"
    elif effect_size == "small":
        return f"Small difference ({magnitude:.1f}% {direction})"
    elif effect_size == "medium":
        return f"Moderate difference ({magnitude:.1f}% {direction})"
    elif effect_size == "large":
        return f"Large difference ({magnitude:.1f}% {direction})"
    else:
        return f"Difference: {magnitude:.1f}% {direction}"

def main():
    # Our equal-length pure comparison experiments
    experiments = {
        'mechanical_baseline': 'experiments/baseline_20250613_214945',
        'standard_llm': 'experiments/llm_baseline_20250613_215046',
        'memory_llm': 'experiments/llm_baseline_20250613_225502'
    }
    
    print("🔬 PAIRWISE COMPARISON ANALYSIS")
    print("=" * 60)
    print("Pure Agent Type Comparison Study")
    print("=" * 60)
    
    # Load data for each experiment
    experiment_data = {}
    for name, exp_dir in experiments.items():
        try:
            # Load final metrics for each run
            final_metrics = load_final_metrics(exp_dir)
            experiment_data[name] = final_metrics
            print(f"✅ Loaded {name}: {len(final_metrics)} runs")
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
            experiment_data[name] = []
    
    # Metrics to analyze
    metrics = ['clusters', 'distance', 'mix_deviation', 'share', 'ghetto_rate']
    
    # Key comparisons
    comparisons = [
        ('mechanical_baseline', 'standard_llm', 'Mechanical vs Standard LLM'),
        ('mechanical_baseline', 'memory_llm', 'Mechanical vs Memory LLM'),
        ('standard_llm', 'memory_llm', 'Standard LLM vs Memory LLM')
    ]
    
    results = []
    
    for group1, group2, comparison_name in comparisons:
        print(f"\n📊 {comparison_name}")
        print("-" * 50)
        
        for metric in metrics:
            result = compare_two_groups(
                experiment_data[group1], 
                experiment_data[group2], 
                group1, group2, metric
            )
            
            if result:
                results.append(result)
                
                # Print summary
                print(f"\n{metric.upper()}:")
                print(f"  {group1}: {result['mean1']:.3f} ± {result['std1']:.3f}")
                print(f"  {group2}: {result['mean2']:.3f} ± {result['std2']:.3f}")
                print(f"  Change: {result['percent_change']:+.1f}%")
                print(f"  Effect: {result['effect_interpretation']}")
                print(f"  P-value: {result['p_value']:.3f} {'*' if result['significant'] else ''}")
                print(f"  Interpretation: {result['practical_interpretation']}")
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv('pairwise_comparison_results.csv', index=False)
    
    print(f"\n✅ Detailed results saved to: pairwise_comparison_results.csv")
    
    # Summary of key findings
    print(f"\n🎯 KEY FINDINGS SUMMARY")
    print("=" * 50)
    
    # Focus on most important comparisons
    key_findings = []
    for result in results:
        if result['group1'] == 'mechanical_baseline':
            if result['metric'] in ['share', 'distance', 'clusters']:
                key_findings.append(result)
    
    print("\n📐 MECHANICAL vs LLM AGENTS:")
    for result in key_findings:
        if result['group2'] in ['standard_llm', 'memory_llm']:
            agent_type = "Standard LLM" if result['group2'] == 'standard_llm' else "Memory LLM"
            print(f"  {result['metric']}: {agent_type} is {result['practical_interpretation']}")
    
    print(f"\n🧠 MEMORY EFFECT:")
    memory_effects = [r for r in results if r['group1'] == 'standard_llm' and r['group2'] == 'memory_llm']
    for result in memory_effects:
        if result['metric'] in ['share', 'distance', 'clusters']:
            print(f"  {result['metric']}: {result['practical_interpretation']}")

if __name__ == "__main__":
    main()