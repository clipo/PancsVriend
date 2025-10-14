#!/usr/bin/env python3
"""
Run noise experiment and create visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import sys

# Add current directory to path
sys.path.append('.')

from schelling_with_noise.noisy_schelling import (
    run_noise_comparison_study, 
    save_noise_study_results,
    analyze_noise_comparison
)

def run_experiment():
    """Run the noise experiment with multiple noise levels"""
    print('🚀 Starting Comprehensive Noise Experiment...')
    
    # Run noise comparison study with mechanical agents
    results = run_noise_comparison_study(
        n_runs=30,  # Reasonable sample size for faster execution
        noise_levels=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],  # Range from no noise to high noise
        max_steps=300,  # Reasonable max steps for convergence
        use_llm=False,  # Using mechanical agents for faster execution
        scenario='baseline'
    )
    
    return results

def create_visualizations(results, output_prefix='noise_experiment'):
    """Create comprehensive visualizations of the noise experiment results"""
    
    # Convert results to DataFrame for easier plotting
    all_data = []
    for noise_level, runs in results['results'].items():
        for run in runs:
            all_data.append({
                'noise_level': noise_level,
                'noise_percent': f"{noise_level:.1%}",
                'converged': run['converged'],
                'final_step': run['final_step'],
                'final_segregation': run['final_segregation'],
                'run_id': run['run_id']
            })
    
    df = pd.DataFrame(all_data)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Noise Effects on Schelling Segregation Model', fontsize=16, fontweight='bold')
    
    # 1. Convergence Rate by Noise Level
    ax1 = axes[0, 0]
    convergence_stats = df.groupby('noise_level')['converged'].agg(['mean', 'std', 'count']).reset_index()
    convergence_stats['error'] = convergence_stats['std'] / np.sqrt(convergence_stats['count'])
    
    bars = ax1.bar(range(len(convergence_stats)), convergence_stats['mean'], 
                   yerr=convergence_stats['error'], capsize=5, alpha=0.7,
                   color=sns.color_palette("husl", len(convergence_stats)))
    
    ax1.set_xlabel('Noise Level')
    ax1.set_ylabel('Convergence Rate')
    ax1.set_title('Convergence Rate vs Noise Level')
    ax1.set_xticks(range(len(convergence_stats)))
    ax1.set_xticklabels([f"{x:.1%}" for x in convergence_stats['noise_level']])
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, convergence_stats['mean'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Final Segregation by Noise Level (Box Plot)
    ax2 = axes[0, 1]
    df_plot = df.copy()
    df_plot['noise_percent_ordered'] = pd.Categorical(df_plot['noise_percent'], 
                                                      categories=[f"{x:.1%}" for x in sorted(df['noise_level'].unique())],
                                                      ordered=True)
    
    sns.boxplot(data=df_plot, x='noise_percent_ordered', y='final_segregation', ax=ax2)
    ax2.set_xlabel('Noise Level')
    ax2.set_ylabel('Final Segregation (Mix Deviation)')
    ax2.set_title('Segregation Distribution by Noise Level')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Steps to Convergence by Noise Level
    ax3 = axes[1, 0]
    converged_df = df[df['converged']].copy()
    
    if not converged_df.empty:
        step_stats = converged_df.groupby('noise_level')['final_step'].agg(['mean', 'std', 'count']).reset_index()
        step_stats['error'] = step_stats['std'] / np.sqrt(step_stats['count'])
        
        bars = ax3.bar(range(len(step_stats)), step_stats['mean'], 
                       yerr=step_stats['error'], capsize=5, alpha=0.7,
                       color=sns.color_palette("husl", len(step_stats)))
        
        ax3.set_xlabel('Noise Level')
        ax3.set_ylabel('Average Steps to Convergence')
        ax3.set_title('Convergence Speed vs Noise Level')
        ax3.set_xticks(range(len(step_stats)))
        ax3.set_xticklabels([f"{x:.1%}" for x in step_stats['noise_level']])
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, step_stats['mean'])):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + step_stats['error'].iloc[i] + 1, 
                    f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No convergent runs found', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Convergence Speed vs Noise Level (No Data)')
    
    # 4. Noise Effect Comparison (Baseline vs Others)
    ax4 = axes[1, 1]
    baseline_segregation = df[df['noise_level'] == 0.0]['final_segregation'].values
    noise_levels = sorted(df['noise_level'].unique())
    
    comparison_data = []
    for noise in noise_levels:
        if noise == 0.0:
            continue
        noise_segregation = df[df['noise_level'] == noise]['final_segregation'].values
        
        # Calculate effect size (Cohen's d)
        if len(baseline_segregation) > 0 and len(noise_segregation) > 0:
            pooled_std = np.sqrt((np.var(baseline_segregation) + np.var(noise_segregation)) / 2)
            cohens_d = (np.mean(baseline_segregation) - np.mean(noise_segregation)) / pooled_std if pooled_std > 0 else 0
            comparison_data.append({'noise_level': noise, 'effect_size': cohens_d})
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        bars = ax4.bar(range(len(comp_df)), comp_df['effect_size'], alpha=0.7,
                       color=['green' if x > 0 else 'red' for x in comp_df['effect_size']])
        
        ax4.set_xlabel('Noise Level')
        ax4.set_ylabel("Effect Size (Cohen's d)")
        ax4.set_title('Effect Size vs Baseline (No Noise)')
        ax4.set_xticks(range(len(comp_df)))
        ax4.set_xticklabels([f"{x:.1%}" for x in comp_df['noise_level']])
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, comp_df['effect_size'])):
            ax4.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.01 if val > 0 else bar.get_height() - 0.03, 
                    f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No effect size data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Effect Size vs Baseline (No Data)')
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f'{output_prefix}_{timestamp}_analysis.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f'📊 Visualization saved to: {plot_filename}')
    
    return plot_filename

def create_summary_table(results):
    """Create a summary table of the noise experiment results"""
    
    print("\n" + "="*80)
    print("📊 NOISE EXPERIMENT SUMMARY RESULTS")
    print("="*80)
    
    # Convert to DataFrame for analysis
    all_data = []
    for noise_level, runs in results['results'].items():
        for run in runs:
            all_data.append({
                'noise_level': noise_level,
                'converged': run['converged'],
                'final_step': run['final_step'],
                'final_segregation': run['final_segregation'],
            })
    
    df = pd.DataFrame(all_data)
    
    # Calculate summary statistics
    summary_stats = []
    for noise_level in sorted(df['noise_level'].unique()):
        subset = df[df['noise_level'] == noise_level]
        converged_subset = subset[subset['converged']]
        
        stats = {
            'Noise Level': f"{noise_level:.1%}",
            'Convergence Rate': f"{subset['converged'].mean():.1%}",
            'Avg Steps (Converged)': f"{converged_subset['final_step'].mean():.1f}" if len(converged_subset) > 0 else "N/A",
            'Avg Segregation': f"{subset['final_segregation'].mean():.3f}",
            'Segregation Std': f"{subset['final_segregation'].std():.3f}",
            'Sample Size': len(subset)
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    print(summary_df.to_string(index=False))
    
    # Statistical significance tests
    baseline_data = df[df['noise_level'] == 0.0]['final_segregation'].values
    print(f"\n🔬 STATISTICAL SIGNIFICANCE TESTS (vs Baseline)")
    print("-" * 50)
    
    try:
        from scipy import stats as scipy_stats
        
        for noise_level in sorted(df['noise_level'].unique()):
            if noise_level == 0.0:
                continue
            
            noise_data = df[df['noise_level'] == noise_level]['final_segregation'].values
            
            if len(baseline_data) > 0 and len(noise_data) > 0:
                t_stat, p_value = scipy_stats.ttest_ind(baseline_data, noise_data)
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
                
                print(f"Noise {noise_level:.1%} vs Baseline: t={t_stat:.3f}, p={p_value:.4f} {significance}")
            
    except ImportError:
        print("(scipy not available for statistical tests)")
    
    return summary_df

def main():
    """Main function to run the experiment and create visualizations"""
    try:
        # Run the experiment
        results = run_experiment()
        
        # Save results to files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file, summary_file = save_noise_study_results(results, f'noise_experiment_{timestamp}')
        
        # Create visualizations
        plot_file = create_visualizations(results, f'noise_experiment_{timestamp}')
        
        # Create summary table
        summary_df = create_summary_table(results)
        
        print(f"\n✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"📁 Files generated:")
        print(f"   • Data: {csv_file}")
        print(f"   • Summary: {summary_file}")
        print(f"   • Plots: {plot_file}")
        
        return results, plot_file
        
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results, plot_file = main()
