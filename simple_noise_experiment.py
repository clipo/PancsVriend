#!/usr/bin/env python3
"""
Simple noise experiment with plotting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Add current directory to path
sys.path.append('.')

from schelling_with_noise.noisy_schelling import NoisySimulation, mechanical_decision_noisy
from base_simulation import Simulation
from baseline_runner import mechanical_decision

def run_simple_noise_experiment():
    """Run a simple noise experiment comparing 0% and 10% noise"""
    
    print("🚀 Running Simple Noise Experiment...")
    
    noise_levels = [0.0, 0.1, 0.2]  # 0%, 10%, 20% noise
    n_runs = 20  # Smaller sample for faster execution
    max_steps = 200
    
    results = {}
    
    for noise_level in noise_levels:
        print(f"\n🔊 Testing noise level: {noise_level:.1%}")
        
        runs = []
        for run_id in range(n_runs):
            if noise_level == 0.0:
                # Regular simulation (no noise)
                from Agent import Agent
                sim = Simulation(
                    run_id=run_id,
                    agent_factory=Agent,
                    decision_func=mechanical_decision,
                    scenario='baseline',
                    random_seed=run_id
                )
            else:
                # Noisy simulation
                sim = NoisySimulation(
                    run_id=run_id,
                    decision_func=mechanical_decision_noisy,
                    noise_probability=noise_level,
                    scenario=f'noisy_{noise_level}',
                    random_seed=run_id
                )
            
            result = sim.run_single_simulation(max_steps=max_steps, show_progress=False)
            
            # Add noise level to result
            result['noise_level'] = noise_level
            
            # Calculate final segregation if not present
            if 'final_segregation' not in result:
                from Metrics import calculate_all_metrics
                final_metrics = calculate_all_metrics(sim.grid)
                result['final_segregation'] = final_metrics.get('mix_deviation', 0.0)
            
            runs.append(result)
            
            print(f"  Run {run_id:2d}: Converged={str(result['converged']):5} "
                  f"Steps={result['final_step']:3d} "
                  f"Segregation={result['final_segregation']:.3f}")
        
        results[noise_level] = runs
    
    return results

def create_simple_plots(results):
    """Create simple comparison plots"""
    
    # Convert to DataFrame
    all_data = []
    for noise_level, runs in results.items():
        for run in runs:
            all_data.append({
                'noise_level': noise_level,
                'noise_percent': f"{noise_level:.0%}",
                'converged': run['converged'],
                'final_step': run['final_step'],
                'final_segregation': run['final_segregation']
            })
    
    df = pd.DataFrame(all_data)
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Noise Effects on Schelling Model', fontsize=14, fontweight='bold')
    
    # Plot 1: Convergence Rate
    convergence_stats = df.groupby('noise_level')['converged'].mean()
    axes[0].bar(range(len(convergence_stats)), convergence_stats.values, alpha=0.7)
    axes[0].set_xlabel('Noise Level')
    axes[0].set_ylabel('Convergence Rate')
    axes[0].set_title('Convergence Rate by Noise Level')
    axes[0].set_xticks(range(len(convergence_stats)))
    axes[0].set_xticklabels([f"{x:.0%}" for x in convergence_stats.index])
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(convergence_stats.values):
        axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Plot 2: Final Segregation Distribution
    df.boxplot(column='final_segregation', by='noise_percent', ax=axes[1])
    axes[1].set_xlabel('Noise Level')
    axes[1].set_ylabel('Final Segregation')
    axes[1].set_title('Segregation Distribution by Noise Level')
    axes[1].grid(axis='y', alpha=0.3)
    plt.setp(axes[1], xlabel='Noise Level')
    
    # Plot 3: Average Segregation
    seg_stats = df.groupby('noise_level')['final_segregation'].agg(['mean', 'std'])
    axes[2].bar(range(len(seg_stats)), seg_stats['mean'], 
                yerr=seg_stats['std'], capsize=5, alpha=0.7)
    axes[2].set_xlabel('Noise Level')
    axes[2].set_ylabel('Average Final Segregation')
    axes[2].set_title('Average Segregation by Noise Level')
    axes[2].set_xticks(range(len(seg_stats)))
    axes[2].set_xticklabels([f"{x:.0%}" for x in seg_stats.index])
    axes[2].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(seg_stats['mean']):
        axes[2].text(i, v + seg_stats['std'].iloc[i] + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f'noise_experiment_simple_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {plot_filename}")
    
    return plot_filename

def print_summary(results):
    """Print summary statistics"""
    
    print("\n" + "="*60)
    print("📊 NOISE EXPERIMENT SUMMARY")
    print("="*60)
    print(f"{'Noise Level':<12} {'Conv Rate':<12} {'Avg Segregation':<16} {'Std Dev':<10}")
    print("-"*60)
    
    for noise_level in sorted(results.keys()):
        runs = results[noise_level]
        conv_rate = sum(1 for r in runs if r['converged']) / len(runs)
        avg_seg = np.mean([r['final_segregation'] for r in runs])
        std_seg = np.std([r['final_segregation'] for r in runs])
        
        print(f"{noise_level:<12.0%} {conv_rate:<12.2f} {avg_seg:<16.4f} {std_seg:<10.4f}")
    
    # Compare to baseline
    if 0.0 in results:
        baseline_seg = [r['final_segregation'] for r in results[0.0]]
        baseline_mean = np.mean(baseline_seg)
        
        print(f"\n🔍 COMPARISON TO BASELINE (0% noise):")
        print("-"*40)
        
        for noise_level in sorted(results.keys()):
            if noise_level == 0.0:
                continue
            
            noise_seg = [r['final_segregation'] for r in results[noise_level]]
            noise_mean = np.mean(noise_seg)
            
            change = noise_mean - baseline_mean
            pct_change = (change / baseline_mean) * 100 if baseline_mean != 0 else 0
            
            print(f"Noise {noise_level:.0%}: {change:+.4f} ({pct_change:+.1f}% change)")

def main():
    """Main function"""
    try:
        print("🎯 Starting Simple Noise Experiment...")
        
        # Run experiment
        results = run_simple_noise_experiment()
        
        # Print summary
        print_summary(results)
        
        # Create plots
        plot_file = create_simple_plots(results)
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"📈 Plot saved: {plot_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
