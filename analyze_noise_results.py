#!/usr/bin/env python3
"""
Analyze and visualize existing noise experiment results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_noise_experiment_data(filename):
    """Load the noise experiment data from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def create_comprehensive_visualization(data):
    """Create comprehensive visualizations of the noise experiment"""
    
    # Note: DataFrame conversion available if needed for future analysis
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Noise Effects on Schelling Segregation Model - Comprehensive Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 1. Convergence Rate by Noise Level
    ax1 = axes[0, 0]
    convergence_rates = []
    noise_levels = []
    
    for noise_level_str, stats in sorted(data.items(), key=lambda x: float(x[0])):
        noise_level = float(noise_level_str)
        noise_levels.append(noise_level)
        convergence_rates.append(stats['convergence_rate'])
    
    bars = ax1.bar(range(len(noise_levels)), convergence_rates, alpha=0.8,
                   color=sns.color_palette("RdYlBu_r", len(noise_levels)))
    
    ax1.set_xlabel('Noise Level', fontsize=12)
    ax1.set_ylabel('Convergence Rate', fontsize=12)
    ax1.set_title('Convergence Rate vs Noise Level', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(noise_levels)))
    ax1.set_xticklabels([f"{x:.0%}" for x in noise_levels])
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, convergence_rates)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.0%}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Segregation Distribution by Noise Level
    ax2 = axes[0, 1]
    
    # Prepare data for box plot
    segregation_by_noise = []
    labels = []
    for noise_level_str, stats in sorted(data.items(), key=lambda x: float(x[0])):
        noise_level = float(noise_level_str)
        segregation_by_noise.append(stats['segregation_values'])
        labels.append(f"{noise_level:.0%}")
    
    box_plot = ax2.boxplot(segregation_by_noise, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = sns.color_palette("RdYlBu_r", len(segregation_by_noise))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Noise Level', fontsize=12)
    ax2.set_ylabel('Final Segregation (Mix Deviation)', fontsize=12)
    ax2.set_title('Segregation Distribution by Noise Level', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Average Segregation with Error Bars
    ax3 = axes[1, 0]
    
    avg_segregation = []
    std_segregation = []
    
    for noise_level_str, stats in sorted(data.items(), key=lambda x: float(x[0])):
        avg_segregation.append(stats['avg_segregation'])
        std_segregation.append(stats['std_segregation'])
    
    bars = ax3.bar(range(len(noise_levels)), avg_segregation, 
                   yerr=std_segregation, capsize=5, alpha=0.8,
                   color=sns.color_palette("RdYlBu_r", len(noise_levels)))
    
    ax3.set_xlabel('Noise Level', fontsize=12)
    ax3.set_ylabel('Average Final Segregation', fontsize=12)
    ax3.set_title('Average Segregation by Noise Level', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(noise_levels)))
    ax3.set_xticklabels([f"{x:.0%}" for x in noise_levels])
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars, avg_segregation, std_segregation)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 4. Effect Size Comparison to Baseline
    ax4 = axes[1, 1]
    
    baseline_values = data['0.0']['segregation_values']
    baseline_mean = np.mean(baseline_values)
    baseline_std = np.std(baseline_values)
    
    effect_sizes = []
    effect_labels = []
    
    for noise_level_str, stats in sorted(data.items(), key=lambda x: float(x[0])):
        noise_level = float(noise_level_str)
        if noise_level == 0.0:
            continue
            
        noise_values = stats['segregation_values']
        noise_mean = np.mean(noise_values)
        noise_std = np.std(noise_values)
        
        # Calculate Cohen's d
        pooled_std = np.sqrt((baseline_std**2 + noise_std**2) / 2)
        cohens_d = (noise_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
        
        effect_sizes.append(cohens_d)
        effect_labels.append(f"{noise_level:.0%}")
    
    # Color bars based on effect size
    colors = ['darkgreen' if x > 0.8 else 'green' if x > 0.5 else 'yellow' if x > 0.2 else 'orange' if x > -0.2 else 'red' 
              for x in effect_sizes]
    
    bars = ax4.bar(range(len(effect_sizes)), effect_sizes, alpha=0.8, color=colors)
    
    ax4.set_xlabel('Noise Level', fontsize=12)
    ax4.set_ylabel("Effect Size (Cohen's d)", fontsize=12)
    ax4.set_title('Effect Size vs Baseline (0% Noise)', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(effect_labels)))
    ax4.set_xticklabels(effect_labels)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
    ax4.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect')
    ax4.grid(axis='y', alpha=0.3)
    ax4.legend(fontsize=8)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, effect_sizes)):
        ax4.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.02 if val > 0 else bar.get_height() - 0.05, 
                f'{val:.2f}', ha='center', 
                va='bottom' if val > 0 else 'top', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f'noise_experiment_analysis_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive analysis plot saved to: {plot_filename}")
    
    return plot_filename

def create_summary_table(data):
    """Create a detailed summary table"""
    
    print("\n" + "="*90)
    print("📊 COMPREHENSIVE NOISE EXPERIMENT ANALYSIS")
    print("="*90)
    print(f"{'Noise':<8} {'Conv%':<8} {'Avg Steps':<12} {'Avg Segregation':<16} {'Std Dev':<10} {'Effect Size':<12} {'Interpretation':<20}")
    print("-"*90)
    
    baseline_values = data['0.0']['segregation_values']
    baseline_mean = np.mean(baseline_values)
    baseline_std = np.std(baseline_values)
    
    for noise_level_str, stats in sorted(data.items(), key=lambda x: float(x[0])):
        noise_level = float(noise_level_str)
        
        conv_rate = stats['convergence_rate']
        avg_steps = stats['avg_steps'] if stats['avg_steps'] != float('inf') else '∞'
        avg_segregation = stats['avg_segregation']
        std_segregation = stats['std_segregation']
        
        if noise_level == 0.0:
            effect_size = 0.0
            interpretation = "Baseline"
        else:
            noise_values = stats['segregation_values']
            noise_mean = np.mean(noise_values)
            noise_std = np.std(noise_values)
            
            pooled_std = np.sqrt((baseline_std**2 + noise_std**2) / 2)
            effect_size = (noise_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
            
            if abs(effect_size) < 0.2:
                interpretation = "Negligible"
            elif abs(effect_size) < 0.5:
                interpretation = "Small"
            elif abs(effect_size) < 0.8:
                interpretation = "Medium"
            else:
                interpretation = "Large"
        
        print(f"{noise_level:<8.0%} {conv_rate:<8.0%} {str(avg_steps):<12} "
              f"{avg_segregation:<16.3f} {std_segregation:<10.3f} {effect_size:<12.2f} {interpretation:<20}")
    
    # Key findings
    print("\n🔍 KEY FINDINGS:")
    print("-" * 50)
    
    # Convergence threshold
    conv_threshold = None
    for noise_level_str, stats in sorted(data.items(), key=lambda x: float(x[0])):
        noise_level = float(noise_level_str)
        if stats['convergence_rate'] < 0.5 and conv_threshold is None:
            conv_threshold = noise_level
            break
    
    if conv_threshold:
        print(f"• Convergence drops dramatically at {conv_threshold:.0%} noise")
    
    # Maximum segregation
    max_segregation_noise = None
    max_segregation_value = 0
    for noise_level_str, stats in sorted(data.items(), key=lambda x: float(x[0])):
        if stats['avg_segregation'] > max_segregation_value:
            max_segregation_value = stats['avg_segregation']
            max_segregation_noise = float(noise_level_str)
    
    print(f"• Peak segregation occurs at {max_segregation_noise:.0%} noise (avg = {max_segregation_value:.3f})")
    
    # Noise effect on segregation
    final_segregation = data[max(data.keys(), key=float)]['avg_segregation']
    segregation_change = final_segregation - baseline_mean
    print(f"• High noise ({max(data.keys(), key=float)}%) changes segregation by {segregation_change:+.3f} vs baseline")
    
    return None

def statistical_analysis(data):
    """Perform statistical analysis"""
    
    print("\n🔬 STATISTICAL ANALYSIS")
    print("-" * 50)
    
    try:
        from scipy import stats
        
        baseline_values = data['0.0']['segregation_values']
        
        print(f"Baseline (0% noise): Mean = {np.mean(baseline_values):.3f}, Std = {np.std(baseline_values):.3f}")
        print(f"Sample size per condition: {len(baseline_values)} runs")
        print()
        
        for noise_level_str, stats_data in sorted(data.items(), key=lambda x: float(x[0])):
            noise_level = float(noise_level_str)
            if noise_level == 0.0:
                continue
                
            noise_values = stats_data['segregation_values']
            
            # T-test
            t_stat, p_value = stats.ttest_ind(baseline_values, noise_values)
            
            # Effect size (Cohen's d)
            baseline_std = np.std(baseline_values)
            noise_std = np.std(noise_values)
            pooled_std = np.sqrt((baseline_std**2 + noise_std**2) / 2)
            cohens_d = (np.mean(noise_values) - np.mean(baseline_values)) / pooled_std
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
            
            print(f"Noise {noise_level:.0%} vs Baseline:")
            print(f"  t-test: t = {t_stat:.3f}, p = {p_value:.4f} {significance}")
            print(f"  Cohen's d = {cohens_d:.3f}")
            print(f"  Mean difference = {np.mean(noise_values) - np.mean(baseline_values):+.3f}")
            print()
            
    except ImportError:
        print("(scipy not available for statistical tests)")

def main():
    """Main analysis function"""
    
    # Load the data
    filename = 'noise_experiment_20250806_182141_summary.json'
    print(f"📁 Loading noise experiment data from: {filename}")
    
    try:
        data = load_noise_experiment_data(filename)
        print(f"✅ Successfully loaded data for {len(data)} noise levels")
        
        # Create visualizations
        plot_file = create_comprehensive_visualization(data)
        
        # Create summary table
        create_summary_table(data)
        
        # Statistical analysis
        statistical_analysis(data)
        
        print("\n✅ ANALYSIS COMPLETED!")
        print(f"📈 Visualization saved: {plot_file}")
        
        return data, plot_file
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    data, plot_file = main()
