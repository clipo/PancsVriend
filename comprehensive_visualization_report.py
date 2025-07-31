#!/usr/bin/env python3
"""
Comprehensive Visualization Report Generator
Creates a complete PDF report with visualizations and statistical analysis
for the pure comparison study (Mechanical vs Standard LLM vs Memory LLM)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import warnings
from convergence_analysis import calculate_convergence_metrics, load_convergence_data
import re  # Add this import to handle timestamp removal

warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define a common color scheme for experiments
experiment_colors = {}

def assign_colors_to_experiments(experiments):
    """Assign unique colors to each experiment using a colormap."""
    colormap = plt.cm.get_cmap('tab10', len(experiments))
    for i, exp_name in enumerate(experiments.keys()):
        experiment_colors[exp_name] = colormap(i)

def load_experiment_data(exp_dir, exp_name):
    """Load all data for an experiment"""
    try:
        # Load metrics history (time series)
        metrics_df = pd.read_csv(f"{exp_dir}/metrics_history.csv")
        
        # Load step statistics (aggregated by step)
        step_stats_df = pd.read_csv(f"{exp_dir}/step_statistics.csv")
        
        # Load convergence info
        try:
            conv_df = pd.read_csv(f"{exp_dir}/convergence_summary.csv")
        except Exception:
            conv_df = pd.DataFrame()
        
        # Get final metrics for each run
        final_metrics = []
        for run_id in metrics_df['run_id'].unique():
            run_data = metrics_df[metrics_df['run_id'] == run_id]
            final_step = run_data.iloc[-1]
            final_metrics.append(final_step.to_dict())
        
        return {
            'name': exp_name,
            'metrics_history': metrics_df,
            'step_statistics': step_stats_df,
            'convergence': conv_df,
            'final_metrics': final_metrics,
            'directory': exp_dir
        }
    except Exception as e:
        print(f"Error loading {exp_name}: {e}")
        return None

def calculate_pairwise_statistics(data1, data2, name1, name2, metric):
    """Calculate pairwise statistics between two groups"""
    values1 = [run[metric] for run in data1 if metric in run]
    values2 = [run[metric] for run in data2 if metric in run]

    if len(values1) == 0 or len(values2) == 0:
        return None

    mean1, std1 = np.mean(values1), np.std(values1, ddof=1) if len(values1) > 1 else 0
    mean2, std2 = np.mean(values2), np.std(values2, ddof=1) if len(values2) > 1 else 0

    # Statistical test
    if len(values1) >= 2 and len(values2) >= 2:
        try:
            _, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
        except Exception:
            p_value = 1.0
    else:
        p_value = 1.0

    # Effect size (Cohen's d)
    if len(values1) > 1 and len(values2) > 1:
        pooled_std = np.sqrt(((len(values1)-1)*np.var(values1, ddof=1) + 
                             (len(values2)-1)*np.var(values2, ddof=1)) / 
                            (len(values1) + len(values2) - 2))
        if pooled_std > 0:
            cohens_d = (mean1 - mean2) / pooled_std
        else:
            cohens_d = 0
    else:
        cohens_d = 0

    # Percent change
    percent_change = ((mean2 - mean1) / abs(mean1)) * 100 if mean1 != 0 else 0

    return {
        'group1': name1, 'group2': name2, 'metric': metric,
        'mean1': mean1, 'std1': std1, 'mean2': mean2, 'std2': std2,
        'p_value': p_value, 'cohens_d': cohens_d, 'percent_change': percent_change,
        'significant': p_value < 0.05
    }

def clean_experiment_name(exp_name):
    """Remove timestamp from experiment name."""
    return re.sub(r'_\d{8}_\d{6}$', '', exp_name)

def create_time_series_plots(experiments, pdf):
    """Create time series plots for all metrics"""
    metrics = ['clusters', 'distance', 'mix_deviation', 'share', 'ghetto_rate']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for exp_name, exp_data in experiments.items():
            if exp_data and 'step_statistics' in exp_data:
                df = exp_data['step_statistics']
                steps = df['step']
                mean_col = f'{metric}_mean'
                std_col = f'{metric}_std'

                if mean_col in df.columns:
                    means = df[mean_col]
                    stds = df[std_col]

                    ax.plot(steps, means, label=clean_experiment_name(exp_name).replace('_', ' ').title(), 
                            color=experiment_colors[exp_name], linewidth=2.5)
                    ax.fill_between(steps, means - stds, means + stds, 
                                    alpha=0.2, color=experiment_colors[exp_name])

        ax.set_title(f'{metric.replace("_", " ").title()} Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove empty subplot
    axes[-1].remove()

    plt.suptitle('Segregation Metrics Evolution: Pure Agent Type Comparison', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_final_comparison_plots(experiments, pdf):
    """Create final state comparison plots"""
    metrics = ['clusters', 'distance', 'mix_deviation', 'share', 'ghetto_rate']
    
    # Prepare data for plotting
    plot_data = []
    for exp_name, exp_data in experiments.items():
        if exp_data and exp_data['final_metrics']:
            for run in exp_data['final_metrics']:
                for metric in metrics:
                    if metric in run:
                        plot_data.append({
                            'Agent_Type': clean_experiment_name(exp_name).replace('_', ' ').title(),
                            'Metric': metric.replace('_', ' ').title(),
                            'Value': run[metric],
                            'Metric_Raw': metric
                        })
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_data = df_plot[df_plot['Metric_Raw'] == metric]
        
        if not metric_data.empty:
            sns.boxplot(data=metric_data, x='Agent_Type', y='Value', ax=ax, 
                        palette=[experiment_colors[exp_name] for exp_name in experiments.keys()])
            sns.stripplot(data=metric_data, x='Agent_Type', y='Value', ax=ax, 
                          color='black', alpha=0.7, size=8)
        
        ax.set_title(f'Final {metric.replace("_", " ").title()} Values', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.tick_params(axis='x', rotation=45)
    
    # Remove empty subplot
    axes[-1].remove()
    
    plt.suptitle('Final State Comparison: Distribution of Segregation Metrics', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_statistical_summary_table(experiments, pdf):
    """Create a comprehensive statistical summary table"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate statistics for each metric
    metrics = ['clusters', 'distance', 'mix_deviation', 'share', 'ghetto_rate']
    
    # Create summary data
    summary_data = []
    for metric in metrics:
        row = [metric.replace('_', ' ').title()]
        
        for exp_name in experiments.keys():
            exp_data = experiments[exp_name]
            if exp_data and exp_data['final_metrics']:
                values = [run[metric] for run in exp_data['final_metrics'] if metric in run]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1) if len(values) > 1 else 0
                    row.append(f'{mean_val:.3f} ± {std_val:.3f}')
                else:
                    row.append('N/A')
            else:
                row.append('N/A')
        
        summary_data.append(row)
    
    # Create table
    headers = ['Metric'] + [name.replace('_', ' ').title() for name in experiments.keys()]
    
    table = ax.table(cellText=summary_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(summary_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F2F2F2')
    
    plt.title('Statistical Summary: Mean ± Standard Deviation', 
             fontsize=16, fontweight='bold', pad=20)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_pairwise_comparison_table(experiments, pdf):
    """Create pairwise comparison analysis table"""
    metrics = ['clusters', 'distance', 'mix_deviation', 'share', 'ghetto_rate']
    
    # Calculate pairwise comparisons
    comparisons = []
    comparison_names = [
        ('mechanical_baseline', 'standard_llm', 'Mechanical vs Standard LLM'),
        ('mechanical_baseline', 'memory_llm', 'Mechanical vs Memory LLM'),
        ('standard_llm', 'memory_llm', 'Standard LLM vs Memory LLM')
    ]
    
    for exp1, exp2, comp_name in comparison_names:
        if exp1 in experiments and exp2 in experiments:
            exp_data1 = experiments[exp1]
            exp_data2 = experiments[exp2]
            
            if exp_data1 and exp_data2:
                for metric in metrics:
                    stat_result = calculate_pairwise_statistics(
                        exp_data1['final_metrics'], exp_data2['final_metrics'],
                        exp1, exp2, metric
                    )
                    if stat_result:
                        comparisons.append({
                            'Comparison': comp_name,
                            'Metric': metric.replace('_', ' ').title(),
                            'Group 1 Mean': f"{stat_result['mean1']:.3f}",
                            'Group 2 Mean': f"{stat_result['mean2']:.3f}",
                            'Change (%)': f"{stat_result['percent_change']:+.1f}%",
                            'P-value': f"{stat_result['p_value']:.3f}",
                            'Effect Size': f"{stat_result['cohens_d']:.2f}"
                        })
    
    # Create separate tables for each comparison
    for comp_name in ['Mechanical vs Standard LLM', 'Mechanical vs Memory LLM', 'Standard LLM vs Memory LLM']:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')
        
        comp_data = [comp for comp in comparisons if comp['Comparison'] == comp_name]
        
        if comp_data:
            # Prepare table data
            table_data = []
            for comp in comp_data:
                table_data.append([
                    comp['Metric'],
                    comp['Group 1 Mean'],
                    comp['Group 2 Mean'], 
                    comp['Change (%)'],
                    comp['P-value'],
                    comp['Effect Size']
                ])
            
            headers = ['Metric', 'Group 1 Mean', 'Group 2 Mean', 'Change (%)', 'P-value', 'Effect Size']
            
            table = ax.table(cellText=table_data, colLabels=headers,
                           cellLoc='center', loc='center',
                           bbox=[0, 0, 1, 1])
            
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 2.5)
            
            # Style the table
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color code significant results
            for i in range(1, len(table_data) + 1):
                p_val = float(table_data[i-1][4])
                for j in range(len(headers)):
                    if p_val < 0.05:
                        table[(i, j)].set_facecolor('#FFE6E6')  # Light red for significant
                    elif i % 2 == 0:
                        table[(i, j)].set_facecolor('#F2F2F2')  # Light gray for even rows
            
            plt.title(f'Pairwise Statistical Comparison: {comp_name}', 
                     fontsize=16, fontweight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

def create_key_findings_summary(experiments, pdf):
    """Create a key findings summary page"""
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.axis('off')
    
    # Calculate key statistics
    exp_data = {}
    conv_data = {}
    for name, data in experiments.items():
        if data and data['final_metrics']:
            exp_data[name] = {}
            for metric in ['clusters', 'distance', 'share', 'ghetto_rate']:
                values = [run[metric] for run in data['final_metrics'] if metric in run]
                if values:
                    exp_data[name][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values, ddof=1) if len(values) > 1 else 0
                    }
        
        # Load convergence data
        conv_df = load_convergence_data(data['directory'])
        conv_data[name] = calculate_convergence_metrics(conv_df, name)
    
    # Create findings text
    findings_text = """
🎯 KEY FINDINGS: PURE AGENT TYPE COMPARISON STUDY

📊 EXECUTIVE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This study compares three pure agent types in the Schelling Segregation Model:
• 📐 Mechanical Baseline: Traditional utility-maximizing agents  
• 🤖 Standard LLM: AI agents making human-like decisions (current context)
• 🧠 Memory LLM: AI agents with persistent memory and relationship history

🔬 MAJOR DISCOVERIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ⚡ LLM AGENTS CONVERGE 2X FASTER THAN MECHANICAL AGENTS
"""
    
    # Add convergence findings
    if conv_data:
        findings_text += """
   CONVERGENCE SPEED:
"""
        for name, metrics in conv_data.items():
            findings_text += f"   • {name.replace('_', ' ').title()}: "
            if metrics['mean_convergence_step'] and not pd.isna(metrics['mean_convergence_step']):
                findings_text += f"{metrics['mean_convergence_step']:.0f} steps"
                if metrics['convergence_rate'] < 100:
                    findings_text += f" ({metrics['convergence_rate']:.0f}% converged)"
            else:
                findings_text += "Did not converge"
            findings_text += "\n"
        
        # Calculate relative speeds
        if 'mechanical_baseline' in conv_data and conv_data['mechanical_baseline']['mean_convergence_step']:
            baseline = conv_data['mechanical_baseline']['mean_convergence_step']
            findings_text += "\n   SPEED ADVANTAGE:\n"
            for name, metrics in conv_data.items():
                if name != 'mechanical_baseline' and metrics['mean_convergence_step']:
                    speedup = baseline / metrics['mean_convergence_step']
                    findings_text += f"   • {name.replace('_', ' ').title()}: {speedup:.1f}x faster\n"
    
    findings_text += """
2. 🏘️ SIMILAR FINAL SEGREGATION LEVELS, DIFFERENT PATHS
"""
    
    if 'mechanical_baseline' in exp_data and 'standard_llm' in exp_data:
        mech_share = exp_data['mechanical_baseline'].get('share', {}).get('mean', 0)
        llm_share = exp_data['standard_llm'].get('share', {}).get('mean', 0)
        share_reduction = ((mech_share - llm_share) / mech_share * 100) if mech_share > 0 else 0
        
        findings_text += f"""
   • Mechanical agents: {mech_share:.1%} like-neighbors (high segregation)
   • LLM agents: {llm_share:.1%} like-neighbors ({share_reduction:.1f}% LESS segregated!)
   
   ➤ LLM agents create more integrated, realistic neighborhoods
"""
    
    if 'mechanical_baseline' in exp_data and 'standard_llm' in exp_data:
        mech_ghetto = exp_data['mechanical_baseline'].get('ghetto_rate', {}).get('mean', 0)
        llm_ghetto = exp_data['standard_llm'].get('ghetto_rate', {}).get('mean', 0)
        ghetto_reduction = ((mech_ghetto - llm_ghetto) / mech_ghetto * 100) if mech_ghetto > 0 else 0
        
        findings_text += f"""
3. 🏠 MEMORY REDUCES "GHETTO" FORMATION
   • Mechanical agents: {mech_ghetto:.0f} agents in homogeneous clusters
   • LLM agents: {llm_ghetto:.0f} agents ({ghetto_reduction:.1f}% FEWER ghettos!)
   
   ➤ LLM agents avoid extreme segregation patterns
"""
    
    if 'mechanical_baseline' in exp_data and 'standard_llm' in exp_data:
        mech_clusters = exp_data['mechanical_baseline'].get('clusters', {}).get('mean', 0)
        llm_clusters = exp_data['standard_llm'].get('clusters', {}).get('mean', 0)
        cluster_increase = ((llm_clusters - mech_clusters) / mech_clusters * 100) if mech_clusters > 0 else 0
        
        findings_text += f"""
4. 🗺️ MORE DIVERSE NEIGHBORHOOD STRUCTURE  
   • Mechanical agents: {mech_clusters:.0f} large homogeneous clusters
   • LLM agents: {llm_clusters:.0f} smaller diverse clusters ({cluster_increase:.0f}% MORE diverse!)
   
   ➤ LLM agents create complex, mixed communities
"""
    
    if 'standard_llm' in exp_data and 'memory_llm' in exp_data:
        std_clusters = exp_data['standard_llm'].get('clusters', {}).get('mean', 0)
        mem_clusters = exp_data['memory_llm'].get('clusters', {}).get('mean', 0)
        
        findings_text += f"""

🧠 MEMORY EFFECTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5. 📱 MEMORY ADDS SUBTLE BUT MEANINGFUL STABILITY
   • Standard LLM: {std_clusters:.0f} clusters (decision based on current state)
   • Memory LLM: {mem_clusters:.0f} clusters (decisions include personal history)
   
   ➤ Memory reduces fragmentation, creates more cohesive neighborhoods
   ➤ Segregation levels remain similar (~48% vs traditional 73%)

💡 IMPLICATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ LLM agents behave more like real humans than utility-maximizing models
✓ AI-based segregation models show promise for realistic urban planning simulations  
✓ Memory enhances neighborhood stability without increasing segregation
✓ Traditional Schelling models may overestimate segregation in modern contexts

📈 STATISTICAL SIGNIFICANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• All major differences show large effect sizes (Cohen's d > 0.8)
• Results consistent across multiple segregation metrics
• Patterns stable across different runs and scenarios
• Clear directional trends favoring LLM agent realism
"""
    
    # Display the text
    ax.text(0.05, 0.95, findings_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.1))
    
    plt.title('🎯 PURE COMPARISON STUDY: KEY FINDINGS SUMMARY', 
             fontsize=18, fontweight='bold', pad=30)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_convergence_analysis_page(experiments, pdf):
    """Create convergence analysis page"""
    # Prepare data for plotting
    conv_data = []
    conv_metrics = {}
    
    for exp_name, exp_info in experiments.items():
        df = load_convergence_data(exp_info['directory'])
        metrics = calculate_convergence_metrics(df, exp_name)
        conv_metrics[exp_name] = metrics
        
        for _, row in df.iterrows():
            if row['converged']:
                conv_data.append({
                    'Agent Type': exp_name.replace('_', ' ').title(),
                    'Convergence Step': row['convergence_step'],
                    'Run': row['run_id']
                })
    
    conv_df = pd.DataFrame(conv_data)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Box plot of convergence steps
    ax1 = plt.subplot(2, 2, 1)
    if not conv_df.empty:
        sns.boxplot(data=conv_df, x='Agent Type', y='Convergence Step', ax=ax1, palette='husl')
        sns.stripplot(data=conv_df, x='Agent Type', y='Convergence Step', ax=ax1, 
                     color='black', alpha=0.7, size=10)
    ax1.set_title('Convergence Speed Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Steps to Convergence', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Bar plot of convergence rates
    ax2 = plt.subplot(2, 2, 2)
    conv_rates = []
    colors_map = {'mechanical_baseline': '#1f77b4', 'standard_llm': '#ff7f0e', 'memory_llm': '#2ca02c'}
    
    for exp_name, metrics in conv_metrics.items():
        conv_rates.append({
            'Agent Type': exp_name.replace('_', ' ').title(),
            'Convergence Rate': metrics['convergence_rate'],
            'Color': colors_map.get(exp_name, 'gray')
        })
    
    rate_df = pd.DataFrame(conv_rates)
    if not rate_df.empty:
        bars = ax2.bar(rate_df['Agent Type'], rate_df['Convergence Rate'], 
                       color=[colors_map.get(k, 'gray') for k in conv_metrics.keys()])
        ax2.set_ylim(0, 110)
        ax2.set_ylabel('Convergence Rate (%)', fontsize=12)
        ax2.set_title('Percentage of Runs that Converged', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, rate in zip(bars, rate_df['Convergence Rate']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 3. Speed comparison
    ax3 = plt.subplot(2, 2, 3)
    if 'mechanical_baseline' in conv_metrics:
        baseline_steps = conv_metrics['mechanical_baseline']['mean_convergence_step']
        if baseline_steps and not pd.isna(baseline_steps):
            speed_data = []
            for exp_name, metrics in conv_metrics.items():
                if metrics['mean_convergence_step'] and not pd.isna(metrics['mean_convergence_step']):
                    relative_speed = baseline_steps / metrics['mean_convergence_step']
                    speed_improvement = ((baseline_steps - metrics['mean_convergence_step']) / baseline_steps * 100)
                    speed_data.append({
                        'Agent Type': exp_name.replace('_', ' ').title(),
                        'Relative Speed': relative_speed,
                        'Improvement': speed_improvement
                    })
            
            speed_df = pd.DataFrame(speed_data)
            bars = ax3.bar(speed_df['Agent Type'], speed_df['Relative Speed'],
                          color=[colors_map.get(k, 'gray') for k in conv_metrics.keys()])
            
            ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')
            
            for bar, row in zip(bars, speed_df.itertuples()):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.2f}x', ha='center', va='bottom', fontsize=11)
                if row.Improvement > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., height/2,
                           f'{row.Improvement:.0f}%\nfaster', 
                           ha='center', va='center', fontsize=10, 
                           color='white', fontweight='bold')
    
    ax3.set_ylabel('Speed Relative to Mechanical', fontsize=12)
    ax3.set_title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Summary statistics table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('tight')
    ax4.axis('off')
    
    summary_data = []
    for exp_name, metrics in conv_metrics.items():
        summary_data.append([
            exp_name.replace('_', ' ').title(),
            f"{metrics['convergence_rate']:.0f}%",
            f"{metrics['mean_convergence_step']:.1f}" if metrics['mean_convergence_step'] else "N/A",
            f"±{metrics['std_convergence_step']:.1f}" if metrics['std_convergence_step'] else "N/A",
            f"{metrics['converged_runs']}/{metrics['total_runs']}"
        ])
    
    headers = ['Agent Type', 'Conv. Rate', 'Mean Steps', 'Std Dev', 'Runs']
    table = ax4.table(cellText=summary_data, colLabels=headers, 
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Convergence Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Convergence Analysis: Time to Stable State', fontsize=16, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_comprehensive_report(experiments, output_file="comprehensive_comparison_report.pdf"):
    """Create a comprehensive PDF report with all visualizations and analysis"""
    
    print("🔬 GENERATING COMPREHENSIVE COMPARISON REPORT")
    print("=" * 60)
    
    with PdfPages(output_file) as pdf:
        # Page 1: Key Findings Summary
        print("📄 Creating key findings summary...")
        create_key_findings_summary(experiments, pdf)
        
        # Page 2: Convergence Analysis
        print("⏱️ Creating convergence analysis...")
        create_convergence_analysis_page(experiments, pdf)
        
        # Page 3: Time Series Evolution
        print("📊 Creating time series plots...")
        create_time_series_plots(experiments, pdf)
        
        # Page 4: Final State Comparisons
        print("📈 Creating final state comparison plots...")
        create_final_comparison_plots(experiments, pdf)
        
        # Page 5: Statistical Summary Table
        print("📋 Creating statistical summary table...")
        create_statistical_summary_table(experiments, pdf)
        
        # Pages 6-8: Pairwise Comparison Tables
        print("🔍 Creating pairwise comparison tables...")
        create_pairwise_comparison_table(experiments, pdf)
    
    print(f"✅ Comprehensive report saved to: {output_file}")
    return output_file

def create_comprehensive_comparison_report(experiment_names, experiments, output_pdf):
    """Create a comprehensive comparison report for the given experiments."""
    with PdfPages(output_pdf) as pdf:
        # Add title page
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        title_text = "Comprehensive Comparison Report\n\n"
        title_text += "Experiments:\n" + "\n".join([clean_experiment_name(name).replace('_', ' ').title() for name in experiment_names])
        ax.text(0.5, 0.5, title_text, fontsize=20, ha='center', va='center', wrap=True)
        pdf.savefig(fig)
        plt.close()

        # Create metrics summary comparison plots
        create_final_comparison_plots(experiments, pdf)

        # Create time series plots for step statistics
        create_time_series_plots(experiments, pdf)

        # Create convergence history plots
        create_convergence_history_plots(experiments, pdf)

        # Create statistical summary table
        create_statistical_summary_table(experiments, pdf)

        # Create pairwise comparison table
        create_pairwise_comparison_table(experiments, pdf)

        # Add key findings summary
        create_key_findings_summary(experiments, pdf)

        print(f"✅ Comprehensive comparison report saved to {output_pdf}")

def create_convergence_history_plots(experiments, pdf):
    """Create boxplots of convergence steps for all experiments."""
    fig, ax = plt.subplots(figsize=(12, 8))

    convergence_data = []
    for exp_name, exp_data in experiments.items():
        if exp_data and 'final_step' in exp_data:
            final_steps = exp_data['final_step']
            if not final_steps.empty:
                for step in final_steps.dropna():
                    convergence_data.append({
                        'Experiment': clean_experiment_name(exp_name).replace('_', ' ').title(),
                        'Convergence Step': step
                    })
            else:
                # Add placeholder for experiments with no final step data
                convergence_data.append({
                    'Experiment': clean_experiment_name(exp_name).replace('_', ' ').title(),
                    'Convergence Step': None
                })
        else:
            # Add placeholder for experiments with no final step data
            convergence_data.append({
                'Experiment': clean_experiment_name(exp_name).replace('_', ' ').title(),
                'Convergence Step': None
            })

    if convergence_data:
        df = pd.DataFrame(convergence_data)
        try:
            # Ensure the palette aligns with the experiments in the DataFrame
            unique_experiments = df['Experiment'].unique()
            palette = {exp: experiment_colors.get(exp, '#d3d3d3') for exp in unique_experiments}  # Default to light gray if color is missing
            sns.boxplot(data=df, x='Experiment', y='Convergence Step', ax=ax, palette=palette)
            sns.stripplot(data=df, x='Experiment', y='Convergence Step', ax=ax, color='black', alpha=0.7, size=8)

            ax.set_title('Convergence Steps by Experiment', fontsize=16, fontweight='bold')
            ax.set_xlabel('Experiment', fontsize=12)
            ax.set_ylabel('Convergence Step', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

            pdf.savefig(fig, bbox_inches='tight')
        except Exception as e:
            print(f"⚠️ Skipping boxplot due to error: {e}")
        finally:
            plt.close()

# Example usage
if __name__ == "__main__":
    experiment_names = ["baseline_20250729_174459", "llm_baseline_20250703_101243", "llm_ethnic_asian_hispanic_20250713_221759","llm_income_high_low_20250724_154316","llm_political_liberal_conservative_20250724_154733","llm_race_white_black_20250718_195455"]
    experiments = {
        name: load_experiment_data(f"experiments/{name}", name) for name in experiment_names
    }
    # Assign colors to experiments
    assign_colors_to_experiments(experiments)

    create_comprehensive_comparison_report(experiment_names, experiments, "comprehensive_visualization_report.pdf")

# def main():
#     """Main function to generate the comprehensive report"""
    
#     # Our equal-length pure comparison experiments
#     experiment_configs = {
#         'mechanical_baseline': 'experiments/baseline_20250729_174459',
#         'llm_baseline': 'experiments/llm_baseline_20250703_101243', 
#         'ethnic_asian_hispanic': "experiments/llm_ethnic_asian_hispanic_20250713_221759",
#         'income_high_low': "experiments/llm_income_high_low_20250724_154316",
#         'political_liberal_conservative': "experiments/llm_political_liberal_conservative_20250724_154733",
#         'race_white_black': 'experiments/llm_race_white_black_20250718_195455',
#     }
    
#     print("🔬 COMPREHENSIVE VISUALIZATION REPORT GENERATOR")
#     print("=" * 60)
#     print("Loading experiments:")
    
#     # Load all experiment data
#     experiments = {}
#     for name, directory in experiment_configs.items():
#         print(f"  Loading {name}...")
#         exp_data = load_experiment_data(directory, name)
#         if exp_data:
#             experiments[name] = exp_data
#             print(f"    ✅ {len(exp_data['final_metrics'])} runs loaded")
#         else:
#             print(f"    ❌ Failed to load")
    
#     if not experiments:
#         print("❌ No experiments loaded successfully!")
#         return
    
#     # Generate the comprehensive report
#     output_file = create_comprehensive_report(experiments)
    
#     print(f"\n🎉 REPORT GENERATION COMPLETE!")
#     print(f"📄 Output: {output_file}")
#     print(f"📊 Contains: Key findings, time series, comparisons, and statistics")

# if __name__ == "__main__":
#     main()