import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os
from plateau_detection import detect_plateau, calculate_convergence_rate
import matplotlib.patches as mpatches
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_metric_evolution(experiments_data, metric, save_path=None):
    """
    Plot the evolution of a specific metric across different experiments
    
    Parameters:
    - experiments_data: Dictionary of experiment names to their data directories
    - metric: The metric to plot
    - save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments_data)))
    
    for idx, (exp_name, exp_dir) in enumerate(experiments_data.items()):
        df = pd.read_csv(f"{exp_dir}/step_statistics.csv")
        
        # Plot mean with confidence interval
        steps = df['step_']
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        
        if mean_col in df.columns:
            mean_values = df[mean_col]
            std_values = df[std_col]
            
            ax.plot(steps, mean_values, label=exp_name, color=colors[idx], linewidth=2)
            ax.fill_between(steps, 
                           mean_values - std_values, 
                           mean_values + std_values, 
                           color=colors[idx], alpha=0.2)
            
            # Detect and mark plateau
            plateau_start, plateau_info = detect_plateau(mean_values.values)
            if plateau_start:
                ax.axvline(x=plateau_start, color=colors[idx], linestyle='--', alpha=0.5)
                ax.text(plateau_start, mean_values.iloc[plateau_start], 
                       f'Plateau: {plateau_start}', 
                       rotation=90, verticalalignment='bottom', fontsize=8)
    
    ax.set_xlabel('Simulation Step', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Evolution of {metric.replace("_", " ").title()} Across Experiments', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_convergence_comparison(experiments_data, save_path=None):
    """
    Create a comprehensive comparison of convergence characteristics
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create subplots for different aspects
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
    
    # 1. Convergence time distribution
    ax1 = fig.add_subplot(gs[0, :])
    convergence_data = []
    
    for exp_name, exp_dir in experiments_data.items():
        conv_df = pd.read_csv(f"{exp_dir}/convergence_summary.csv")
        conv_steps = conv_df['convergence_step'].dropna()
        convergence_data.append({
            'Experiment': exp_name,
            'Convergence Step': conv_steps.values
        })
    
    # Box plot of convergence times
    conv_df_combined = pd.concat([
        pd.DataFrame({'Experiment': d['Experiment'], 'Convergence Step': d['Convergence Step']})
        for d in convergence_data
    ])
    
    sns.boxplot(data=conv_df_combined, x='Experiment', y='Convergence Step', ax=ax1)
    ax1.set_title('Distribution of Convergence Times', fontsize=14)
    ax1.set_xlabel('Experiment Type', fontsize=12)
    ax1.set_ylabel('Convergence Step', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Plateau values comparison
    metrics = ['clusters', 'distance', 'mix_deviation', 'share']
    
    for idx, metric in enumerate(metrics):
        row = (idx // 2) + 1
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        plateau_data = []
        
        for exp_name, exp_dir in experiments_data.items():
            df = pd.read_csv(f"{exp_dir}/step_statistics.csv")
            mean_values = df[f'{metric}_mean'].values
            
            plateau_start, plateau_info = detect_plateau(mean_values)
            if plateau_info and 'value' in plateau_info:
                plateau_data.append({
                    'Experiment': exp_name,
                    'Plateau Value': plateau_info['value']
                })
        
        if plateau_data:
            plateau_df = pd.DataFrame(plateau_data)
            sns.barplot(data=plateau_df, x='Experiment', y='Plateau Value', ax=ax)
            ax.set_title(f'{metric.replace("_", " ").title()} - Final Plateau Values', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Convergence Characteristics Comparison', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_scenario_comparison(llm_experiments, metrics_to_compare=['distance', 'mix_deviation', 'ghetto_rate'], 
                           save_path=None):
    """
    Compare different LLM scenario outcomes
    """
    n_metrics = len(metrics_to_compare)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    scenario_names = list(llm_experiments.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(scenario_names)))
    
    for idx, metric in enumerate(metrics_to_compare):
        ax = axes[idx]
        
        final_values = []
        errors = []
        
        for scenario in scenario_names:
            df = pd.read_csv(f"{llm_experiments[scenario]}/metrics_history.csv")
            
            # Get final values for each run
            final_df = df.groupby('run_id').last()
            values = final_df[metric].values
            
            final_values.append(np.mean(values))
            errors.append(np.std(values))
        
        # Create bar plot
        x = np.arange(len(scenario_names))
        bars = ax.bar(x, final_values, yerr=errors, capsize=5, color=colors)
        
        ax.set_xlabel('Scenario', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} by Scenario', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, final_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('LLM Scenario Impact on Segregation Metrics', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_comprehensive_report(baseline_dir, llm_dirs, output_path='segregation_analysis_report.pdf'):
    """
    Create a comprehensive PDF report comparing all experiments
    """
    with PdfPages(output_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.6, 'Schelling Segregation Model Analysis', 
                ha='center', va='center', fontsize=24, weight='bold')
        fig.text(0.5, 0.5, 'Comparison of Mechanical and LLM-based Agents', 
                ha='center', va='center', fontsize=18)
        fig.text(0.5, 0.4, f'with Different Social Context Scenarios', 
                ha='center', va='center', fontsize=16)
        
        import datetime
        fig.text(0.5, 0.2, f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                ha='center', va='center', fontsize=12)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 1. Baseline vs LLM comparison
        all_experiments = {'Baseline (Mechanical)': baseline_dir}
        all_experiments.update(llm_dirs)
        
        # Plot each metric evolution
        metrics = ['clusters', 'switch_rate', 'distance', 'mix_deviation', 'share', 'ghetto_rate']
        
        for metric in metrics:
            fig = plot_metric_evolution(all_experiments, metric)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # 2. Convergence comparison
        fig = plot_convergence_comparison(all_experiments)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 3. Scenario comparison (LLM only)
        if len(llm_dirs) > 1:
            fig = plot_scenario_comparison(llm_dirs)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # 4. Statistical summary page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('tight')
        ax.axis('off')
        
        # Create summary table
        summary_data = []
        
        for exp_name, exp_dir in all_experiments.items():
            conv_df = pd.read_csv(f"{exp_dir}/convergence_summary.csv")
            metrics_df = pd.read_csv(f"{exp_dir}/metrics_history.csv")
            
            # Calculate summary statistics
            final_metrics = metrics_df.groupby('run_id').last()
            
            summary_data.append({
                'Experiment': exp_name,
                'Avg Convergence': f"{conv_df['convergence_step'].mean():.1f} ± {conv_df['convergence_step'].std():.1f}",
                'Final Distance': f"{final_metrics['distance'].mean():.3f} ± {final_metrics['distance'].std():.3f}",
                'Final Mix Dev': f"{final_metrics['mix_deviation'].mean():.3f} ± {final_metrics['mix_deviation'].std():.3f}",
                'Final Ghetto Rate': f"{final_metrics['ghetto_rate'].mean():.1f} ± {final_metrics['ghetto_rate'].std():.1f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create table
        table = ax.table(cellText=summary_df.values,
                        colLabels=summary_df.columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        ax.set_title('Summary Statistics Across All Experiments', fontsize=16, pad=20)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 5. Add metadata
        d = pdf.infodict()
        d['Title'] = 'Schelling Segregation Model Analysis'
        d['Author'] = 'Segregation Simulation Framework'
        d['Subject'] = 'Comparison of Mechanical and LLM-based Agent Behaviors'
        d['Keywords'] = 'Schelling, Segregation, LLM, Agent-Based Model'
        d['CreationDate'] = datetime.datetime.now()
    
    print(f"Report saved to: {output_path}")

def plot_realtime_comparison(baseline_dir, llm_dir, metric='distance'):
    """
    Create an animated comparison of baseline vs LLM simulation
    """
    from matplotlib.animation import FuncAnimation
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Load data
    baseline_df = pd.read_csv(f"{baseline_dir}/step_statistics.csv")
    llm_df = pd.read_csv(f"{llm_dir}/step_statistics.csv")
    
    # Initialize plots
    line1, = ax1.plot([], [], 'b-', label='Baseline', linewidth=2)
    line2, = ax2.plot([], [], 'r-', label='LLM', linewidth=2)
    
    # Set limits
    max_steps = min(len(baseline_df), len(llm_df))
    ax1.set_xlim(0, max_steps)
    ax2.set_xlim(0, max_steps)
    
    y_min = min(baseline_df[f'{metric}_mean'].min(), llm_df[f'{metric}_mean'].min()) * 0.9
    y_max = max(baseline_df[f'{metric}_mean'].max(), llm_df[f'{metric}_mean'].max()) * 1.1
    
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    
    ax1.set_xlabel('Step')
    ax2.set_xlabel('Step')
    ax1.set_ylabel(metric.replace('_', ' ').title())
    
    ax1.set_title('Baseline (Mechanical) Agents')
    ax2.set_title('LLM Agents')
    
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    def animate(frame):
        line1.set_data(baseline_df['step_'][:frame], baseline_df[f'{metric}_mean'][:frame])
        line2.set_data(llm_df['step_'][:frame], llm_df[f'{metric}_mean'][:frame])
        return line1, line2
    
    anim = FuncAnimation(fig, animate, frames=max_steps, interval=50, blit=True)
    
    plt.tight_layout()
    return fig, anim

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize segregation simulation results")
    parser.add_argument('--baseline-dir', type=str, required=True, help='Baseline experiment directory')
    parser.add_argument('--llm-dirs', type=str, nargs='+', help='LLM experiment directories')
    parser.add_argument('--output', type=str, default='segregation_analysis_report.pdf', 
                        help='Output PDF filename')
    
    args = parser.parse_args()
    
    # Process LLM directories into dictionary
    llm_experiments = {}
    if args.llm_dirs:
        for llm_dir in args.llm_dirs:
            # Extract scenario name from directory
            scenario = llm_dir.split('/')[-1].split('_')[1]
            llm_experiments[scenario] = llm_dir
    
    create_comprehensive_report(args.baseline_dir, llm_experiments, args.output)