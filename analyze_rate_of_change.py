import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.signal import savgol_filter
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define scenarios and their paths
scenarios = {
    'baseline': 'llm_baseline_20250703_101243',
    'ethnic_asian_hispanic': 'llm_ethnic_asian_hispanic_20250713_221759',
    'income_high_low': 'llm_income_high_low_20250724_154316',
    'political_liberal_conservative': 'llm_political_liberal_conservative_20250724_154733',
    'race_white_black': 'llm_race_white_black_20250718_195455'
}

scenario_labels = {
    'baseline': 'Baseline',
    'ethnic_asian_hispanic': 'Ethnic',
    'income_high_low': 'Income',
    'political_liberal_conservative': 'Political',
    'race_white_black': 'Race'
}

# Color scheme
scenario_colors = {
    'baseline': '#1f77b4',
    'race_white_black': '#ff7f0e',
    'ethnic_asian_hispanic': '#2ca02c',
    'income_high_low': '#d62728',
    'political_liberal_conservative': '#9467bd'
}

def calculate_rate_of_change(series, window=5):
    """Calculate smoothed rate of change"""
    if len(series) < window * 2:
        return np.zeros_like(series)
    
    # Smooth the series first
    smoothed = savgol_filter(series, window_length=min(len(series), window*2+1), polyorder=3)
    
    # Calculate rate of change
    roc = np.gradient(smoothed)
    
    return roc

def analyze_dynamics():
    """Analyze the dynamics of metric changes across contexts"""
    
    metrics = ['clusters', 'switch_rate', 'distance', 'mix_deviation', 'share', 'ghetto_rate']
    metric_labels = {
        'clusters': 'Number of Clusters',
        'switch_rate': 'Switch Rate',
        'distance': 'Average Distance',
        'mix_deviation': 'Mix Deviation',
        'share': 'Segregation Share',
        'ghetto_rate': 'Ghetto Formation Rate'
    }
    
    # Collect data for all scenarios
    all_data = {}
    
    for scenario_name, folder in scenarios.items():
        filepath = Path(f'experiments/{folder}/metrics_history.csv')
        if filepath.exists():
            df = pd.read_csv(filepath)
            all_data[scenario_name] = df
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(5, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # 1. Rate of Change Heatmap
    ax1 = fig.add_subplot(gs[0, :])
    
    # Calculate average rate of change for each metric/scenario
    roc_matrix = []
    scenario_order = ['income_high_low', 'baseline', 'race_white_black', 
                     'ethnic_asian_hispanic', 'political_liberal_conservative']
    
    for scenario in scenario_order:
        if scenario in all_data:
            df = all_data[scenario]
            roc_row = []
            
            for metric in metrics:
                # Get mean trajectory
                mean_trajectory = df.groupby('step')[metric].mean()
                
                # Calculate rate of change
                roc = calculate_rate_of_change(mean_trajectory.values)
                
                # Take absolute mean rate of change
                avg_roc = np.mean(np.abs(roc))
                roc_row.append(avg_roc)
            
            roc_matrix.append(roc_row)
    
    # Create heatmap
    roc_array = np.array(roc_matrix)
    im = ax1.imshow(roc_array, cmap='YlOrRd', aspect='auto')
    
    ax1.set_xticks(range(len(metrics)))
    ax1.set_yticks(range(len(scenario_order)))
    ax1.set_xticklabels([metric_labels[m] for m in metrics], rotation=45, ha='right')
    ax1.set_yticklabels([scenario_labels[s] for s in scenario_order])
    
    # Add values
    for i in range(len(scenario_order)):
        for j in range(len(metrics)):
            text = ax1.text(j, i, f'{roc_array[i, j]:.3f}',
                           ha="center", va="center", color="black" if roc_array[i, j] < 0.05 else "white")
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Average Absolute Rate of Change', rotation=270, labelpad=20)
    ax1.set_title('Rate of Change Intensity Across Metrics and Contexts', fontsize=16, fontweight='bold')
    
    # 2. Early vs Late Stage Dynamics
    ax2 = fig.add_subplot(gs[1, :])
    
    early_cutoff = 20
    late_cutoff = 80
    
    early_late_data = []
    
    for scenario in scenario_order:
        if scenario in all_data:
            df = all_data[scenario]
            
            for metric in ['share', 'ghetto_rate']:  # Focus on key segregation metrics
                # Early stage
                early_data = df[df['step'] <= early_cutoff].groupby('step')[metric].mean()
                if len(early_data) > 5:
                    early_roc = calculate_rate_of_change(early_data.values)
                    early_volatility = np.std(early_roc)
                else:
                    early_volatility = 0
                
                # Late stage
                late_data = df[df['step'] >= late_cutoff].groupby('step')[metric].mean()
                if len(late_data) > 5:
                    late_roc = calculate_rate_of_change(late_data.values)
                    late_volatility = np.std(late_roc)
                else:
                    late_volatility = 0
                
                early_late_data.append({
                    'Scenario': scenario_labels[scenario],
                    'Metric': metric_labels[metric],
                    'Stage': 'Early (≤20 steps)',
                    'Volatility': early_volatility
                })
                
                early_late_data.append({
                    'Scenario': scenario_labels[scenario],
                    'Metric': metric_labels[metric],
                    'Stage': 'Late (≥80 steps)',
                    'Volatility': late_volatility
                })
    
    early_late_df = pd.DataFrame(early_late_data)
    
    # Create grouped bar plot
    pivot_df = early_late_df.pivot_table(values='Volatility', 
                                         index=['Scenario', 'Metric'], 
                                         columns='Stage')
    pivot_df.plot(kind='bar', ax=ax2, rot=45)
    ax2.set_ylabel('Volatility (Std of Rate of Change)')
    ax2.set_title('Early vs Late Stage Dynamics by Context', fontsize=14, fontweight='bold')
    ax2.legend(title='Stage')
    
    # 3. Trajectory Patterns for Each Metric
    for idx, metric in enumerate(metrics):
        row = 2 + idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        max_step = 150  # Focus on first 150 steps
        
        for scenario in scenario_order:
            if scenario in all_data:
                df = all_data[scenario]
                
                # Get mean trajectory
                mean_trajectory = df[df['step'] <= max_step].groupby('step')[metric].mean()
                
                # Calculate rate of change
                roc = calculate_rate_of_change(mean_trajectory.values, window=3)
                
                # Create twin axis for rate of change
                ax2 = ax.twinx()
                
                # Plot trajectory
                line1 = ax.plot(mean_trajectory.index, mean_trajectory.values, 
                               label=scenario_labels[scenario], 
                               color=scenario_colors[scenario], 
                               linewidth=2, alpha=0.8)
                
                # Plot rate of change as area
                ax2.fill_between(mean_trajectory.index[:len(roc)], 0, roc, 
                               alpha=0.2, color=scenario_colors[scenario])
        
        ax.set_xlabel('Step')
        ax.set_ylabel(metric_labels[metric])
        ax2.set_ylabel('Rate of Change', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax.set_title(f'{metric_labels[metric]} Evolution and Rate of Change', fontsize=12)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Dynamics of Segregation: How Different Contexts Evolve', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('experiments/rate_of_change_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('experiments/rate_of_change_analysis.pdf', bbox_inches='tight')
    
    # Additional Analysis: Phase Transitions
    fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for scenario in scenario_order:
            if scenario in all_data:
                df = all_data[scenario]
                
                # Get mean trajectory
                mean_trajectory = df.groupby('step')[metric].mean()
                
                # Calculate acceleration (second derivative)
                roc = calculate_rate_of_change(mean_trajectory.values)
                acceleration = np.gradient(roc)
                
                # Smooth for visualization
                if len(acceleration) > 20:
                    acceleration_smooth = savgol_filter(acceleration, 
                                                      window_length=min(21, len(acceleration)), 
                                                      polyorder=3)
                else:
                    acceleration_smooth = acceleration
                
                # Find phase transitions (peaks in acceleration)
                ax.plot(mean_trajectory.index[:len(acceleration_smooth)], 
                       acceleration_smooth, 
                       label=scenario_labels[scenario],
                       color=scenario_colors[scenario],
                       linewidth=2, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Acceleration')
        ax.set_title(f'{metric_labels[metric]} - Phase Transitions', fontsize=12)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
    
    plt.suptitle('Phase Transitions: When Segregation Patterns Shift', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('experiments/phase_transitions_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('experiments/phase_transitions_analysis.pdf', bbox_inches='tight')
    
    # Summary Statistics
    print("\nRATE OF CHANGE ANALYSIS SUMMARY")
    print("=" * 60)
    
    print("\nAverage Rate of Change by Context (across all metrics):")
    for i, scenario in enumerate(scenario_order):
        avg_roc_scenario = np.mean(roc_array[i])
        print(f"{scenario_labels[scenario]:30s}: {avg_roc_scenario:.4f}")
    
    print("\nMost Dynamic Metric by Context:")
    for i, scenario in enumerate(scenario_order):
        max_idx = np.argmax(roc_array[i])
        max_metric = metrics[max_idx]
        max_value = roc_array[i][max_idx]
        print(f"{scenario_labels[scenario]:30s}: {metric_labels[max_metric]} ({max_value:.4f})")
    
    print("\nEarly vs Late Stage Volatility Ratio:")
    for scenario in scenario_order:
        scenario_data = early_late_df[early_late_df['Scenario'] == scenario_labels[scenario]]
        if not scenario_data.empty:
            early_vol = scenario_data[scenario_data['Stage'].str.contains('Early')]['Volatility'].mean()
            late_vol = scenario_data[scenario_data['Stage'].str.contains('Late')]['Volatility'].mean()
            if late_vol > 0:
                ratio = early_vol / late_vol
                print(f"{scenario_labels[scenario]:30s}: {ratio:.2f}x more volatile early")

if __name__ == "__main__":
    analyze_dynamics()
    print("\nAnalysis complete! Check experiments/ folder for visualizations.")