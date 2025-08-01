import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define scenarios
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

def calculate_stability_score(df, metric, window=10):
    """Calculate stability score based on variance in rolling windows"""
    grouped = df.groupby('step')[metric]
    mean_values = grouped.mean()
    
    if len(mean_values) < window:
        return 0
    
    # Calculate rolling variance
    rolling_var = mean_values.rolling(window=window).var()
    
    # Stability score is inverse of average variance
    stability = 1 / (1 + rolling_var.mean())
    
    return stability

def analyze_stability():
    """Analyze stability and volatility patterns across contexts"""
    
    metrics = ['share', 'ghetto_rate', 'switch_rate', 'clusters']
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Analyze each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        stability_data = []
        
        for scenario_name, folder in scenarios.items():
            filepath = Path(f'experiments/{folder}/metrics_history.csv')
            if filepath.exists():
                df = pd.read_csv(filepath)
                
                # Calculate stability over time windows
                time_windows = [(0, 20), (20, 50), (50, 100), (100, 200)]
                
                for start, end in time_windows:
                    window_df = df[(df['step'] >= start) & (df['step'] < end)]
                    if len(window_df) > 0:
                        stability = calculate_stability_score(window_df, metric)
                        stability_data.append({
                            'Scenario': scenario_labels[scenario_name],
                            'Window': f'{start}-{end}',
                            'Stability': stability,
                            'Metric': metric
                        })
        
        # Create visualization
        stability_df = pd.DataFrame(stability_data)
        pivot_df = stability_df.pivot(index='Scenario', columns='Window', values='Stability')
        
        # Create heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Stability Score'}, ax=ax)
        ax.set_title(f'{metric.replace("_", " ").title()} Stability Over Time', fontsize=14)
        ax.set_xlabel('Time Window (steps)')
        ax.set_ylabel('Social Context')
    
    plt.suptitle('Stability Analysis: How Consistent Are Segregation Patterns?', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('experiments/stability_analysis.png', dpi=300, bbox_inches='tight')
    
    # Create trajectory variance plot
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    for scenario_name, folder in scenarios.items():
        filepath = Path(f'experiments/{folder}/metrics_history.csv')
        if filepath.exists():
            df = pd.read_csv(filepath)
            
            # Calculate coefficient of variation for each step
            cv_by_step = []
            steps = sorted(df['step'].unique())[:150]  # First 150 steps
            
            for step in steps:
                step_data = df[df['step'] == step]['share']
                if len(step_data) > 1:
                    cv = step_data.std() / step_data.mean() if step_data.mean() > 0 else 0
                    cv_by_step.append(cv)
                else:
                    cv_by_step.append(0)
            
            ax.plot(steps[:len(cv_by_step)], cv_by_step, 
                   label=scenario_labels[scenario_name], linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Coefficient of Variation (across runs)')
    ax.set_title('Trajectory Consistency: How Similar Are Different Runs?', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/trajectory_variance.png', dpi=300, bbox_inches='tight')
    
    print("\nSTABILITY ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Print overall stability rankings
    overall_stability = []
    
    for scenario_name, folder in scenarios.items():
        filepath = Path(f'experiments/{folder}/metrics_history.csv')
        if filepath.exists():
            df = pd.read_csv(filepath)
            
            # Calculate average stability across all metrics
            stabilities = []
            for metric in metrics:
                stab = calculate_stability_score(df, metric)
                stabilities.append(stab)
            
            avg_stability = np.mean(stabilities)
            overall_stability.append((scenario_labels[scenario_name], avg_stability))
    
    # Sort by stability
    overall_stability.sort(key=lambda x: x[1], reverse=True)
    
    print("\nOverall Stability Ranking (most to least stable):")
    for i, (scenario, stability) in enumerate(overall_stability, 1):
        print(f"{i}. {scenario:30s}: {stability:.4f}")
    
    # Analyze which contexts stabilize quickly vs slowly
    print("\nQuick vs Slow Stabilizers:")
    
    for scenario_name, folder in scenarios.items():
        filepath = Path(f'experiments/{folder}/metrics_history.csv')
        if filepath.exists():
            df = pd.read_csv(filepath)
            
            # Compare early vs late variance
            early_df = df[df['step'] <= 30]
            late_df = df[df['step'] >= 100]
            
            if len(early_df) > 0 and len(late_df) > 0:
                early_var = early_df.groupby('step')['share'].mean().var()
                late_var = late_df.groupby('step')['share'].mean().var()
                
                if early_var > late_var * 2:
                    pattern = "Quick stabilizer"
                elif late_var > early_var * 2:
                    pattern = "Slow stabilizer"
                else:
                    pattern = "Gradual stabilizer"
                
                print(f"{scenario_labels[scenario_name]:30s}: {pattern}")

if __name__ == "__main__":
    analyze_stability()
    print("\nStability analysis complete!")