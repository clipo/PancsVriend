import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import cast
# from scipy import stats  # unused
# import matplotlib.patches as mpatches  # unused
from analysis_tools.experiment_list_for_analysis import (
    SCENARIOS as scenarios,
    SCENARIO_LABELS as scenario_labels,
)
from analysis_tools.output_paths import get_reports_dir

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# scenarios and labels now imported from centralized module

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
    out_dir = get_reports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'stability_analysis.png', dpi=300, bbox_inches='tight')
    
    # Create trajectory variance plot
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    for scenario_name, folder in scenarios.items():
        filepath = Path(f'experiments/{folder}/metrics_history.csv')
        if filepath.exists():
            df = pd.read_csv(filepath)
            
            # Calculate coefficient of variation for each step
            cv_by_step = []
            steps = sorted(df['step'].unique())[:150]  # First 150 steps
            # steps = sorted(df['step'].unique()) 
            
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
    plt.savefig(out_dir / 'trajectory_variance.png', dpi=300, bbox_inches='tight')
    
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
                
                early_val_raw = cast(float, early_var) if early_var is not None else 0.0
                late_val_raw = cast(float, late_var) if late_var is not None else 0.0
                early_val = float(early_val_raw)
                late_val = float(late_val_raw)

                if early_val > late_val * 2:
                    pattern = "Quick stabilizer"
                elif late_val > early_val * 2:
                    pattern = "Slow stabilizer"
                else:
                    pattern = "Gradual stabilizer"
                
                print(f"{scenario_labels[scenario_name]:30s}: {pattern}")

if __name__ == "__main__":
    analyze_stability()
    print("\nStability analysis complete!")