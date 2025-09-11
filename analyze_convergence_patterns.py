import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define scenarios and their paths
scenarios = {
    'baseline': 'llm_baseline_20250703_101243',
    'ethnic_asian_hispanic': 'llm_ethnic_asian_hispanic_20250713_221759',
    'income_high_low': 'llm_income_high_low_20250724_154316',
    'economic_high_working' : "llm_economic_high_working_20250728_220134",
    'political_liberal_conservative': 'llm_political_liberal_conservative_20250724_154733',
    'race_white_black': 'llm_race_white_black_20250718_195455'
}

scenario_labels = {
    'baseline': 'Baseline (Control)',
    'ethnic_asian_hispanic': 'Ethnic (Asian/Hispanic)',
    'income_high_low': 'Income (High/Low)',
    'economic_high_working': 'Economic (High/Working)',
    'political_liberal_conservative': 'Political (Liberal/Conservative)',
    'race_white_black': 'Race (White/Black)'
}

# Create figure for time series
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

metrics = ['clusters', 'switch_rate', 'distance', 'mix_deviation', 'share', 'ghetto_rate']
metric_labels = {
    'clusters': 'Number of Clusters',
    'switch_rate': 'Switch Rate',
    'distance': 'Average Distance',
    'mix_deviation': 'Mix Deviation',
    'share': 'Segregation Share',
    'ghetto_rate': 'Ghetto Formation Rate'
}

# For each metric, plot convergence patterns
for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    for scenario_name, folder in scenarios.items():
        filepath = Path(f'experiments/{folder}/metrics_history.csv')
        if filepath.exists():
            df = pd.read_csv(filepath)
            
            # Calculate mean and confidence interval at each step
            grouped = df.groupby('step')[metric]
            mean_values = grouped.mean()
            std_values = grouped.std()
            count_values = grouped.count()
            
            # Calculate 95% confidence interval
            ci = 1.96 * std_values / np.sqrt(count_values)
            
            # Plot only up to step 100 for clarity
            max_step = min(1000, mean_values.index.max())
            steps = mean_values.index[mean_values.index <= max_step]
            
            # Plot mean line
            ax.plot(steps, mean_values[steps], label=scenario_labels[scenario_name], 
                   linewidth=2, alpha=0.8)
            
            # Add confidence interval
            ax.fill_between(steps, 
                          mean_values[steps] - ci[steps],
                          mean_values[steps] + ci[steps],
                          alpha=0.2)
    
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel(metric_labels[metric])
    ax.set_title(f'{metric_labels[metric]} Over Time', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Convergence Patterns of Segregation Metrics Across Scenarios', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('experiments/convergence_patterns.png', dpi=300, bbox_inches='tight')
plt.savefig('experiments/convergence_patterns.pdf', bbox_inches='tight')

# Calculate convergence speed (steps to reach 90% of final value)
print("\nCONVERGENCE ANALYSIS:")
print("=" * 60)

convergence_data = {}

for scenario_name, folder in scenarios.items():
    filepath = Path(f'experiments/{folder}/metrics_history.csv')
    if filepath.exists():
        df = pd.read_csv(filepath)
        convergence_data[scenario_name] = {}
        
        print(f"\n{scenario_labels[scenario_name]}:")
        
        for metric in metrics:
            # Get final values for each run
            final_values = df.groupby('run_id')[metric].last()
            
            # For each run, find when it reaches 90% of its final value
            convergence_steps = []
            
            for run_id in df['run_id'].unique():
                run_data = df[df['run_id'] == run_id]
                final_val = run_data[metric].iloc[-1]
                initial_val = run_data[metric].iloc[0]
                
                # Calculate 90% of the change
                if final_val != initial_val:
                    target_val = initial_val + 0.9 * (final_val - initial_val)
                    
                    # Find first step where this is reached
                    if final_val > initial_val:
                        conv_data = run_data[run_data[metric] >= target_val]
                    else:
                        conv_data = run_data[run_data[metric] <= target_val]
                    
                    if not conv_data.empty:
                        convergence_steps.append(conv_data['step'].iloc[0])
            
            if convergence_steps:
                mean_conv = np.mean(convergence_steps)
                std_conv = np.std(convergence_steps)
                convergence_data[scenario_name][metric] = mean_conv
                print(f"  {metric}: {mean_conv:.1f} ± {std_conv:.1f} steps")

# Create convergence speed comparison chart
fig2, ax2 = plt.subplots(figsize=(12, 8))

# Prepare data for grouped bar chart
x = np.arange(len(metrics))
width = 0.15
multiplier = 0

for scenario_name in ['baseline', 'income_high_low', 'race_white_black', 
                     'ethnic_asian_hispanic', 'political_liberal_conservative']:
    if scenario_name in convergence_data:
        values = [convergence_data[scenario_name].get(metric, 0) for metric in metrics]
        offset = width * multiplier
        ax2.bar(x + offset, values, width, label=scenario_labels[scenario_name])
        multiplier += 1

ax2.set_xlabel('Segregation Metric', fontsize=12)
ax2.set_ylabel('Steps to 90% Convergence', fontsize=12)
ax2.set_title('Convergence Speed Comparison Across Scenarios', fontsize=14, fontweight='bold')
ax2.set_xticks(x + width * 2)
ax2.set_xticklabels([metric_labels[m] for m in metrics], rotation=45, ha='right')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('experiments/convergence_speed_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('experiments/convergence_speed_comparison.pdf', bbox_inches='tight')

print("\n\nFigures saved to:")
print("  - experiments/convergence_patterns.png")
print("  - experiments/convergence_speed_comparison.png")