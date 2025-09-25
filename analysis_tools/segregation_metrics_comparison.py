import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from experiment_list_for_analysis import (
    SCENARIO_LABELS as scenario_labels,
)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load combined results from reports
OUT_DIR = Path('reports')
OUT_DIR.mkdir(parents=True, exist_ok=True)
combined_df = pd.read_csv(OUT_DIR / 'combined_final_metrics.csv')

# Scenario labels imported from shared module

# Define metric labels
metric_labels = {
    'clusters': 'Number of Clusters',
    'switch_rate': 'Switch Rate',
    'distance': 'Average Distance',
    'mix_deviation': 'Mix Deviation',
    'share': 'Segregation Share',
    'ghetto_rate': 'Ghetto Formation Rate'
}

# Create figure with subplots for each metric
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

metrics = ['clusters', 'switch_rate', 'distance', 'mix_deviation', 'share', 'ghetto_rate']

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    # Prepare data for plotting
    plot_data = []
    plot_labels = []
    
    for scenario in scenario_labels.keys():
        scenario_data = combined_df[combined_df['scenario'] == scenario][metric].values
        if len(scenario_data) > 0:
            plot_data.append(scenario_data)
            plot_labels.append(scenario_labels[scenario])
    
    # Create violin plot with box plot overlay
    parts = ax.violinplot(plot_data, positions=range(len(plot_labels)), 
                         showmeans=True, showmedians=True, showextrema=False)
    
    # Customize violin plots
    for pc in parts['bodies']:
        pc.set_alpha(0.6)
    
    # Add box plots on top
    bp = ax.boxplot(plot_data, positions=range(len(plot_labels)), 
                    widths=0.15, patch_artist=True, showfliers=False)
    
    # Color box plots
    colors = sns.color_palette("husl", len(plot_labels))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    # Customize plot
    ax.set_xticks(range(len(plot_labels)))
    ax.set_xticklabels(plot_labels, rotation=45, ha='right')
    ax.set_ylabel(metric_labels[metric])
    ax.set_title(f'{metric_labels[metric]} by Scenario', fontsize=12, fontweight='bold')
    
    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    
    # Add significance indicators
    if metric in ['clusters', 'switch_rate', 'distance', 'mix_deviation', 'share', 'ghetto_rate']:
        # Perform pairwise t-tests against baseline
        baseline_data = combined_df[combined_df['scenario'] == 'baseline'][metric].values
        y_max = ax.get_ylim()[1]
        
        for i, scenario in enumerate(['ethnic_asian_hispanic', 'income_high_low', 
                                     'political_liberal_conservative', 'race_white_black']):
            if scenario in combined_df['scenario'].values:
                scenario_data = combined_df[combined_df['scenario'] == scenario][metric].values
                if len(scenario_data) > 0 and len(baseline_data) > 0:
                    _, p_value = stats.ttest_ind(baseline_data, scenario_data)
                    if p_value < 0.001:
                        ax.text(i+1, y_max * 0.95, '***', ha='center', va='bottom', fontsize=10)
                    elif p_value < 0.01:
                        ax.text(i+1, y_max * 0.95, '**', ha='center', va='bottom', fontsize=10)
                    elif p_value < 0.05:
                        ax.text(i+1, y_max * 0.95, '*', ha='center', va='bottom', fontsize=10)

plt.suptitle('Segregation Metrics Comparison Across Social Context Scenarios', 
             fontsize=16, fontweight='bold', y=0.98)

# Add legend for significance
significance_text = '* p < 0.05, ** p < 0.01, *** p < 0.001 (vs. baseline)'
plt.figtext(0.99, 0.01, significance_text, ha='right', va='bottom', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig(OUT_DIR / 'segregation_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'segregation_metrics_comparison.pdf', bbox_inches='tight')

"""Heatmap summarizing normalized segregation metrics across scenarios present in data."""
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Calculate mean values for heatmap using only scenarios actually present
heatmap_data = []
scenarios_present = []
for scenario in scenario_labels.keys():
    if scenario in combined_df['scenario'].values:
        scenario_df = combined_df[combined_df['scenario'] == scenario]
        if scenario_df.empty:
            continue
        means = []
        for metric in metrics:
            metric_data = scenario_df[metric]
            all_metric_data = combined_df[metric]
            denom = (all_metric_data.max() - all_metric_data.min())
            if denom == 0 or np.isnan(denom):
                normalized_mean = 0.0  # avoid divide-by-zero; all values identical
            else:
                normalized_mean = (metric_data.mean() - all_metric_data.min()) / denom
            means.append(normalized_mean)
        heatmap_data.append(means)
        scenarios_present.append(scenario)

if len(heatmap_data) == 0:
    print("No scenarios present for heatmap; skipping heatmap generation.")
else:
    heatmap_array = np.array(heatmap_data)
    im = ax2.imshow(heatmap_array, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)

    # Set ticks and labels aligned with present scenarios
    ax2.set_xticks(np.arange(len(metrics)))
    ax2.set_yticks(np.arange(len(scenarios_present)))
    ax2.set_xticklabels([metric_labels[m] for m in metrics], rotation=45, ha='right')
    ax2.set_yticklabels([scenario_labels[s] for s in scenarios_present])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Normalized Segregation Level', rotation=270, labelpad=20)

    # Add text annotations safely
    for i in range(len(scenarios_present)):
        for j in range(len(metrics)):
            ax2.text(j, i, f'{heatmap_array[i, j]:.2f}',
                     ha="center", va="center", color="black" if heatmap_array[i, j] < 0.5 else "white")

    ax2.set_title('Normalized Segregation Metrics Heatmap\n(Higher values indicate more segregation)',
                  fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUT_DIR / 'segregation_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'segregation_heatmap.pdf', bbox_inches='tight')

# Print summary statistics
print("\nKEY FINDINGS:")
print("=" * 60)

# Find scenarios with highest and lowest segregation for each metric
for metric in metrics:
    metric_means = combined_df.groupby('scenario')[metric].mean()
    max_scenario = metric_means.idxmax()
    min_scenario = metric_means.idxmin()
    
    print(f"\n{metric_labels[metric]}:")
    # Ensure keys are strings when accessing scenario_labels
    max_key = str(max_scenario)
    min_key = str(min_scenario)
    max_label = scenario_labels.get(max_key, max_key)
    min_label = scenario_labels.get(min_key, min_key)
    print(f"  Highest: {max_label} ({metric_means[max_scenario]:.3f})")
    print(f"  Lowest: {min_label} ({metric_means[min_scenario]:.3f})")
    print(f"  Ratio: {metric_means[max_scenario]/metric_means[min_scenario]:.2f}x")

print("\n\nFigures saved to:")
print(f"  - {OUT_DIR / 'segregation_metrics_comparison.png'}")
print(f"  - {OUT_DIR / 'segregation_heatmap.png'}")