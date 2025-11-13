import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from experiment_list_for_analysis import (
    SCENARIO_ORDER,
    SCENARIO_LABELS,
    SCENARIO_COLORS,
)
from analysis_tools.output_paths import get_reports_dir

# Publication-ready style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "normal",
    "axes.labelsize": 14,
    "axes.titlesize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "axes.labelpad": 6,
    "xtick.major.pad": 4,
    "ytick.major.pad": 4,
})

# Load combined results from reports
OUT_DIR = get_reports_dir()
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
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

metrics = ['clusters', 'switch_rate', 'distance', 'mix_deviation', 'share', 'ghetto_rate']

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    # Prepare data and order
    plot_data = []
    plot_labels = []
    plot_keys = []

    # Follow canonical scenario order from experiment list
    for scenario in SCENARIO_ORDER:
        vals = combined_df[combined_df['scenario'] == scenario][metric].values
        if len(vals) > 0:
            plot_data.append(vals)
            plot_labels.append(SCENARIO_LABELS[scenario])
            plot_keys.append(scenario)

    positions = np.arange(len(plot_labels))

    # Violin plot
    parts = ax.violinplot(plot_data, positions=positions,
                          showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        col = SCENARIO_COLORS.get(plot_keys[i], '#999999')
        pc.set_facecolor(col)
        pc.set_edgecolor(col)
        pc.set_alpha(0.35)
        pc.set_linewidth(1.0)

    # Box plot overlay
    bp = ax.boxplot(plot_data, positions=positions,
                    widths=0.18, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp['boxes']):
        col = SCENARIO_COLORS.get(plot_keys[i], '#999999')
        patch.set_facecolor(col)
        patch.set_edgecolor(col)
        patch.set_alpha(0.65)
        patch.set_linewidth(1.0)
    for i, med in enumerate(bp['medians']):
        med.set_color('black')
        med.set_linewidth(1.2)
    for wl in bp['whiskers']:
        wl.set_color('#777777')
        wl.set_linewidth(1.0)
    for cap in bp['caps']:
        cap.set_color('#777777')
        cap.set_linewidth(1.0)

    # Axes formatting
    ax.set_xticks(positions)
    if idx < 3:
        # Top row: hide x tick labels to avoid repetition
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', length=0, labelbottom=False)
    else:
        # Bottom row: show scenario labels once per column
        ax.set_xticklabels(plot_labels, rotation=30, ha='right')
    ax.set_ylabel(metric_labels[metric])
    ax.set_title(f'{metric_labels[metric]} by Scenario', pad=8)
    ax.grid(True, axis='y', alpha=0.25)
    sns.despine(ax=ax)

    # Significance markers removed for cleaner, publication-focused visuals

plt.suptitle('Segregation Metrics Comparison Across Social Context Scenarios', y=0.98)

# Add legend for significance
# Significance footnote removed

plt.tight_layout(rect=(0.0, 0.04, 1.0, 0.94), h_pad=1.5)
plt.savefig(OUT_DIR / 'segregation_metrics_comparison.png', dpi=300, bbox_inches='tight')

"""Heatmap summarizing normalized segregation metrics across scenarios present in data."""
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Calculate mean values for heatmap using only scenarios actually present
heatmap_data = []
scenarios_present = []
# Follow canonical scenario order from experiment list
for scenario in SCENARIO_ORDER:
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
    ax2.set_yticklabels([SCENARIO_LABELS[s] for s in scenarios_present])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Normalized Segregation Level', rotation=270, labelpad=18)

    # Add text annotations safely
    for i in range(len(scenarios_present)):
        for j in range(len(metrics)):
            ax2.text(j, i, f'{heatmap_array[i, j]:.2f}',
                     ha="center", va="center", color="black" if heatmap_array[i, j] < 0.5 else "white")

    ax2.set_title('Normalized Segregation Metrics Heatmap\n(Higher values indicate more segregation)',
                  pad=14)
    sns.despine(ax=ax2, left=False, bottom=False)

plt.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))
plt.savefig(OUT_DIR / 'segregation_heatmap.png', dpi=300, bbox_inches='tight')

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
    max_label = SCENARIO_LABELS.get(max_key, max_key)
    min_label = SCENARIO_LABELS.get(min_key, min_key)
    print(f"  Highest: {max_label} ({metric_means[max_scenario]:.3f})")
    print(f"  Lowest: {min_label} ({metric_means[min_scenario]:.3f})")
    print(f"  Ratio: {metric_means[max_scenario]/metric_means[min_scenario]:.2f}x")

print("\n\nFigures saved to:")
print(f"  - {OUT_DIR / 'segregation_metrics_comparison.png'}")
print(f"  - {OUT_DIR / 'segregation_heatmap.png'}")