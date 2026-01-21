import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from experiment_list_for_analysis import (
    SCENARIO_ORDER,
    SCENARIOS as scenarios,
    SCENARIO_LABELS as scenario_labels,
    SCENARIO_COLORS as scenario_colors,
)
from analysis_tools.output_paths import get_reports_dir

# Publication-ready seaborn theme
sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    # Embed fonts in vector outputs and avoid Type 3 fonts
    "ps.fonttype": 42,
    # Ticks and lines
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "normal",
    # Font sizes for small-print readability
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
    "figure.titlesize": 18,
    # Spacing to reduce clutter
    "axes.labelpad": 8,
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
    "legend.handletextpad": 0.8,
    "legend.columnspacing": 1.6,
    "legend.borderaxespad": 1.0,
    "legend.borderpad": 0.6,
    "legend.labelspacing": 0.8,
    "legend.handlelength": 2.0,
})

# Scenarios, labels, and colors are imported from shared module

# Create figure for time series
metric_labels = {
    'clusters': 'Number of Clusters',
    'switch_rate': 'Switch Rate',
    'distance': 'Average Distance',
    'mix_deviation': 'Mix Deviation',
    'share': 'Segregation Share',
    'ghetto_rate': 'Ghetto Formation Rate',
    'dissimilarity_index': 'Dissimilarity Index',
}

# Load cached dissimilarity outputs if available
dissim_path = get_reports_dir() / 'dissimilarity_index' / 'dissimilarity_by_step_all.csv.gz'
dissim_ts = pd.read_csv(dissim_path) if dissim_path.exists() else None
include_dissimilarity = dissim_ts is not None

metrics = ['clusters', 'switch_rate', 'distance', 'mix_deviation', 'share', 'ghetto_rate']
# if dissim_ts is not None:
#     metrics.append('dissimilarity_index')
speed_metrics = metrics + (['dissimilarity_index'] if include_dissimilarity else [])

# Cache metrics_history per scenario to avoid repeated reads
metrics_history_cache = {}
if dissim_ts is not None:
    dissim_by_scenario = {s: dissim_ts[dissim_ts['scenario'] == s].copy() for s in dissim_ts['scenario'].unique()}
else:
    dissim_by_scenario = {}

n_cols = 3
n_rows = math.ceil(len(metrics) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
axes = axes.flatten()

# For each metric, plot convergence patterns
for idx, metric in enumerate(metrics):
    ax = axes[idx]

    for scenario_name, folder in scenarios.items():
        if metric == 'dissimilarity_index':
            df = dissim_by_scenario.get(scenario_name)
        else:
            filepath = Path(f'experiments/{folder}/metrics_history.csv')
            df = None
            if filepath.exists():
                if scenario_name in metrics_history_cache:
                    df = metrics_history_cache[scenario_name]
                else:
                    df = pd.read_csv(filepath)
                    metrics_history_cache[scenario_name] = df

        if df is None or df.empty:
            continue

        # Calculate mean and confidence interval at each step
        grouped = df.groupby('step')[metric]
        mean_values = grouped.mean()
        std_values = grouped.std()
        count_values = grouped.count().replace(0, np.nan)

        # Calculate 95% confidence interval
        ci = 1.96 * std_values / np.sqrt(count_values)

        # Limit steps for visual clarity
        max_step = min(1000, mean_values.index.max())
        steps = mean_values.index[mean_values.index <= max_step]

        color = scenario_colors[scenario_name]

        # Plot mean line
        ax.plot(steps, mean_values[steps],
                label=scenario_labels[scenario_name],
                linewidth=2.2, alpha=0.95, color=color)

        # Add confidence interval
        ax.fill_between(steps,
                        mean_values[steps] - ci[steps],
                        mean_values[steps] + ci[steps],
                        alpha=0.15, color=color)

    ax.set_xlabel('Simulation Step', )
    ax.set_ylabel(metric_labels[metric], )
    ax.set_title(f'{metric_labels[metric]} Over Time', pad=12)
    # ax.grid(False, axis='y', alpha=0.25)
    ax.grid(False)

# Hide any unused subplot slots
for extra_ax in axes[len(metrics):]:
    extra_ax.axis('off')


handles, labels = [], []
for ax in axes:
    h_i, lbls_i = ax.get_legend_handles_labels()
    handles += h_i
    labels += lbls_i

# Deduplicate while preserving order
seen = set()
unique = [(h_i, lbl) for h_i, lbl in zip(handles, labels) if not (lbl in seen or seen.add(lbl))]
uhandles, ulabels = zip(*unique) if unique else ([], [])

if uhandles:
    fig.legend(uhandles, ulabels, loc='lower center', ncol=min(4, len(ulabels)),
               frameon=False, bbox_to_anchor=(0.5, -0.04),
               labelspacing=0.8, borderaxespad=1.0, columnspacing=1.6, handlelength=2.0)

plt.suptitle('Convergence Patterns of Segregation Metrics Across Scenarios', y=0.98)
sns.despine(fig=fig)
plt.tight_layout(rect=(0.0, 0.08, 1.0, 0.94), h_pad=2.0)
OUT_DIR = get_reports_dir()
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_DIR / 'convergence_patterns.png', dpi=300, bbox_inches='tight')

if include_dissimilarity:
    # Dedicated dissimilarity index convergence figure for detailed inserts
    fig_di, ax_di = plt.subplots(figsize=(10, 4.5))
    for scenario_name in SCENARIO_ORDER:
        df = dissim_by_scenario.get(scenario_name)
        if df is None or df.empty:
            continue

        grouped = df.groupby('step')['dissimilarity_index']
        mean_values = grouped.mean()
        std_values = grouped.std()
        count_values = grouped.count().replace(0, np.nan)
        ci = 1.96 * std_values / np.sqrt(count_values)

        max_step = min(1000, mean_values.index.max())
        steps = mean_values.index[mean_values.index <= max_step]
        color = scenario_colors[scenario_name]

        ax_di.plot(steps, mean_values[steps],
                    label=scenario_labels[scenario_name],
                    linewidth=2.4, alpha=0.95, color=color)
        ax_di.fill_between(steps,
                            mean_values[steps] - ci[steps],
                            mean_values[steps] + ci[steps],
                            alpha=0.18, color=color)

    ax_di.set_xlabel('Simulation Step')
    ax_di.set_ylabel(metric_labels['dissimilarity_index'])
    ax_di.set_title('Dissimilarity Index Convergence', pad=10)
    ax_di.grid(False)
    sns.despine(ax=ax_di)
    handles_di, labels_di = ax_di.get_legend_handles_labels()
    if handles_di:
        ax_di.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                     frameon=False, ncol=1, borderaxespad=0.0)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'dissimilarity_index_convergence.png', dpi=300, bbox_inches='tight')

# Calculate convergence speed (steps to reach 90% of final value)
print("\nCONVERGENCE ANALYSIS:")
print("=" * 60)

convergence_data = {}

for scenario_name, folder in scenarios.items():
    if scenario_name in metrics_history_cache:
        df_base = metrics_history_cache[scenario_name]
    else:
        path = Path(f'experiments/{folder}/metrics_history.csv')
        df_base = pd.read_csv(path) if path.exists() else None
        if df_base is not None:
            metrics_history_cache[scenario_name] = df_base

    df_dissim = dissim_by_scenario.get(scenario_name)
    if df_base is None and df_dissim is None:
        continue

    convergence_data[scenario_name] = {}
    print(f"\n{scenario_labels[scenario_name]}:")

    for metric in speed_metrics:
        df = df_dissim if metric == 'dissimilarity_index' else df_base
        if df is None or df.empty:
            continue
        final_values = df.groupby('run_id')[metric].last()
        convergence_steps = []

        for run_id in df['run_id'].unique():
            run_data = df[df['run_id'] == run_id].sort_values('step')
            final_val = run_data[metric].iloc[-1]
            initial_val = run_data[metric].iloc[0]

            if final_val == initial_val:
                continue

            target_val = initial_val + 0.9 * (final_val - initial_val)
            conv_data = run_data[run_data[metric] >= target_val] if final_val > initial_val else run_data[run_data[metric] <= target_val]

            if not conv_data.empty:
                convergence_steps.append(conv_data['step'].iloc[0])

        if convergence_steps:
            mean_conv = np.mean(convergence_steps)
            std_conv = np.std(convergence_steps)
            convergence_data[scenario_name][metric] = mean_conv
            print(f"  {metric}: {mean_conv:.1f} ± {std_conv:.1f} steps")

# Create convergence speed comparison chart
fig2, ax2 = plt.subplots(figsize=(12, 8))

# Prepare data for grouped bar chart in canonical scenario order
x = np.arange(len(metrics))
scenario_order = [s for s in SCENARIO_ORDER if s in convergence_data]
n_scenarios = len(scenario_order)
width = min(0.8 / max(1, n_scenarios), 0.22)  # keep groups within 80% of tick width
multiplier = 0

for scenario_name in scenario_order:
    values = [convergence_data[scenario_name].get(metric, 0) for metric in metrics]
    # Center the bars around each tick
    offset = (multiplier - (n_scenarios - 1) / 2) * width
    ax2.bar(
        x + offset,
        values,
        width,
        label=scenario_labels[scenario_name],
        color=scenario_colors[scenario_name],
        edgecolor='white',
        linewidth=0.6,
        alpha=0.95,
    )
    multiplier += 1

ax2.set_xlabel('Segregation Metric', fontsize=13)
ax2.set_ylabel('Steps to 90% Convergence', fontsize=13)
ax2.set_title('Convergence Speed Comparison Across Scenarios')
ax2.set_xticks(x)
ax2.set_xticklabels([metric_labels[m] for m in metrics], rotation=25, ha='right')
ax2.grid(True, alpha=0.25, axis='y')
sns.despine(ax=ax2)

# Legend at the bottom of the bar chart figure
handles2, labels2 = ax2.get_legend_handles_labels()
if handles2:
    fig2.legend(handles2, labels2, loc='lower center', ncol=min(4, len(labels2)),
                frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=12,
                labelspacing=0.8, borderaxespad=1.0, columnspacing=1.6, handlelength=2.0)

plt.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
plt.savefig(OUT_DIR / 'convergence_speed_comparison.png', dpi=300, bbox_inches='tight')

# Dedicated convergence speed figure for dissimilarity index
dissim_speed_generated = False
if include_dissimilarity:
    dissim_scenarios = [s for s in SCENARIO_ORDER
                        if 'dissimilarity_index' in convergence_data.get(s, {})]
    if dissim_scenarios:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        x_d = np.arange(len(dissim_scenarios))
        values_d = [convergence_data[s]['dissimilarity_index'] for s in dissim_scenarios]
        colors_d = [scenario_colors[s] for s in dissim_scenarios]

        ax3.bar(x_d, values_d, color=colors_d, edgecolor='white', linewidth=0.8, alpha=0.95)
        ax3.set_xticks(x_d)
        ax3.set_xticklabels([scenario_labels[s] for s in dissim_scenarios], rotation=20, ha='right')
        ax3.set_ylabel('Steps to 90% Convergence')
        ax3.set_title('Dissimilarity Index Convergence Speed')
        ax3.grid(True, axis='y', alpha=0.25)
        sns.despine(ax=ax3)

        plt.tight_layout()
        plt.savefig(OUT_DIR / 'dissimilarity_index_convergence_speed.png',
                    dpi=300, bbox_inches='tight')
    dissim_speed_generated = True

print("\n\nFigures saved to:")
print(f"  - {OUT_DIR / 'convergence_patterns.png'}")
print(f"  - {OUT_DIR / 'convergence_speed_comparison.png'}")
if include_dissimilarity:
    print(f"  - {OUT_DIR / 'dissimilarity_index_convergence.png'}")
    if dissim_speed_generated:
        print(f"  - {OUT_DIR / 'dissimilarity_index_convergence_speed.png'}")