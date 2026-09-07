import math
import pandas as pd
import run_files
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from experiment_list_for_analysis import (
    SCENARIO_ORDER,
    SCENARIOS as scenarios,
    SCENARIO_LABELS as scenario_labels,
    SCENARIO_COLORS as scenario_colors,
)
from analysis_tools.output_paths import get_reports_dir
from mpl_toolkits.axes_grid1 import make_axes_locatable
try:                                   # bare import matches the orchestrator's sys.path
    from plot_style import step_stats_forward_filled, steps_to_fraction_of_final
except ImportError:                    # package form, for direct invocation
    from analysis_tools.plot_style import step_stats_forward_filled, steps_to_fraction_of_final

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


def _draw_active_strip(host_ax, active_by_scenario, colors, labels_map, max_step=None):
    """Hang an active-run strip under host_ax (shared x).

    Forward-filling makes the mean unbiased, but it cannot show how much of the
    curve is new information versus frozen runs persisting. Measured spread is
    wide: gemma falls under 10 active runs by step 48 while its axis runs to
    999, whereas deepseek still has ~70 active at 999 so its late drift is real.
    """
    if not active_by_scenario:
        return None
    strip = make_axes_locatable(host_ax).append_axes(
        "bottom", size="22%", pad=0.08, sharex=host_ax)
    n_runs = 0
    for name, series in active_by_scenario.items():
        s2 = series if max_step is None else series[series.index <= max_step]
        strip.step(s2.index, s2.values, where='post', linewidth=1.3, alpha=0.85,
                   color=colors.get(name, '#999999'))
        n_runs = max(n_runs, int(series.max()) if len(series) else 0)
    strip.set_ylim(0, max(n_runs, 1) * 1.08)
    strip.set_ylabel('active\nruns', fontsize=8)
    strip.grid(True, alpha=0.25)
    strip.tick_params(axis='y', labelsize=7)
    if n_runs:
        strip.set_yticks([0, n_runs // 2, n_runs])
        strip.axhline(n_runs * 0.1, color='#999999', lw=0.8, ls=':')
    host_ax.tick_params(axis='x', labelbottom=False)
    return strip


n_cols = 3
n_rows = math.ceil(len(metrics) / n_cols)
# Active-run count is metric-INDEPENDENT (it depends only on how long each run
# lived), so the grid needs ONE strip for the whole figure. It gets its own
# full-width gridspec row rather than being hung off a subplot with
# make_axes_locatable, which would steal that subplot's space and overlap its
# axis labels.
fig = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows + 1.3))
_gs = fig.add_gridspec(n_rows + 1, n_cols,
                       height_ratios=[*([4.5] * n_rows), 1.0], hspace=0.55)
axes = np.array([fig.add_subplot(_gs[r, c])
                 for r in range(n_rows) for c in range(n_cols)])
ax_grid_active = fig.add_subplot(_gs[n_rows, :])
active_by_scenario = {}

# For each metric, plot convergence patterns
for idx, metric in enumerate(metrics):
    ax = axes[idx]

    for scenario_name, folder in scenarios.items():
        if metric == 'dissimilarity_index':
            df = dissim_by_scenario.get(scenario_name)
        else:
            filepath = Path(run_files.metrics_history_path(f'experiments/{folder}'))
            df = None
            if filepath.exists():
                if scenario_name in metrics_history_cache:
                    df = metrics_history_cache[scenario_name]
                else:
                    df = pd.read_csv(filepath)
                    metrics_history_cache[scenario_name] = df

        if df is None or df.empty:
            continue

        # Forward-filled: a bare groupby('step') averages only the runs still
        # alive at that step, so the trace drifts upward from attrition rather
        # than from dynamics (see plot_style.step_stats_forward_filled).
        mean_values, ci, n_active_series = step_stats_forward_filled(df, metric)
        active_by_scenario[scenario_name] = n_active_series

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

for _sc, _series in active_by_scenario.items():
    ax_grid_active.step(_series.index, _series.values, where='post',
                        linewidth=1.3, alpha=0.85,
                        color=scenario_colors.get(_sc, '#999999'))
_nr = max((int(v.max()) for v in active_by_scenario.values() if len(v)), default=0)
ax_grid_active.set_ylim(0, max(_nr, 1) * 1.08)
ax_grid_active.set_ylabel('active runs', fontsize=9)
# No x-label: all six subplots above already carry 'Simulation Step', and a
# seventh would collide with the figure legend beneath.
ax_grid_active.set_title('Runs still evolving (shared by all metrics above)',
                         fontsize=10, pad=4)
ax_grid_active.grid(True, alpha=0.25)
if _nr:
    ax_grid_active.set_yticks([0, _nr // 2, _nr])
    ax_grid_active.axhline(_nr * 0.1, color='#999999', lw=0.8, ls=':')
plt.suptitle('Convergence Patterns of Segregation Metrics Across Scenarios', y=0.98)
sns.despine(fig=fig)
plt.tight_layout(rect=(0.0, 0.08, 1.0, 0.94), h_pad=2.0)
OUT_DIR = get_reports_dir()
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_DIR / 'convergence_patterns.png', dpi=300, bbox_inches='tight')
plt.close(fig)

if include_dissimilarity:
    # Dedicated dissimilarity index convergence figure for detailed inserts
    fig_di, ax_di = plt.subplots(figsize=(10, 4.5))
    ax_di_active = make_axes_locatable(ax_di).append_axes(
        "bottom", size="22%", pad=0.08, sharex=ax_di)
    di_active_by_scenario = {}
    for scenario_name in SCENARIO_ORDER:
        df = dissim_by_scenario.get(scenario_name)
        if df is None or df.empty:
            continue

        mean_values, ci, n_active_series = step_stats_forward_filled(
            df, 'dissimilarity_index')
        di_active_by_scenario[scenario_name] = n_active_series

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
    for _sc, _series in di_active_by_scenario.items():
        ax_di_active.step(_series.index, _series.values, where='post',
                          linewidth=1.3, alpha=0.85,
                          color=scenario_colors.get(_sc, '#999999'))
    _nr = max((int(v.max()) for v in di_active_by_scenario.values() if len(v)), default=0)
    ax_di_active.set_ylim(0, max(_nr, 1) * 1.08)
    ax_di_active.set_ylabel('active\nruns', fontsize=8)
    ax_di_active.set_xlabel('Simulation Step')
    ax_di_active.grid(True, alpha=0.25)
    ax_di_active.tick_params(axis='y', labelsize=7)
    if _nr:
        ax_di_active.set_yticks([0, _nr // 2, _nr])
        ax_di_active.axhline(_nr * 0.1, color='#999999', lw=0.8, ls=':')
    ax_di.tick_params(axis='x', labelbottom=False)
    ax_di.set_title('Dissimilarity Index Convergence', pad=10)
    ax_di.grid(False)
    sns.despine(ax=ax_di)
    handles_di, labels_di = ax_di.get_legend_handles_labels()
    if handles_di:
        ax_di.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                     frameon=False, ncol=1, borderaxespad=0.0)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'convergence_patterns_dissimilarity_index.png', dpi=300, bbox_inches='tight')
    plt.close(fig_di)

# Calculate convergence speed (steps to reach 90% of final value)
print("\nCONVERGENCE ANALYSIS:")
print("=" * 60)

convergence_data = {}
convergence_step_records = {}

for scenario_name, folder in scenarios.items():
    if scenario_name in metrics_history_cache:
        df_base = metrics_history_cache[scenario_name]
    else:
        path = Path(run_files.metrics_history_path(f'experiments/{folder}'))
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
        # Steps to 90% of each run's final value; see plot_style for why
        # this is not a per-run filter loop.
        convergence_steps = steps_to_fraction_of_final(df, metric)

        if convergence_steps:
            convergence_step_records.setdefault(scenario_name, {})[metric] = convergence_steps
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
plt.close(fig2)

# Generate per-step 90% convergence progress plots (CDF-style)
progress_fig_generated = False
progress_fig_path = OUT_DIR / 'convergence_progress_90pct.png'

cdf_fig, cdf_axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
cdf_axes = cdf_axes.flatten()
metric_has_data = []

for idx, metric in enumerate(metrics):
    ax = cdf_axes[idx]
    plotted = False
    for scenario_name in SCENARIO_ORDER:
        steps_list = convergence_step_records.get(scenario_name, {}).get(metric)
        if not steps_list:
            continue

        plotted = True
        sorted_steps = np.sort(steps_list)
        cumulative = np.arange(1, len(sorted_steps) + 1) / len(sorted_steps)
        ax.plot(
            sorted_steps,
            cumulative,
            label=scenario_labels[scenario_name],
            linewidth=2.2,
            alpha=0.95,
            color=scenario_colors[scenario_name],
        )

    if plotted:
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Runs >= 90%')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(0, 1.01)
        ax.set_title(f'Time to 90%: {metric_labels[metric]}', pad=12)
        ax.grid(True, axis='y', alpha=0.25)
    else:
        ax.axis('off')
    metric_has_data.append(plotted)

for extra_ax in cdf_axes[len(metrics):]:
    extra_ax.axis('off')

if any(metric_has_data):
    handles_cdf, labels_cdf = [], []
    for ax in cdf_axes:
        h_ax, l_ax = ax.get_legend_handles_labels()
        handles_cdf += h_ax
        labels_cdf += l_ax

    seen = set()
    unique_cdf = [(h, lbl) for h, lbl in zip(handles_cdf, labels_cdf) if not (lbl in seen or seen.add(lbl))]
    if unique_cdf:
        h_unique, l_unique = zip(*unique_cdf)
        cdf_fig.legend(
            h_unique,
            l_unique,
            loc='lower center',
            ncol=min(4, len(l_unique)),
            frameon=False,
            bbox_to_anchor=(0.5, -0.04),
            labelspacing=0.8,
            borderaxespad=1.0,
            columnspacing=1.6,
            handlelength=2.0,
        )

    plt.suptitle('Share of Runs Reaching 90% Convergence by Step', y=0.98)
    plt.tight_layout(rect=(0.0, 0.08, 1.0, 0.94), h_pad=2.0)
    plt.savefig(progress_fig_path, dpi=300, bbox_inches='tight')
    progress_fig_generated = True
    plt.close(cdf_fig)
else:
    plt.close(cdf_fig)

# Dedicated convergence progress figure for dissimilarity index if present
dissim_progress_generated = False
dissim_progress_path = OUT_DIR / 'convergence_progress_90pct_dissimilarity_index.png'
if include_dissimilarity:
    fig_cdf_di, ax_cdf_di = plt.subplots(figsize=(10, 4.5))
    plotted_di = False
    for scenario_name in SCENARIO_ORDER:
        steps_list = convergence_step_records.get(scenario_name, {}).get('dissimilarity_index')
        if not steps_list:
            continue
        plotted_di = True
        sorted_steps = np.sort(steps_list)
        cumulative = np.arange(1, len(sorted_steps) + 1) / len(sorted_steps)
        ax_cdf_di.plot(
            sorted_steps,
            cumulative,
            label=scenario_labels[scenario_name],
            linewidth=2.4,
            alpha=0.95,
            color=scenario_colors[scenario_name],
        )

    if plotted_di:
        ax_cdf_di.set_xlabel('Simulation Step')
        ax_cdf_di.set_ylabel('Runs >= 90%')
        ax_cdf_di.yaxis.set_major_formatter(PercentFormatter(1))
        ax_cdf_di.set_ylim(0, 1.01)
        ax_cdf_di.set_title('Dissimilarity Index 90% Convergence Progress', pad=10)
        ax_cdf_di.grid(True, axis='y', alpha=0.25)
        sns.despine(ax=ax_cdf_di)
        handles_di, labels_di = ax_cdf_di.get_legend_handles_labels()
        if handles_di:
            ax_cdf_di.legend(
                handles_di,
                labels_di,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                ncol=1,
                borderaxespad=0.0,
            )
        plt.tight_layout()
        plt.savefig(dissim_progress_path, dpi=300, bbox_inches='tight')
        dissim_progress_generated = True
        plt.close(fig_cdf_di)
    else:
        plt.close(fig_cdf_di)

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
        plt.savefig(OUT_DIR / 'convergence_speed_dissimilarity_index.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig3)
    dissim_speed_generated = True

print("\n\nFigures saved to:")
print(f"  - {OUT_DIR / 'convergence_patterns.png'}")
print(f"  - {OUT_DIR / 'convergence_speed_comparison.png'}")
if include_dissimilarity:
    print(f"  - {OUT_DIR / 'convergence_patterns_dissimilarity_index.png'}")
    if dissim_speed_generated:
        print(f"  - {OUT_DIR / 'convergence_speed_dissimilarity_index.png'}")
if progress_fig_generated:
    print(f"  - {progress_fig_path}")
if dissim_progress_generated:
    print(f"  - {dissim_progress_path}")