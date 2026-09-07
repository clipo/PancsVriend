import pandas as pd
import run_files
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from pathlib import Path
from scipy.signal import savgol_filter
from experiment_list_for_analysis import (
    SCENARIOS as scenarios,
    SCENARIO_LABELS as scenario_labels,
    SCENARIO_COLORS as scenario_colors,
)
from analysis_tools.output_paths import get_reports_dir
try:                                   # bare import matches the orchestrator's sys.path
    from plot_style import steps_to_fraction_of_final
except ImportError:                    # package form, for direct invocation
    from analysis_tools.plot_style import steps_to_fraction_of_final
try:                                   # bare import matches how the orchestrator
    from plot_style import violin_box_points          # puts analysis_tools/ on sys.path
except ImportError:                    # package form, for direct invocation
    from analysis_tools.plot_style import violin_box_points

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

BASE_DIR = Path(__file__).resolve().parent
# cwd-relative, matching the dissimilarity_index_over_time step: the analysis
# orchestrator runs with cwd at the run root, where experiments/ lives.
# (Was BASE_DIR / "experiments" = analysis_tools/experiments — a directory
# that never exists, so this step silently produced NOTHING in every campaign
# while reporting status=ok. Fixed 2026-08-22.)
def _resolve_exp_dir():
    """experiments/ under the CURRENT working dir (the analysis orchestrator
    runs with cwd at the run root), falling back to the repo root so the
    script also works when invoked from anywhere else."""
    cwd_rel = Path("experiments")
    if cwd_rel.is_dir():
        return cwd_rel
    repo_rel = BASE_DIR.parent / "experiments"
    return repo_rel if repo_rel.is_dir() else cwd_rel


EXP_DIR = _resolve_exp_dir()
# Dedicated subfolder, same convention as normality/ and
# movement_decision_counts/ (see c0142b4).
OUT_DIR = get_reports_dir() / "metric_panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Scenarios, labels, colors imported from shared module

metrics = ['clusters', 'switch_rate', 'distance', 'mix_deviation', 'share', 'ghetto_rate']
metric_labels = {
    'clusters': 'Number of Clusters',
    'switch_rate': 'Switch Rate',
    'distance': 'Average Distance',
    'mix_deviation': 'Mix Deviation',
    'share': 'Segregation Share',
    'ghetto_rate': 'Ghetto Formation Rate',
    'dissimilarity_index': 'Dissimilarity Index',
}


def calculate_rate_of_change(series: np.ndarray, window: int = 7) -> np.ndarray:
    if series is None or len(series) == 0:
        return np.array([])
    if len(series) < window * 2 + 1:
        # Fall back to simple gradient if too short for Savitzky-Golay
        return np.gradient(series)
    smoothed = savgol_filter(series, window_length=window * 2 + 1, polyorder=3)
    return np.gradient(smoothed)


def load_data():
    data = {}
    for scenario_name, folder in scenarios.items():
        filepath = Path(run_files.metrics_history_path(EXP_DIR / folder))
        if filepath.exists():
            df = pd.read_csv(filepath)
            data[scenario_name] = df
    return data


def load_di_data():
    """Per-scenario DI frames (run_id, step, dissimilarity_index), from the
    table the dissimilarity_index_over_time step writes earlier in the suite.
    DI is not in metrics_history.csv, so its panel reads this instead; the
    table's scenario keys (llm_baseline, mech_baseline, ...) already match the
    shared SCENARIOS mapping. Returns {} when the DI step has not run."""
    di_path = get_reports_dir() / 'dissimilarity_index' / 'dissimilarity_by_step_all.csv.gz'
    if not di_path.exists():
        return {}
    df = pd.read_csv(di_path)
    return {str(sc): grp[['run_id', 'step', 'dissimilarity_index']].copy()
            for sc, grp in df.groupby('scenario') if str(sc) in scenarios}


def compute_convergence_steps(df: pd.DataFrame, metric: str) -> list:
    """Steps to 90% of each run's final value (plot_style.steps_to_fraction_of_final)."""
    return steps_to_fraction_of_final(df, metric)


def make_metric_panel(metric: str, data_by_scenario: dict):
    # Prepare figure: 1x3 layout (all three panels in a single row)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    ax_ts, ax_box, ax_conv = axes
    # Strip under panel A carrying the ACTIVE-run count — the "number at risk"
    # row from survival-analysis convention. Forward-filling makes the mean
    # unbiased (always n=100) but it cannot show how much of the curve is new
    # information versus frozen runs persisting, and that distinction is large
    # here: gemma drops under 10 active runs by step 48 yet the axis runs to
    # 999, while deepseek still has 13 active at 999 so its late drift is real.
    ax_alive = make_axes_locatable(ax_ts).append_axes(
        "bottom", size="22%", pad=0.08, sharex=ax_ts)

    # Panel A: Convergence time series with 95% CI
    #
    # Converged runs stop writing rows, so averaging with a bare
    # groupby('step') averages only the SURVIVORS — the atypical runs still in
    # motion — and the trace drifts upward purely from attrition. Measured on
    # mistral/income_high_low (2026-08-28): 100 runs alive at step 4 with mean
    # DI 0.244, but only 2 alive at step 9 with mean 0.337. The apparent rise
    # was entirely survivorship; panel B correctly showed every run's final at
    # 0.25, and the two panels disagreed. A converged run's grid is FROZEN, so
    # carrying its last value forward is the honest continuation rather than an
    # approximation — the same convention vf_simulation_evaluation.load_batch
    # uses. Filling to a panel-wide max keeps the scenario traces comparable.
    present = [s for s in scenarios.keys() if s in data_by_scenario]
    panel_max_step = max((int(data_by_scenario[s]['step'].max()) for s in present),
                         default=0)
    for scenario in present:
        df = data_by_scenario[scenario]
        pv = df.pivot_table(index='run_id', columns='step', values=metric)
        pv = pv.reindex(columns=range(int(df['step'].min()), panel_max_step + 1))
        # Active = still writing rows at this step, i.e. not yet converged.
        # Taken BEFORE the ffill, which is what makes it informative.
        n_active = pv.notna().sum(axis=0)
        ax_alive.step(n_active.index, n_active.values, where='post',
                      color=scenario_colors.get(scenario, None),
                      linewidth=1.4, alpha=0.85)
        pv = pv.ffill(axis=1)
        mean_values = pv.mean(axis=0)
        std_values = pv.std(axis=0)
        count_values = pv.count(axis=0).replace(0, np.nan)
        ci = 1.96 * std_values / np.sqrt(count_values)

        steps = mean_values.index
        ax_ts.plot(steps, mean_values.values, label=scenario_labels[scenario],
                   color=scenario_colors.get(scenario, None), linewidth=2.5, alpha=0.9)
        if ci.notna().any():
            ax_ts.fill_between(steps,
                               (mean_values - ci).values,
                               (mean_values + ci).values,
                               color=scenario_colors.get(scenario, None), alpha=0.15)
    ax_ts.set_title(f"A. {metric_labels[metric]} — Convergence Over Time", fontsize=13, fontweight='bold')
    ax_ts.set_ylabel(metric_labels[metric])
    ax_ts.grid(True, alpha=0.3)
    ax_ts.tick_params(axis='x', labelbottom=False)   # x labels belong to the strip

    # Active-run strip: how many runs are still evolving at each step. Read the
    # trace above together with this — where the strip is near zero the curve is
    # frozen runs persisting, not ongoing dynamics.
    n_runs = max((data_by_scenario[s]['run_id'].nunique() for s in present), default=0)
    ax_alive.set_ylim(0, max(n_runs, 1) * 1.08)
    ax_alive.set_ylabel('active\nruns', fontsize=8)
    ax_alive.set_xlabel('Simulation Step')
    ax_alive.grid(True, alpha=0.25)
    ax_alive.tick_params(axis='y', labelsize=7)
    if n_runs:
        ax_alive.set_yticks([0, n_runs // 2, n_runs])
        # 10% of runs: below this the mean is carried by a handful of runs.
        ax_alive.axhline(n_runs * 0.1, color='#999999', lw=0.8, ls=':')

    # Panel B: Final value distribution (boxplot) across scenarios
    final_rows = []
    for scenario in scenarios.keys():
        if scenario not in data_by_scenario:
            continue
        df = data_by_scenario[scenario]
        # final step per run
        final_per_run = df.loc[df.groupby('run_id')['step'].idxmax(), ['run_id', metric]].copy()
        final_per_run['scenario'] = scenario_labels[scenario]
        final_rows.append(final_per_run)
    if final_rows:
        final_df = pd.concat(final_rows, ignore_index=True)
        # House distribution style (violin + narrow box + every run as a point),
        # shared with segregation_metrics_comparison via plot_style so the two
        # figures showing this same data cannot drift apart. Was a bare seaborn
        # boxplot until 2026-08-28.
        keys = [k for k in scenarios.keys() if k in data_by_scenario]
        plot_labels = [scenario_labels[k] for k in keys]
        plot_colors = [scenario_colors.get(k, '#999999') for k in keys]
        plot_data = [final_df.loc[final_df.scenario == scenario_labels[k], metric]
                     .dropna().to_numpy() for k in keys]
        positions = np.arange(len(keys))
        violin_box_points(ax_box, plot_data, positions, plot_colors)
        ax_box.set_title(f"B. {metric_labels[metric]} — Final Values Across Scenarios", fontsize=13, fontweight='bold')
        ax_box.set_xlabel('Scenario')
        ax_box.set_ylabel(metric_labels[metric])
        # Tilted labels must be RIGHT-aligned and anchor-rotated so the END of
        # the text sits on its tick, rather than the text being centred under
        # the box — centred rotation drifts each label left of its own box and
        # it reads as belonging to the neighbour. rotation_mode='anchor' pivots
        # about the alignment point, which is what makes ha='right' land the
        # text end on the tick. Panel C already did this; panel B was the odd
        # one out. Explicit tick marks give the eye something to trace back to.
        ax_box.set_xticks(positions)
        ax_box.set_xticklabels(plot_labels, rotation=30, ha='right',
                               rotation_mode='anchor')
        ax_box.tick_params(axis='x', which='major', length=5, width=1,
                           direction='out', bottom=True)
        ax_box.grid(True, axis='y', alpha=0.3)
    else:
        ax_box.text(0.5, 0.5, 'No data for boxplot', ha='center')
        ax_box.axis('off')

    # Panel C: Convergence speed (steps to reach 90% of final)
    conv_values = []
    conv_labels = []
    bar_colors = []
    for scenario in scenarios.keys():
        if scenario not in data_by_scenario:
            continue
        df = data_by_scenario[scenario]
        steps_list = compute_convergence_steps(df, metric)
        if len(steps_list) > 0:
            conv_values.append(float(np.mean(steps_list)))
            conv_labels.append(scenario_labels[scenario])
            bar_colors.append(scenario_colors.get(scenario, '#999999'))
    if conv_values:
        x = np.arange(len(conv_values))
        ax_conv.bar(x, conv_values, color=bar_colors)
        ax_conv.set_xticks(x)
        ax_conv.set_xticklabels(conv_labels, rotation=30, ha='right')
        ax_conv.set_ylabel('Steps to 90% Convergence')
        ax_conv.set_title(f"C. {metric_labels[metric]} — Convergence Speed", fontsize=13, fontweight='bold')
        ax_conv.grid(True, axis='y', alpha=0.3)
        # annotate bars
        for i, v in enumerate(conv_values):
            ax_conv.text(i, v + max(conv_values) * 0.02, f"{v:.0f}", ha='center', va='bottom', fontsize=9)
    else:
        ax_conv.text(0.5, 0.5, 'Insufficient data to compute convergence speed', ha='center')
        ax_conv.axis('off')

    plt.suptitle(f"Comprehensive Panel — {metric_labels[metric]}", fontsize=16, fontweight='bold')
    # Figure-level legend on the right, not inside panel A. The scenario colours
    # key ALL of A, B and C, so a legend parked in one panel both under-claims
    # its scope and covers that panel's data. Reserve the right margin via the
    # tight_layout rect so the legend sits beside the axes rather than over them.
    handles, labels = ax_ts.get_legend_handles_labels()
    plt.tight_layout(rect=(0.0, 0.02, 0.90, 0.95))
    if handles:
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.905, 0.5),
                   fontsize=10, frameon=True, framealpha=0.9,
                   title='Scenario', title_fontsize=11)

    # Save
    out_png = OUT_DIR / f"metric_panel_{metric}.png"
    fig.savefig(str(out_png), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {out_png}")


def main():
    data = load_data()
    if not data:
        print("No experiment data found. Ensure metrics_history.csv exist under experiments/<scenario>/.")
        return
    for metric in metrics:
        make_metric_panel(metric, data)
    di_data = load_di_data()
    if di_data:
        make_metric_panel('dissimilarity_index', di_data)
    else:
        print("dissimilarity_index panel skipped: dissimilarity_by_step_all.csv.gz "
              "not found (run the dissimilarity_index_over_time step first)")


if __name__ == "__main__":
    main()
