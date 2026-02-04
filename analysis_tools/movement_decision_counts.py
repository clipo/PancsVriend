"""Aggregate and plot agent move counts per scenario.

How to use
==========
1. Ensure each experiment listed in ``SCENARIOS`` has move logs under
    ``experiments/<folder>/move_logs`` (JSON or CSV, gzip optional) plus a
    corresponding ``states`` directory. The script reads the existing logs; no
    recomputation is performed.
2. Activate the project virtual environment if you have one:
         source .venv/bin/activate
3. Generate the plot directly (add ``--no-recompute`` to reuse the cached
    summary CSV when it already exists):
         python analysis_tools/movement_decision_counts.py
    The script writes both the PNG plot and a compressed CSV summary into
    ``reports/movement_decision_counts/`` (or the path defined by
    ``PANCSVRIEND_REPORTS_DIR``).
4. Alternatively, include the step in the full scenario pipeline via:
         python analysis_tools/run_all_scenario_analysis.py --output-folder reports
    The runner imports this module, so no extra flags are required.
    Passing ``--no-recompute`` to the runner carries through, allowing this
    module to reuse the cached summary CSV when it already exists.

Outputs
-------
* ``movement_decision_counts.png`` – mean move percentages per step with 95% CIs.
* ``movement_decision_counts_summary.csv.gz`` – per-step statistics (mean %, std %,
  run counts) for both move and stay decisions.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from numbers import Integral

from experiment_list_for_analysis import (
    SCENARIOS,
    SCENARIO_ORDER,
    SCENARIO_LABELS,
    SCENARIO_COLORS,
)
from analysis_tools.output_paths import get_reports_dir
from analysis_tools.analyze_agent_movement import (
    iter_move_logs_json,
    iter_move_logs_csv,
    load_move_log_json,
    load_move_log_csv,
)

# Reuse the convergence plotting aesthetics for visual consistency
sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "normal",
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
    "figure.titlesize": 18,
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

MOVEMENT_TYPES = ("move", "stay")
PLOTTED_MOVEMENTS = ("move",)
MOVEMENT_TITLES = {
    "move": "Agents Moving Per Step (%)",
    "stay": "Agents Staying Per Step (%)",
}
MAX_STEPS = 50
OUTPUT_SUBDIR = "movement_decision_counts"
SUMMARY_FILENAME = "movement_decision_counts_summary.csv.gz"
PLOT_FILENAME = "movement_decision_counts.png"


def _ordered_scenarios() -> List[str]:
    seen = set()
    ordered: List[str] = []
    for scenario in SCENARIO_ORDER:
        if scenario in SCENARIOS and scenario not in seen:
            ordered.append(scenario)
            seen.add(scenario)
    for scenario in SCENARIOS:
        if scenario not in seen:
            ordered.append(scenario)
            seen.add(scenario)
    return ordered


def _normalize_moved(value) -> Optional[bool]:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral):
        return value != 0
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return abs(value) > 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "move", "moved", "yes", "y", "t"}:
            return True
        if text in {"false", "0", "stay", "stayed", "no", "n", "f"}:
            return False
    return None


def _load_run_decisions(experiment_dir: Path, run_id: int) -> Optional[pd.DataFrame]:
    log_json = load_move_log_json(experiment_dir, run_id)
    if log_json is not None:
        df = pd.DataFrame(log_json)
    else:
        df = load_move_log_csv(experiment_dir, run_id)
    if df is None or df.empty:
        return None

    df = df.iloc[1:].copy()  # Skip the initial dummy entry
    if df.empty or "step" not in df.columns or "moved" not in df.columns:
        return None

    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step"])
    if df.empty:
        return None

    df["moved"] = df["moved"].apply(_normalize_moved)
    df = df.dropna(subset=["moved"])
    if df.empty:
        return None

    df["step"] = df["step"].astype(int)
    df["movement"] = np.where(df["moved"], "move", "stay")
    df["run_id"] = run_id
    return df[["run_id", "step", "movement"]]


def _collect_decisions_for_scenario(folder_name: str) -> Optional[pd.DataFrame]:
    experiment_dir = Path("experiments") / folder_name
    if not experiment_dir.exists():
        print(f"movement_decision_counts: experiment folder '{folder_name}' missing; skipping")
        return None

    print(f"movement_decision_counts: scanning '{folder_name}' for move logs")
    run_ids = sorted(set(iter_move_logs_json(experiment_dir)) | set(iter_move_logs_csv(experiment_dir)))
    if not run_ids:
        print(f"movement_decision_counts: no move logs found under '{folder_name}'")
        return None
    print(f"movement_decision_counts: found {len(run_ids)} run logs under '{folder_name}'")

    frames: List[pd.DataFrame] = []
    for run_id in run_ids:
        df_run = _load_run_decisions(experiment_dir, run_id)
        if df_run is not None and not df_run.empty:
            frames.append(df_run)
            print(f"movement_decision_counts:   run {run_id} contributed {len(df_run)} decision rows")
        else:
            print(f"movement_decision_counts:   run {run_id} missing usable move log; skipped")
    if not frames:
        print(f"movement_decision_counts: no usable move data remained for '{folder_name}'")
        return None
    combined = pd.concat(frames, ignore_index=True)
    print(
        "movement_decision_counts:   compiled "
        f"{combined['run_id'].nunique()} runs | {len(combined)} decisions from '{folder_name}'"
    )
    return combined


def _aggregate_statistics(decisions: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    per_run = (
        decisions.groupby(["run_id", "step", "movement"]).size()
        .unstack("movement", fill_value=0)
        .reindex(columns=MOVEMENT_TYPES, fill_value=0)
        .reset_index()
    )

    per_run["total"] = per_run[list(MOVEMENT_TYPES)].sum(axis=1)
    for movement in MOVEMENT_TYPES:
        per_run[f"{movement}_pct"] = np.where(
            per_run["total"] > 0,
            per_run[movement] / per_run["total"] * 100,
            np.nan,
        )

    stats: Dict[str, pd.DataFrame] = {}
    grouped = per_run.groupby("step", sort=True)
    for movement in MOVEMENT_TYPES:
        stats[movement] = grouped[f"{movement}_pct"].agg(["mean", "std", "count"]).reset_index()
    return stats


def _plot(stats_by_scenario: Dict[str, Dict[str, pd.DataFrame]], out_path: Path, max_steps: int) -> bool:
    n_cols = len(PLOTTED_MOVEMENTS)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 7))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    axis_used = [False] * n_cols
    plotted_any = False
    ordered_scenarios = _ordered_scenarios()

    for idx, movement in enumerate(PLOTTED_MOVEMENTS):
        ax = axes[idx]
        max_upper = None
        for scenario_name in ordered_scenarios:
            scenario_stats = stats_by_scenario.get(scenario_name)
            if not scenario_stats:
                continue
            df_stats = scenario_stats.get(movement)
            if df_stats is None or df_stats.empty:
                continue
            df_plot = df_stats[df_stats["step"] <= max_steps].copy()
            if df_plot.empty:
                continue

            counts = df_plot["count"].replace(0, np.nan)
            ci = 1.96 * df_plot["std"].fillna(0) / np.sqrt(counts)
            ci = ci.fillna(0)

            label = SCENARIO_LABELS.get(scenario_name, scenario_name)
            color = SCENARIO_COLORS.get(scenario_name)
            ax.plot(
                df_plot["step"],
                df_plot["mean"],
                label=label,
                linewidth=2.2,
                alpha=0.95,
                color=color,
            )
            lower = np.clip(df_plot["mean"] - ci, 0, 100)
            upper = np.clip(df_plot["mean"] + ci, 0, 100)
            upper_max = upper.max()
            if not np.isnan(upper_max):
                max_upper = upper_max if max_upper is None else max(max_upper, upper_max)
            ax.fill_between(
                df_plot["step"],
                lower,
                upper,
                alpha=0.15,
                color=color,
            )
            axis_used[idx] = True
            plotted_any = True

        ax.set_xlabel("Simulation Step")
        ax.set_ylabel("Average Agent Share (%)")
        ax.set_title(MOVEMENT_TITLES[movement], pad=12)
        ax.grid(False)
        if max_upper is not None:
            padded_max = max_upper * 1.05
            ax.set_ylim(0, min(100, padded_max))

    for ax, used in zip(axes, axis_used):
        if not used:
            ax.axis("off")

    if not plotted_any:
        plt.close(fig)
        return False

    handles: List = []
    labels: List[str] = []
    for ax in axes:
        h_ax, l_ax = ax.get_legend_handles_labels()
        handles += h_ax
        labels += l_ax

    seen = set()
    unique = []
    for h_item, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        unique.append((h_item, label))

    if unique:
        uhandles, ulabels = zip(*unique)
        fig.legend(
            uhandles,
            ulabels,
            loc="center left",
            ncol=1,
            frameon=False,
            bbox_to_anchor=(1.02, 0.5),
            labelspacing=0.8,
            borderaxespad=0.6,
            columnspacing=1.2,
            handlelength=2.0,
        )

    plt.suptitle("Move Decisions Percentage Per Step Across Scenarios", y=0.98)
    sns.despine(fig=fig)
    plt.tight_layout(rect=(0.0, 0.08, 1.0, 0.94), w_pad=2.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _build_summary(stats_by_scenario: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for scenario_name, movement_dict in stats_by_scenario.items():
        for movement, df_stats in movement_dict.items():
            if df_stats is None or df_stats.empty:
                continue
            df = df_stats.copy()
            df["scenario"] = scenario_name
            df["scenario_label"] = SCENARIO_LABELS.get(scenario_name, scenario_name)
            df["movement"] = movement
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    summary = pd.concat(rows, ignore_index=True)
    summary = summary.rename(columns={
        "mean": "mean_pct",
        "std": "std_pct",
        "count": "run_count",
    })
    cols = [
        "scenario",
        "scenario_label",
        "movement",
        "step",
        "mean_pct",
        "std_pct",
        "run_count",
    ]
    return summary[cols]


def _load_cached_summary(summary_path: Path) -> Optional[pd.DataFrame]:
    if not summary_path.exists():
        print(f"movement_decision_counts: cached summary not found at {summary_path}")
        return None
    try:
        df = pd.read_csv(summary_path)
    except Exception as exc:
        print(f"movement_decision_counts: failed to read cached summary {summary_path}: {exc}")
        return None

    required = {"scenario", "movement", "step", "mean_pct", "std_pct", "run_count"}
    missing = required.difference(df.columns)
    if missing:
        print(
            "movement_decision_counts: cached summary missing required columns: "
            f"{', '.join(sorted(missing))}"
        )
        return None
    return df


def _summary_to_stats(summary_df: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
    stats: Dict[str, Dict[str, pd.DataFrame]] = {}
    if summary_df is None or summary_df.empty:
        return stats

    summary_df = summary_df.copy()
    summary_df["step"] = pd.to_numeric(summary_df["step"], errors="coerce")
    summary_df = summary_df.dropna(subset=["step"])
    if summary_df.empty:
        return stats
    summary_df["step"] = summary_df["step"].astype(int)

    for scenario_name, scen_df in summary_df.groupby("scenario"):
        scenario_stats: Dict[str, pd.DataFrame] = {}
        for movement, move_df in scen_df.groupby("movement"):
            clean = (
                move_df.sort_values("step")[
                    ["step", "mean_pct", "std_pct", "run_count"]
                ]
                .rename(columns={
                    "mean_pct": "mean",
                    "std_pct": "std",
                    "run_count": "count",
                })
                .reset_index(drop=True)
            )
            scenario_stats[movement] = clean
        stats[scenario_name] = scenario_stats
    return stats


def _compute_stats_from_logs() -> Dict[str, Dict[str, pd.DataFrame]]:
    stats_by_scenario: Dict[str, Dict[str, pd.DataFrame]] = {}
    ordered = _ordered_scenarios()
    print("movement_decision_counts: starting movement decision aggregation")
    for scenario_name in ordered:
        folder = SCENARIOS.get(scenario_name)
        if not folder:
            print(f"movement_decision_counts: scenario '{scenario_name}' has no experiment folder; skipping")
            continue
        print(f"movement_decision_counts: processing scenario '{scenario_name}' ({folder})")
        decisions = _collect_decisions_for_scenario(folder)
        if decisions is None or decisions.empty:
            print(f"movement_decision_counts: no move logs for scenario '{scenario_name}' ({folder}); skipping")
            continue
        run_count = decisions['run_id'].nunique()
        print(
            "movement_decision_counts: "
            f"scenario '{scenario_name}' aggregated {run_count} runs and {len(decisions)} decision rows"
        )
        stats = _aggregate_statistics(decisions)
        stats_by_scenario[scenario_name] = stats
        max_step = max((df['step'].max() for df in stats.values() if df is not None and not df.empty), default=None)
        if max_step is not None:
            print(f"movement_decision_counts: scenario '{scenario_name}' includes steps up to {int(max_step)}")
    return stats_by_scenario


def main(max_steps: int = MAX_STEPS, recompute: bool = True):
    reports_dir = get_reports_dir() / OUTPUT_SUBDIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    plot_path = reports_dir / PLOT_FILENAME
    summary_path = reports_dir / SUMMARY_FILENAME

    stats_by_scenario: Dict[str, Dict[str, pd.DataFrame]] = {}
    summary_df: Optional[pd.DataFrame] = None
    summary_file: Optional[Path] = None

    if not recompute:
        summary_df = _load_cached_summary(summary_path)
        if summary_df is not None and not summary_df.empty:
            print(f"movement_decision_counts: reusing cached summary at {summary_path}")
            stats_by_scenario = _summary_to_stats(summary_df)
            summary_file = summary_path
        else:
            print("movement_decision_counts: cache missing or empty; recomputing from logs")
            recompute = True

    if recompute:
        stats_by_scenario = _compute_stats_from_logs()
        if not stats_by_scenario:
            print("movement_decision_counts: nothing to process; no plots generated")
            return None
        summary_df = _build_summary(stats_by_scenario)
        if not summary_df.empty:
            print(f"movement_decision_counts: writing summary CSV to {summary_path}")
            summary_df.to_csv(summary_path, index=False, compression="gzip")
            print(f"movement_decision_counts: wrote summary CSV to {summary_path}")
            summary_file = summary_path
        else:
            summary_file = None
            print("movement_decision_counts: summary CSV not written (no data)")
    else:
        if not stats_by_scenario:
            print("movement_decision_counts: cached summary had no usable entries; no plots generated")
            return None

    print(f"movement_decision_counts: rendering plots to {plot_path}")
    plotted = _plot(stats_by_scenario, plot_path, max_steps=max_steps)

    if plotted:
        print(f"movement_decision_counts: saved plot to {plot_path}")
    else:
        plot_path = None
        print("movement_decision_counts: plot skipped (no valid data)")

    return {
        "figure_path": plot_path,
        "summary_path": summary_file,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot average move counts per scenario")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="Maximum simulation step to include in the plot (default: 1000)",
    )
    parser.add_argument(
        "--no-recompute",
        action="store_true",
        help="Reuse cached summary CSV when available instead of re-reading move logs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    main(max_steps=_args.max_steps, recompute=not _args.no_recompute)
