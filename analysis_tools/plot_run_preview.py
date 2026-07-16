"""Preview plots for an in-progress (or completed) llama.cpp production run.

Regenerate any time — works on partial data (only completed runs are read):

    .venv/bin/python analysis_tools/plot_run_preview.py \
        --run-dir experiments_with_llama_cpp/run_20260716_045616_Llama-3.3-70B-Instruct-Q4_K_M-a3

Outputs into <run_dir>/plots/:
    preview_metrics_<scenario>.png   6 segregation metrics + move rate + parse
                                     failures vs step (mean +/- 95% CI, faint
                                     per-run traces)
    preview_grids_<scenario>.png     initial vs final grid per completed run

Reads states/states_run_*.npz (one int-grid snapshot per agent decision;
-1 empty, 0/1 agent types) and move_logs/agent_moves_run_*.json.gz (per-decision
records incl. step, moved, llm_parse_status), via the shared loaders in
analysis_tools.analyze_agent_movement (single source of truth for run-file
formats). The per-step grid is the LAST snapshot within each step. Metrics come
from Metrics.calculate_all_metrics — the same definitions used everywhere else
in the project.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from Metrics import calculate_all_metrics  # noqa: E402
from analysis_tools.analyze_agent_movement import (  # noqa: E402
    load_move_log_json,
    load_states_for_run,
)

METRIC_KEYS = ["clusters", "switch_rate", "distance", "mix_deviation", "share", "ghetto_rate"]
GRID_CMAP = ListedColormap(["#f2f2f2", "#d62728", "#1f77b4"])  # empty, type 0, type 1


class _CellAgent:
    """Minimal stand-in so Metrics (which expects .type_id objects) accepts int grids."""
    __slots__ = ("type_id",)

    def __init__(self, type_id):
        self.type_id = type_id


def int_grid_to_object_grid(int_grid):
    obj = np.empty(int_grid.shape, dtype=object)
    for r in range(int_grid.shape[0]):
        for c in range(int_grid.shape[1]):
            v = int(int_grid[r, c])
            obj[r, c] = _CellAgent(v) if v >= 0 else None
    return obj


def load_run(exp_dir, run_id):
    """Return (per_step_grids, per_step_move_rate, per_step_parse_fail) or None."""
    states = load_states_for_run(Path(exp_dir), run_id)
    entries = load_move_log_json(Path(exp_dir), run_id)
    if states is None or entries is None:
        return None

    last_idx_by_step, decisions_by_step = {}, {}
    for idx, e in enumerate(entries):
        if idx >= len(states):
            break
        step = e.get("step")
        if step is None:
            continue
        step = int(step)
        last_idx_by_step[step] = idx
        if e.get("reason") == "initial_state":
            continue
        moved = bool(e.get("moved"))
        parse_ok = str(e.get("llm_parse_status")) == "OK"
        decisions_by_step.setdefault(step, []).append((moved, parse_ok))

    steps = sorted(last_idx_by_step)
    grids = [states[last_idx_by_step[s]] for s in steps]
    move_rate, parse_fail = [], []
    for s in steps:
        ds = decisions_by_step.get(s, [])
        move_rate.append(np.mean([m for m, _ in ds]) if ds else np.nan)
        parse_fail.append(np.mean([not ok for _, ok in ds]) if ds else np.nan)
    return steps, grids, move_rate, parse_fail


def collect_experiment(exp_dir):
    run_ids = sorted(
        int(os.path.basename(p).split("_run_")[1].split(".")[0])
        for p in glob.glob(os.path.join(exp_dir, "states", "states_run_*.npz"))
    )
    runs = {}
    for rid in run_ids:
        loaded = load_run(exp_dir, rid)
        if loaded:
            runs[rid] = loaded
    return runs


def mean_ci(matrix):
    """(n_runs, n_steps) -> mean, half-width of 95% CI (nan-aware)."""
    mean = np.nanmean(matrix, axis=0)
    n = np.sum(~np.isnan(matrix), axis=0)
    sd = np.nanstd(matrix, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hw = 1.96 * sd / np.sqrt(np.maximum(n, 1))
    return mean, hw


def plot_metrics(scenario, runs, out_path, title_suffix=""):
    n_steps = max(len(v[0]) for v in runs.values())
    series = {k: np.full((len(runs), n_steps), np.nan) for k in METRIC_KEYS}
    move = np.full((len(runs), n_steps), np.nan)
    pfail = np.full((len(runs), n_steps), np.nan)

    for row, (rid, (steps, grids, mrate, pf)) in enumerate(sorted(runs.items())):
        for col, g in enumerate(grids):
            m = calculate_all_metrics(int_grid_to_object_grid(g))
            for k in METRIC_KEYS:
                series[k][row, col] = m[k]
        move[row, :len(mrate)] = mrate
        pfail[row, :len(pf)] = pf

    panels = METRIC_KEYS + ["move_rate", "parse_fail_rate"]
    data = {**series, "move_rate": move, "parse_fail_rate": pfail}
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    for ax, key in zip(axes.flat, panels):
        mat = data[key]
        for row in range(mat.shape[0]):
            ax.plot(mat[row], color="gray", alpha=0.25, lw=0.7)
        mean, hw = mean_ci(mat)
        x = np.arange(n_steps)
        ax.plot(x, mean, color="#d62728", lw=1.8, label="mean")
        ax.fill_between(x, mean - hw, mean + hw, color="#d62728", alpha=0.2, label="95% CI")
        ax.set_title(key)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{scenario} — {len(runs)} completed runs{title_suffix}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_grids(scenario, runs, out_path, title_suffix=""):
    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n + 2, 4.2), squeeze=False)
    for col, (rid, (steps, grids, _, _)) in enumerate(sorted(runs.items())):
        for row, (g, label) in enumerate([(grids[0], "initial"), (grids[-1], "final")]):
            ax = axes[row][col]
            ax.imshow(g + 1, cmap=GRID_CMAP, vmin=0, vmax=2)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"run {rid}", fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
    fig.suptitle(f"{scenario} — initial vs final grids{title_suffix}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True,
                    help="run_<timestamp>_<model> directory (contains experiments/)")
    ap.add_argument("--out", default=None, help="output dir (default <run_dir>/plots)")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    exp_dirs = sorted(glob.glob(os.path.join(args.run_dir, "experiments", "llm_*")))
    if not exp_dirs:
        sys.exit(f"no experiments/llm_* dirs under {args.run_dir}")

    for exp_dir in exp_dirs:
        scenario = os.path.basename(exp_dir).replace("llm_", "").rsplit("_", 2)[0]
        runs = collect_experiment(exp_dir)
        if not runs:
            print(f"[skip] {scenario}: no completed runs yet")
            continue
        m_path = os.path.join(out_dir, f"preview_metrics_{scenario}.png")
        g_path = os.path.join(out_dir, f"preview_grids_{scenario}.png")
        plot_metrics(scenario, runs, m_path)
        plot_grids(scenario, runs, g_path)
        print(f"[done] {scenario}: {len(runs)} runs -> {m_path}, {g_path}")


if __name__ == "__main__":
    main()
