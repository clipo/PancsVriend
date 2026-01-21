#!/usr/bin/env python3
"""
Analyze agent movement likelihood by type and by neighbor ratio across experiments.

What it does
- Scans each experiment in experiments/ (or a provided path)
- For every run, pairs agent move logs with the corresponding states to reconstruct
  the PRE-decision neighborhood for each agent decision
- Computes:
  1) P(move) by agent type (Type 0 vs Type 1)
  2) P(move) vs. share of same-type neighbors (8-neighborhood), separately for each type
- Saves per-experiment plots under <experiment>/analysis/
- Optionally writes combined summary plots under an output directory (default: reports/movement_analysis)

Input expectations (from base_simulation.py):
- Each experiment folder contains:
  experiments/<name>/
    - move_logs/agent_moves_run_<id>.json.gz (or .csv)
    - states/states_run_<id>.npz (array name 'states')
    - config.json (optional; includes 'scenario')

Usage examples
  python analyze_agent_movement.py
  python analyze_agent_movement.py --experiments-dir experiments --out-dir reports/movement_analysis
  python analyze_agent_movement.py --only "llm_race_white_black_20250718_195455"

Notes
- We derive PRE-decision neighbor stats from states[i-1] for the i-th move entry.
  The first log entry is a dummy 'initial_state' and is skipped.
- If JSON logs are missing, we fall back to CSV. If states are missing, that run is skipped.
- Aggregated summaries are written as gzip-compressed CSV (.csv.gz) files; legacy uncompressed
    caches are still read automatically when present.
"""

from __future__ import annotations

import argparse
import json
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from context_scenarios import CONTEXT_SCENARIOS
from analysis_tools.output_paths import get_reports_dir


def load_config(experiment_dir: Path) -> Dict:
    cfg_path = experiment_dir / "config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text())
        except Exception:
            return {}
    return {}


def list_experiments(experiments_dir: Path, only: Optional[List[str]] = None) -> List[Path]:
    if not experiments_dir.exists():
        return []
    all_dirs = [p for p in experiments_dir.iterdir() if p.is_dir()]
    if only:
        names = set(only)
        return [p for p in all_dirs if p.name in names]
    # Sort by mtime desc (newest first)
    return sorted(all_dirs, key=lambda p: p.stat().st_mtime, reverse=True)


def load_states_for_run(experiment_dir: Path, run_id: int) -> Optional[np.ndarray]:
    states_path = experiment_dir / "states" / f"states_run_{run_id}.npz"
    if not states_path.exists():
        return None
    try:
        data = np.load(states_path)
        # Expect key 'states'
        if 'states' in data:
            return data['states']
        # Fallback to first key
        keys = list(data.keys())
        return data[keys[0]] if keys else None
    except Exception:
        return None


def iter_move_logs_json(experiment_dir: Path) -> List[int]:
    move_dir = experiment_dir / "move_logs"
    if not move_dir.exists():
        return []
    run_ids = []
    for pattern in ("agent_moves_run_*.json.gz", "agent_moves_run_*.json"):
        for f in move_dir.glob(pattern):
            name = f.name
            if name.endswith(".json.gz"):
                prefix = name[:-8]  # strip .json.gz
            elif name.endswith(".json"):
                prefix = name[:-5]  # strip .json
            else:
                continue
            try:
                rid = int(prefix.split("_")[-1])
            except ValueError:
                continue
            run_ids.append(rid)
    return sorted(set(run_ids))


def iter_move_logs_csv(experiment_dir: Path) -> List[int]:
    """Return run ids for CSV or compressed CSV move logs."""
    move_dir = experiment_dir / "move_logs"
    if not move_dir.exists():
        return []
    run_ids = set()
    for ext in (".csv.gz", ".csv"):
        for f in move_dir.glob(f"agent_moves_run_*{ext}"):
            try:
                rid = int(f.stem.split("_")[-1].split(".")[0])
            except ValueError:
                continue
            run_ids.add(rid)
    return sorted(run_ids)


def load_move_log_json(experiment_dir: Path, run_id: int) -> Optional[List[dict]]:
    path = experiment_dir / "move_logs" / f"agent_moves_run_{run_id}.json.gz"
    if not path.exists():
        return None
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def load_move_log_csv(experiment_dir: Path, run_id: int) -> Optional[pd.DataFrame]:
    """Load CSV (optionally gzip-compressed) move logs for a run."""
    move_dir = experiment_dir / "move_logs"
    candidates = [
        move_dir / f"agent_moves_run_{run_id}.csv.gz",
        move_dir / f"agent_moves_run_{run_id}.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            return pd.read_csv(path, compression="infer")
        except Exception:
            continue
    return None


def count_neighbors(pre_grid: np.ndarray, r: int, c: int, agent_type: int) -> Tuple[int, int]:
    # 8-neighborhood
    like = 0
    unlike = 0
    h, w = pre_grid.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                val = pre_grid[nr, nc]
                if val >= 0:
                    if val == agent_type:
                        like += 1
                    else:
                        unlike += 1
    return like, unlike


def compute_summary_for_run(experiment_dir: Path, run_id: int) -> Optional[pd.DataFrame]:
    """Return a dataframe with per-move records including pre-decision neighbor shares.

    Columns: run_id, index_in_run, step, type_id, moved, reason, row, col,
             share_same, like_neighbors, unlike_neighbors
    """
    states = load_states_for_run(experiment_dir, run_id)
    if states is None or len(states) == 0:
        return None

    # Prefer JSON (keeps types), else CSV
    log_json = load_move_log_json(experiment_dir, run_id)
    if log_json is not None:
        entries = log_json
    else:
        df_csv: Optional[pd.DataFrame] = load_move_log_csv(experiment_dir, run_id)
        if df_csv is None or df_csv.empty:
            return None
        entries = df_csv.to_dict('records')

    # Safety: lengths should match (including initial_state row)
    # If mismatch, we'll still iterate over min length
    n = min(len(entries), len(states))
    if n <= 1:
        return None

    rows = []
    for i in range(1, n):  # skip index 0 (initial_state)
        e = entries[i]
        step = e.get('step', None)
        type_id = e.get('type_id', None)
        moved = e.get('moved', False)
        reason = e.get('reason', '')
        curr_pos = e.get('current_position')

        # Normalize current_position which may be stored as list/tuple/str
        r = c = None
        if isinstance(curr_pos, (list, tuple)) and len(curr_pos) == 2:
            r, c = curr_pos
        elif isinstance(curr_pos, str):
            try:
                # Expect format like "(r, c)" or "[r, c]"
                s = curr_pos.strip().replace('(', '[').replace(')', ']')
                parsed = json.loads(s)
                if isinstance(parsed, list) and len(parsed) == 2:
                    r, c = parsed
            except Exception:
                pass

        # Skip malformed entries or non-typed
        if type_id not in (0, 1) or r is None or c is None:
            continue

        # pre-decision grid is the previous state snapshot
        pre_grid = states[i - 1]
        # Guard bounds
        if not (0 <= r < pre_grid.shape[0] and 0 <= c < pre_grid.shape[1]):
            continue

        # Ensure the agent was indeed of that type at pre-grid
        pre_val = pre_grid[r, c]
        if pre_val != type_id:
            # If mismatch, still compute neighbors around (r, c) but skip strict check
            # This can happen rarely due to logging order; we don't fail hard
            pass

        like, unlike = count_neighbors(pre_grid, r, c, type_id)
        total_nbrs = like + unlike
        share_same = (like / total_nbrs) if total_nbrs > 0 else np.nan

        rows.append({
            'run_id': run_id,
            'index_in_run': i,
            'step': step,
            'type_id': type_id,
            'moved': bool(moved),
            'reason': reason,
            'row': r,
            'col': c,
            'like_neighbors': like,
            'unlike_neighbors': unlike,
            'share_same': share_same,
        })

    if not rows:
        return None
    return pd.DataFrame(rows)


def analyze_experiment(experiment_dir: Path, summary_out_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Analyze a single experiment directory and write per-experiment plots.

    Returns the aggregated dataframe of decisions with neighbor shares (for summary across experiments).
    """
    print(f"Analyzing experiment: {experiment_dir.name}")
    cfg = load_config(experiment_dir)
    scenario = cfg.get('scenario', None)
    # Derive scenario from folder name if missing
    if not scenario:
        name = experiment_dir.name
        # Assume name like llm_<scenario>_<timestamp>
        parts = name.split('_')
        scenario = '_'.join(parts[1:-2]) if len(parts) > 3 else name

    json_runs = iter_move_logs_json(experiment_dir)
    csv_runs = iter_move_logs_csv(experiment_dir)
    run_ids = sorted(set(json_runs) | set(csv_runs))
    if not run_ids:
        print(f"  No move logs found in {experiment_dir}")
        return None

    per_run = []
    for rid in run_ids:
        df = compute_summary_for_run(experiment_dir, rid)
        if df is not None and not df.empty:
            per_run.append(df)

    if not per_run:
        print(f"  No analyzable runs in {experiment_dir}")
        return None

    df_all = pd.concat(per_run, ignore_index=True)
    df_all['experiment'] = experiment_dir.name
    df_all['scenario'] = scenario

    # Map type_id -> scenario-specific labels
    scen_info = CONTEXT_SCENARIOS.get(scenario, {}) if scenario else {}
    type_names = {
        0: scen_info.get('type_a', 'Type 0'),
        1: scen_info.get('type_b', 'Type 1'),
    }
    df_all['type_label'] = df_all['type_id'].map(type_names)

    # Create per-experiment output dir under the configured reports directory
    base_reports = get_reports_dir() / 'movement_analysis'
    analysis_dir = base_reports / experiment_dir.name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 1) P(move) by type (bar plot)
    by_type = (
        df_all.dropna(subset=['share_same'])
             .groupby('type_label')['moved']
             .mean()
             .reset_index()
    )
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=by_type, x='type_label', y='moved', hue='type_label', palette='Set2')
    plt.title(f"P(move) by type — \n{experiment_dir.name}")
    plt.xlabel("Agent type")
    plt.ylabel("Probability of moving")
    plt.ylim(0, 1)
    # Place legend inside subplot
    # Only add legend if there are labeled handles
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(handles, labels, loc='upper right', frameon=True, title='Type')
    plt.tight_layout()
    out1 = analysis_dir / "movement_probability_by_type.png"
    plt.savefig(out1, dpi=150)
    plt.close()

    # 2) P(move) vs neighbor share (per type) — bin share_same into deciles
    df_bins = df_all.dropna(subset=['share_same']).copy()
    # Clip to [0,1] and bin
    df_bins['share_same'] = df_bins['share_same'].clip(0, 1)
    bins_list = [i / 10 for i in range(11)]
    df_bins['share_bin'] = pd.cut(df_bins['share_same'], bins=bins_list, include_lowest=True)

    prob_vs_ratio = (
        df_bins.groupby(['type_label', 'share_bin'], observed=False)['moved']
               .mean()
               .reset_index()
    )
    # Midpoint for plotting
    prob_vs_ratio['share_mid'] = prob_vs_ratio['share_bin'].apply(lambda b: (b.left + b.right) / 2 if hasattr(b, 'left') else np.nan)

    plt.figure(figsize=(7, 5))
    ax2 = sns.lineplot(data=prob_vs_ratio, x='share_mid', y='moved', hue='type_label', marker='o', palette='Set1')
    plt.title(f"P(move) vs share of same-type neighbors — \n{experiment_dir.name}")
    plt.xlabel("Share of same-type neighbors (8-neighborhood)")
    plt.ylabel("Probability of moving")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    # Place legend inside subplot
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles2 and labels2:
        ax2.legend(handles2, labels2, loc='lower right', frameon=True, title='Type')
    plt.tight_layout()
    out2 = analysis_dir / "move_probability_vs_neighbor_ratio.png"
    plt.savefig(out2, dpi=150)
    plt.close()

    # Save aggregated CSV for this experiment (compressed to reduce footprint)
    per_exp_path = analysis_dir / "movement_neighbor_summary.csv.gz"
    df_all.to_csv(per_exp_path, index=False, compression="gzip")

    # Write to summary_out_dir if provided (for combined plots later)
    if summary_out_dir is not None:
        summary_out_dir.mkdir(parents=True, exist_ok=True)
    return df_all


def make_summary_plots(df_all_exp: pd.DataFrame, out_dir: Path) -> None:
    if df_all_exp.empty:
        print("No data provided for summary plots; skipping stitching.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = (
        df_all_exp.get('experiment', pd.Series(dtype=str))
                 .dropna()
                 .astype(str)
                 .unique()
                 .tolist()
    )

    if not experiments:
        print("No experiment identifiers found in dataframe; skipping summary plots.")
        return

    experiments = sorted(experiments)

    # Each analyze_experiment call creates two per-experiment plots. Stitch them across experiments.
    plot_specs = [
        {
            'filename': 'movement_probability_by_type.png',
            'summary_name': 'summary_movement_probability_by_type.png',
            'title': 'P(move) by type — summary',
        },
        {
            'filename': 'move_probability_vs_neighbor_ratio.png',
            'summary_name': 'summary_move_probability_vs_neighbor_ratio.png',
            'title': 'P(move) vs share same-type neighbors — summary',
        },
    ]

    for spec in plot_specs:
        images: List[np.ndarray] = []
        labels: List[str] = []
        missing: List[Path] = []

        for exp in experiments:
            img_path = out_dir / exp / spec['filename']
            if img_path.exists():
                try:
                    images.append(plt.imread(img_path))
                    labels.append(exp)
                except Exception as exc:
                    print(f"  Warning: failed to read {img_path}: {exc}")
            else:
                missing.append(img_path)

        if missing:
            for path in missing:
                print(f"  Info: per-experiment plot missing for summary: {path}")

        if not images:
            print(f"No images available to stitch for {spec['filename']}; skipping.")
            continue

        n_images = len(images)
        n_cols = min(3, n_images)
        n_rows = math.ceil(n_images / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        axes_arr = np.array(axes).reshape(-1)

        for ax in axes_arr[n_images:]:
            ax.axis('off')

        for ax, img, label in zip(axes_arr, images, labels):
            ax.imshow(img)
            ax.set_title(label, fontsize=10)
            ax.axis('off')

        fig.suptitle(spec['title'], fontsize=14, weight='bold')
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

        summary_path = out_dir / spec['summary_name']
        try:
            fig.savefig(str(summary_path), dpi=150)
            print(f"Summary plot written: {summary_path}")
        except Exception as exc:
            print(f"Warning: failed to save summary plot {summary_path}: {exc}")
        finally:
            plt.close(fig)




def main():
    parser = argparse.ArgumentParser(description="Analyze movement probability by type and neighbor ratio across experiments")
    parser.add_argument("--experiments-dir", type=str, default="experiments", help="Path to experiments directory")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(get_reports_dir() / "movement_analysis"),
        help="Directory for combined summary outputs",
    )
    parser.add_argument("--only", type=str, nargs="*", help="Limit to specific experiment folder names (space-separated or a single comma-separated string)")
    parser.add_argument("--no-recompute", action="store_true", help="Load cached analyzed dataframe instead of recomputing")
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    summary_out_dir = Path(args.out_dir)

    # Normalize --only values: allow either space-separated or a single comma-separated string
    only_names: Optional[List[str]] = None
    if args.only:
        # Flatten and split comma lists
        flattened: List[str] = []
        for item in args.only:
            if "," in item:
                flattened.extend([x.strip() for x in item.split(",") if x.strip()])
            else:
                if item.strip():
                    flattened.append(item.strip())
        only_names = flattened if flattened else None

    # Cached combined dataframe path (compressed)
    combined_path = summary_out_dir / "movement_neighbor_summary_all.csv.gz"

    if args.no_recompute:
        # Load cached dataframe for plotting only
        if combined_path.exists():
            df_all_exp = pd.read_csv(combined_path, compression="infer")
            make_summary_plots(df_all_exp, summary_out_dir)
            print(f"Loaded cached analyzed data from: {combined_path}")
            print(f"Summary plots written to: {summary_out_dir}")
            return 0
        # Backward compatibility: fallback to legacy uncompressed cache if present
        legacy_combined = summary_out_dir / "movement_neighbor_summary_all.csv"
        if legacy_combined.exists():
            df_all_exp = pd.read_csv(legacy_combined)
            make_summary_plots(df_all_exp, summary_out_dir)
            print(f"Loaded cached analyzed data from legacy CSV: {legacy_combined}")
            print(f"Summary plots written to: {summary_out_dir}")
            return 0
        print(f"Cached analyzed dataframe not found at {combined_path}.")
        print("Run without --no-recompute to generate it.")
        return 1

    # Recompute workflow
    exp_dirs = list_experiments(experiments_dir, only=only_names)
    if not exp_dirs:
        if only_names:
            print(f"No matching experiments for: {only_names}")
        else:
            print(f"No experiments found under: {experiments_dir}")
        return 1

    all_frames = []
    for exp in exp_dirs:
        df = analyze_experiment(exp, summary_out_dir=None)
        if df is not None and not df.empty:
            all_frames.append(df)

    if not all_frames:
        print("No data available to create summary plots.")
        return 0

    df_all_exp = pd.concat(all_frames, ignore_index=True)
    # Save combined analyzed dataframe for future --no-recompute runs
    try:
        summary_out_dir.mkdir(parents=True, exist_ok=True)
        df_all_exp.to_csv(combined_path, index=False, compression="gzip")
        print(f"Saved combined analyzed dataframe to: {combined_path}")
        # Clean up any legacy uncompressed cache to avoid stale large files
        legacy_combined = summary_out_dir / "movement_neighbor_summary_all.csv"
        if legacy_combined.exists():
            legacy_combined.unlink()
    except Exception as e:
        print(f"Warning: failed to write combined dataframe: {e}")

    make_summary_plots(df_all_exp, summary_out_dir)
    print(f"Summary plots written to: {summary_out_dir}")
    return 0

# Early entrypoint to avoid running duplicate blocks below if present
if __name__ == "__main__":
    raise SystemExit(main())
