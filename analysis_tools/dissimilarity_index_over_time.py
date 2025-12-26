"""Compute Dissimilarity Index per step for each experiment/run.

Outputs
- Per-experiment CSVs under reports/dissimilarity_index/*_dissimilarity_by_step.csv.gz
- Combined CSV with all experiments: reports/dissimilarity_index/dissimilarity_by_step_all.csv.gz
- Final values per run: reports/dissimilarity_index/dissimilarity_final_by_run.csv.gz

Usage
    python analysis_tools/dissimilarity_index_over_time.py
    python analysis_tools/dissimilarity_index_over_time.py --only llm_baseline mech_baseline

The script expects experiments to follow the layout produced by the simulation
runs (move_logs/*.json.gz or .csv + states/*.npz).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from experiment_list_for_analysis import SCENARIOS as scenarios
from analysis_tools.output_paths import get_reports_dir
from analysis_tools.analyze_agent_movement import (
    load_states_for_run,
    load_move_log_json,
    load_move_log_csv,
    iter_move_logs_json,
    iter_move_logs_csv,
)

# Precomputed tract map for 10x10 grid (see DissimilarityIndex.py)
TRACT_MAP = np.array([
    [0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
    [0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
    [0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
    [3, 3, 3, 4, 4, 4, 4, 5, 5, 5],
    [3, 3, 3, 4, 4, 4, 4, 5, 5, 5],
    [3, 3, 3, 4, 4, 4, 4, 5, 5, 5],
    [3, 3, 3, 4, 4, 4, 4, 5, 5, 5],
    [6, 6, 6, 7, 7, 7, 7, 8, 8, 8],
    [6, 6, 6, 7, 7, 7, 7, 8, 8, 8],
    [6, 6, 6, 7, 7, 7, 7, 8, 8, 8],
], dtype=np.int8)
TRACT_MAP_FLAT = TRACT_MAP.reshape(-1)


def compute_dissimilarity_from_int_grid(grid: np.ndarray) -> float:
    """Fast dissimilarity index computation for int grids (-1 empty, 0/1 types)."""
    if grid.shape != TRACT_MAP.shape:
        raise ValueError(f"Expected grid shape {TRACT_MAP.shape}, got {grid.shape}")

    flat = grid.reshape(-1)
    mask0 = flat == 0
    mask1 = flat == 1
    total0 = int(mask0.sum())
    total1 = int(mask1.sum())

    if total0 == 0 or total1 == 0:
        return 1.0 if (total0 + total1) > 0 else 0.0

    counts0 = np.bincount(TRACT_MAP_FLAT[mask0], minlength=9)
    counts1 = np.bincount(TRACT_MAP_FLAT[mask1], minlength=9)
    return 0.5 * np.abs(counts0 / total0 - counts1 / total1).sum()


def load_moves(experiment_dir: Path, run_id: int) -> Optional[pd.DataFrame]:
    log_json = load_move_log_json(experiment_dir, run_id)
    if log_json is not None:
        return pd.DataFrame(log_json)
    df_csv = load_move_log_csv(experiment_dir, run_id)
    if df_csv is not None and not df_csv.empty:
        return df_csv
    return None


def compute_run_timeseries(
    experiment_dir: Path,
    run_id: int,
    scenario_key: str,
    recompute: bool,
) -> Optional[pd.DataFrame]:
    states = load_states_for_run(experiment_dir, run_id)
    if states is None or len(states) == 0:
        print(f"  [WARN] Missing states for run {run_id} in {experiment_dir.name}; skipping run.")
        return None

    moves_df = load_moves(experiment_dir, run_id)
    if moves_df is None or moves_df.empty or 'step' not in moves_df.columns:
        print(f"  [WARN] Missing move log for run {run_id} in {experiment_dir.name}; skipping run.")
        return None

    moves_df = moves_df.reset_index(drop=True)
    n = min(len(states), len(moves_df))
    if n == 0:
        return None

    moves_df = moves_df.iloc[:n].copy()
    states = np.array(states[:n])

    # Map each step to the index of its last move entry
    moves_df['state_idx'] = moves_df.index
    last_indices = moves_df.groupby('step')['state_idx'].max().sort_index()

    rows: List[dict] = []
    for step_val, state_idx in last_indices.items():
        try:
            grid_int = np.array(states[int(state_idx)], dtype=int)
            dis_val = compute_dissimilarity_from_int_grid(grid_int)
            rows.append({
                'scenario': scenario_key,
                'experiment': experiment_dir.name,
                'run_id': int(run_id),
                'step': int(step_val),
                'dissimilarity_index': float(dis_val),
            })
        except Exception as exc:
            print(f"  [WARN] Failed to compute dissimilarity for run {run_id}, step {step_val}: {exc}")
            continue

    if not rows:
        return None
    return pd.DataFrame(rows)


def process_experiment(
    experiments_dir: Path,
    scenario_key: str,
    folder: str,
    recompute: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    exp_dir = experiments_dir / folder
    if not exp_dir.exists():
        print(f"[INFO] Experiment folder not found for {scenario_key}: {exp_dir}")
        return None, None

    out_dir = get_reports_dir() / 'dissimilarity_index'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip recompute if outputs already exist
    cached_ts = out_dir / f"{exp_dir.name}_dissimilarity_by_step.csv.gz"
    cached_final = out_dir / f"{exp_dir.name}_dissimilarity_final.csv.gz"
    if not recompute and cached_ts.exists() and cached_final.exists():
        try:
            ts_df = pd.read_csv(cached_ts, compression="infer")
            final_df = pd.read_csv(cached_final, compression="infer")
            print(f"[SKIP] {scenario_key}: using cached dissimilarity outputs ({cached_ts.name})")
            return ts_df, final_df
        except Exception:
            print(f"[WARN] Failed to read cached dissimilarity for {scenario_key}; recomputing.")

    print(f"[RUN] {scenario_key}: computing dissimilarity index over time from {exp_dir}")
    run_ids = sorted(set(iter_move_logs_json(exp_dir) + iter_move_logs_csv(exp_dir)))
    if not run_ids:
        print(f"  [WARN] No move logs found in {exp_dir}; skipping.")
        return None, None

    frames = []
    for rid in run_ids:
        df = compute_run_timeseries(exp_dir, rid, scenario_key, recompute=recompute)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        print(f"  [WARN] No usable runs for {exp_dir}; no output written.")
        return None, None

    per_run = pd.concat(frames, ignore_index=True)

    # Final values per run
    final_rows = per_run.loc[per_run.groupby('run_id')['step'].idxmax()].copy()
    final_rows = final_rows.rename(columns={'step': 'final_step'})

    per_run.to_csv(out_dir / f"{exp_dir.name}_dissimilarity_by_step.csv.gz", index=False, compression='gzip')
    final_rows.to_csv(out_dir / f"{exp_dir.name}_dissimilarity_final.csv.gz", index=False, compression='gzip')

    return per_run, final_rows


def run_all(
    experiments_dir: Path,
    only: Optional[Iterable[str]] = None,
    recompute: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    only_set = set(only) if only else None
    all_ts: List[pd.DataFrame] = []
    all_final: List[pd.DataFrame] = []

    for scenario_key, folder in scenarios.items():
        if only_set and (scenario_key not in only_set and folder not in only_set):
            continue
        ts_df, final_df = process_experiment(experiments_dir, scenario_key, folder, recompute=recompute)
        if ts_df is not None:
            all_ts.append(ts_df)
        if final_df is not None:
            all_final.append(final_df)

    if not all_ts:
        print("[INFO] No dissimilarity data produced.")
        return None, None

    out_dir = get_reports_dir() / 'dissimilarity_index'
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_all = pd.concat(all_ts, ignore_index=True)
    ts_all.to_csv(out_dir / 'dissimilarity_by_step_all.csv.gz', index=False, compression='gzip')

    if all_final:
        final_all = pd.concat(all_final, ignore_index=True)
        final_all.to_csv(out_dir / 'dissimilarity_final_by_run.csv.gz', index=False, compression='gzip')
    else:
        final_all = None

    print(f"[DONE] Wrote dissimilarity time series to {out_dir}/dissimilarity_by_step_all.csv.gz")
    if final_all is not None:
        print(f"[DONE] Wrote final dissimilarity values to {out_dir}/dissimilarity_final_by_run.csv.gz")
    return ts_all, final_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Dissimilarity Index per step for each experiment.")
    parser.add_argument('--experiments-dir', type=str, default='experiments', help='Path to experiments directory')
    parser.add_argument('--only', nargs='*', help='Limit to specific scenario keys or folder names')
    parser.add_argument('--no-recompute', action='store_true', help='Skip recomputing if cached outputs exist')
    args = parser.parse_args()

    ts_all, _ = run_all(Path(args.experiments_dir), only=args.only, recompute=not args.no_recompute)
    return 0 if ts_all is not None else 1


if __name__ == '__main__':
    raise SystemExit(main())
