"""Compute Dissimilarity Index per step for each experiment/run.

Outputs
- Per-experiment CSVs under reports/dissimilarity_index/*_dissimilarity_by_step.csv.gz
- Combined CSV with all experiments: reports/dissimilarity_index/dissimilarity_by_step_all.csv.gz
- Final values per run: reports/dissimilarity_index/dissimilarity_final_by_run.csv.gz

Usage
    python analysis_tools/dissimilarity_index_over_time.py
    python analysis_tools/dissimilarity_index_over_time.py --only llm_baseline mech_baseline

The script expects experiments to follow the layout produced by the simulation
runs: move_logs/ + states/ in either the per-step or the older per-move
format, read through run_files.load_step_frames.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from experiment_list_for_analysis import SCENARIOS as scenarios
from analysis_tools.output_paths import get_reports_dir
from run_files import list_run_ids, load_step_frames

# Tract map + DI now live in Metrics.py (batch C, 2026-08-25): the simulation
# computes the index alongside the other six metrics, and this module reuses
# the SAME definition instead of a second hardcoded 10x10 copy. tract_map()
# reproduces the historical 10x10 map exactly and generalizes to any size.
from Metrics import compute_dissimilarity_from_int_grid  # noqa: E402


def _process_pool_context() -> Any:
    for method in ("spawn", "forkserver"):
        try:
            return mp.get_context(method)
        except ValueError:
            continue
    return mp.get_context()


def compute_run_timeseries(
    experiment_dir: Path,
    run_id: int,
    scenario_key: str,
    recompute: bool,
) -> Optional[pd.DataFrame]:
    loaded = load_step_frames(experiment_dir, run_id)
    if loaded is None:
        print(f"  [WARN] Missing states or move log for run {run_id} in {experiment_dir.name}; skipping run.")
        return None
    steps, grids = loaded

    rows: List[dict] = []
    for step_val, grid in zip(steps, grids):
        try:
            dis_val = compute_dissimilarity_from_int_grid(np.asarray(grid, dtype=int))
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


def _compute_run_timeseries_worker(task: Tuple[str, int, str, bool]) -> Optional[pd.DataFrame]:
    experiment_dir_str, run_id, scenario_key, recompute = task
    return compute_run_timeseries(
        experiment_dir=Path(experiment_dir_str),
        run_id=run_id,
        scenario_key=scenario_key,
        recompute=recompute,
    )


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
    run_ids = list_run_ids(exp_dir)
    if not run_ids:
        print(f"  [WARN] No move logs found in {exp_dir}; skipping.")
        return None, None

    workers_env = os.environ.get('DISSIMILARITY_WORKERS', '')
    if workers_env.strip():
        try:
            requested_workers = int(workers_env)
        except (TypeError, ValueError):
            requested_workers = 1
        max_workers = max(1, min(requested_workers, len(run_ids)))
    else:
        max_workers = min(len(run_ids), max(1, os.cpu_count() or 1))

    tasks: List[Tuple[str, int, str, bool]] = [
        (str(exp_dir), rid, scenario_key, recompute) for rid in run_ids
    ]

    frames: List[pd.DataFrame] = []
    if max_workers > 1 and len(tasks) > 1:
        print(f"  [INFO] Processing {len(tasks)} runs with {max_workers} workers")
        try:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=_process_pool_context()) as executor:
                for df in executor.map(_compute_run_timeseries_worker, tasks):
                    if df is not None and not df.empty:
                        frames.append(df)
        except Exception as exc:
            print(f"  [WARN] Parallel processing failed ({exc}); falling back to sequential")
            frames = []
            for rid in run_ids:
                df = compute_run_timeseries(exp_dir, rid, scenario_key, recompute=recompute)
                if df is not None and not df.empty:
                    frames.append(df)
    else:
        for rid in run_ids:
            df = compute_run_timeseries(exp_dir, rid, scenario_key, recompute=recompute)
            if df is not None and not df.empty:
                frames.append(df)

    if not frames:
        print(f"  [WARN] No usable runs for {exp_dir}; no output written.")
        return None, None

    per_run = pd.concat(frames, ignore_index=True)
    per_run = per_run.sort_values(['scenario', 'experiment', 'run_id', 'step'], kind='mergesort').reset_index(drop=True)

    # Final values per run
    final_rows = per_run.loc[per_run.groupby('run_id')['step'].idxmax()].copy()
    final_rows = final_rows.rename(columns={'step': 'final_step'})
    final_rows = final_rows.sort_values(['scenario', 'experiment', 'run_id', 'final_step'], kind='mergesort').reset_index(drop=True)

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
    ts_all = ts_all.sort_values(['scenario', 'experiment', 'run_id', 'step'], kind='mergesort').reset_index(drop=True)
    ts_all.to_csv(out_dir / 'dissimilarity_by_step_all.csv.gz', index=False, compression='gzip')

    if all_final:
        final_all = pd.concat(all_final, ignore_index=True)
        final_all = final_all.sort_values(['scenario', 'experiment', 'run_id', 'final_step'], kind='mergesort').reset_index(drop=True)
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
