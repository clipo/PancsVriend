#!/usr/bin/env python3
"""Check that an experiment directory's run files are internally consistent.

For every run listed by run_files.list_run_ids the checker asserts:

1. the move log and states/states_run_<id>.npz are both readable;
2. per-step runs: the npz has exactly one frame more than the step log has
   rows (frame 0 = initial grid, frame k+1 = after step k) and the steps are
   contiguous; full-format runs: one frame per move record, and any inline
   'grid' copies (pre-2026-09-01 logs) equal the npz frames;
3. the last logged step equals `final_step` in convergence_summary.csv.

Usage:
    python test_latest_experiment_output_format.py [EXPERIMENT_DIR]

Without an argument the newest directory under ./experiments is checked.
Exit status 0 when every run passes, 1 otherwise.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import run_files


def latest_experiment_dir(root="experiments"):
    dirs = [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]
    if not dirs:
        raise FileNotFoundError(f"no experiment directories under {root}")
    return max(dirs, key=os.path.getmtime)


def check_run(output_dir, run_id, final_steps):
    """List of problems for one run (empty when it is consistent)."""
    problems = []
    step_log = run_files.load_step_log(output_dir, run_id)
    frames = run_files.load_frames(output_dir, run_id)
    if step_log is None or step_log.empty:
        return [f"run {run_id}: no readable move log"]
    if frames is None:
        return [f"run {run_id}: no readable states file"]

    steps = [int(s) for s in step_log["step"]]
    if steps != list(range(steps[0], steps[0] + len(steps))):
        problems.append(f"run {run_id}: steps are not contiguous ({steps[0]}..{steps[-1]}, {len(steps)} rows)")

    if os.path.exists(run_files.step_log_path(output_dir, run_id)):
        if len(frames) != len(steps) + 1:
            problems.append(f"run {run_id}: {len(frames)} frames for {len(steps)} steps (expected steps + 1)")
    else:
        records = run_files.load_move_log_json(output_dir, run_id)
        if records is None:
            problems.append(f"run {run_id}: CSV move log has no frame pairing to check")
        else:
            if len(records) != len(frames):
                problems.append(f"run {run_id}: {len(records)} move records but {len(frames)} frames")
            inline = [(i, r["grid"]) for i, r in enumerate(records) if "grid" in r]
            mismatched = sum(1 for i, grid in inline
                             if i < len(frames) and not np.array_equal(np.asarray(grid), frames[i]))
            if mismatched:
                problems.append(f"run {run_id}: {mismatched} inline grid copies differ from the npz frames")

    if run_id in final_steps and steps[-1] != final_steps[run_id]:
        problems.append(f"run {run_id}: last logged step {steps[-1]} != final_step {final_steps[run_id]}")
    return problems


def validate_experiment(output_dir):
    """True when every run in output_dir passes; prints each problem found."""
    run_ids = run_files.list_run_ids(output_dir)
    if not run_ids:
        print(f"no move logs found in {output_dir}")
        return False

    # run_summary.csv is the corrected source (see CLAUDE.md): the live
    # convergence_summary.csv of a pre-2026-09-01 run records the step cap,
    # not the last simulated step, for max_steps runs.
    final_steps = {}
    for name in ("run_summary.csv", "convergence_summary.csv"):
        summary_path = os.path.join(output_dir, name)
        if not os.path.exists(summary_path):
            continue
        summary = pd.read_csv(summary_path)
        if {"run_id", "final_step"} <= set(summary.columns):
            final = pd.to_numeric(summary["final_step"], errors="coerce")   # 'unknown' placeholders
            final_steps = {int(r): int(f) for r, f in zip(summary["run_id"], final) if pd.notna(f)}
            break
    else:
        print("no run_summary.csv or convergence_summary.csv; skipping the final_step check")

    problems = []
    for run_id in run_ids:
        problems.extend(check_run(output_dir, run_id, final_steps))

    for problem in problems:
        print(f"  FAIL {problem}")
    print(f"{len(run_ids)} runs checked in {output_dir}: "
          f"{'all consistent' if not problems else f'{len(problems)} problems'}")
    return not problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment_dir", nargs="?", help="defaults to the newest under ./experiments")
    args = parser.parse_args(argv)
    output_dir = args.experiment_dir or latest_experiment_dir()
    return 0 if validate_experiment(output_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
