#!/usr/bin/env python3
"""Pack completed experiments' per-run files into the packed containers.

    python analysis_tools/pack_run_records.py EXPERIMENT_DIR [EXPERIMENT_DIR ...]
    python analysis_tools/pack_run_records.py --all [--root experiments]
    python analysis_tools/pack_run_records.py --all --root experiments_with_llama_cpp

Runners pack their own experiment at the end (llm_runner / baseline_runner,
unless PACK_RUN_RECORD=0); this is for experiments recorded before 2026-09-05,
of which there are ~1M per-run files on disk. Per experiment it folds
move_logs/step_moves_run_<id>.csv + states/states_run_<id>.npz into
move_logs/step_moves_packed.csv.gz + states/states_packed.npz — same rows,
same arrays, verified member by member before any per-run file is deleted —
and every reader in run_files answers identically afterwards. Full-format
(per-move) runs are left untouched. --dry-run only counts. With --all,
directories under --root are searched recursively for a move_logs/ folder,
so campaign trees (run_<stamp>/experiments/<experiment>) are covered.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_files  # noqa: E402


def find_experiments(root):
    for dirpath, dirnames, _ in os.walk(root):
        if "move_logs" in dirnames:
            yield dirpath
            dirnames[:] = [d for d in dirnames if d not in ("move_logs", "states")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("experiment_dirs", nargs="*")
    ap.add_argument("--all", action="store_true", help="every experiment under --root")
    ap.add_argument("--root", default="experiments")
    ap.add_argument("--dry-run", action="store_true", help="count per-run files; pack nothing")
    args = ap.parse_args(argv)

    targets = list(args.experiment_dirs)
    if args.all:
        targets.extend(find_experiments(args.root))
    if not targets:
        ap.error("give experiment directories or --all")

    total_runs = total_files = 0
    t0 = time.time()
    for exp_dir in targets:
        per_run = [n for n in os.listdir(os.path.join(exp_dir, "move_logs"))
                   if n.startswith("step_moves_run_")] if os.path.isdir(os.path.join(exp_dir, "move_logs")) else []
        if not per_run:
            continue
        if args.dry_run:
            print(f"[pack] {exp_dir}: {len(per_run)} per-run step logs (dry run)")
            total_runs += len(per_run)
            continue
        packed = run_files.pack_run_record(exp_dir)
        total_runs += packed
        total_files += 2 * packed
    print(f"[pack] {'would pack' if args.dry_run else 'packed'} {total_runs} run(s) "
          f"across {len(targets)} director{'y' if len(targets) == 1 else 'ies'} "
          f"in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
