"""Build (or rebuild) run_summary.csv for experiment directories.

New experiments get run_summary.csv automatically — Simulation.analyze_results
writes it alongside metrics_history.csv and convergence_summary.csv. This CLI
is for the ones that already exist, and for regenerating after a change to the
summary logic.

Usage
-----
    # every experiment directory under experiments/
    .venv/bin/python analysis_tools/build_run_summary.py --all

    # a single experiment
    .venv/bin/python analysis_tools/build_run_summary.py experiments/llm_baseline_20260202_171530

    # every experiment named by a run manifest, plus a combined CSV
    .venv/bin/python analysis_tools/build_run_summary.py \
        --manifest-file experiments/manifests/20260901_120000_run_manifest.json \
        --out reports/run_summary_combined.csv

Rows whose stored convergence bookkeeping is a resume placeholder
(final_step == 'unknown') or whose metrics rows are missing are rebuilt from
the run's move log, so the retrofit also repairs the legacy rows that recorded
the LAST step of the no-move window instead of the first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_summary import combine_run_summaries, write_run_summary  # noqa: E402


def _experiment_dirs_from_manifest(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    dirs: list[str] = []
    for record in payload.get("experiments", []):
        if record.get("status") != "success":
            continue
        output_dir = record.get("output_dir")
        if output_dir:
            dirs.append(output_dir)
    if not dirs:
        # Older manifests only carry the selected_experiments mapping.
        for experiment_name in (payload.get("selected_experiments") or {}).values():
            dirs.append(str(Path("experiments") / experiment_name))
    return dirs


def _all_experiment_dirs(experiments_dir: Path) -> list[str]:
    if not experiments_dir.is_dir():
        return []
    return [
        str(path) for path in sorted(experiments_dir.iterdir())
        if path.is_dir() and (path / "convergence_summary.csv").exists()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('experiment_dirs', nargs='*',
                        help='Experiment directories to summarize')
    parser.add_argument('--all', action='store_true',
                        help='Summarize every experiment directory under --experiments-dir')
    parser.add_argument('--experiments-dir', default='experiments',
                        help="Root of the experiment folders (default: experiments)")
    parser.add_argument('--manifest-file', default=None,
                        help='Run manifest JSON naming the experiments to summarize')
    parser.add_argument('--out', default=None,
                        help='Also write a combined CSV of every summarized run to this path')
    parser.add_argument('--force', action='store_true',
                        help='Rebuild from raw move logs instead of reusing rows already '
                             'in run_summary.csv (slow for large experiments)')
    parser.add_argument('--quiet', action='store_true', help='Suppress per-experiment output')
    args = parser.parse_args(argv)

    targets: list[str] = list(args.experiment_dirs)
    if args.manifest_file:
        targets.extend(_experiment_dirs_from_manifest(Path(args.manifest_file)))
    if args.all:
        targets.extend(_all_experiment_dirs(Path(args.experiments_dir)))

    # Preserve order, drop duplicates.
    seen: set[str] = set()
    ordered: list[str] = []
    for target in targets:
        resolved = str(Path(target))
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)

    if not ordered:
        parser.error("No experiments selected: pass directories, --all, or --manifest-file")

    verbose = not args.quiet
    written = 0
    for output_dir in ordered:
        if not Path(output_dir).is_dir():
            print(f"[run_summary] Skipping missing directory: {output_dir}")
            continue
        df = write_run_summary(output_dir, verbose=verbose, reuse_existing=not args.force)
        if not df.empty:
            written += 1

    print(f"[run_summary] Summarized {written}/{len(ordered)} experiment directory(ies)")

    if args.out:
        combine_run_summaries(ordered, out_path=args.out, verbose=verbose)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
