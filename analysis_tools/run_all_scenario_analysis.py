"""Run all PancsVriend analyses using the shared scenario metadata.

Quick start (from repo root):
    python analysis_tools/run_all_scenario_analysis.py \
        [--no-recompute] [--include-movement | --movement-only] \
        [--output-folder <path>] [--quiet]

Flags:
    --no-recompute      Skip forced metric recompute; use existing CSVs if present.
    --include-movement  Add movement analysis (slower; needs move_logs + states).
    --movement-only     Run only movement analysis (overrides other steps).
    --output-folder     Base reports directory (default: reports). Respects
                        env PANCSVRIEND_REPORTS_DIR if set.
    --quiet             Suppress per-step progress; summary still printed.

Pipeline order (when not movement-only):
    0) dissimilarity_index_over_time
    1) combined_final_metrics
    2) analyze_agent_movement (optional via --include-movement)
    3) analyze_stability_patterns
    4) convergence_patterns_and_speed
    5) per_metric_panels
    6) segregation_metrics_comparison

Config lives in `analysis_tools/experiment_list_for_analysis.py`
(scenarios, labels, colors). This file only orchestrates execution.
"""

from __future__ import annotations

import importlib
import argparse
import sys
from pathlib import Path
from typing import Union

from analysis_tools.output_paths import set_reports_dir

# Importing ensures metadata is available to downstream modules that rely on it
from experiment_list_for_analysis import SCENARIOS  # noqa: F401


def run_all_analyses(
    recompute: bool = True,
    movement_only: bool = False,
    include_movement: bool = False,
    verbose: bool = True,
    output_folder: Union[str, Path] = "reports",
):
    """Run the suite of analysis scripts in a recommended order."""
    at_path = Path(__file__).resolve().parent
    if str(at_path) not in sys.path:
        sys.path.append(str(at_path))

    target_output = output_folder or "reports"
    reports_dir = set_reports_dir(target_output)
    reports_dir.mkdir(parents=True, exist_ok=True)

    steps = []  # (name, callable, kwargs)

    # Movement step definition (shared by both branches)
    def _run_movement():
        mod = importlib.import_module('analysis_tools.analyze_agent_movement')
        original_argv = sys.argv[:]
        module_path = getattr(mod, '__file__', None)
        prog_name = str(Path(module_path)) if module_path else mod.__name__

        only_list = list(SCENARIOS.values())
        argv = [prog_name, '--experiments-dir', 'experiments', '--out-dir', str(reports_dir / 'movement_analysis')]
        if only_list:
            argv.extend(['--only', ','.join(only_list)])
        if not recompute:
            argv.append('--no-recompute')

        try:
            sys.argv = argv
            result = mod.main()
        finally:
            sys.argv = original_argv

        if isinstance(result, int) and result != 0:
            raise RuntimeError(f"movement analysis exited with code {result}")

    movement_step = ("analyze_agent_movement", _run_movement, {})

    # If movement-only, skip all other analyses
    if movement_only:
        steps.append(movement_step)
        return _execute_steps(steps, verbose=verbose)

    # 0. Dissimilarity index over time (requires states + move logs)
    def _run_dissimilarity_index():
        mod = importlib.import_module('analysis_tools.dissimilarity_index_over_time')
        mod.run_all(Path('experiments'), recompute=recompute)

    steps.append(("dissimilarity_index_over_time", _run_dissimilarity_index, {}))

    # 1. Combined metrics (foundation for many downstream plots)
    def _run_combined_final_metrics():
        mod = importlib.import_module('analysis_tools.combined_final_metrics')
        mod.process_scenarios(recompute=recompute)

    steps.append(("combined_final_metrics", _run_combined_final_metrics, {}))

    # 2. Movement analysis (can be heavy) – optional
    if include_movement:
        steps.append(movement_step)

    # 3. Stability patterns
    def _run_stability():
        mod = importlib.import_module('analysis_tools.analyze_stability_patterns')
        mod.analyze_stability()
    steps.append(("analyze_stability_patterns", _run_stability, {}))

    # 4. Convergence patterns & speed
    def _run_convergence():
        import importlib as _il
        _il.import_module('analysis_tools.convergence_patterns_and_speed')  # code runs on import
    steps.append(("convergence_patterns_and_speed", _run_convergence, {}))

    # 5. Per-metric panels
    def _run_per_metric_panels():
        mod = importlib.import_module('analysis_tools.per_metric_panels')
        mod.main()
    steps.append(("per_metric_panels", _run_per_metric_panels, {}))

    # 6. Segregation metrics comparison (depends on combined_final_metrics output)
    def _run_segregation_metrics_comparison():
        import importlib as _il
        _il.import_module('analysis_tools.segregation_metrics_comparison')  # executes on import
    steps.append(("segregation_metrics_comparison", _run_segregation_metrics_comparison, {}))

    return _execute_steps(steps, verbose=verbose)


def _execute_steps(steps, verbose: bool = True):
    import time
    import traceback

    results = []  # (name, status, seconds, error)
    start_all = time.time()
    for name, func, kwargs in steps:
        t0 = time.time()
        status = 'ok'
        err_msg = ''
        if verbose:
            print(f"\n[RUN] {name} ...")
        try:
            func(**kwargs)
        except Exception as e:
            status = 'fail'
            err_msg = f"{e.__class__.__name__}: {e}"
            if verbose:
                traceback.print_exc(limit=2)
        dt = time.time() - t0
        results.append((name, status, dt, err_msg))
        if verbose:
            print(f"[DONE] {name} status={status} time={dt:.1f}s")

    total = time.time() - start_all
    if verbose:
        print("\nSUMMARY:")
        width = max(len(r[0]) for r in results) + 2
        for name, status, dt, err in results:
            line = f"  {name.ljust(width)} {status.upper():5s} {dt:7.1f}s"
            if err:
                line += f"  - {err}"[:120]
            print(line)
        print(f"\nTotal time: {total:.1f}s for {len(results)} steps")
    return results


def _parse_args_and_run():
    p = argparse.ArgumentParser(description="Run all PancsVriend analyses from a single entry point.")
    p.add_argument('--no-recompute', action='store_true', default=False, help='Do not force recomputation of metrics in combined_final_metrics.')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--movement-only', action='store_true', default=False, help='Run only movement analysis.')
    p.add_argument('--include-movement', action='store_true', default=False, help='Include movement analysis (off by default).')
    p.add_argument('--output-folder', type=str, default='reports', help='Base directory to write analysis artifacts (default: reports).')
    p.add_argument('--quiet', action='store_true', help='Suppress verbose step output.')
    args = p.parse_args()
    run_all_analyses(
        recompute=not args.no_recompute,
        movement_only=args.movement_only,
        include_movement=args.include_movement,
        verbose=not args.quiet,
        output_folder=args.output_folder,
    )


if __name__ == '__main__':
    _parse_args_and_run()
