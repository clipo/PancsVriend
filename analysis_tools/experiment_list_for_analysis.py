"""
Centralized list of experiments (scenarios), human-friendly labels, and colors
to be reused across analysis scripts.

Usage:
    from analysis_tools.experiment_list_for_analysis import (
        SCENARIOS, SCENARIO_LABELS, SCENARIO_COLORS,
    )

Notes:
    - Keys in SCENARIOS are the canonical scenario identifiers used across the
      codebase (e.g., 'baseline', 'race_white_black', ...).
    - Values in SCENARIOS are the folder names under 'experiments/'.
    - SCENARIO_LABELS provides readable labels for plots.
    - SCENARIO_COLORS provides a consistent color per scenario for plots.
"""

from __future__ import annotations

# Canonical scenario -> experiment folder mapping
SCENARIOS = {
    # Baselines
    # 'baseline': 'llm_baseline_20250703_101243',
    'llm_baseline': 'llm_baseline_20250703_101243',
    'mech_baseline': 'baseline_20250729_174459',

    # Social contexts
    'green_yellow': 'llm_green_yellow_20250912_072712',
    'ethnic_asian_hispanic': 'llm_ethnic_asian_hispanic_20250713_221759',
    'income_high_low': 'llm_income_high_low_20251006_150254',
    # 'economic_high_working': 'llm_economic_high_working_20250728_220134',
    'political_liberal_conservative': 'llm_political_liberal_conservative_20250724_154733',
    'race_white_black': 'llm_race_white_black_20250718_195455',
}

# Human-friendly labels for plots
SCENARIO_LABELS = {
    # 'baseline': 'Baseline (Control)',
    'llm_baseline': 'Color (Red/Blue)',
    'mech_baseline': 'Mechanical Baseline',
    'political_liberal_conservative': 'Political (Liberal/Conservative)',
    'ethnic_asian_hispanic': 'Ethnic (Asian/Hispanic)',
    'race_white_black': 'Racial (White/Black)',
    'income_high_low': 'Economic (High/Low Income)',
    # 'economic_high_working': 'Economic (High/Working)',
    'green_yellow': 'Color (Green/Yellow)',
}
SCENARIO_ORDER = [
    'mech_baseline',
    'llm_baseline',
    'green_yellow',
    'political_liberal_conservative',
    'race_white_black',
    'ethnic_asian_hispanic',
    # 'economic_high_working',
    'income_high_low',
]
# Consistent colors for scenarios (hex codes)
SCENARIO_COLORS = {
    'mech_baseline': '#34495E',       # Dark slate - contrasts with LLM baseline
    'llm_baseline': '#E74C3C',        # Light gray - neutral baseline
    'green_yellow': '#27AE60',        # Emerald green - vibrant, nature-inspired
    'political_liberal_conservative': '#E67E22', # Warm orange - distinctive, accessible
    'race_white_black': '#95A5A6',    # Vivid red - high contrast, attention-grabbing
    'ethnic_asian_hispanic': '#9B59B6', # Purple - distinct from red/blue politics
    # 'economic_high_working': '#3498DB', # Bright blue - professional, clear
    'income_high_low': '#3498DB', # Bright blue - professional, clear
}

# ---------------------------------------------------------------------------
# Orchestrator to run all analysis scripts from one place
# ---------------------------------------------------------------------------
def run_all_analyses(
    recompute: bool = True,
    movement_only: bool = False,
    include_movement: bool = False,
    verbose: bool = True,
):
    """Run the suite of analysis scripts in a recommended order.

    Parameters
    ----------
    recompute : bool
        Passed to combined_final_metrics to (re)compute metrics if missing.
    movement_only : bool
        If True, only run movement analysis (analyze_agent_movement).
    include_movement : bool
        If True, include movement analysis (disabled by default for speed).
    verbose : bool
        Print progress messages.
    """
    import importlib
    import sys
    from pathlib import Path

    # Ensure analysis_tools on path if executing from repo root
    at_path = Path(__file__).resolve().parent
    if str(at_path) not in sys.path:
        sys.path.append(str(at_path))

    steps = []  # (name, callable, kwargs)

    # 1. Combined metrics (foundation for many downstream plots)
    def _run_combined_final_metrics():
        mod = importlib.import_module('analysis_tools.combined_final_metrics')
        mod.process_scenarios(recompute=recompute)

    steps.append(("combined_final_metrics", _run_combined_final_metrics, {}))

    # 2. Movement analysis (can be heavy) – optional
    if include_movement:
        def _run_movement():
            mod = importlib.import_module('analysis_tools.analyze_agent_movement')
            # Use default CLI behavior (reports/movement_analysis)
            # We'll call main() with adjusted argv semantics
            import sys

            original_argv = sys.argv[:]
            module_path = getattr(mod, '__file__', None)
            prog_name = str(Path(module_path)) if module_path else mod.__name__
            try:
                sys.argv = [prog_name]
                result = mod.main()
            finally:
                sys.argv = original_argv

            if isinstance(result, int) and result != 0:
                raise RuntimeError(f"movement analysis exited with code {result}")
        steps.append(("analyze_agent_movement", _run_movement, {}))

    if movement_only:
        return _execute_steps(steps, verbose=verbose)

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

    # # 5. Rate of change detailed
    # def _run_rate_of_change():
    #     mod = importlib.import_module('analysis_tools.rate_of_change')
    #     mod.analyze_dynamics()
    # steps.append(("rate_of_change", _run_rate_of_change, {}))

    # # 6. Rate of change clear narrative
    # def _run_rate_of_change_clear():
    #     mod = importlib.import_module('analysis_tools.rate_of_change_clear')
    #     mod.create_clear_dynamics_visualization()
    # steps.append(("rate_of_change_clear", _run_rate_of_change_clear, {}))

    # 7. Per-metric panels
    def _run_per_metric_panels():
        mod = importlib.import_module('analysis_tools.per_metric_panels')
        mod.main()
    steps.append(("per_metric_panels", _run_per_metric_panels, {}))

    # 8. Segregation metrics comparison (depends on combined_final_metrics output)
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
    import argparse
    p = argparse.ArgumentParser(description="Run all PancsVriend analyses from a single entry point.")
    p.add_argument('--no-recompute', action='store_true', help='Do not force recomputation of metrics in combined_final_metrics.')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--movement-only', action='store_true', help='Run only movement analysis.')
    p.add_argument('--include-movement', action='store_true', help='Include movement analysis (off by default).')
    p.add_argument('--quiet', action='store_true', help='Suppress verbose step output.')
    args = p.parse_args()
    run_all_analyses(
        recompute=not args.no_recompute,
        movement_only=args.movement_only,
        include_movement=args.include_movement,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    _parse_args_and_run()
