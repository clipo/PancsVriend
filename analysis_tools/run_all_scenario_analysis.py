"""Run all PancsVriend analyses using the shared scenario metadata.

Quick start (from repo root):
    python analysis_tools/run_all_scenario_analysis.py \
        [--no-recompute] [--include-movement | --movement-only] \
        [--output-folder <path>] [--llm-model <name>] \
        [--manifest-file <path/to/run_manifest.json>] [--quiet]

Flags:
    --no-recompute      Skip forced metric recompute; use existing CSVs if present.
    --include-movement  Add movement analysis (slower; needs move_logs + states).
    --movement-only     Run only movement analysis (overrides other steps).
    --output-folder     Base reports directory (default: reports). Respects
                        env PANCSVRIEND_REPORTS_DIR if set.
    --llm-model          Filter experiments to the specified LLM model. When set
                        and --output-folder is omitted, outputs go to
                        reports_{modelname} (sanitized for filesystem safety).
    --manifest-file     Analyze exactly the experiments in a run manifest JSON.
                        Also writes manifest-based experiment parameter reports.
    --quiet             Suppress per-step progress; summary still printed.

Manifest report outputs (when --manifest-file is used):
    experiment_list_{model}.txt
    experiment_details_{model}.txt
    experiment_details_{model}.json

Pipeline order (when not movement-only):
    0) repair_stored_metrics (unless --no-recompute; --force-recompute rebuilds all)
    1) dissimilarity_index_over_time
    2) run_summary (per-experiment run_summary.csv + run_summary_by_run.csv)
    2a) anova_by_metric
    2b) normality_tests
    3) analyze_agent_movement (optional via --include-movement)
    4) analyze_stability_patterns
    5) convergence_patterns_and_speed
    6) movement_decision_counts
    7) per_metric_panels
    8) segregation_metrics_comparison

Config lives in `analysis_tools/experiment_list_for_analysis.py`
(scenarios, labels, colors). This file only orchestrates execution.
"""

from __future__ import annotations

import importlib
import argparse
import json
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass  # Older Pythons or non-tty streams

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from analysis_tools.output_paths import set_reports_dir

import experiment_list_for_analysis as experiment_list
import run_files


def run_all_analyses(
    recompute: bool = True,
    movement_only: bool = False,
    include_movement: bool = False,
    verbose: bool = True,
    output_folder: Union[str, Path] = "reports",
    llm_model: Optional[str] = None,
    force: bool = False,
):
    """Run the suite of analysis scripts in a recommended order.

    recompute: verify each selected experiment's stored metrics_history
    against its step logs first and rebuild only the runs that disagree
    (Simulation.repair_stored_metrics). force: rebuild every run from its
    frames regardless (the pre-2026-09-05 behaviour of every pass).
    """
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

        only_list = list(experiment_list.SCENARIOS.values())
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

    # 0. Stored metrics_history verified against the run record; only runs
    #    that disagree are rebuilt (everything downstream reads that file).
    if recompute or force:
        def _run_repair():
            repair_selected_experiments(force=force, verbose=verbose)
        steps.append(("repair_stored_metrics", _run_repair, {}))

    # 1. Dissimilarity index over time (stored column; frames for legacy runs)
    def _run_dissimilarity_index():
        mod = importlib.import_module('analysis_tools.dissimilarity_index_over_time')
        mod.run_all(Path('experiments'), recompute=recompute)

    steps.append(("dissimilarity_index_over_time", _run_dissimilarity_index, {}))

    # 2. Per-run summary: convergence step + every metric at the final step,
    #    one row per run. Each experiment folder gets its own run_summary.csv
    #    (rebuilt here so pre-2026-09-01 experiments and resume placeholders
    #    are repaired), and the selected experiments are rolled up into a
    #    single run_summary_by_run.csv for this model — the file every
    #    cross-scenario step below reads.
    def _run_run_summary():
        build_run_summaries_for_selection(reports_dir, verbose=verbose)

    steps.append(("run_summary", _run_run_summary, {}))

    # 2a. One-way ANOVA per metric across scenarios (from the roll-up)
    def _run_anova():
        mod = importlib.import_module('analysis_tools.anova_by_metric')
        mod.anova_by_metric(scenario_order=list(experiment_list.SCENARIOS))

    steps.append(("anova_by_metric", _run_anova, {}))

    # 2b. Normality diagnostics (consumes run_summary_by_run.csv): Shapiro-Wilk
    #     table + Q-Q/histograms per metric. Diagnostic only since 2026-09-05;
    #     the ranking table's paired t does not depend on it.
    def _run_normality_tests():
        mod = importlib.import_module('analysis_tools.normality_tests')
        mod.run_from_summary_csv()

    steps.append(("normality_tests", _run_normality_tests, {}))

    # 2. Scenario ranking + significance table for the selected model run
    def _run_scenario_ranking_significance_table():
        _write_single_model_scenario_ranking_table(
            reports_dir=reports_dir,
            llm_model=llm_model,
        )

    steps.append(("scenario_ranking_significance", _run_scenario_ranking_significance_table, {}))

    # 3. Movement analysis (can be heavy) – optional
    if include_movement:
        steps.append(movement_step)

    # 4. Stability patterns
    def _run_stability():
        mod = importlib.import_module('analysis_tools.analyze_stability_patterns')
        mod.analyze_stability()
    steps.append(("analyze_stability_patterns", _run_stability, {}))

    # 5. Convergence patterns & speed
    def _run_convergence():
        import importlib as _il
        mod = _il.import_module('analysis_tools.convergence_patterns_and_speed')
        _il.reload(mod)  # code runs on import; reload to regenerate per-model outputs
    steps.append(("convergence_patterns_and_speed", _run_convergence, {}))

    # 6. Movement decision counts (requires move logs)
    def _run_movement_decision_counts():
        mod = importlib.import_module('analysis_tools.movement_decision_counts')
        mod.main(recompute=recompute)
    steps.append(("movement_decision_counts", _run_movement_decision_counts, {}))

    # 7. Per-metric panels
    def _run_per_metric_panels():
        mod = importlib.import_module('analysis_tools.per_metric_panels')
        mod.main()
    steps.append(("per_metric_panels", _run_per_metric_panels, {}))

    # 8. Segregation metrics comparison (depends on run_summary_by_run.csv)
    def _run_segregation_metrics_comparison():
        import importlib as _il
        mod = _il.import_module('analysis_tools.segregation_metrics_comparison')
        _il.reload(mod)  # executes on import; reload to regenerate per-model outputs
    steps.append(("segregation_metrics_comparison", _run_segregation_metrics_comparison, {}))

    return _execute_steps(steps, verbose=verbose)


def repair_selected_experiments(experiments_dir: Union[str, Path] = 'experiments',
                                force: bool = False, verbose: bool = True) -> Dict[str, list]:
    """Verify (or, with force, fully rebuild) every selected experiment's
    stored metrics_history. Returns {experiment folder: run ids rebuilt}."""
    from base_simulation import Simulation

    rebuilt: Dict[str, list] = {}
    experiments_root = Path(experiments_dir)
    for scenario_key, folder in dict(experiment_list.SCENARIOS).items():
        exp_dir = experiments_root / folder
        if not (exp_dir / 'move_logs').is_dir():
            if verbose:
                print(f"[repair] Skipping '{scenario_key}': {exp_dir / 'move_logs'} not found")
            continue
        try:
            if force:
                Simulation.load_and_analyze_results(str(exp_dir), force_recompute=True)
                rebuilt[folder] = run_files.list_run_ids(str(exp_dir))
            else:
                rebuilt[folder] = Simulation.repair_stored_metrics(str(exp_dir))
        except Exception as exc:
            print(f"[repair] WARN {scenario_key}: {exc}")
    return rebuilt


def build_run_summaries_for_selection(reports_dir: Path,
                                      experiments_dir: Union[str, Path] = 'experiments',
                                      verbose: bool = True) -> pd.DataFrame:
    """run_summary.csv per selected experiment + one combined CSV for the model.

    The selection is whatever experiment_list.SCENARIOS currently holds, which
    _update_scenarios_for_run has already narrowed to this model/manifest — the
    same source the movement step reads. Summaries are rebuilt rather than
    read, so experiments predating run_summary.csv and runs whose convergence
    rows are resume placeholders are repaired in place.
    """
    from run_summary import combine_run_summaries, write_run_summary

    experiments_root = Path(experiments_dir)
    output_dirs: List[str] = []
    scenario_key_by_experiment: Dict[str, str] = {}
    for scenario_key, folder in dict(experiment_list.SCENARIOS).items():
        exp_dir = experiments_root / folder
        if not exp_dir.is_dir():
            print(f"[run_summary] Skipping '{scenario_key}': {exp_dir} not found")
            continue
        write_run_summary(str(exp_dir), verbose=verbose)
        output_dirs.append(str(exp_dir))
        scenario_key_by_experiment[exp_dir.name] = scenario_key

    combined = combine_run_summaries(output_dirs, verbose=verbose)
    if not combined.empty:
        # The per-experiment 'scenario' comes from that experiment's
        # config.json ('baseline'); the analysis keys are what distinguishes
        # llm_baseline from mech_baseline downstream.
        combined.insert(
            1, 'scenario_key',
            combined['experiment'].map(scenario_key_by_experiment).fillna(combined['scenario']))

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / 'run_summary_by_run.csv'
    combined.to_csv(out_path, index=False)
    print(f"[run_summary] Wrote {len(combined)} run(s) to {out_path}")
    return combined


def _execute_steps(steps, verbose: bool = True):
    import time
    import traceback

    results = []  # list of dict(name,status,seconds,error,warnings)
    start_all = time.time()
    for name, func, kwargs in steps:
        t0 = time.time()
        status = 'ok'
        err_msg = ''
        step_warnings = []
        if verbose:
            print(f"\n[RUN] {name} ...")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('default')
            try:
                func(**kwargs)
            except Exception as e:
                status = 'fail'
                err_msg = f"{e.__class__.__name__}: {e}"
                if verbose:
                    traceback.print_exc(limit=2)
        for warn in caught:
            msg = f"{warn.category.__name__}: {warn.message}"
            step_warnings.append(msg)
        dt = time.time() - t0
        results.append({
            'name': name,
            'status': status,
            'seconds': dt,
            'error': err_msg,
            'warnings': step_warnings,
        })
        if verbose:
            print(f"[DONE] {name} status={status} time={dt:.1f}s")

    total = time.time() - start_all
    if verbose:
        print("\nSUMMARY:")
        width = max(len(r['name']) for r in results) + 2
        for entry in results:
            line = f"  {entry['name'].ljust(width)} {entry['status'].upper():5s} {entry['seconds']:7.1f}s"
            if entry['error']:
                line += f"  - {entry['error']}"[:120]
            print(line)
        warning_count = sum(len(r['warnings']) for r in results)
        if warning_count:
            print("\nWARNINGS BY STEP:")
            for entry in results:
                if not entry['warnings']:
                    continue
                print(f"  {entry['name']} ({len(entry['warnings'])} warning(s))")
                for warn_msg in entry['warnings']:
                    print(f"    - {warn_msg}")
        failures = [r for r in results if r['status'] != 'ok']
        if failures:
            print("\nERROR SUMMARY:")
            for entry in failures:
                print(f"  {entry['name']}: {entry['error']}")
        print(f"\nTotal time: {total:.1f}s for {len(results)} steps")
    return results


def _parse_args_and_run():
    p = argparse.ArgumentParser(description="Run all PancsVriend analyses from a single entry point.")
    p.add_argument('--no-recompute', action='store_true', default=False, help='Skip verifying stored metrics_history against the step logs.')
    p.add_argument('--force-recompute', action='store_true', default=False, help="Rebuild every run's metrics_history from its frames (slow; the pre-2026-09-05 default).")
    group = p.add_mutually_exclusive_group()
    group.add_argument('--movement-only', action='store_true', default=False, help='Run only movement analysis.')
    p.add_argument('--include-movement', action='store_true', default=False, help='Include movement analysis (off by default).')
    p.add_argument('--output-folder', type=str, default=None, help='Base directory to write analysis artifacts (default: reports).')
    p.add_argument('--llm-model', type=str, default=None, help='Filter experiments to a specific LLM model name.')
    p.add_argument('--manifest-file', type=str, default=None, help='Use an explicit simulation run manifest JSON to select experiments for analysis.')
    p.add_argument('--quiet', action='store_true', help='Suppress verbose step output.')
    args = p.parse_args()

    if args.manifest_file and args.llm_model and args.llm_model.strip().lower() == "all":
        p.error("--manifest-file cannot be combined with --llm-model all")

    experiments_dir = Path('experiments')

    if args.manifest_file:
        selected, matching, unused, manifest_meta, manifest_payload = _select_experiments_from_manifest(
            Path(args.manifest_file),
            experiments_dir,
        )
        if not selected:
            raise RuntimeError(
                f"Manifest '{args.manifest_file}' did not contain any valid successful experiments to analyze."
            )
        output_folder = _resolve_output_folder(
            args.output_folder,
            args.llm_model or manifest_meta.get("llm_model"),
        )
        _update_scenarios_for_run(selected)
        _write_experiment_list_report(
            Path(output_folder),
            args.llm_model or str(manifest_meta.get("llm_model") or "manifest"),
            matching,
            selected,
            unused,
        )
        _write_manifest_experiment_details_report(
            reports_dir=Path(output_folder),
            llm_model=args.llm_model or str(manifest_meta.get("llm_model") or "manifest"),
            selected=selected,
            manifest_meta=manifest_meta,
            manifest_payload=manifest_payload,
        )
        _apply_scenarios_to_plot()
        run_all_analyses(
            recompute=not args.no_recompute,
            force=args.force_recompute,
            movement_only=args.movement_only,
            include_movement=args.include_movement,
            verbose=not args.quiet,
            output_folder=output_folder,
            llm_model=args.llm_model or str(manifest_meta.get("llm_model") or "manifest"),
        )
        return

    if args.llm_model and args.llm_model.strip().lower() == "all":
        models = _collect_llm_models(experiments_dir)
        models = _filter_models_for_all(models)
        if not models:
            if not args.quiet:
                print("No LLM models found in experiments; nothing to run.")
            return
        for model in models:
            output_folder = _resolve_output_folder(args.output_folder, model)
            selected, matching, unused = _select_experiments_for_model(
                experiments_dir,
                model,
            )
            _update_scenarios_for_run(selected)
            _write_experiment_list_report(
                Path(output_folder),
                model,
                matching,
                selected,
                unused,
            )
            _apply_scenarios_to_plot()
            run_all_analyses(
                recompute=not args.no_recompute,
                movement_only=args.movement_only,
                include_movement=args.include_movement,
                verbose=not args.quiet,
                output_folder=output_folder,
                llm_model=model,
            )
        _run_llm_model_comparison_plots(experiments_dir, Path("reports_all_llm_models"), quiet=args.quiet)
        return

    output_folder = _resolve_output_folder(args.output_folder, args.llm_model)
    if args.llm_model:
        selected, matching, unused = _select_experiments_for_model(
            experiments_dir,
            args.llm_model,
        )
        _update_scenarios_for_run(selected)
        _write_experiment_list_report(
            Path(output_folder),
            args.llm_model,
            matching,
            selected,
            unused,
        )
    _apply_scenarios_to_plot()
    run_all_analyses(
        recompute=not args.no_recompute,
        movement_only=args.movement_only,
        include_movement=args.include_movement,
        verbose=not args.quiet,
        output_folder=output_folder,
        llm_model=args.llm_model,
    )


def _write_single_model_scenario_ranking_table(
    reports_dir: Path,
    llm_model: Optional[str],
) -> None:
    from analysis_tools.anova_by_metric import load_final_metrics

    combined_df = load_final_metrics(reports_dir)
    if combined_df.empty:
        raise RuntimeError(f"run_summary_by_run.csv is empty in {reports_dir}")
    if "scenario" not in combined_df.columns:
        raise RuntimeError("run_summary_by_run.csv is missing required 'scenario' column")

    metrics = [
        metric
        for metric in [
            "dissimilarity_index",
            "clusters",
            "switch_rate",
            "distance",
            "mix_deviation",
            "share",
            "ghetto_rate",
        ]
        if metric in combined_df.columns
    ]
    if not metrics:
        raise RuntimeError("No ranking metrics found in run_summary_by_run.csv")

    model_display = (llm_model or "unknown-model").strip() or "unknown-model"
    safe_model = _sanitize_model_for_path_component(model_display)
    output_csv = reports_dir / f"segregation_scenario_rankings_{safe_model}.csv"
    output_md = reports_dir / f"segregation_scenario_rankings_{safe_model}.md"

    # Adjacent-pair comparison: paired t-test on the per-run differences,
    # matched by run_id (run k shares its seed, hence its initial grid,
    # across scenarios). Until 2026-09-05 this was an UNPAIRED Welch t /
    # Mann-Whitney U gated on normality_tests.csv; the pairing removes the
    # shared initial-configuration variance, and the normality gate was
    # dropped for the reasons in cross_model_vf_comparison.rank_and_test.
    # The normality step still runs and its Q-Q figures stay as diagnostics.
    def _stars(p: float) -> str:
        return ("***" if p < 0.001 else "**" if p < 0.01
                else "*" if p < 0.05 else "")

    rows: List[dict] = []
    markdown_lines: List[str] = [
        f"# Segregation Ranking by Scenario ({model_display})",
        "",
        "Each scenario is compared with the NEXT-ranked one with a paired",
        "t-test on the per-run differences (runs matched by run_id: the same",
        "seed, hence the same initial grid, in every scenario), and with its",
        "own INITIAL grids — a random allocation, i.e. chance — by the same",
        "paired t (final - initial). 'Excess' is the mean final - initial in",
        "the segregating direction; a scenario not significantly above its",
        "initial grids shows no segregation on that metric.",
        "Significance levels: `***` p<0.001, `**` p<0.01, `*` p<0.05.",
        "",
        "Metrics are ordered with dissimilarity first when available.",
        "",
    ]

    for metric in metrics:
        scenario_values: Dict[str, np.ndarray] = {}
        scenario_series: Dict[str, pd.Series] = {}      # indexed by run_id, for pairing
        for scenario, scenario_df in combined_df.groupby("scenario"):
            vals = scenario_df[metric].dropna().to_numpy(dtype=float)
            if len(vals) > 0:
                scenario_values[str(scenario)] = vals
                if "run_id" in scenario_df.columns:
                    scenario_series[str(scenario)] = (
                        scenario_df.dropna(subset=[metric]).set_index("run_id")[metric])

        if not scenario_values:
            continue

        ranking = sorted(
            scenario_values.items(),
            key=lambda item: float(np.mean(item[1])),
            reverse=True,
        )

        markdown_lines.extend([
            f"## {metric}",
            "",
            "| Rank | Scenario | Mean | Std dev | Runs | Sig. vs next | p-value vs next | Test | Excess vs chance | Sig. vs chance | p-value vs chance |",
            "|---:|---|---:|---:|---:|:---:|---:|---|---:|:---:|---:|",
        ])

        init_col = f"initial_{metric}"
        for idx, (scenario, values) in enumerate(ranking):
            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            n_runs = int(len(values))

            # Chance as a paired scenario: final vs the run's own initial grid.
            chance_excess = chance_p = None
            chance_marker = ""
            sdf = combined_df[combined_df["scenario"] == scenario]
            if init_col in sdf.columns:
                fin = sdf[metric].to_numpy(dtype=float)
                ini = sdf[init_col].to_numpy(dtype=float)
                ok = ~np.isnan(fin) & ~np.isnan(ini)
                if ok.sum() >= 2:
                    direction = -1 if metric in ("clusters", "switch_rate") else 1
                    diff = fin[ok] - ini[ok]
                    chance_excess = float(direction * diff.mean())
                    chance_p = 1.0 if np.allclose(diff, 0.0) else float(
                        stats.ttest_rel(fin[ok], ini[ok]).pvalue)
                    chance_marker = _stars(chance_p) if chance_excess > 0 else ""

            sig_marker = ""
            p_value = None
            test_used = None
            if idx + 1 < len(ranking):
                next_name, next_values = ranking[idx + 1]
                if len(values) >= 2 and len(next_values) >= 2:
                    try:
                        a = scenario_series.get(scenario)
                        b = scenario_series.get(next_name)
                        common = (a.index.intersection(b.index)
                                  if a is not None and b is not None else [])
                        if len(common) >= 2:
                            diff = (a.loc[common] - b.loc[common]).to_numpy(dtype=float)
                            if np.allclose(diff, 0.0):
                                test_used, p_value = "degenerate (all differences zero)", 1.0
                            else:
                                test_used = "paired t"
                                _, p_value = stats.ttest_rel(a.loc[common], b.loc[common],
                                                             nan_policy="omit")
                        else:                       # no run_id to pair on (legacy roll-up)
                            test_used = "Welch t (unpaired)"
                            _, p_value = stats.ttest_ind(values, next_values, equal_var=False)
                        p_value = float(p_value)
                        sig_marker = _stars(p_value)
                    except Exception:
                        p_value = None
                        test_used = None

            scenario_label = experiment_list.SCENARIO_LABELS.get(
                scenario,
                scenario.replace("_", " ").title(),
            )
            rows.append({
                "llm_model": model_display,
                "metric": metric,
                "scenario": scenario,
                "scenario_label": scenario_label,
                "rank": idx + 1,
                "mean_value": round(mean_value, 6),
                "std_value": round(std_value, 6),
                "n_runs": n_runs,
                "significant_vs_next": sig_marker,
                "p_value_vs_next": None if p_value is None else round(float(p_value), 6),
                "test_vs_next": test_used,
                "excess_vs_chance": None if chance_excess is None else round(chance_excess, 6),
                "significant_vs_chance": chance_marker,
                "p_value_vs_chance": None if chance_p is None else round(chance_p, 6),
            })

            p_value_display = "" if p_value is None else f"{p_value:.6f}"
            excess_display = "" if chance_excess is None else f"{chance_excess:+.4f}"
            chance_p_display = "" if chance_p is None else f"{chance_p:.6f}"
            markdown_lines.append(
                f"| {idx + 1} | {scenario_label} | {mean_value:.4f} | {std_value:.4f} | {n_runs} | {sig_marker} | {p_value_display} | {test_used or ''} | {excess_display} | {chance_marker} | {chance_p_display} |"
            )

        markdown_lines.append("")

    if not rows:
        raise RuntimeError("No per-metric scenario rankings could be computed")

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    output_md.write_text("\n".join(markdown_lines), encoding="utf-8")


def _resolve_output_folder(output_folder: Optional[str], llm_model: Optional[str]) -> str:
    if output_folder:
        return output_folder
    if llm_model:
        safe_model = _sanitize_model_for_path_component(llm_model)
        return f"reports_{safe_model}"
    return "reports"


def _normalize_model_name(name: str) -> str:
    return name.strip().lower()


def _sanitize_model_for_path_component(name: str) -> str:
    # Windows-disallowed chars for a path segment: < > : " / \ | ? * and controls.
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '-', name.strip())
    sanitized = sanitized.rstrip(' .')
    sanitized = re.sub(r'-{2,}', '-', sanitized)
    return sanitized or "unknown-model"


def _scenario_key_from_config(folder_name: str, scenario: str) -> str:
    if scenario == "baseline" and folder_name.startswith("llm_baseline"):
        return "llm_baseline"
    return scenario


def _select_experiments_for_model(
    experiments_dir: Path,
    llm_model: str,
) -> Tuple[Dict[str, str], List[dict], List[dict]]:
    model_key = _normalize_model_name(llm_model)
    matching: List[dict] = []

    for entry in sorted(experiments_dir.iterdir()):
        if not entry.is_dir():
            continue
        config_path = entry / "config.json"
        if not config_path.exists():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        config_model = payload.get("llm_model")
        if not config_model:
            continue
        if _normalize_model_name(config_model) != model_key:
            continue
        scenario = payload.get("scenario")
        if not scenario:
            continue
        scenario_key = _scenario_key_from_config(entry.name, str(scenario))
        matching.append({
            "folder": entry.name,
            "scenario": scenario_key,
            "raw_scenario": scenario,
            "timestamp": payload.get("timestamp"),
            "llm_model": config_model,
        })

    selected: Dict[str, str] = {}
    unused: List[dict] = []
    for item in matching:
        scenario_key = item["scenario"]
        if scenario_key in selected:
            unused.append(item)
            continue
        selected[scenario_key] = item["folder"]

    mech_folder = _find_mech_baseline_experiment(experiments_dir)
    if mech_folder and "mech_baseline" not in selected:
        selected["mech_baseline"] = mech_folder
        matching.append({
            "folder": mech_folder,
            "scenario": "mech_baseline",
            "raw_scenario": "mech_baseline",
            "timestamp": None,
            "llm_model": "mechanical",
        })

    return selected, matching, unused


def _select_experiments_from_manifest(
    manifest_path: Path,
    experiments_dir: Path,
) -> Tuple[Dict[str, str], List[dict], List[dict], Dict[str, str], Dict[str, Any]]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Unable to read manifest file '{manifest_path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Manifest file '{manifest_path}' is not valid JSON: {exc}") from exc

    selected_raw = payload.get("selected_experiments")
    if not isinstance(selected_raw, dict):
        raise RuntimeError(
            f"Manifest file '{manifest_path}' is missing a valid 'selected_experiments' mapping."
        )

    selected: Dict[str, str] = {}
    matching: List[dict] = []
    unused: List[dict] = []

    for scenario_key, folder_name in selected_raw.items():
        if not isinstance(scenario_key, str) or not isinstance(folder_name, str):
            continue
        folder = folder_name.strip()
        scenario = scenario_key.strip()
        if not folder or not scenario:
            continue
        if not (experiments_dir / folder).exists():
            raise RuntimeError(
                f"Manifest references experiment folder '{folder}' for scenario '{scenario}', "
                f"but it does not exist under '{experiments_dir}'."
            )
        mapped_scenario = _scenario_key_from_config(folder, scenario)
        selected[mapped_scenario] = folder
        matching.append({
            "folder": folder,
            "scenario": mapped_scenario,
            "raw_scenario": scenario,
            "timestamp": payload.get("created_at"),
            "llm_model": payload.get("llm_model") or "manifest",
        })

    manifest_meta = {
        "llm_model": str(payload.get("llm_model") or ""),
        "created_at": str(payload.get("created_at") or ""),
        "manifest_path": str(manifest_path),
    }
    return selected, matching, unused, manifest_meta, payload


def _find_mech_baseline_experiment(experiments_dir: Path) -> Optional[str]:
    configured = experiment_list.SCENARIOS.get("mech_baseline")
    if configured:
        candidate = experiments_dir / configured
        if candidate.exists():
            return configured

    for entry in sorted(experiments_dir.iterdir()):
        if not entry.is_dir():
            continue
        config_path = entry / "config.json"
        if not config_path.exists():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        scenario = payload.get("scenario")
        if not scenario:
            continue
        scenario_name = str(scenario).strip().lower()
        if scenario_name == "mech_baseline":
            return entry.name
        if scenario_name == "baseline" and not entry.name.startswith("llm_baseline"):
            return entry.name
    return None


def _collect_llm_models(experiments_dir: Path) -> List[str]:
    models: List[str] = []
    seen = set()
    for entry in sorted(experiments_dir.iterdir()):
        if not entry.is_dir():
            continue
        config_path = entry / "config.json"
        if not config_path.exists():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        config_model = payload.get("llm_model")
        if not config_model:
            continue
        key = _normalize_model_name(str(config_model))
        if key in seen:
            continue
        seen.add(key)
        models.append(str(config_model))
    return models


def _filter_models_for_all(models: List[str]) -> List[str]:
    allowed = getattr(experiment_list, "LLM_MODELS_TO_PLOT", None)
    if not allowed:
        return models
    allowed_map = {
        _normalize_model_name(str(name)): str(name)
        for name in allowed
    }
    filtered = []
    for model in models:
        key = _normalize_model_name(model)
        if key in allowed_map:
            filtered.append(allowed_map[key])
    return filtered


def _run_llm_model_comparison_plots(
    experiments_dir: Path,
    output_dir: Path,
    quiet: bool = False,
) -> None:
    if not quiet:
        print(f"\n[RUN] llm_model_comparison_plots -> {output_dir}")
    try:
        mod = importlib.import_module("analysis_tools.llm_model_comparison_plots")
        mod.generate_llm_model_comparisons(experiments_dir, output_dir)
    except Exception as exc:
        if not quiet:
            print(f"[WARN] LLM model comparison plots failed: {exc}")


def _update_scenarios_for_run(selected: Dict[str, str]) -> None:
    experiment_list.SCENARIOS.clear()
    experiment_list.SCENARIOS.update(selected)

    for scenario in selected.keys():
        if scenario not in experiment_list.SCENARIO_LABELS:
            experiment_list.SCENARIO_LABELS[scenario] = scenario.replace("_", " ").title()
        if scenario not in experiment_list.SCENARIO_COLORS:
            experiment_list.SCENARIO_COLORS[scenario] = "#999999"


def _apply_scenarios_to_plot() -> None:
    scenarios_to_plot = getattr(experiment_list, "SCENARIOS_TO_PLOT", None)
    if not scenarios_to_plot:
        return

    allowed = set(scenarios_to_plot)

    for key in list(experiment_list.SCENARIOS.keys()):
        if key not in allowed:
            experiment_list.SCENARIOS.pop(key, None)
    for key in list(experiment_list.SCENARIO_LABELS.keys()):
        if key not in allowed:
            experiment_list.SCENARIO_LABELS.pop(key, None)
    for key in list(experiment_list.SCENARIO_COLORS.keys()):
        if key not in allowed:
            experiment_list.SCENARIO_COLORS.pop(key, None)

    # Intentionally do NOT mutate SCENARIO_ORDER here.
    # Plot ordering must always come from the canonical SCENARIO_ORDER list,
    # while SCENARIOS_TO_PLOT only controls which scenarios are included.


def _write_experiment_list_report(
    reports_dir: Path,
    llm_model: str,
    matching: List[dict],
    selected: Dict[str, str],
    unused: List[dict],
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    safe_model = _sanitize_model_for_path_component(llm_model)
    output_path = reports_dir / f"experiment_list_{safe_model}.txt"

    used_rows = []
    for scenario, folder in selected.items():
        used_rows.append(f"- {scenario}: {folder}")
    if not used_rows:
        used_rows.append("- (none)")

    unused_rows = []
    for item in unused:
        unused_rows.append(
            f"- NOT USED {item['scenario']}: {item['folder']}"
        )
    if not unused_rows:
        unused_rows.append("- (none)")

    all_rows = []
    for item in matching:
        all_rows.append(
            f"- {item['scenario']}: {item['folder']}"
        )
    if not all_rows:
        all_rows.append("- (none)")

    content = "\n".join([
        f"LLM model: {llm_model}",
        "",
        "All experiments for model:",
        *all_rows,
        "",
        "Used experiments (first per scenario):",
        *used_rows,
        "",
        "Unused experiments (NOT USED):",
        *unused_rows,
        "",
    ])

    output_path.write_text(content, encoding="utf-8")


def _write_manifest_experiment_details_report(
    reports_dir: Path,
    llm_model: str,
    selected: Dict[str, str],
    manifest_meta: Dict[str, str],
    manifest_payload: Dict[str, Any],
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    safe_model = _sanitize_model_for_path_component(llm_model)
    output_path_txt = reports_dir / f"experiment_details_{safe_model}.txt"
    output_path_json = reports_dir / f"experiment_details_{safe_model}.json"

    manifest_experiments = manifest_payload.get("experiments")
    records = manifest_experiments if isinstance(manifest_experiments, list) else []

    selected_by_folder = {folder: scenario for scenario, folder in selected.items()}
    selected_records: List[dict] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        folder = str(record.get("experiment_name") or "").strip()
        if not folder:
            output_dir = str(record.get("output_dir") or "").strip()
            if output_dir:
                folder = Path(output_dir).name
        if not folder or folder not in selected_by_folder:
            continue
        selected_records.append(record)

    lines: List[str] = [
        f"LLM model: {llm_model}",
        f"Manifest path: {manifest_meta.get('manifest_path', '')}",
        f"Manifest created_at: {manifest_meta.get('created_at', '')}",
        "",
        "Selected experiment details (from manifest):",
    ]

    if not selected_records:
        lines.extend([
            "- (none found in manifest experiments list)",
            "",
        ])
    else:
        for record in selected_records:
            folder = str(record.get("experiment_name") or "").strip()
            output_dir = str(record.get("output_dir") or "").strip()
            if not folder and output_dir:
                folder = Path(output_dir).name
            scenario = selected_by_folder.get(folder, str(record.get("scenario") or "unknown"))
            status = str(record.get("status") or "")
            run_count = record.get("run_count")
            effective_config = record.get("effective_config")

            lines.append(f"- scenario: {scenario}")
            lines.append(f"  folder: {folder or '(unknown)'}")
            lines.append(f"  status: {status or '(unknown)'}")
            lines.append(f"  output_dir: {output_dir or '(unknown)'}")
            lines.append(f"  run_count: {run_count if run_count is not None else '(unknown)'}")
            lines.append("  effective_config:")
            if isinstance(effective_config, dict):
                config_json = json.dumps(effective_config, indent=2, sort_keys=True)
                for line in config_json.splitlines():
                    lines.append(f"    {line}")
            else:
                lines.append("    (missing)")
            lines.append("")

    output_path_txt.write_text("\n".join(lines), encoding="utf-8")

    details_payload = {
        "llm_model": llm_model,
        "manifest": {
            "path": manifest_meta.get("manifest_path", ""),
            "created_at": manifest_meta.get("created_at", ""),
        },
        "selected_experiments": [
            {
                "scenario": selected_by_folder.get(
                    str(record.get("experiment_name") or "").strip()
                    or Path(str(record.get("output_dir") or "")).name,
                    str(record.get("scenario") or "unknown"),
                ),
                "experiment_name": str(record.get("experiment_name") or "").strip()
                or Path(str(record.get("output_dir") or "")).name,
                "status": str(record.get("status") or ""),
                "output_dir": str(record.get("output_dir") or ""),
                "run_count": record.get("run_count"),
                "effective_config": record.get("effective_config"),
            }
            for record in selected_records
            if isinstance(record, dict)
        ],
    }
    output_path_json.write_text(json.dumps(details_payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == '__main__':
    _parse_args_and_run()
