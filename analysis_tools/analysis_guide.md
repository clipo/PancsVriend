# PancsVriend Analysis Guide

This guide explains how to generate all analysis outputs (figures & CSV summaries) from the simulation experiments using a single entry point.

## 1. Overview

All analysis scripts run behind one orchestrator:

`analysis_tools/run_all_scenario_analysis.py`

(`analysis_tools/experiment_list_for_analysis.py` is its configuration — the
scenario list, labels and colours — not an entry point.) Run standalone, it
reads `experiments/` and writes under `reports/`. Run as the analysis stage
of `run_llm_probability_simulation_analysis.py` (the normal case since the
value-function pipeline), it reads the run's own `experiments/` and writes
into `<run_dir>/analysis/`, after which the pipeline moves every figure into
`<run_dir>/plots/`. The layout of a finished run folder is in
`experiments_with_llama_cpp/README.md`.

## 2. Prerequisites

1. Ensure experiments have completed and contain `run_summary.csv` and
   `metrics_history.csv.gz` under `experiments/<experiment_id>/` (or
   `<run_dir>/experiments/<experiment_id>/` for a pipeline run):
	- `move_logs/` + `states/` (only required for movement analysis and for
	  repairing a stale `metrics_history`)

## 3. One-Click Full Analysis

From the project root:

```bash
python analysis_tools/run_all_scenario_analysis.py
```

Need the reports somewhere else? Add `--output-folder <path>` (e.g.
`--output-folder reports_gemma3_27b`). `--llm-model <name>` filters to one
model (and, without `--output-folder`, writes to `reports_<name>`);
`--manifest-file <run_manifest.json>` analyses exactly the experiments of one
pipeline run and also writes the `experiment_list_*` / `experiment_details_*`
reports. The pipeline passes both.

This will run (in order):

0. `repair_stored_metrics` — verifies each experiment's `metrics_history` against its step logs (see §5)
1. `dissimilarity_index_over_time` — DI per step and final DI per run
2. `run_summary` — the per-run roll-up (`run_summary_by_run.csv`), then `anova_by_metric` and `normality_tests`
3. (`analyze_agent_movement` — skipped unless `--include-movement`)
4. ~~`analyze_stability_patterns`~~ — removed from the pipeline 2026-09-15 (code kept, step commented out)
5. `convergence_patterns_and_speed`
6. `movement_decision_counts`
7. `per_metric_panels`
8. `segregation_metrics_comparison`

Each step prints progress and a summary table is shown at the end with timing and status.

## 4. Including Movement Analysis

Movement analysis is computationally heavier (needs parsing move logs & states). Enable it explicitly:

```bash
python analysis_tools/run_all_scenario_analysis.py --include-movement
```

Run ONLY movement analysis (skipping everything else):

```bash
python analysis_tools/run_all_scenario_analysis.py --movement-only
```

## 5. Controlling Metric Recomputation

By default, the `repair_stored_metrics` step verifies each experiment's stored
`metrics_history` against its step logs (`run_files.stale_metrics_runs`: one
row per logged step, same first/last step, all seven metrics present) and
rebuilds only the runs that disagree (`Simulation.repair_stored_metrics`).
On a complete 10k-run experiment that is a ~1 s read-only check; before
2026-09-05 every pass rebuilt every run from its frames (750-960 s per
model campaign for byte-identical rows). Experiments predating the
`dissimilarity_index` column, missing runs, resume placeholders and
front-truncated histories are still rebuilt. `dissimilarity_index_over_time`
applies the same check and reads the stored column for complete runs.

To skip the check and only read existing CSVs, or to force the old full rebuild:

```bash
python analysis_tools/run_all_scenario_analysis.py --no-recompute
python analysis_tools/run_all_scenario_analysis.py --force-recompute
```

## 6. Quiet Mode

Suppress per-step progress output (summary still shown):

```bash
python analysis_tools/run_all_scenario_analysis.py --quiet
```

## 7. Output Locations

Everything is written under the output folder: `reports/` standalone, or
`<run_dir>/analysis/` in a pipeline run, where the pipeline then moves every
`.png` into `<run_dir>/plots/` (the table below shows the pipeline
placement; standalone, the plots stay next to the tables).

| Step | Tables (`analysis/`) | Figures (`plots/`) |
|------|----------------------|--------------------|
| dissimilarity_index_over_time | `dissimilarity_index/<experiment>_dissimilarity_{by_step,final}.csv.gz`, `dissimilarity_by_step_all.csv.gz`, `dissimilarity_final_by_run.csv.gz` | — |
| run_summary | `run_summary_by_run.csv` (one row per run, all scenarios; the same rows as the six per-scenario `run_summary.csv`) | — |
| anova_by_metric | `anova_results_by_metric.csv` / `.md` | — |
| normality_tests | `normality_tests.csv`, `segregation_scenario_rankings_<model>.csv` / `.md` | `normality/normality_<metric>.png` |
| movement analysis (if enabled) | `movement_analysis/<experiment>/...` | summary plots |
| convergence_patterns_and_speed | — | `convergence_patterns.png`, `convergence_patterns_dissimilarity_index.png`, `convergence_speed_comparison.png`, `convergence_speed_dissimilarity_index.png` |
| movement_decision_counts | `movement_decision_counts/movement_decision_counts_summary.csv.gz` | `movement_decision_counts/movement_decision_counts.png` |
| per_metric_panels | — | `metric_panels/metric_panel_<metric>.png` |
| segregation_metrics_comparison | — | `segregation_metrics_comparison.png`, `segregation_metrics_comparison_dissimilarity_index.png`, `segregation_heatmap.png` |
| manifest reports (with `--manifest-file`) | `experiment_list_<model>.txt`, `experiment_details_<model>.{txt,json}` | — |
| ~~stability patterns~~ | no longer generated (2026-09-15); `convergence_progress_90pct*.png` likewise removed from `convergence_patterns_and_speed` | |
| rate-of-change (if re-enabled manually) | `rate_of_change_analysis.*`, `phase_transitions_analysis.*` | |

> Note: The rate-of-change scripts are currently commented out in the orchestrator for runtime reduction. Uncomment if needed.

Two pipeline stages that run AFTER this script also write into the run
folder: rank stability (`analysis/rank_stability/`, `vf_rank_stability.py
--exact`: is the DI ordering of the scenarios SETTLED?) and the cross-model
stage (§7b). Neither is part of `run_all_scenario_analysis.py`.

## 7b. Cross-model figures (automatic)

The per-campaign pipeline above is one model. The cross-model comparison —
scenario ORDERING per model (bump charts) and the metric LEVELS behind it
(level charts), plus their test tables — is regenerated by the orchestrator's
final `cross_model` stage every time any model's
`run_llm_probability_simulation_analysis.py` pipeline completes, from every
model's newest full run under that run's `run_root`
(`value_functions/comparison/cross_model_vf_comparison.py --run-root <run_root>`):

| Output (`experiments_with_llama_cpp/cross_model/`, i.e. `<run_root>/cross_model/`) | Content |
|---|---|
| `cross_model_bump_all_metrics.png` | 7 bump-chart panels, one per metric |
| `cross_model_level_all_metrics.png` | 7 level-chart panels, same grid (y inverted for clusters / switch rate) |
| `cross_model_bump_<metric>.png` | bump chart per metric, with gap bars binned on Cohen's d (mean gap / pooled SD of the two scenarios' final values; 0.2 / 0.5 / 0.8) |
| `cross_model_level_<metric>.png` | level chart per metric |
| `cross_model_chance_tests.csv`, `cross_model_pairwise_tests.csv`, `cross_model_rankings.csv` | the paired t-tests behind every marker |
| `cross_model_vf-lp_sources.csv` | which run folder each model's rows came from (added 2026-09-15; the choice is "newest of the largest", below) |

Significance rules (decided 2026-09-05). "Chance" is tested as a scenario
like any other: each run's `initial_<metric>` (frame 0, a uniformly random
allocation) is its paired draw from the chance distribution, and a scenario
not significantly above its initial grids "shows no segregation". Every
significance statement in the ranking table and the bump charts is a paired
t-test on per-run differences (final − initial for chance, final − final
for scenario pairs), Holm-corrected, with no normality gate and no practical
floor: solid = statistically distinguishable, not necessarily large. The
0.01 DI FLOOR-TIE exists only in `vf_rank_stability`'s certification, where
it bounds GPU top-up quotes.

The comparison is built from the EXACT token-probability tables (runs with
`llm_model` suffix `-vf-lp`, tables from
`value_functions/logprob/logprob_value_function.py`); the stage derives the family
from the run's suffix. The sampled `-vf-r3` tables are superseded
(2026-09-06): sampled at concurrency > 1 they carry the llama-server
batch-numerics artifact and are not treated as a result or a ground truth,
so a `-vf-r3` run records the stage as skipped. `--family r3` still writes
them as `cross_model_sampled_*` (title says "superseded") for the artifact
write-up only; the two families are never mixed in one figure. The
multi-split sufficiency ruler only exists for sampled tables — exact tables
carry no sampling error, so their rank-stability stage runs `--exact` and no
multisplit check happens for them.

Each model contributes its newest run among its LARGEST (by row count), so a
smoke-test run never displaces the production run. Concurrent pipelines
(should two ever run at once) serialise on `<run_root>/.cross_model_figures.lock`.
`skip_cross_model: true` in the run yaml disables the stage; `cross_model_args`
passes `--out-dir` / `--dpi`. To refresh by hand:

```bash
python value_functions/comparison/cross_model_vf_comparison.py               # exact tables   -> cross_model_*
python value_functions/comparison/cross_model_vf_comparison.py --family r3   # sampled tables -> cross_model_sampled_* (superseded)
python value_functions/comparison/cross_model_vf_comparison.py --family s    # sequential sanity runs (-vf-s, 100 runs) -> cross_model_sanity_*
```

The `-vf-s` family (registered 2026-09-07) is the sequential sanity
cross-check ON the exact tables — 100 draws per cell, 100 runs per
scenario — and like `r3` it is skipped by the automatic stage; only `lp`
regenerates the unprefixed canonical set. An unregistered suffix falls back
to `r3`, i.e. to being skipped, so a new family can never overwrite the
exact figures by accident.

## 8. Adding New Analyses

To add a new analysis script to the one-click pipeline:
1. Implement it under `analysis_tools/your_script.py` with either a `main()` or top-level side-effect logic.

2. Edit `run_all_scenario_analysis.py` and append a new step inside `run_all_analyses`:

```python
def _run_new():
	import importlib
	mod = importlib.import_module('analysis_tools.your_script')
	mod.main()  # or executes on import
steps.append(("your_script", _run_new, {}))
```

1. Re-run the orchestrator.

## 9. Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Missing `metrics_history.csv.gz` | Simulation not finished or wrong path | Check `experiments/<id>/` contents; the pipeline verifies and repairs it from `states/` (§5) |
| Movement step fails | Missing `move_logs/` or `states/` | Re-run simulation with logging enabled |
| Empty figures | Metrics columns have NaNs | Inspect source CSV; validate simulation outputs |
| Very slow run | Movement + large number of experiments | Run without movement first, then add `--include-movement` |
| No heatmap generated | Not enough scenarios present | Ensure combined metrics contains multiple scenario rows |

## 10. Example Workflow

```bash
# 1. Run simulations (example)
python run_all_contexts.py --runs 10 --processes 5 --llm-model phi4:latest

# 2. One-click analysis without movement (fast)
python analysis_tools/run_all_scenario_analysis.py

# 3. Add movement analysis later
python analysis_tools/run_all_scenario_analysis.py --include-movement

# 4. Open figures (Linux example)
xdg-open reports/convergence_patterns.png
```

## 11. Best Practices

1. Commit generated CSVs if you need reproducible downstream statistical work (figures can be regenerated).
2. `--no-recompute` skips the completeness check entirely; it is rarely needed now that the check itself is cheap.
3. Keep experiments tidy—archive or move old runs if they clutter the reports.
4. Add docstrings and clear function names in any new analysis script so orchestration stays readable.

---
