# PancsVriend Analysis Guide

This guide explains how to generate all analysis outputs (figures & CSV summaries) from the simulation experiments using a single entry point.

## 1. Overview

All analysis scripts have been unified behind an orchestrator located in:

`analysis_tools/experiment_list_for_analysis.py`

You can now run almost the full analysis pipeline with a single command instead of invoking each script manually. Outputs are centralized under the `reports/` directory (and subfolders like `reports/movement_analysis/`).

## 2. Prerequisites

1. Ensure experiments have completed and contain expected files under `experiments/<experiment_id>/`:
	- `move_logs/` + `states/` (only required for movement analysis)

## 3. One-Click Full Analysis

From the project root:

```bash
python analysis_tools/experiment_list_for_analysis.py
```

Need the reports somewhere else? Add `--output-folder <path>` to point the orchestrator at a different base directory (e.g., `--output-folder reports_gemma3_27b`).

This will run (in order):

1. `combined_final_metrics` (aggregates final metrics across scenarios)
2. (Movement analysis is skipped by default for speed)
3. `analyze_stability_patterns`
4. `convergence_patterns_and_speed`
5. `per_metric_panels`
6. `segregation_metrics_comparison`

Each step prints progress and a summary table is shown at the end with timing and status.

## 4. Including Movement Analysis

Movement analysis is computationally heavier (needs parsing move logs & states). Enable it explicitly:

```bash
python analysis_tools/experiment_list_for_analysis.py --include-movement
```

Run ONLY movement analysis (skipping everything else):

```bash
python analysis_tools/experiment_list_for_analysis.py --movement-only
```

## 5. Controlling Metric Recomputation

By default, `combined_final_metrics` will attempt to (re)compute metrics for scenarios (if the simulation `Simulation` helper is available) to ensure files exist. To skip recomputation and only read existing CSVs:

```bash
python analysis_tools/experiment_list_for_analysis.py --no-recompute
```

## 6. Quiet Mode

Suppress per-step progress output (summary still shown):

```bash
python analysis_tools/experiment_list_for_analysis.py --quiet
```

## 7. Output Locations

All generated artifacts are written under `reports/`:

| Component | Outputs |
|-----------|---------|
| combined_final_metrics | `reports/combined_final_metrics.csv` |
| movement analysis (if enabled) | `reports/movement_analysis/<experiment>/...` (per-experiment) + summary plots |
| stability patterns | `reports/stability_analysis.png`, `reports/trajectory_variance.png` |
| convergence patterns | `reports/convergence_patterns.(png\|pdf)` |
| convergence speed | `reports/convergence_speed_comparison.(png\|pdf)` |
| rate-of-change (if re-enabled manually) | `reports/rate_of_change_analysis.*`, `reports/phase_transitions_analysis.*` |
| per metric panels | `reports/metric_panel_<metric>.(png\|pdf)` |
| segregation metrics comparison | `reports/segregation_metrics_comparison.*`, `reports/segregation_heatmap.*` |

> Note: The rate-of-change scripts are currently commented out in the orchestrator for runtime reduction. Uncomment if needed.

## 8. Adding New Analyses

To add a new analysis script to the one-click pipeline:
1. Implement it under `analysis_tools/your_script.py` with either a `main()` or top-level side-effect logic.

2. Edit `experiment_list_for_analysis.py` and append a new step inside `run_all_analyses`:

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
| Missing `metrics_history.csv` | Simulation not finished or wrong path | Check `experiments/<id>/` contents |
| Movement step fails | Missing `move_logs/` or `states/` | Re-run simulation with logging enabled |
| Empty figures | Metrics columns have NaNs | Inspect source CSV; validate simulation outputs |
| Very slow run | Movement + large number of experiments | Run without movement first, then add `--include-movement` |
| No heatmap generated | Not enough scenarios present | Ensure combined metrics contains multiple scenario rows |

## 10. Example Workflow

```bash
# 1. Run simulations (example)
python run_experiments.py --config some_config.yaml

# 2. One-click analysis without movement (fast)
python analysis_tools/experiment_list_for_analysis.py

# 3. Add movement analysis later
python analysis_tools/experiment_list_for_analysis.py --include-movement

# 4. Open figures (Linux example)
xdg-open reports/convergence_patterns.png
```

## 11. Best Practices

1. Commit generated CSVs if you need reproducible downstream statistical work (figures can be regenerated).
2. Use `--no-recompute` when you are sure the metrics are up-to-date to save time.
3. Keep experiments tidy—archive or move old runs if they clutter the reports.
4. Add docstrings and clear function names in any new analysis script so orchestration stays readable.

---
