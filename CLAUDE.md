# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git policy — NEVER commit or push

NEVER run `git commit` or `git push` yourself, under any circumstances. Leave
all changes uncommitted in the working tree so the user can review them; the
user commits and pushes manually. This also applies to scripts you author:
do not add auto-commit/auto-push steps to runner or queue scripts.

## Project Overview

This is a research framework comparing traditional utility-maximizing agents with LLM-based agents in the classic Schelling Segregation Model. The project studies how different social contexts (race, income, politics) affect residential segregation patterns.

## Key Commands

### Setup and Configuration
```bash
# Install dependencies
pip install -r requirements.txt

# Test LLM connectivity (uses config.py by default)
python check_llm.py

# Test LLM connectivity with custom configuration
python check_llm.py --llm-model "gpt-4" --llm-url "https://api.openai.com/v1/chat/completions" --llm-api-key "your-key"

# Test parallel LLM processing robustness
python test_llm_parallel.py

# Test with custom LLM configuration
python test_llm_parallel.py --llm-model "claude-3-sonnet" --llm-url "https://api.anthropic.com/v1/messages"
```

### Running Experiments
```bash
# Full experiment suite (uses config.py by default)
python run_experiments.py

# Full experiment suite with custom LLM
python run_experiments.py --llm-model "gpt-4o" --llm-url "https://api.openai.com/v1/chat/completions" --llm-api-key "your-key"

# Quick test version
python run_experiments.py --quick-test

# Custom configuration with multiple parameters
python run_experiments.py --baseline-runs 50 --llm-runs 20 --scenarios baseline race_white_black --llm-model "claude-3-sonnet"

# Individual components
python baseline_runner.py --runs 100
python llm_runner.py --scenario race_white_black --runs 10

# Interactive GUI simulation
python SchellingSim.py
```

### Analysis and Visualization
```bash
# Generate statistical analysis
python statistical_analysis.py

# Create visualization reports
python visualization.py --baseline-dir experiments/baseline_xxx --llm-dirs experiments/llm_*
```

## Architecture

### Core Components
- **Agent.py**: Traditional utility-maximizing agents with best-response dynamics
- **LLMAgent.py**: LLM-powered agents that make authentic human-like housing decisions
- **Metrics.py**: Six segregation metrics (clusters, switch rate, distance, mix deviation, share, ghetto rate)
- **config.py**: Central configuration for all simulation parameters

### Experiment Framework
- **run_experiments.py**: Master orchestrator for complete experiment suites
- **baseline_runner.py**: Runs mechanical agent simulations
- **llm_runner.py**: Runs LLM agent simulations with social context scenarios
- **plateau_detection.py**: Detects convergence points and calculates segregation speeds

### Analysis Pipeline
- **statistical_analysis.py**: ANOVA, effect sizes, multivariate analysis
- **visualization.py**: Comprehensive PDF reports with time series and comparisons
- **SchellingSim.py**: Interactive GUI for real-time simulation viewing

## Social Context Scenarios

The framework supports multiple social contexts defined in `llm_runner.py`:
- **baseline**: Red vs blue teams (control)
- **race_white_black**: White middle class vs Black families
- **ethnic_asian_hispanic**: Asian American vs Hispanic/Latino families
- **economic_high_working**: High-income vs working-class households
- **political_liberal_conservative**: Liberal vs conservative households

## LLM Configuration

### Default Configuration (config.py)
LLM settings are configured in `config.py`:
- `OLLAMA_MODEL`: Model identifier (default: "qwen2.5-coder:32B")
- `OLLAMA_URL`: API endpoint
- `OLLAMA_API_KEY`: Authentication key

### Command-Line Override
All scripts support command-line LLM configuration that overrides `config.py`:
- `--llm-model`: Specify different model (e.g., "gpt-4", "claude-3-sonnet", "llama2")
- `--llm-url`: Specify different API endpoint
- `--llm-api-key`: Specify different API key

### Supported LLM Providers
The framework works with any OpenAI-compatible API, including:
- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude-3-sonnet, Claude-3-haiku (via proxy)
- **Local models**: Ollama, LM Studio, vLLM
- **Cloud providers**: Azure OpenAI, AWS Bedrock (via proxy)

The system includes robust error handling with circuit breakers and automatic fallback to mechanical agents when LLM services fail.

## Output Structure

Experiments generate structured outputs:
```
experiments/
├── baseline_[timestamp]/       # Mechanical agent results
├── llm_[scenario]_[timestamp]/ # LLM scenario results
│   ├── metrics_history.csv.gz  # every metric at every step (.csv before 2026-09-03; run_files.metrics_history_path finds either)
│   ├── run_summary.csv         # ONE ROW PER RUN (see below): the per-run bookkeeping
│   ├── move_logs/step_moves_run_<id>.csv   # per-step decision counts (see below)
│   └── states/states_run_<id>.npz          # grid frames: 0 = initial, k+1 = after step k
├── manifests/
│   ├── [timestamp]_run_manifest.json
│   └── [timestamp]_run_summary_[model].csv   # campaign roll-up across scenarios
└── ...

reports/
├── comprehensive_report_[timestamp].pdf
├── statistical_analysis_[timestamp].txt
├── run_summary_by_run.csv      # roll-up for the analysed model selection
└── experiment_summary_[timestamp].json
```

### Run files: move_logs/ and states/

Every run's record is read through `run_files.py`; nothing else parses these
files. Two formats exist:

* **Per-step** (default since 2026-09-02, all value-function and mechanical
  runs): `step_moves_run_<id>.csv` with one row per step
  (`step, decisions, moved, parse_failed, successful_move, target_occupied,
  invalid_target, chose_to_stay, same_position`) and one grid frame per step
  in the npz. Runs are seeded by `run_id`, so per-move detail is regenerable —
  under the decision-RNG scheme recorded as `rng_scheme` in the run's
  `config.json`: `keyed` (default since 2026-09-04; each value-function
  decision's uniform is a function of `(run_id, step, agent_id)`, giving
  common random numbers across tables and scenarios) or `shared` (the
  earlier shared `random` stream). Set `VF_RNG_SCHEME=shared` to regenerate a
  run made before 2026-09-04. See `keyed_uniform` in `llm_runner.py`.
* **Full** (per-move): `agent_moves_run_<id>.json.gz` with one record per
  agent decision (incl. the raw LLM reply) and one frame per record. Written
  by live-LLM runs always, by anything else with `--full-move-log` /
  `FULL_MOVE_LOG=1`, and by every run before 2026-09-02.

`run_files.load_step_log` / `load_step_frames` / `load_final_grid` return the
same per-step tables for both, so analysis code never branches on format.
`python test_latest_experiment_output_format.py [EXPERIMENT_DIR]` checks a
directory's files are mutually consistent.

* **Packed** (since 2026-09-05): when a value-function or mechanical
  experiment completes, the runner folds its per-run files into
  `move_logs/step_moves_packed.csv.gz` (every step log, `run_id` column
  first) and `states/states_packed.npz` (member `run_<id>` = that run's
  frames) and deletes the per-run files — same rows, same arrays, verified
  member by member before anything is removed. A 10k-run experiment goes
  from ~20k files / 84 MB on disk to 7 files / 11 MB. The readers try a
  run's own files first, then the containers, so runs added or re-run
  after packing land as per-run files until the next pack, and resume,
  analysis and the format checker are unchanged. Per-run files remain the
  write format (parallel workers; an aborted campaign keeps finished
  runs). `PACK_RUN_RECORD=0` disables the automatic pack;
  `python analysis_tools/pack_run_records.py --all [--root DIR]` packs
  existing experiments (~1M per-run files predate this). Full-format
  (per-move) runs are never packed.

Re-running an incomplete experiment without `--new` resumes it. Aborted
live-LLM runs continue from their last saved grid and the files keep the
pre-abort steps (`Simulation.preload_record`); aborted value-function runs
are redone from scratch, since a resumed run reseeds mid-way and would no
longer be reproducible from `run_id`.

### run_summary.csv

Written automatically by `Simulation.analyze_results`, so every experiment
directory has one, and rolled up per model campaign by `run_all_contexts.py`
and the `run_summary` step of `analysis_tools/run_all_scenario_analysis.py`.
Columns:

```
run_id, scenario, converged, convergence_step, dissimilarity_index,
clusters, switch_rate, distance, mix_deviation, share, ghetto_rate,
initial_dissimilarity_index, ... initial_ghetto_rate,
final_step, n_steps, stop_reason, experiment, llm_model, metrics_source
```

* `initial_<metric>` (2026-09-05) — the same seven metrics on the run's
  initial grid (frame 0), a uniformly random allocation: the run's own
  paired draw from the chance distribution. "Chance" is tested as a scenario
  like any other — paired t on final − initial across runs, Holm-corrected —
  in the ranking table and the cross-model bump charts; a scenario not
  significantly above its initial grids "shows no segregation". Every
  significance statement in those outputs is a paired t on per-run
  differences with no normality gate and no practical floor (solid =
  statistically distinguishable, not necessarily large); the 0.01 DI
  FLOOR-TIE remains only in `vf_rank_stability`'s certification, where it
  bounds GPU top-up quotes.

* `convergence_step` is the **first** of the `NO_MOVE_THRESHOLD` consecutive
  zero-move steps, so `final_step == convergence_step + NO_MOVE_THRESHOLD - 1`
  (= +4 by default) for every converged run. `base_simulation.convergence_from_step_moves`
  is the single definition; before 2026-09-01 the live loop recorded the last
  step of that window instead, and `run_summary` corrects such legacy rows.
* The metric columns are the values at `final_step`; missing ones (runs
  predating `dissimilarity_index`, or resume placeholders with no metrics
  rows) are recomputed from the run's final grid, flagged via `metrics_source`.
* `final_step` is always the last step actually simulated — the lower of the
  convergence-implied last step and where the run stopped. A run capped
  mid-streak (3 no-move steps at step 999) is `converged=False` with an empty
  `convergence_step`, but `final_step` is still 999.
* `stop_reason` is `converged` / `max_steps` / `incomplete` — the `+4` identity
  only applies to `converged` rows.
* Rebuilding a run from its move log is expensive (a few hundred MB gzipped for
  a placeholder-heavy experiment), so rows already in `run_summary.csv` are
  reused on later passes; `build_run_summary.py --force` re-parses the logs.

Regenerate for existing experiments with
`python analysis_tools/build_run_summary.py --all`.

Two files stopped being written on 2026-09-05 and every reader now takes
`run_summary.csv` instead: `convergence_summary.csv` (its four columns are
in `run_summary.csv`, and its `final_step` carried the live loop's
one-past-the-end value for capped runs) and `step_statistics.csv` (a
survivor-biased per-step average; plot from `metrics_history` with
`plot_style.step_stats_forward_filled`). Readers fall back to a legacy
`convergence_summary.csv` only in directories that have no
`run_summary.csv` yet. At the campaign level the roll-up is
`analysis/run_summary_by_run.csv` (`scenario_key` = the analysis scenario);
`combined_final_metrics.csv` — the same rows under another name — is gone
with it, and `analysis_tools/anova_by_metric.py` computes the ANOVA table
from the roll-up.

### Metrics are whole-array functions of the int grid

Every metric in `Metrics.py` (and the DI) is computed with numpy on the
int grid that `DissimilarityIndex.as_int_grid` produces from either
representation; `Simulation.run_step` builds that grid once per step and
shares it with the frame log. The pre-2026-09-05 object-grid loops live in
`tests/test_metrics.py` as the oracle the vectorised versions are pinned
to, bit for bit (real campaign frames + adversarial grids). Per 20x20 step:
4.4 ms -> 0.3 ms; a value-function run halves overall.

### Logging cadence

Value-function runs print no per-run lines; `[run-progress]` appears every
`llm_runner.PROGRESS_EVERY` (100) completions and always on the last (the
watchers read the latest line). Live-LLM runs keep per-run lines and a
progress line per run. A 60k-run campaign log is a few hundred lines, not
240k.

### metrics_history is a cache, verified rather than regenerated

`metrics_history.csv.gz` is a pure function of the frames in `states/` (the
same `calculate_all_metrics` on the same grids). The analysis pipeline checks
it against the run record with `run_files.stale_metrics_runs` — one row per
logged step, same first and last step, all seven metric columns — and
`Simulation.repair_stored_metrics` rebuilds only the runs that fail (all of
them when the file lacks a column). Since 2026-09-05; before that
the `combined_final_metrics` step rebuilt every run on every pass,
750–960 s per campaign for byte-identical rows; it is now the pipeline's
`repair_stored_metrics` step (`--no-recompute` skips it, `--force-recompute`
rebuilds everything). Count and first step matter, not just the
last: a live-LLM run resumed after an abort stores only its post-resume rows.
The merge readers in `analyze_results` / `run_summary` parse floats with
`float_precision='round_trip'` so a merge never perturbs stored values.

## Development Notes

- The system uses parallel processing for LLM queries with configurable batch sizes
- All simulations include plateau detection to identify convergence points
- Statistical analysis includes normality testing, ANOVA, and effect size calculations
- Visualization generates comprehensive PDF reports with confidence intervals
- The codebase follows a modular design with clear separation between mechanical and LLM agent logic