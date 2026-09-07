# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Git policy — NEVER commit or push

NEVER run `git commit` or `git push` yourself, under any circumstances. Leave
all changes uncommitted so the user can review them; the user commits and
pushes manually. Do not add auto-commit/auto-push steps to scripts you author.

## Project overview

A research framework comparing utility-maximizing agents with LLM-based agents
in the Schelling segregation model, studying how social context (race, income,
politics) affects segregation. The headline metric is the dissimilarity index
(`DissimilarityIndex.py`); `Metrics.py` adds six more (clusters, switch rate,
distance, mix deviation, share, ghetto rate).

## Key commands

```bash
pip install -r requirements.txt
python check_llm.py                       # LLM connectivity (config.py, or --llm-model/--llm-url/--llm-api-key)

# Orchestrated value-function pipeline (simulation + analysis + cross-model stage)
python run_llm_probability_simulation_analysis.py --config-yaml configs/vf_run_gemma_lp.yaml --config-profile production

python run_all_contexts.py --runs 10 --processes 5 --llm-model phi4:latest   # multi-scenario live-LLM runner
python baseline_runner.py --runs 100                                          # mechanical agents
python llm_runner.py --scenario race_white_black --runs 10                    # one LLM scenario
python SchellingSim.py                                                        # interactive GUI
python statistical_analysis.py                                                # ANOVA / effect sizes
```

Every script that talks to an LLM takes `--llm-model`, `--llm-url` and
`--llm-api-key`, overriding `OLLAMA_MODEL` / `OLLAMA_URL` / `OLLAMA_API_KEY`
in `config.py`. Any OpenAI-compatible endpoint works. There is no fallback to
mechanical decisions when the LLM fails; a failed decision is an error.

## Architecture

- **Agent.py** / **LLMAgent.py**: mechanical best-response agents / LLM agents.
- **config.py**: simulation parameters (grid, `NO_MOVE_THRESHOLD`, LLM endpoint).
- **context_scenarios.py**: `CONTEXT_SCENARIOS` — `baseline` (red vs blue,
  control), `race_white_black`, `ethnic_asian_hispanic`, `income_high_low`,
  `political_liberal_conservative`, `green_yellow`.
- **run_all_contexts.py**: multi-scenario runner (resume, manifests, run-summary roll-up);
  **baseline_runner.py** / **llm_runner.py**: single-track runners;
  **plateau_detection.py**: convergence points and segregation speeds.
- **run_llm_probability_simulation_analysis.py**: the orchestrated pipeline
  (vf build → contexts → scenario analysis → rank stability → cross-model).
  Its `cross_model` stage regenerates
  `experiments_with_llama_cpp/cross_model/cross_model_{bump,level}_*.png` and
  `cross_model_*_tests.csv` from every model's newest full run, so those
  figures are never hand-run (`analysis_tools/analysis_guide.md` §7b).
- **analysis_tools/**: `run_all_scenario_analysis.py` (per-campaign analysis,
  `analysis_guide.md`), `build_run_summary.py`, `pack_run_records.py`,
  `anova_by_metric.py`, `plot_style.py`.

### Value functions (`value_functions/`)

- `sampling/`: `build_value_function.py` samples P(MOVE | context) into vf-1
  tables (the retired `-vf-r3` route; `vf_multisplit_check.py` rulers,
  `vf_ruler_scaling.py`, the vfS/vfH plots).
- `logprob/`: `logprob_value_function.py` extracts the exact tables from token
  log-probabilities (`-vf-lp`, the result).
- `batch_numerics/`: the probe showing served probabilities depend on the
  batch, which retired sampling (results tracked).
- `comparison/`: `vf_rank_stability.py` (per-run certification) and
  `cross_model_vf_comparison.py` (bump/level charts across models).
- `results/` (gitignored): `sampled/`, `llm_logprob/`, `figures/`,
  `chance_null/`. Every location is defined once in `value_functions/paths.py`.
- `prompt_refinement/` keeps the prompt templates and the sampling harness
  (`sampling_common.py`, `ratio_prompt_templates.py`, `evaluate_ratio_prompts.py`).

Only the exact `-vf-lp` tables are a result: the sampled `-vf-r3` tables carry
the batch-numerics artifact, so the cross-model stage skips them (`--family r3`
writes `cross_model_sampled_*` for the artifact write-up only; families are
never mixed). Exact tables have no multi-split ruler, so rank stability runs
`--exact`.

## Output structure

```
experiments/<name>_<timestamp>/
├── config.json                 # incl. rng_scheme for value-function runs
├── metrics_history.csv.gz      # every metric at every step (a cache, see below)
├── run_summary.csv             # one row per run
├── move_logs/                  # step_moves_run_<id>.csv, or step_moves_packed.csv.gz
└── states/                     # states_run_<id>.npz (frame 0 = initial, k+1 = after step k), or states_packed.npz
experiments/manifests/<timestamp>_run_manifest.json, <timestamp>_run_summary_<model>.csv   # campaign roll-ups
```

### Run files: move_logs/ and states/

Every run record is read through `run_files.py`; nothing else parses these
files, and analysis code never branches on per-step vs full vs packed format.

Per-step runs are seeded by `run_id`, so per-move detail is regenerable under
the `rng_scheme` recorded in the run's `config.json` (`keyed_uniform` in
`llm_runner.py`); set `VF_RNG_SCHEME=shared` to regenerate a run made before
2026-09-04. Live-LLM runs always keep the full per-move record including the
raw LLM reply.

`python test_latest_experiment_output_format.py [EXPERIMENT_DIR]` checks a
directory's files are mutually consistent.

Re-running an incomplete experiment without `--new` resumes it. Aborted
live-LLM runs continue from their last saved grid (`Simulation.preload_record`);
aborted value-function runs are redone from scratch, since a resumed run would
no longer be reproducible from `run_id`.

### run_summary.csv and metrics_history

`run_summary.csv` (one row per run, written by `Simulation.analyze_results`)
is the per-run bookkeeping every reader takes; `convergence_summary.csv` and
`step_statistics.csv` are legacy and no longer written. Column semantics are
in the `analysis_tools/build_run_summary.py` docstring; regenerate with
`python analysis_tools/build_run_summary.py --all`. The campaign roll-up is
`analysis/run_summary_by_run.csv`, from which `anova_by_metric.py` computes
the ANOVA table.

`metrics_history.csv.gz` is a cache derived from the frames in `states/`;
never edit or regenerate it by hand, the pipeline verifies and repairs it.
The significance rules behind the ranking table and bump charts are in
`analysis_tools/analysis_guide.md`.

### Metrics are whole-array functions of the int grid

Every metric is computed with numpy on the int grid from
`DissimilarityIndex.as_int_grid`; `tests/test_metrics.py` pins the vectorised
versions bit for bit to the original loop implementations kept there as the
oracle.

### Logging cadence

Value-function runs print only a `[run-progress]` line every
`llm_runner.PROGRESS_EVERY` completions and on the last; watchers read the
latest line.
