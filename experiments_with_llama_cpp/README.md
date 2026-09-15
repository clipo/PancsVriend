# experiments_with_llama_cpp/ — what is in here and which file to analyse

Every folder here is one pipeline run of
`run_llm_probability_simulation_analysis.py` for one model: six scenarios,
simulated from that model's value-function tables, then analysed. The
folder name is `run_<YYYYMMDD_HHMMSS>_<model>-vf-<family>`.

| Suffix | What it is | Status |
| --- | --- | --- |
| `-vf-lp` | 10,000 runs per scenario from the model's **exact** value function (token log-probabilities). | **The result.** Nine models, all rank-stability `SETTLED`. |
| `-vf-s` | 100 runs per scenario from the sequential **sanity** resample of the same model (n=100 draws per cell). | A check on the `-vf-lp` tables, read against them, not a result. |
| `cross_model/` | Bump and level charts and the paired tests across every model's newest `-vf-lp` run. | Regenerated automatically by the pipeline's last stage; never hand-run. |

## The file for further analysis

**`<run>/experiments/<scenario>/run_summary.csv`** — one row per simulation
run, the values every downstream reader takes. Six files per run, one per
scenario, 10,000 rows each for `-vf-lp`. 

Alternatively, **`<run>/experiments/run_summary_by_run_all_scenarios.csv`** contains the same data for all six scenarios in one table (60,000 rows for `-vf-lp`), plus a `scenario_key` column. 
It is what the ANOVA, normality, metrics-comparison and
cross-model steps read.

Columns (`analysis_tools/build_run_summary.py` is the single definition):

| Column | Meaning |
| --- | --- |
| `run_id` | Seed of the run; per-move detail is regenerable from it under the `rng_scheme` in `config.json`. |
| `scenario` | `baseline`, `race_white_black`, `ethnic_asian_hispanic`, `income_high_low`, `political_liberal_conservative`, `green_yellow`. |
| `converged`, `convergence_step` | Whether the run hit `NO_MOVE_THRESHOLD` (5) consecutive zero-move steps, and the FIRST step of that streak. |
| `dissimilarity_index`, `clusters`, `switch_rate`, `distance`, `mix_deviation`, `share`, `ghetto_rate` | The seven metrics on the final grid. DI is the headline metric. |
| `initial_<metric>` | The same seven on frame 0, a uniformly random allocation: the run's own paired draw from the chance distribution. |
| `final_step`, `n_steps` | Last step simulated; capped runs stop at `max_steps` = 1000. |
| `stop_reason` | `converged` / `max_steps` / `incomplete`. |
| `experiment`, `llm_model`, `metrics_source` | Scenario folder, model slug, and whether metrics came from the live run or were recomputed from the final grid. |

## One run folder

```
run_<ts>_<model>-vf-lp/
├── run_config_source.yaml        the yaml the run was launched from (configs/vf_run_<model>_lp.yaml)
├── run_config_effective.yaml     that yaml with the chosen profile resolved (production: 10000 runs, 1000 steps, 20x20, 160+160)
├── vf_build_config_source.yaml   the value-function sampling config named by the yaml (provenance only; exact tables are not sampled)
├── run_layout_manifest.json      what the pipeline did: stages run/skipped, exact CLI passed to each, rank-stability verdict, cross-model status
├── scenarios_a2.py               the scenario definitions frozen with the run (agent labels per scenario)
├── value_functions/              the decision tables the run simulated FROM, frozen so the folder is self-contained
│   ├── vf_<label>-lp__<scenario>__R3_dual_count.json   P(MOVE | n_similar, n_occupied) per role, one per scenario
│   ├── TABLE_HASHES.json         sha256 of each table as loaded
│   └── value_function_heatmaps.png
├── experiments/                  one folder per scenario
│   └── llm_<scenario>_<ts>/
│       ├── config.json           board, steps, model, rng_scheme, value_function_file and its meta
│       ├── run_summary.csv       <- THE FILE (see above)
│       ├── metrics_history.csv.gz   every metric at every step (a cache derived from states/; not in git)
│       ├── states/               states_packed.npz, every grid frame of every run (not in git)
│       └── move_logs/            step_moves_packed.csv.gz, every move of every run (not in git)
├── manifest/
│   ├── <run>_run_manifest.json   the six scenario folders this run comprises, with the effective args
│   └── <run>_run_summary_<model>.csv   roll-up duplicate of the six run_summary.csv (not in git)
├── analysis/                     outputs of analysis_tools/run_all_scenario_analysis.py
│   ├── anova_results_by_metric.{csv,md}          one-way ANOVA per metric across scenarios
│   ├── segregation_scenario_rankings_<model>.{csv,md}   scenario ranking per metric
│   ├── normality_tests.csv
│   ├── rank_stability/           vf_rank_stability.py --exact: is the DI ordering of scenarios settled?
│   │   ├── rank_status.json      verdict (SETTLED / FIXABLE / UNMEASURED) and per-pair classes
│   │   ├── rank_pairs.csv, di_levels.csv, metric_levels.csv
│   │   └── RANK_STABILITY_NOTES.md
│   ├── dissimilarity_index/
│   │   ├── *_dissimilarity_final.csv.gz, dissimilarity_final_by_run.csv.gz   final DI per run
│   │   └── *_dissimilarity_by_step.csv.gz, dissimilarity_by_step_all.csv.gz   DI at every step (not in git)
│   ├── movement_decision_counts/movement_decision_counts_summary.csv.gz
│   ├── experiment_details_<model>.{json,txt}, experiment_list_<model>.txt
│   └── run_summary_by_run_all_scenarios.csv    all six scenarios' run_summary rows in one table (in git)
└── plots/
    ├── segregation_metrics_comparison.png, ..._dissimilarity_index.png   final-value distributions by scenario
    ├── segregation_heatmap.png
    ├── convergence_patterns.png, ..._dissimilarity_index.png             metric trajectories over steps
    ├── convergence_speed_comparison.png, convergence_speed_dissimilarity_index.png
    ├── metric_panels/metric_panel_<metric>.png                            one panel per metric
    ├── normality/normality_<metric>.png
    └── movement_decision_counts/movement_decision_counts.png
```

## cross_model/

| File | Content |
| --- | --- |
| `cross_model_bump_<metric>.png`, `cross_model_bump_all_metrics.png` | Scenario ranking per model, gap bars binned on Cohen's d. |
| `cross_model_level_<metric>.png`, `cross_model_level_all_metrics.png` | Scenario means per model with the chance level. |
| `cross_model_pairwise_tests.csv` | Paired t-tests between adjacent scenarios, per model and metric. |
| `cross_model_chance_tests.csv` | Each scenario against its own paired chance draws (`initial_<metric>`). |
| `cross_model_rankings.csv` | The ranking table behind the bump charts. |
| `cross_model_vf-lp_sources.csv` | Which run folder each model's rows came from (newest of the largest). |

Significance rules: `analysis_tools/analysis_guide.md`.