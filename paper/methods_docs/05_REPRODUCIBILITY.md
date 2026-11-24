# 05 — Reproducibility: environment, commands, data locations, caveats

## 1. Environment

- llama.cpp `llama-server`, build **b1-a4ce259**; NVIDIA GB10, full offload,
  121 GB unified memory. Build and model path are recorded in every table's
  `meta`.
- Models: the nine gguf files of 04 §1, under `llms/`.
- libfaketime (`/srv/shared/schelling/tools/libfaketime`) for the clock pin.
- Python: the repo `.venv`. No GPU is needed for simulations or analysis.
- Bitwise replication of a table is promised **per (build, model file,
  hardware, pinned date)** only.

## 2. Commands, stage by stage

**Extract the exact tables** (server needed; one model at a time, sequential):

```bash
python value_functions/logprob/logprob_value_function.py --label gemma-4-31b-chat-grammar
```

For Llama and Mistral the server must be started under its pinned clock
(`value_functions.serving.PinnedServer`). A new model or quantisation goes
through download → server → exact tables → batch-numerics probe → server
down → simulation pipeline with `python value_functions/new_model_chain.py
<key>`.

**Simulate + analyse** (no server needed):

```bash
python run_llm_probability_simulation_analysis.py \
    --config-yaml configs/vf_run_gemma_lp.yaml --config-profile production
```

One command per model (`configs/vf_run_<model>_lp.yaml`): 6 scenarios ×
10,000 runs on the 20×20 grid, per-campaign analysis (`<run_dir>/analysis/`),
the DI rank-stability verdict (`<run_dir>/analysis/rank_stability/`) and, as
the final stage, the cross-model bump/level charts and test tables
(`experiments_with_llama_cpp/cross_model/cross_model_*`) rebuilt from every
model's newest full run. `cross_model_vf-lp_sources.csv` records which run
folder supplied each model.

**Validate** (server needed, pinned clock, sequential):

```bash
python value_functions/sampling/census.py --models all
```

The census goodness of fit reported in the paper (also the `census_gof`
pipeline stage; reads only stored tables, no server):

```bash
python value_functions/comparison/census_goodness_of_fit.py
```

**Paper tables and figures**:

```bash
python paper/schelling-llm-paper-JEIC/regenerate-all-tables-and-plots.py
```

(a wrapper for `analysis_tools/ccs2026_presentation/regenerate_all_tables_and_plots.py`).

**Historical** (reproduce qualitatively only): the prompt sweeps
(`prompt_refinement/evaluate_prompts.py`, `evaluate_ratio_prompts.py`) and
the sampled route (`value_functions/sampling/build_value_function.py`).

## 3. Data locations

| data | location |
|---|---|
| exact tables (vf-1 JSON) | `value_functions/results/llm_logprob/tables/` |
| extraction records (every path and state) | `value_functions/results/llm_logprob/raw/` |
| sequential-check verdicts and cells | `value_functions/results/llm_logprob/validation_data/` |
| sequential census tables and raw replies | `value_functions/results/sampled_small/` |
| batch-numerics probe data and plots | `value_functions/batch_numerics/results/` (tracked) |
| date-sensitivity evidence | `value_functions/results/date_sensitivity/` |
| production runs (one folder per model) | `experiments_with_llama_cpp/run_<timestamp>_<model>-vf-lp/` |
| frozen decision tables per run | `<run_dir>/value_functions/` (+ `TABLE_HASHES.json`) |
| cross-model charts and test tables | `experiments_with_llama_cpp/cross_model/` |
| sweep tables, raws, figures | `prompt_refinement/results/` |

`value_functions/results/` is gitignored; every location is defined once in
`value_functions/paths.py`. Simulation output is large (a 10k-run batch is
gigabytes) and is regenerable exactly from the frozen config and tables,
since runs are seeded by `run_id`. A public archive of tables and runs for
the Data Availability Statement is not yet deposited.

The nine production runs behind the paper:

| model | run folder |
|---|---|
| gemma-4-31b | `run_20260906_003730_gemma-4-31b-vf-lp` |
| phi-4-14b | `run_20260906_010130_phi-4-14b-vf-lp` |
| granite-4.2-30b | `run_20260906_020531_granite-4.2-30b-vf-lp` |
| hermes-4.3-36b | `run_20260906_043033_hermes-4.3-36b-vf-lp` |
| deepseek-v4-flash | `run_20260906_092955_deepseek-v4-flash-vf-lp` |
| qwen3.6-27b | `run_20260906_124956_qwen3.6-27b-vf-lp` |
| olmo-2-32b | `run_20260906_131256_olmo-2-32b-vf-lp` |
| llama-3.3-70b | `run_20260912_200213_llama-3.3-70b-vf-lp` |
| mistral-small-4-119b | `run_20260913_134219_mistral-small-4-119b-vf-lp` |

## 4. Caveats registry

Each is documented where it is established; this is the index.

| # | caveat | where |
|---|---|---|
| 1 | Batch-numerics artifact — probabilities served with >1 request in flight are batch-dependent; only sequential measurements are citable | 04 §4 |
| 2 | Date injection — Llama 3.3 and Mistral prompts carry the server date; tables are valid for the pinned date only | 04 §3 |
| 3 | Quantisation differs by model (Q8_0 to IQ3_XXS) and is part of each model's identity; its effect on the choice function is not yet measured (Gemma Q8 comparison prepared) | 04 §1 |
| 4 | Forced choice — grammar + reasoning off: the model cannot refuse or explain; one prompt wording (R3 dual-count), one temperature (0.3) | 02, 04 §2 |
| 5 | Sweep-era numbers (prompt sweeps, sampled tables, KV-cache study) are not citable; they chose the protocol only | 02, 03 §5 |
| 6 | Two significance procedures coexist: paired t (Holm) in the cross-model stage vs independent Welch t (Holm) in the script that writes the paper's ordering table. Neither has a practical floor: at n = 10,000 a ~0.002 DI gap is significant, so "distinguishable" is not "large" | 01 §3b |
| 7 | Finals are the value at the last simulated step; runs that hit the 1,000-step cap (most Olmo runs, DeepSeek racial) are not necessarily stationary | 01 §2 |
| 8 | DI is partition-dependent — levels do not compare across grid sizes; random-placement DI at 20×20 is 0.125 | 01 §3 |
| 9 | `ghetto_rate` is a count, not a rate; the 4- vs 8-neighbourhood split between metrics is historical | 01 §3 |
| 10 | Ratio-axis curves are a projection of the composition surface; scale-dependence family counts were measured on retired tables | 03 §3 |
| 11 | Sequential census: the paper reports the whole-table goodness of fit (all nine pass); it is weak against a single wrong interior cell. The per-cell fresh-draw check is internal: no flagged cell disagrees, but cells near p = 0 or 1 cannot be resolved by sampling at n ≤ 1,200 (the September "discrepant cells" were an artifact of the retired pooled escalation) | 03 §4 |
| 12 | Live-LLM campaigns (A2/A3) predate all of the above and are not used in the paper | — |
