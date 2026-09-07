# 05 — Reproducibility: environment, commands, data locations, caveats

## 1. Environment

- llama.cpp `llama-server`, build **b1-a4ce259**; GPU NVIDIA GB10, full
  offload. The build and model path are recorded in every artifact's
  `meta.sampling_protocol.server`.
- Model: `llms/gemma-4-31B-it-Q5_K_M.gguf`.
- Python: the repo `.venv`. No GPU is needed for simulations or analysis.
- Bitwise LLM replication is promised **per (build, model file, hardware)**
  only; see the reproducibility table in `LLAMA_CPP_SERVING_NOTES.md`.

## 2. Commands, stage by stage

**Serve** (measurement protocol):

```bash
llama-server -m llms/gemma-4-31B-it-Q5_K_M.gguf --alias gemma-4-31B-it-Q5_K_M \
  -ngl -1 -fa on -np 4 -c 8192 --host 127.0.0.1 --port 8085 --jinja --reasoning off
```

**Build the value functions** (pilot → ±2 pp top-up loop → figures):

```bash
python prompt_refinement/build_value_function.py --config configs/value_function_scenarios_gemma.yaml --plot
python prompt_refinement/build_value_function.py --config <same> --top-up --precision 0.02
python prompt_refinement/plot_value_functions.py --label gemma-4-31b-chat-grammar
```

Repeat the top-up until `--dry-run` reports zero deficits.

**Simulate + analyse** (no server needed):

```bash
python run_llm_probability_simulation_analysis.py \
    --config-yaml configs/vf_run_gemma_lp.yaml --config-profile production
```

**Bitwise audit** (same server build):

```bash
python prompt_refinement/replication_kit.py --verify
```

**Prompt sweeps** (historical, cache-on era — reproduce qualitatively):
`prompt_refinement/evaluate_prompts.py` and `evaluate_ratio_prompts.py`; slot
calibration: `python slot_sweep/sweep_slots.py --model <label>`.

Shell wrappers exist for the multi-stage campaigns but are machine-specific
and not versioned; the Python commands above are the reproducible interface.

## 3. Data locations

| data | location |
|---|---|
| value-function artifacts (vf-1 JSON) + figures | `prompt_refinement/results/value_functions/` |
| per-reply raw logs (with seeds) | `prompt_refinement/results/value_functions/raw/` |
| archived cache-on artifacts (evidence) | `prompt_refinement/results/value_functions_cacheon_archive/` |
| contamination map | `prompt_refinement/results/value_functions/vf_contamination_map.{csv,png}` |
| bitwise replication reference | `prompt_refinement/results/replication_reference_*.json` |
| sweep tables, raws, figures | `prompt_refinement/results/` |
| simulation experiments | `experiments/` (ad hoc), `experiments_with_llama_cpp/run_*/` (orchestrated) |
| frozen decision tables per run | `<run_dir>/value_functions/` (+ `TABLE_HASHES.json`) |
| slot sweep results | `slot_sweep/results/<label>/` |

Simulation result data is deliberately **not** version-controlled: a single
10k-run batch is ~3 GB across 20k files. Runs are seeded by `run_id`, so a
batch is regenerated exactly from its frozen config and value functions.

## 4. Caveats registry

Each is documented where it is established; this is the index.

| # | caveat | where |
|---|---|---|
| 1 | KV-cache sampling artifact — cache-on measurements biased at transition cells | 04 |
| 2 | Plain-endpoint silent misparse (~0.08% overall, biased toward MOVE); grammar arms immune | 02 |
| 3 | Forced-choice arms — grammar readings on models that will not answer unconstrained are protocol-constructed | 02 |
| 4 | `step_statistics.csv` (pre-2026-09-05 dirs) is survivor-biased; use `metrics_history.csv.gz` with forward-fill | 01 |
| 5 | `ghetto_rate` is a count, not a rate; 4- vs 8-neighbourhood split is historical | 01 |
| 6 | Two run-record formats; read both through `run_files.py` | 01 |
| 7 | DI is partition-dependent — levels do not compare across grid sizes | 01 |
| 8 | Live-LLM production runs (A2/A3) ran cache-on: extra near-tie noise, no stable bias | 04 |
| 9 | Multi-slot FP jitter flips ~3% of near-tie draws; bitwise claims need the serial audit tier | 04 |
