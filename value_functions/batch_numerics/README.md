# Batch-numerics artifact — does a llama-server probability depend on the batch?

Everything about this artifact lives here: the probe, its data and its plots.

* `batch_numerics_probe.py` — for one model (server up with the campaign's
  flags), reads P(MOVE) from `/v1/chat/completions` post-sampling probabilities
  for the most sensitive unsaturated cells under three conditions — sequential
  (one request in flight), concurrent (four identical requests in flight),
  interleaved (mixed with other cells' requests) — plus sequential
  campaign-payload sampling, and the campaign's own rate with its Wilson CI.
* `results/probe_<label>.csv` — every request (condition, repeat, P(MOVE), prompt tokens).
* `results/probe_<label>_summary.csv` — per cell: sequential value and its
  range over repeats, concurrent / interleaved mean-sd-min-max, campaign p and
  CI, sequential-sampling p and CI, campaign − sequential.
* `results/probe_<label>.json` — server build/flags and aggregates.
* `results/probe_<label>.png` — the per-cell picture.

* A label with no sampled campaign (every model added since the sampling
  route was retired) is probed with `--from-exact`: the candidate cells are
  ranked on its exact tables and the campaign columns are NaN.
  `--match-cells <label>` adds the cells another probe measured, so two
  quantisations of one model (`gemma-4-31b-q8-chat-grammar` vs
  `gemma-4-31b-chat-grammar`, 2026-09-07) are compared cell for cell.
* `run_batch_numerics_study.py` — the study driver: for each model, starts
  its llama-server with the campaign flags, runs the probe, stops the server
  (`--dry-run` prints the plan, `--wait` defers until the port is free,
  positional labels select models). Not part of any pipeline; the study ran
  once (all nine models, 2026-09-06/07) and the CSVs are kept. Python since
  2026-09-07; the shell driver it replaces is gone.

All requests use the chat endpoint, the campaign's own; the first measurements
of this effect (2026-09-05) went through `/completion` and were discarded as
not comparable.

Finding (see `../KV_CACHE_SAMPLING_ARTIFACT.md` §8): sequential requests are
bit-reproducible; with four in flight the same request returns probabilities
spread over tens of percentage points on transition cells, flash attention on
or off, whatever shares the batch. The sampling campaigns ran four in flight,
so their `vf_*.json` tables are batch-state averages; the exact tables in
`../results/value_functions_logprob/` are sequential.

Across the nine models (2026-09-06/07) the size of the artifact tracks the
weight quantisation, not the architecture: the four Q8_0 models (phi-4,
OLMo-2, Hermes-4.3, granite-4.2 — all dense) have a max concurrent span of
0.15–0.37, every Q4_K_M / Q5_K_M / IQ3_XXS model (llama-3.3, qwen3.6, the
dense gemma-4-31B, and the two MoE models mistral-small-4 and
deepseek-v4-flash) 0.30–1.00. Spans near 1.0 are partly the T = 0.3
amplification (post-temperature logits are raw/0.3): gemma's 0.027→0.94
swing is ~1.9 raw nats, phi's ±0.05 is ~0.09. The Q8_0 Gemma probe
(`probe_gemma-4-31b-q8-chat-grammar_*`, same cells as the Q5_K_M one via
`--match-cells`) is the test of that reading.
