# 04 — Serving infrastructure: llama.cpp, grammar, cache artifact, seeds

Deep notes: `prompt_refinement/LLAMA_CPP_SERVING_NOTES.md`,
`GRAMMAR_GBNF_NOTES.md`, `KV_CACHE_SAMPLING_ARTIFACT.md`.

## 1. Serving stack

Local llama.cpp `llama-server` (build `b1-a4ce259`), one model per server;
production model `gemma-4-31B-it-Q5_K_M.gguf`. Standard flags:

```
-ngl -1 -fa on -np 4 -c 8192 --host 127.0.0.1 --port 8085 --jinja --reasoning off
```

— full GPU offload, flash attention, 4 parallel slots, the model's own chat
template with reasoning channels closed (required for the Gemma/Qwen/DeepSeek
chat arms; Llama is natively non-reasoning). The sampler is pinned to pure
temperature (T=0.3; top_k=0, top_p=1, min_p=0, all penalties off) via
`llm_runner.SAMPLER_PARAMS`, identically in every harness.

Slot count and client concurrency come from `slot_sweep/sweep_slots.py`, which
saturates a dedicated server with the real payload: decode is
memory-bandwidth-bound, so batching is nearly free until bandwidth saturates,
after which extra slots only add latency. The campaign used `-np 4` with
matching concurrency 4.

## 2. Grammar (GBNF)

Replies are constrained to `optional-whitespace + MOVE|STAY (any casing)` by
`llm_runner.MOVE_STAY_GRAMMAR`; llama.cpp masks illegal tokens and
renormalises, guaranteeing parseability (0 bad in 473,625 samples) and
halting. Caveats — forced-choice measurement, per-token renormalisation is not
conditioning on grammaticality, reasoning-model interaction — are in
`GRAMMAR_GBNF_NOTES.md`. Grammar does **not** protect against §3.

## 3. The KV-cache sampling artifact (found and remediated 2026-08-22)

With server prompt caching on (the default), the sampled MOVE probability of a
byte-identical prompt depended on slot cache history: episode-stable shifts of
up to ~44 pp at near-tie compositions, zero effect at saturated ones.
Temperature amplifies logit perturbations by 1/T in log-odds, which is why the
effect concentrates where the model is torn.

It was detected by the top-up recalibration check (a >5σ shift in the pooled
p̂), diagnosed from the stored raw replies, and causally confirmed by a
cache-off A/B and a cache-priming experiment. Remediation: a **clean
protocol** for all measurement — `cache_prompt:false` plus deterministic
per-request seeds (crc32 of the sample's identity tuple) — and a full 540-cell
resample. The archived cache-on artifacts and a per-cell clean-vs-archived
contamination map, with saturated cells as negative controls, are released as
evidence (`KV_CACHE_SAMPLING_ARTIFACT.md`).

This matters beyond this study: the bias is stable within an episode and
therefore invisible to any within-batch consistency check.

## 4. Reproducibility tiers

- **Simulations** — fully deterministic. `random_seed = run_id` seeds numpy
  and stdlib random per run, so run k is paired across batches, which is what
  makes the paired half-vs-full sufficiency test and the paired mechanical
  comparisons valid.
- **LLM sampling, campaign tier** — concurrency 4, cache off, per-request
  seeds: distribution-replicable. Residual multi-slot floating-point jitter
  flips ~3% of near-tie draws on an exact rerun (44 vs 47 of 100 on an
  identical seeded batch), absorbed by the ±2 pp CIs.
- **LLM sampling, audit tier** — `prompt_refinement/replication_kit.py` runs a
  fixed 600-request manifest STRICTLY SERIALLY, which is bitwise reproducible
  on a given server build (20/20 identical across passes, immune to
  interleaving and to adversarial cache priming). `--generate` stores
  reference outputs, sha256 and the server fingerprint; `--verify` requires
  bitwise identity and refuses cross-build comparisons. This is the
  third-party verification artifact.
