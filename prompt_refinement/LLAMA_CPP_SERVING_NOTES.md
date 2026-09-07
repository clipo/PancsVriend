# llama.cpp serving concepts: slots, KV cache, seeds — function, advantages, issues

How our llama-server deployment actually serves requests, what each mechanism
buys, and where each one can bite. Written 2026-08-22 after the KV-cache
sampling artifact was found (full forensic account:
`KV_CACHE_SAMPLING_ARTIFACT.md`). Companions: `GRAMMAR_GBNF_NOTES.md`,
`NOTES.md`, `SAMPLING_METHODOLOGY.md`.

Server invocation used throughout this campaign:
`llama-server -m <gguf> -ngl -1 -fa on -np 4 -c 8192 --jinja --reasoning off`.

## 1. Slots and continuous batching

**Function.** `-np 4` gives the server 4 parallel "slots", each a generation
stream with its own KV-cache region (context divided: 8192/4 = 2048 per
slot). Continuous batching co-processes whatever tokens all active slots need
each step.

**Advantage.** Throughput: ~4 concurrent requests; our measured sampling rate
was ~200 requests/min with caching.

**Issues.**
- **Multi-slot ⇒ nondeterminism — and NOT small (2026-09-05, KV_CACHE_SAMPLING_ARTIFACT.md §8: tens of percentage points on transition cells; one request in flight is bit-reproducible; measure sequentially).** Logits are not bit-identical across batch
  compositions (reduction order, kernel tiling differ with what else is in
  flight). The llama.cpp docs/discussions state plainly that results are not
  guaranteed reproducible with `cache_prompt`/multi-slot serving
  ([Discussion #10311](https://github.com/ggml-org/llama.cpp/discussions/10311)).
  At temperature T the effect on a two-way decision is amplified 1/T in
  log-odds (T=0.3 ⇒ ×3.3) — negligible at saturated decisions, real at
  near-ties (we measured a repeat of an identical seeded batch giving 44/100
  vs 47/100).
- Slot *selection* is by longest-common-prefix similarity
  (`--slot-prompt-similarity`, log lines `sim_best=... f_keep=...`), which
  couples a request's numerical fate to which slot's history it lands on —
  see §2.

## 2. KV cache and prompt caching (`cache_prompt`)

**Function.** The KV cache stores per-token key/value tensors. Prompt caching
reuses a slot's cached prefix for a new request sharing that prefix,
recomputing only the tail; partially matching caches are salvaged via
approximate repositioning (cache shift / RoPE re-rotation).

**Advantage.** Large prompt-eval savings: our ratio prompts re-evaluated only
~16.5 of ~150+ tokens per request (~10× less prompt compute).

**Issues.**
1. **State-dependent logits ⇒ biased sampling at near-tie decisions (our
   finding).** Cached KV carries the floating-point history of the batches
   that computed it, and cache shifts are explicitly approximate — so the
   logits of a decision token depend on what the slot served before. Measured
   effect on one transition cell: episode-stable MOVE rates of 0.46 / 0.02 /
   0.25 / 0.02-0.10 across sampling episodes with byte-identical prompts and
   parameters; `cache_prompt: false` restores a stable, binomially-scattering
   0.42–0.47 in the same sessions. Full evidence, including a cache-priming
   experiment: `KV_CACHE_SAMPLING_ARTIFACT.md`.
2. **Documented cache-contamination precedent.** The same mechanism family
   produced silently wrong outputs when cached prompt KV was reused across
   requests selecting different LoRA adapters
   ([Issue #26207](https://github.com/ggml-org/llama.cpp/issues/26207)) —
   i.e., cache reuse trusting an equivalence ("same prefix text ⇒ same
   state") that does not actually hold.
3. **Gemma/SWA complications.** Sliding-window-attention models (the Gemma
   family) have known cache-reuse limitations — the server may refuse or
   mishandle reuse ([Issue #21468](https://github.com/ggml-org/llama.cpp/issues/21468),
   [Issue #21831](https://github.com/ggml-org/llama.cpp/issues/21831)) —
   making cache behaviour on our primary model doubly version-sensitive.

**Operating rule (this project).** MEASUREMENT (value-function sampling,
prompt sweeps): `cache_prompt: false` — distribution fidelity over speed.
PRODUCTION live-LLM simulations: caching may stay on for throughput, with the
explicit caveat that near-tie decisions carry cache-state noise (record this
in any writeup using those runs).

## 3. Sampling seeds and reproducibility

**Function.** Each request may carry a `seed`; the default is `-1` = a random
seed per request ([server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
[llama-server manpage](https://manpages.debian.org/testing/llama.cpp-tools/llama-server.1.en.html)).
Historical note: some past versions silently ignored per-request seeds
([Issue #7381](https://github.com/ggml-org/llama.cpp/issues/7381)) — worth a
one-off verification per server build (send two requests with the same seed
at T≫0 and confirm identical output when run serially).

**Status of this campaign's runs.**
- LLM sampling runs (all sweeps + value-function sampling to date): **no
  seed sent** ⇒ random per request ⇒ not bit-replicable; with caching on,
  not even distribution-replicable (the distribution depended on cache
  state).
- Clean protocol (cache off + explicit distinct per-request seeds):
  distribution-replicable, approximately seed-replicable; exact bitwise
  replay is still not guaranteed under multi-slot batching (§1). Single-slot
  serial replay is the only fully deterministic configuration.
- Simulations driven by value functions: fully replicable since 2026-08-21
  (`random_seed=run_id` seeds numpy + stdlib random per run).

## 4. Can the cache be kept without the §2.1 problem?

Candidate middle grounds, none currently validated:
- **Exact-full-prefix reuse only** (no partial keep/shift): removes the
  approximate-shift channel but keeps FP-history in retained KV — plausibly
  much smaller effect; would need the same priming-style validation before
  trust.
- **Single slot (`-np 1`) with cache**: removes cross-request batch
  variation; ~4× throughput cost; prompt-vs-generation batch-size difference
  remains.
- **Cache off** (chosen): the only configuration with demonstrated stable,
  pilot-consistent, binomially-behaved measurements; ~2× per-request cost on
  our prompt lengths.

## 5. Quick reference: what is reproducible when

| configuration | same numbers on rerun? |
|---|---|
| cache on, no seed, -np 4 concurrent (historical sweeps) | no — and distribution itself state-dependent at near-ties |
| cache off, no seed, -np 4 concurrent | **distribution NO** (batch-dependent, tens of points at transition cells — §8 of the KV doc); counts no |
| cache off, fixed seeds, -np 4 concurrent | approximately — measured 44 vs 47/100 on an identical seeded batch (multi-slot batch-composition FP noise flips ~3% of near-tie draws) |
| cache off, fixed seeds, SERIAL submission (one in flight; works even on an -np 4 server) | **yes — bitwise, measured**: 20/20 identical outputs across two passes; same seed ×10 → identical (2026-08-22, this build) |

## Sources

- [Discussion #10311 — cache_prompt & determinism](https://github.com/ggml-org/llama.cpp/discussions/10311)
- [Issue #26207 — prompt cache reused across different LoRA adapters, contaminated output](https://github.com/ggml-org/llama.cpp/issues/26207)
- [Issue #21468 — cache reuse unsupported for Gemma 4 (SWA)](https://github.com/ggml-org/llama.cpp/issues/21468)
- [Issue #21831 — forced full prompt re-processing (SWA/recurrent memory)](https://github.com/ggml-org/llama.cpp/issues/21831)
- [Issue #7381 — per-request seed ignored (historical)](https://github.com/ggml-org/llama.cpp/issues/7381)
- [llama-server README (request fields, defaults)](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [llama-server(1) manpage (seed default -1 = random)](https://manpages.debian.org/testing/llama.cpp-tools/llama-server.1.en.html)
- [KV cache reuse tutorial (Discussion #13606)](https://github.com/ggml-org/llama.cpp/discussions/13606)
