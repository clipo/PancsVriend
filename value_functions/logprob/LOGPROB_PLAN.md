# Plan: exact value functions from grammar-masked logprobs

Status: IMPLEMENTED 2026-09-05 as `value_functions/logprob/logprob_value_function.py`
(+ `run_logprob_vf_campaign.sh`, `batch_numerics/`). The plan below is kept as
written; what was actually built differs in three ways, recorded here:

1. **Server, not in-process.** The path sum is evaluated against the campaign's
   own llama-server, through **`/v1/chat/completions`** with the user prompt as
   the single message — so template, BOS, reasoning prefill, tokenizer and
   sampler chain are the campaign's by construction. Prefix states are
   assistant prefills. (A first version used `/completion` on the
   `/apply-template` rendering; it matched the chat path for qwen but NOT for
   gemma-4 — same token count, P(MOVE) 0.999 vs 0.022 — so every output from
   that path was discarded and re-extracted.) Per-state probabilities come
   from `post_sampling_probs` (after temperature); the grammar mask is applied
   client-side because llama.cpp applies grammar by rejection.
2. **Validation redefined.** Comparing exact values to the campaign surface is
   an ARTIFACT MAP, not the test: the campaigns ran four requests in flight
   and this server returns batch-dependent probabilities (tens of points on
   transition cells; `KV_CACHE_SAMPLING_ARTIFACT.md` §8, `batch_numerics/`).
   The pass/fail test is sequential re-sampling of the 12 most-disagreeing
   cells (300 draws each, one in flight): exact must lie inside ≥ 10/12 Wilson
   CIs, and saturated cells inside their bounds. All extraction requests are
   sequential.
3. **Consequence for the sampled tables.** They are batch-state averages;
   the exact tables replace them for simulation (`-vf-lp` runs, rank
   stability in `--exact` mode). Whether any ordering moved is measured, not
   assumed (qwen: none).

## Why now

The sampled value functions (`vf_*.json`, R3_dual_count, chat+grammar, T=0.3)
are fine where they are unsaturated (top-up to ±2 pp) and blind where they are
saturated: 76–98 % of cells are 0/100 or 100/100, whose true p is only known to
be < 0.03 (rule of three). Whether a frozen model freezes at all, and whether a
certified ordering survives value-function error, depends on those cells.
Sampling cannot buy that: bounding p < θ costs ~3/θ draws per cell (n ≈ 1000
per cell for θ = 0.003 → ~50 GPU-h per model), and the multi-split ruler
assigns those cells exactly zero uncertainty by construction.

A model we can query for logits gives P(MOVE) exactly. The grammar is what
makes this clean: under the grammar arm the sampler at every step masks the
vocabulary to tokens consistent with `whitespace* · (MOVE|STAY)` (any casing)
and samples from softmax(logits/T) over the allowed set. The sampled effective
rate is therefore, exactly,

    P(MOVE) = Σ_{grammar-valid paths ending in MOVE} Π_k p_T(token_k | prefix_k, allowed_k)

— a sum over a CLOSED, small language. The earlier attempts failed for
identifiable reasons, not because the quantity is ill-defined:

* `llm_token_probabilities.py` (in the former `llm_utility_approximation/`, folder removed 2026-09-05) read the FIRST token
  only. The user observed the failure mode directly: `"\n"` can carry most of
  the first-token mass and the answer after `"\n"` can favour the other word,
  so the first-token argmax/ratio is not P(MOVE). (Superseded; ignore.)
* `branching_probability_estimator.py` (same folder, removed 2026-09-05) was the right
  idea — enumerate token paths with in-process llama_cpp, credit mass to
  MOVE/STAY — but enumerates FREE generation: full-vocabulary softmax, beam
  width 16, `unknown_mass` for pruned paths, substring `parse_decision`, the
  old 3×3-map prompt fed as raw completion text. It therefore estimates a
  different process from the grammar arm, and validated it against
  free-generation sampling (retries, junk tokens). That is why the
  Gemma-4-31B branching-vs-sampling comparison was inconclusive.

## Method

For each (scenario, role, composition) — 6 × 2 × 45 = 540 prompts per model:

1. **Prompt bytes identical to the sampled arm.** Render with
   `evaluate_ratio_prompts.render_prompt(R3_dual_count, ...)` (same
   `role_keywords`, same `scenarios_a2.py` labels), then apply the model's
   OWN chat template exactly as llama-server did for the chat+grammar arm
   (`--jinja`; gemma with `--reasoning off`). Assert token-sequence equality
   against the server's `/apply-template` (or `/tokenize` of the rendered
   text) for a sample of prompts before any measurement. This is the single
   biggest risk (CHAT_TEMPLATE_EFFECTS.md) and the likeliest cause of the old
   inconclusive result.
2. **Grammar state machine, not substring parsing.** State = text generated so
   far. A token is allowed iff `text + detok(token)` is a prefix of some string
   in L = WS* · (MOVE|STAY) with case-insensitive word matching, where WS
   tokens are allowed only before the word starts. A path is complete when the
   word is complete (the grammar ends there; nothing after the word counts).
   Match `sampling_common.MOVE_STAY_GRAMMAR` exactly (verify the admitted
   casings/whitespace against the GBNF, GRAMMAR_GBNF_NOTES.md).
3. **Per-step masked, renormalised distribution at the production
   temperature.** From the full logit vector (`llm.scores` in-process — not
   `n_probs`, which truncates), keep the allowed tokens, `softmax(logits/T)`
   over them. Scale-then-mask equals mask-then-scale for temperature alone;
   the campaign's sampler params (top_k 0, top_p 1, min_p 0, penalties off)
   make temperature the only transform, so this reproduces the server's
   sampling distribution.
4. **Exact expansion with a mass bound.** Expand states in order of mass.
   Whitespace prefixes decay geometrically; stop when the unexpanded mass is
   < 1e-6 and report it as the bound on the error of P(MOVE). No beam, no
   pruning, no "unknown" bucket. Expect 5–20 forward passes per prompt,
   ~5k per model, minutes on the GPU (KV prefix cache makes prefix
   extensions cheap).
5. **Store per-state allowed-token logits**, so the value function can be
   re-rendered at any temperature offline (the "renormalisable to any T"
   property of FUTURE_EXPLORATIONS §5 holds only with the logits kept).

## Validation protocol (before anything replaces a sampled table)

Per model, all 540 cells:

* `|P_logprob − p_sampled|` against the sampled cell's Wilson 95 % CI. Pass
  criterion: ≥ 95 % of UNSATURATED cells inside their CI, and no cell outside
  by more than 2 × its half-width. This is a sharp test because the two
  compute the same process; a systematic miss points at a specific defect
  (template, tokenisation, temperature order, EOS handling), which the
  per-path breakdown (kept for every cell) will localise.
* Saturated cells: P_logprob must lie below the 0/n upper bound (or above the
  n/n lower bound). Report the distribution of P_logprob on 0/100 cells —
  this is the number that decides whether "frozen" models are frozen.
* Order of models: qwen first (most unsaturated cells → the test has the most
  power), then gemma (chat template + reasoning-off, the riskiest template),
  then the rest.

## Outputs

`value_functions/results/llm_logprob/`  (moved here 2026-09-07 from
`prompt_refinement/results/value_functions_logprob/`, then briefly
`llm_log_probs/value_functions_logprob/`; since 2026-09-07 it is
`value_functions/results/llm_logprob/`, see value_functions/paths.py)
  `vflp_<label>__<scenario>__R3_dual_count.json`   schema vf-lp-1: per cell
      p_move (exact), mass_bound, n_paths, per-path breakdown, allowed-token
      logits per state, provenance (gguf sha, template hash, T, grammar sha)
  `seqcheck_<label>.csv` + `seqcheck_plots/seqcheck_<label>.png`   THE TEST:
      adversarially chosen cells re-sampled SEQUENTIALLY (n=300, escalated to
      900 on a first-stage CI miss). The plot shows the sequential arm only.
      `validation_<label>.json` carries the PASS/FAIL verdict.
  `OUTDATED_artifact_samples_vs_exact_<label>.csv` + `.png`   one-off map of
      the superseded concurrency-4 samples against the exact tables (was
      `validation_*`, then `exact_vs_concurrent_campaign_*`, renamed
      2026-09-07). Evidence OF the batch-numerics artifact; not a pass/fail
      record, not a value function, and not used going forward.
  `raw/`                                           per-prompt traces (jsonl.gz)
The script: `value_functions/logprob/logprob_value_function.py`, standalone, reusing
the branching estimator's model loading / expansion / trace / resume
skeleton. It must not touch `results/value_functions/` (the sampled store).

## What happens if it validates

* Build "exact" artifacts in the vf-1 schema (so `llm_runner --value-function`
  consumes them unchanged) with `p_move_effective` from logprobs and the
  sampled counts kept alongside for provenance.
* Re-run the production simulations from the exact tables (the orchestrator
  chain; ~1 h per model at 18 processes). The ruler is then exactly zero —
  `vf_rank_stability` already handles `s_obs <= 0` — and the multi-split,
  the sampling plan and the keep-fraction diagnostic are retired for any
  model with logit access.
* For future models (including a from-scratch one): value function = 540
  prompts, minutes, no campaign; temperature becomes a free axis.

## What happens if it does not validate

Stop and report the disagreement pattern. Disagreement on unsaturated cells
after the template check passes would mean the server's grammar sampling is
not the renormalised softmax assumed here (e.g. sampler order, a
grammar-implementation detail) — a finding about the measurement channel that
must be understood before either number is trusted.

## Risks, in order

1. Chat-template mismatch between in-process rendering and the server arm.
2. Tokeniser splits of MOVE/STAY differing between casings (handled by the
   state machine, but must be verified against the GBNF's admitted set).
3. Quantised GPU inference is not bit-deterministic across batch layouts;
   differences are ~1e-6 in p and irrelevant, but the validation should not
   assert exact reproducibility, only CI containment.
   **Revised 2026-09-12:** the "~1e-6" estimate is right for the numerics.
   What moved llama's table by up to 0.45 in p between 2026-09-06 and 09-11
   was the chat template: llama.cpp injects today's date and the Llama-3
   template prints it, so the prompt changed daily. The server's clock is
   pinned (libfaketime) for Llama-3 and Mistral, the trace records the
   rendered prompt, and a second extraction in a fresh session
   (`comparison/reextract_diff.py`, 540/540 within 1e-6) is the determinism
   test. See LLAMA_CPP_SERVING_NOTES.md §6.
4. Temperature/mask order if any sampler other than temperature were active
   (they are not, per the artifacts' `sampler_params`).
5. EOS: the grammar terminates at the word; the estimator must not require
   or credit an EOS token after it.
