# 03 — The exact choice function: what is computed, validation, coupling

Extractor: `value_functions/logprob/logprob_value_function.py` (plan:
`value_functions/logprob/LOGPROB_PLAN.md`), consumed by `llm_runner.py
--value-function`. "Value function" in the code and "choice function" in the
paper are the same object. Tables carry the label suffix `-lp` and runs the
model suffix `-vf-lp`.

## 1. What is computed

Rather than calling the model live inside every simulation, the model is
measured once:

> p(role, composition, scenario) = P(the model answers MOVE | it is told it is
> a `role` resident of a `scenario` facing `n_similar` like and
> `n_occupied − n_similar` unlike neighbours).

A composition is a pair (n_similar, n_occupied) with 0 ≤ n_similar ≤
n_occupied ≤ 8: **45 compositions**, so **540 cells per model** across 2
roles × 6 scenarios. Every composition is its own cell — 1-of-2 and 2-of-4
are never pooled, even though they share the ratio 1/2.

A live agent's reply is generated token by token: at each step llama.cpp
draws from softmax(logits / T) restricted to the tokens the MOVE/STAY grammar
still allows. P(MOVE) is therefore an exact, finite sum over that closed
language:

    P(MOVE) = Σ over grammar-valid token paths ending in MOVE of
              Π_k p_T(token_k | prefix_k, allowed_k)

The extractor evaluates the sum directly:

- **Same request as a live agent.** `/v1/chat/completions`, the prompt as the
  single user message, so template rendering, BOS handling and any reasoning
  prefill are the server's own. A partial reply (a prefix state) is sent as a
  trailing assistant message that the server continues.
- **Per state, one request** returns the top 64 tokens with their
  probabilities after the sampler chain, i.e. after temperature
  (`post_sampling_probs`). T = 0.3, pure temperature sampling (top_k 0, top_p
  1, min_p 0, penalties off).
- **The grammar mask is applied client-side.** llama.cpp enforces grammar by
  rejection (draw, then resample if invalid), so the returned probabilities
  are not masked. A token is kept iff prefix + token is still a prefix of a
  string in the language (optional whitespace, then MOVE or STAY in any
  casing); the kept mass is renormalised. That is exactly the distribution
  the accept/resample loop realises.
- **Truncation is bounded, not ignored.** States are expanded in order of
  mass down to a floor of 1e-7 and depth 8; mass outside the returned top-64
  or below the floor accumulates in `mass_bound`. The largest `mass_bound` in
  any of the 9 × 540 cells is 1.2e-7, so the tables are exact to ~1e-7.
- **Cost.** 2–3 requests per cell (≈250 per scenario, ≈1,500 per model),
  issued strictly one at a time (04 §4).

Nothing is read from the sampled tables: scenarios come from
`scenarios_a2.py`, the temperature from `--temperature`, and a new model
needs no sampling campaign.

## 2. The artifact

`value_functions/results/llm_logprob/`:

- `tables/vf_<label>-lp__<scenario>__R3_dual_count.json` — the table the
  simulation loads (schema vf-1). `meta` records model file, endpoint arm,
  temperature, sampler params, grammar hash, scenario and identity labels,
  role→type map, server build and the extraction parameters;
  `compositions.{red,blue}` holds 45 rows with the exact rate in
  `p_move_effective` and ± `mass_bound` as its interval; `meta.source =
  "logprob"`. `ratios.{red,blue}` (the 23 reachable fractions, each the
  equal-weight mean of its member compositions) is a plotting summary only,
  not the policy.
- `raw/vflp_<label>__<scenario>__R3_dual_count_states.jsonl.gz` — the
  extraction record (schema vf-lp-1): one line per (role, cell) with p_move,
  p_stay, mass_bound, request count, prompt hash, every path with its mass
  and every state's top-n table.
- `validation_data/` — the checks of §4.

The nine tables extracted on 2026-09-05/06 still carry informational
`p_move_sampled` / `ci95_sampled` slots from the retired route; nothing reads
them.

## 3. Composition-level lookup, and why

The simulation reads the **composition** surface — the agent's exact
(n_similar, n_occupied) cell — with no aggregation. Ratio-level lookup was
removed (2026-08-25) because the decision is not a function of the fraction
alone: members of a ratio family (1/2 = 1-of-2, 2-of-4, 3-of-6, 4-of-8) share
a fraction but can differ sharply in P(MOVE), typically rising with absolute
count (one unlike neighbour and nothing else means STAY; several mean MOVE).

Two consequences for the paper. The 1-D "choice function" curves average
each ratio's member compositions, so their sawtooth is a projection artifact,
not noise; the heatmaps show the surface the simulation consumes. And the
ratio abstraction of the classical model is only an approximation of how
these agents behave.

(The family counts quoted in earlier revisions — gemma 15 of 72 families
rising, llama 17, qwen 26, none falling — were measured on the retired
sampled tables with `analysis_tools/vf_scale_dependence_probe.py` and have
not been recomputed on the exact tables; recompute before citing.)

## 4. Validation

The exact table is a computation, so the question is whether it predicts what
the server actually samples.

- **Sequential check (per extraction).** For 16–18 cells per model — the
  least saturated ones, chosen from the exact surface, plus 4–6 saturated
  controls — replies are sampled with the production payload, one request in
  flight, and the sampled rate compared with the exact value, escalating
  borderline cells. In all nine models every checked cell's exact value lies
  inside the sampled 95% interval; the median absolute difference per model
  is 0.001–0.015. Verdicts: `validation_data/validation_<label>.json`, cells
  in `seqcheck_<label>.csv`.
- **Sequential census (`value_functions/sampling/census.py`, `-vf-s`).** All
  540 cells × n = 100, one request in flight, under the model's pinned clock,
  and a 100-run sanity simulation from the sampled tables. Stores:
  `value_functions/results/sampled_small/`, runs `run_*_vf-s`. The census is
  judged by **two checks with two different jobs** (decision 2026-11-16):

  1. **Goodness of fit — the census result reported in the paper.** One test
     per model over all 540 cells, census draws only: the binomial deviance G
     of the census against the exact table (each cell's null is the exact
     value's mass-bound interval), against 10,000 censuses simulated from the
     exact table (fixed seed). PASS if p > 0.05. All nine models pass
     (p = 0.27–0.94):

     | Model | G | null mean | null 95th pct | p |
     |---|---|---|---|---|
     | gemma-4-31B | 7.3 | 10.4 | 19.2 | 0.71 |
     | qwen3.6-27B | 188.5 | 177.6 | 210.1 | 0.27 |
     | llama-3.3-70B | 47.9 | 47.9 | 64.7 | 0.47 |
     | hermes-4.3-36B | 144.1 | 149.2 | 179.1 | 0.59 |
     | deepseek-v4-flash | 140.0 | 138.0 | 166.8 | 0.43 |
     | mistral-small-4-119B | 40.0 | 40.0 | 55.9 | 0.46 |
     | granite-4.2-30B | 28.7 | 37.2 | 52.8 | 0.84 |
     | phi-4-14B | 28.8 | 41.0 | 56.6 | 0.94 |
     | olmo-2-32B | 526.6 | 515.5 | 565.3 | 0.36 |

     `value_functions/comparison/census_goodness_of_fit.py` writes
     `experiments_with_llama_cpp/cross_model/census_goodness_of_fit.{csv,png}`;
     it is the `census_gof` stage of the orchestrated pipeline, regenerated
     after the cross-model stage. What it detects (measured on hermes): every
     interior cell off by 0.02 is caught 72% of the time, by 0.05 always; a
     near-0 cell at 0.10 instead of 0.001, 93%. A single interior cell off by
     0.2 is caught only 26% of the time — a whole-table test is built to
     catch a failing extraction method, which would move many cells at once.
  2. **Per-cell check — an internal sanity check, not a paper result.** Is the
     extracted table a good cell-by-cell replacement for sampling? Cells whose
     census Wilson interval misses the exact value get ONE round of fresh,
     never-issued-seed draws (sized from the census for 80% power at an odds
     ratio of 2, 300–1,200 draws), tested on the fresh draws only (two-sided
     exact binomial, Holm across the flagged cells). Result (September census):
     72 flagged cells across the
     nine models, **none disagrees**; most of the rest are near p = 0 or 1,
     where 1,200 draws cannot resolve a doubling (labelled BEYOND_CAP —
     unverified, not confirmed). Verdicts in
     `seqcheck_plots/sanity_vs_exact_<label>.json` (`goodness_of_fit` and
     `sanity_check` blocks).

  The September escalation's own verdict (hermes and olmo "DISCREPANCY") is
  retired: it re-tested the flagged cells on pooled draws that kept the
  draws which flagged them, and replayed seeds across rounds; see the
  `sanity_vs_exact.py` docstring.
- **Rank stability (`value_functions/comparison/vf_rank_stability.py
  --exact`).** Certifies each run's scenario ordering on the dissimilarity
  index. Exact tables carry no sampling error, so there is no multi-split
  ruler; the 0.01 DI floor-tie exists only in this certification.

## 5. The retired sampled route

Until 2026-09-06 the tables were *sampled* (`-vf-r3`): a pilot of 100 replies
per cell, topped up until every Wilson 95% interval was within ±2 pp
(`value_functions/sampling/build_value_function.py`,
`VALUE_FUNCTION_SAMPLING_FOR_REVIEW.md`). They were collected with four
requests in flight and so carry the batch-numerics artifact (04 §4): they are
batch-state averages, not the sequential probabilities. They are not a
result, are never mixed with exact tables in any figure, and the store was
deleted on 2026-09-15; the evidence that retired them is in
`value_functions/batch_numerics/results/`.

## 6. Simulation coupling

`llm_runner.py --value-function <file | '{scenario}' template | directory>`
loads the table, maps roles via `meta.role_to_type`, and at each decision
draws MOVE with the probability stored for the agent's exact composition
cell (a keyed uniform draw, 01 §2). All 45 cells per role must carry a
defined rate, validated at load time; a scenario-mismatch guard refuses a
table extracted under a different social framing. No server or GPU is
involved.

The YAML orchestrator (`configs/vf_run_<model>_lp.yaml`) runs simulation and
analysis as one command and **freezes** the resolved tables, their figures, a
composition-heatmap grid and a `TABLE_HASHES.json` into
`<run_dir>/value_functions/`, pointing the simulation at the frozen copies.
Every run folder therefore carries the exact decision tables it simulated
from, and two runs used the same tables iff their hashes match.
