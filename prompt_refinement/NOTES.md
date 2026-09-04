# Prompt-refinement analysis notes — caveats for reading the results

Short reference for interpreting the ratio-sweep outputs (`ratio_comparison_*`,
`ratio_consistency_*`, figR1–figR5). Longer companions:
`CHAT_TEMPLATE_EFFECTS.md` (what chat/jinja wraps around the prompt),
`SAMPLING_METHODOLOGY.md` (two-stage top-up sampling design + n(p) chart),
`GRAMMAR_GBNF_NOTES.md` (constrained decoding: function/advantages/caveats),
`LLAMA_CPP_SERVING_NOTES.md` (slots, KV cache, seeds — and their issues),
`KV_CACHE_SAMPLING_ARTIFACT.md` (the cache-state sampling bias: full forensics),
`FUTURE_EXPLORATIONS.md` (deferred probes).

## Percent styles (R4_percent_opposite, R5_percent_similar) carry baggage

They are **artifact probes, not equal candidates**. Know this before using
their curves:

1. **Genre priors.** "62.5% of your neighbors are X" reads as
   demographic-statistics prose; models have strong learned continuations for
   that genre that need not reflect a residential decision. Count forms
   ("5 of your 8 neighbors") are closer to lived description.
2. **Tokenization / round-number anchors.** Decimals ("62.5") tokenize
   irregularly; 50% / 75% / 100% act as attractors.
3. **No absolute count — deliberate aliasing.** The percent styles state no
   occupancy, so "50%" is the same prompt for 1-of-2 and 4-of-8. Any
   occupancy-dependence in their surfaces is therefore *invisible to the
   model* and shows up as unexplained spread.
4. **Rounding aliasing off-eighths.** Non-eighth ratios render to 1 decimal
   (2/3 → "66.7%"), so nearby ratios can collide across occupancies.
5. **Winner guard (analyze_ratio_consistency.py).** A percent style may rank
   first on smoothness/centrality *because of* the genre prior — plausible
   curves, wrong object. The winner rule therefore bars percent styles unless
   they AGREE with the count forms (warning printed, human decides). If they
   agree, choosing them is harmless; if they disagree while ranking first,
   the disagreement is the finding.
6. **Pilot evidence (qwen, comp+grammar, N=25):** the encoding effect was
   small — R1≈R4 and R2≈R5 within valence — so the guard likely never binds.
   The large phrasing effect was **valence** (opposite-framed R1/R4 threshold
   ≈0.6–0.65 vs similar-framed R2/R5 ≈1.0), not number format.

## Other caveats when reading sweep results (one-liners)

- **Role asymmetry dominates everything** (pilot): blue-role active in all
  styles, red near-frozen; max ΔP = 1.0. Never average roles; the historical
  `prompt_comparison_*` sweep is RED-ONLY evidence.
- **Pooling rule (winner selection): aggregate ONLY over presentation
  dimensions** — prompt styles and endpoint/grammar arms. Never across models,
  never across agent roles, never (when added) across social contexts: those
  are the measured objects, each with its own value function. Winners are
  chosen per (model, role).
- **G0 (grid anchor) shows empties**; its interior brightness is a
  vacancy-as-MOVE-cue effect the ratio styles deliberately exclude — G0 vs
  ratio differences bundle representation + vacancy visibility.
- **Chat arms are wrapped by model-specific templates** — llama injects a
  dates system block, mistral injects a ~2.4k-char "Le Chat" assistant
  identity (conflicts with our persona). See CHAT_TEMPLATE_EFFECTS.md before
  attributing chat-arm anomalies to the styles.
- **Resolution is ~1/N per cell** (N=25 pilot, N=50 Phase A); at T=0.3 the
  sampler is near-deterministic, so a 0% cell means "no movement at this
  composition", not missing data.
- **All analysis and plots use the EFFECTIVE rate n_move/(n_move+n_stay)** —
  the production-faithful quantity (production retries until parseable). The
  ambiguous column name `move_rate` is ELIMINATED (2026-08-01): new ratio CSVs
  ship only `move_rate_raw` (single-shot, bad parses in denominator) and
  `move_rate_effective`; the loader normalizes legacy files to the same two
  names and drops `move_rate`, so stale code fails loudly instead of silently
  using the raw rate. Old-harness (wide) CSVs gain `move_rate_eff_{k}of8`
  columns in new runs; fig1/fig2 prefer them and every figure states which
  rate it shows in its title. All-unparseable cells have NO effective rate —
  they appear as gaps (production would crash there, not decide).
  Both are computable from the long-format CSVs. Checked 2026-08-01: where
  parsing is healthy the two coincide and plain≈grammar (llama both-roles:
  mean |d| 0.005–0.025 ≈ sampling noise). Where parsing is unhealthy the
  conditional correction does NOT reconcile plain with grammar (deepseek plain
  completions: mean |d| 0.217 even conditional; qwen plain completions: cells
  with zero valid replies) — the parseable subset is selection-biased, so
  unhealthy plain arms are NOT comparable to grammar arms at all, corrected or
  not. Grammar doesn't merely "repair" those arms; it is a different, and the
  only well-measured, channel there.
- **KV-cache caveat on historical numbers** (2026-08-22): all sweeps before
  the clean protocol ran with server prompt caching — transition-cell values
  and figR3 implied thresholds carry episode bias
  (KV_CACHE_SAMPLING_ARTIFACT.md). Clean-protocol value functions supersede.
- **Mechanical reference** = MOVE iff n_opposite/n_occupied > 0.5 (occupied
  denominator, empties/walls excluded), STAY at zero neighbors.

## Prefer GRAMMAR arms — the plain parser can misparse SILENTLY (2026-08-13)

Measured across every `results/raw/*_raw.jsonl.gz`. This is a *parser* defect,
independent of the forced-choice question below, and it is the strongest
argument for making grammar the default production channel.

The MOVE/STAY rule (`evaluate_prompts.py:parse`, and identically
`llm_runner.py` ~line 667) is a bare substring test:

    has_move, has_stay = "MOVE" in text_upper, "STAY" in text_upper

Both present -> AMBIGUOUS (counted bad, retried). One -> that decision. Two ways
this records a WRONG decision while reporting a clean parse:

1. **Truncated enumeration.** `max_tokens=5`, so a model that lists the options
   is cut off before the second keyword: `'MOVE\nor\nST'`, `'> MOVE\n\nST'`,
   `' (MOVE or ST'`. The ambiguity guard never fires and the reply is recorded
   as a confident MOVE. **358 / 460,800 plain-arm samples (0.078%)**, but
   concentrated: llama-3.3-70b grid completions **0.82%** (235/28,800),
   deepseek grid **0.29–0.47%**. Ratio family is near-immune (~0.01%) — the
   short composition sentence does not invite enumeration the way the grid
   legend does. Essentially every instance parses as MOVE, so it biases
   **P(MOVE) upward**.
2. **Keyword inside a longer word.** `MOVEMENT` (59), `MOVEMENTS` (26),
   `ASSISTANCEMOVE` (20), `REMOVED` (11). Of 145 such samples, 68 were recorded
   MOVE and 9 STAY; only the 71 containing both words were correctly flagged.

**Neither is visible in any bad-parse statistic** — that is what makes them
dangerous. A bad parse is counted and retried; these enter the move rate as
real decisions. Note the model that looks *cleanest* by bad-parse rate (llama,
0.01% bad) is the one worst affected: the metric evades exactly this failure.

**Grammar arms are structurally immune to MISPARSE** — the GBNF admits only
whitespace plus MOVE/STAY, so neither enumeration nor stray prose can occur.
**0 occurrences in 473,625 grammar samples.** Chat arms also measured 0, but
nothing structurally guarantees that. Immune to misparse is NOT immune to bad
parses — see the next section.

**So: yes, prefer grammar — but it is not a free win.** Grammar removes parse
artifacts; it does NOT make an unwilling model willing. Where the plain arm is
unhealthy, the grammar reading is *protocol-constructed* (the forced-choice
verdict in `ratio_consistency_summary.md`), i.e. behaviour UNDER the production
protocol, not preference. The two facts combine into the operating rule:

- **Run production on grammar** (only channel free of silent misparse).
- **Keep a plain arm as a validity probe**, never as the measurement — it is
  what tells you whether the grammar reading is voluntary or constructed.
- **Do not compare plain-arm numbers across models**: their misparse rates
  differ by ~80x, so part of any plain-arm gap is parser noise, not behaviour.

Fixing the plain path (word-boundary regex + treat `finish_reason == "length"`
as ambiguous rather than decisive) is deferred — it would shift the effective
rates in affected plain arms and break comparability with published figures.
See FUTURE_EXPLORATIONS.md item 6.

## Grammar arms DO produce bad parses — completions only (2026-08-21)

`analyze_grammar_truncation.py` -> `results/grammar_truncation.md`. Measured over
every `results/raw/*_raw.jsonl.gz`. **272 bad parses in 473,625 grammar samples
(0.057%), and every one of them is on the `/completions` endpoint.** They are not
grammar violations; they are `max_tokens` truncation, allowed by the grammar's own
unbounded whitespace production:

    root ::= ws answer
    ws   ::= [ \t\n]*          <-- UNBOUNDED

Whitespace is legal without limit, so a constrained model can spend its whole
5-token budget on it and get cut before reaching the keyword. All 272 have
`finish_reason == "length"` and `completion_tokens == 5`.

**So llm_runner.py:59 is wrong.** "Generation halts at the word boundary, so
max_tokens=5 becomes a never-binding safety ceiling" — it does bind, on the same
grammar production shipped in `MOVE_STAY_GRAMMAR`. Production pays it as retries.

The chain, each step measured:

1. **Endpoint.** Leading whitespace occurs ONLY on `/completions`: chat+grammar
   shows **0 leading-whitespace replies in 243,000 samples**, across every model.
   The chat template ends the prompt with the assistant-turn header, so the first
   emitted token is already the word. Raw `/completions` has no such scaffolding
   and some models open with their own newline. **This is why the figR6/fig3
   parse-error bars appear on compl+grammar and never on chat+grammar.**
2. **Run length, not presence.** Leading whitespace alone is harmless. Qwen emits
   it on ~12% of completions replies but always 1-2 tokens, and truncates ZERO
   times. Only Gemma produces long runs (832 replies at 3-4 tokens in the ratio
   sweep, 232 cut at 5). Gemma is the only affected model in the whole corpus.
3. **Prompt.** Gemma's truncations concentrate in the SIMILAR-framed ratio styles
   at low similar-share: R2/R5, almost all at `n_similar = 1`. Natural experiment
   in the data: R5 renders `n_sim=1,n_occ=3` and `n_sim=2,n_occ=6` to the SAME
   string ("33.3%"), and the two cells truncate at the same rate (blue 21 vs 22,
   red 4 vs 4) — so this is driven by the prompt string, not the composition.

**Bias direction.** Of the truncated replies that reached a letter, all are
`ST`/`STA` (21) and none are `MO`. Truncated samples were headed for STAY, so
dropping them from the effective rate biases P(MOVE) *up*. The affected cells are
already STAY-dominated, so the realised shift is small; the pessimistic bound if
every truncated reply were forced the other way is 0.64 in the single worst cell
(R5, red, n_sim=1, n_occ=7), which the fragment evidence rules out.

**Fix (deferred, same reasoning as item 6 in FUTURE_EXPLORATIONS.md):** bound the
whitespace production (`ws ::= [ \t\n]{0,2}`) or raise `max_tokens` for grammar
arms. Either changes the sampler's legal set and so breaks comparability with
every published figure; do it at a clean re-run boundary, not in place.

## The `--no-jinja` probe is settled: equivalent, dropped from figures (2026-08-21)

`llama-3.3-70b-nojinja` is a SERVER FLAG, not a model. `--no-jinja` makes
llama.cpp take the built-in C++ chat path (src/llama-chat.cpp:485-493), which
omits the "Cutting Knowledge Date / Today Date" system block that Llama-3.3's
embedded jinja template auto-inserts whenever no system message is supplied (we
never supply one). Same weights, sampler and prompts; only that block differs.
Chat arms only — raw /completions applies no chat template at all.

Measured against the jinja default across every style, role and chat arm:

- per-cell mean |dP| **0.0004 – 0.026** (max in any single cell 0.76, but those
  sit on the step edge where a <0.2-neighbour threshold shift is a large vertical
  gap — geometry, not behaviour)
- implied P=0.5 thresholds at full occupancy agree within **0.05 of a neighbour
  in 9 of 12** (style, role) pairs; largest deviations R1/red +0.18, G0/blue -0.12
- **0.00% bad parses** either way

So removing the assistant-deployment date preamble does not meaningfully move
Llama's value function. **Decision: plot the regular jinja version only.** The
probe is excluded from every ratio figure via `PROBE_MODELS` in
plot_ratio_results.py (`--include-probes` restores it); its CSVs and raw logs are
kept, because they are the evidence for this claim.

Caveat unchanged (FUTURE_EXPLORATIONS.md item 4): this probe is clean ONLY for
llama. On qwen/gemma/deepseek `--no-jinja` silently disables `--reasoning off`
(reasoning control lives in the jinja render context), so it would confound
template effects with an unclosed think block. Those need a custom
`--chat-template`, not `--no-jinja`.

## Ratio-datapoint aggregation: equal weights (2026-08-23)

The vf ratio datapoints are the SIMPLE (equal-weight) average of their
member compositions' rates, not the sample-weighted pool. Reason: the ±2 pp
top-up gives members unequal n, so pooled counts silently re-weight the
datapoint toward whichever member needed more samples (baseline red 1/1:
pooled 0.55 vs equal-weight 0.81 — members 0.00/0.46/1.00×6). Artifacts
carry both (`p_move_effective` = equal-weight, `p_move_pooled` alongside);
CI on the mean is variance-propagated from member Wilson intervals.
`llm_runner --vf-lookup composition` bypasses aggregation entirely (exact
cell rates); paired-seed A/B batches in `experiments/vf_lookup_comparison_*/`
test whether the granularity choice moves simulation metrics.

Production lookup default = composition (2026-08-23): llm_runner
--vf-lookup defaults to the exact cell rates; the A/B campaign
(experiments/vf_lookup_comparison_20260823_065748) showed ratio averaging
inflates segregation in 5/6 scenarios and significantly flips the
income-vs-political ordering on mix_deviation/share/ghetto_rate/DI. The
ratio line plot stays as visualization; heatmaps show the surface the
simulation uses. Resume restores value_function_file and vf_lookup from
the experiment config, so old ratio-mode batches resume faithfully.

## Scale matters at fixed ratio — gemma, llama, qwen (2026-08-24, corrected + extended 2026-08-25)

Probe analysis_tools/vf_scale_dependence_probe.py --label <vf-1 label>
(standalone, not part of any pipeline; runs on any model's artifacts). Correct test = hold the RATIO fixed and vary scale
within a ratio family; a pure ratio rule predicts flat families. Result over
72 multi-member families (6 scenarios x 2 roles): 57 flat, 15 rising, 0
falling. The rises are 0->1 jumps, concentrated in the all-opposite family
(ratio 1, rising in 11/12 scenario-roles: 1 opposite = STAY, >=2 = FLEE) and
ratio 3/4 (4 rising, e.g. political red (1,4)=0.00 vs (2,8)=0.99). So most of
the surface is ratio-consistent, but absolute counts flip behaviour where the
counts are small. REPLICATES ACROSS MODELS (2026-08-25): llama-3.3-70b
17/72 rising, qwen3.6-27b 26/72 rising, gemma 15/72 — and ZERO falling
families in any model, i.e. more opposite neighbours at the same ratio
never reduces P(MOVE). The ratio line plot's sawtooth is a projection artifact of
averaging families whose members disagree. Figure:
results/figures/vf_<label>_scale_dependence.png (gemma, llama, qwen). Paper note in
paper/methods_docs/03_VALUE_FUNCTION.md. (An earlier version of this probe
held allies fixed and varied the opposite count — confounded, since that
varies the ratio too; superseded.)
