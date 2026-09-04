# Future explorations (deferred sub-experiments)

Running list of experiments considered during the ratio-prompt / sampled-value-
function study but deliberately deferred. Each entry notes why it was excluded
from the main design, so the exclusion can be revisited with its rationale.

## 1. Response to nearby vacancy

Does *stating* adjacent empty houses shift P(MOVE), independent of composition?

The ratio sweep's prompts intentionally never mention empties or vacancy
(neighbor-count-only wording): local vacancy is mechanically irrelevant in this
model (relocation is global; `Agent._unlike_ratio` excludes empties), and a
stated vacancy is itself a plausible MOVE cue that would confound the phrasing
comparison. That makes vacancy response a clean standalone question: hold
composition fixed (e.g. 3 of 6 similar) and vary only a vacancy sentence
("2 of the houses next to yours are empty") vs its absence. If LLM agents react
strongly to stated vacancy, that is a behavioral deviation from the canonical
model worth its own writeup.

## 2. Majority-quantifier framing probe (rejected style R5)

An `R5_majority_verbal` style ("Most of your 8 neighbors…") was rejected from
the sweep: the quantifier pre-frames majority status — the judgment under
study — and its wording necessarily switches at the k=4/5 boundary
("a minority / half / most"), confounding the style with the threshold region.
As a standalone probe it is well-posed: R3_dual_count vs R3 with a quantifier
sentence prepended, ONE wording change, 1–2 models, and measures how much a
supplied social framing shifts the implied threshold.

## 2b. Blue-role extension of the original grid sweep — PROMOTED TO QUEUED
(2026-07-31, user decision: full completeness over a targeted A2/A3 probe.)
The historical prompt_comparison_* sweep is red-only; run_grid_blue_sweep.sh
re-runs ALL 8 candidates x 4 arms x 5 models as the blue resident (N=100,
labels <old>-blue) and auto-replots fig1/fig2 blue variants. Queued LAST:
gated behind the production chain and the Phase A ratio sweep.

## 3. Social-context transfer of the winning style

The robustness sweep runs on the baseline red/blue context only. Whether
phrasing-robustness transfers to loaded contexts is untested — "62.5% of your
neighbors are Black families" plausibly interacts with social priors far more
than "…are blue team residents". Before trusting the winning style across the
full scenario campaign, re-run a reduced sweep (winner + one alternative,
1–2 arms) on one loaded context (e.g. race_white_black).

## 4. Chat-template tagging effects (--no-jinja probe; llama-cpp-python legacy)

CORRECTED PREMISE (2026-07-30): the Jul-15 sweep and all production chat runs
used the SAME tagging — this box's llama-server build (a4ce259, built Jul 14,
unchanged since) has `--jinja` ON BY DEFAULT, so the sweep's flag-less servers
still rendered the models' embedded Jinja templates. The explicit `--jinja` in
production-era scripts is belt-and-braces, not a behavior change. Two probes
remain genuinely open:
(a) `--no-jinja` (the C++ built-in template path, which for Llama-3.3 omits
    the auto-inserted system preamble with knowledge-cutoff/date).
    NOW IN THE SWEEP: run_ratio_sweep.sh carries a `llama-3.3-70b-nojinja`
    entry (chat arms only — completions never applies a chat template).
    CAVEAT — the probe is clean ONLY for llama: reasoning control lives in the
    jinja render context (`enable_thinking`, common/chat.cpp), which the
    built-in path cannot express, so on qwen/gemma/deepseek `--no-jinja`
    silently disables `--reasoning off` — plain chat would die in an unclosed
    think block, and grammar arms would force MOVE/STAY out of a
    wants-to-think distribution (renormalization artifact). Isolating template
    effects on those models needs a custom `--chat-template` (their real
    template minus the studied element), not `--no-jinja`.
    RELATED DISCOVERY (2026-07-30): Mistral-Small-4's embedded template is the
    extreme injection case — a bare user message gets ~2.3k chars of default
    system prompt ("You are Mistral-Small-4-119B… You power an AI assistant
    called Le Chat", web/tool/multimodal instructions) plus
    `[MODEL_SETTINGS]{"reasoning_effort": "none"}` (it is hybrid-reasoning,
    not non-thinking). Every Mistral chat-arm measurement therefore carries a
    built-in identity conflict with our persona prompt — interpret its chat
    results accordingly, and note llama is the only truly non-thinking chat
    model in the fleet.
(b) The June-5 legacy run used `python -m llama_cpp.server` — a third template
    implementation (llama-cpp-python's own chat handler) plus no grammar and
    the old 0_current prompt; it froze completely (final_step 4 everywhere,
    DI = initial grid). Its deadness is already explained by 0_current-on-chat
    (the sweep measured that combination at 0.0 flat), so re-running it has
    low value; listed here only as the historical data point.
Related open observation from production: llama-chat's freeze is
scenario-dependent (total on race/ethnic/political, partial on
baseline/income/green_yellow) — a prompt-content effect the baseline-only
sweep cannot see (see item 3).

## 5. Token-logprob estimation of P(MOVE)

Rejected for the main study: accuracy vs sampling is unproven here (the
`llm_log_probs/` branching-vs-sampling comparison for Gemma-4-31B was started
for this reason and is not conclusive). If it were validated, the entire
45-cell surface would cost 1 call/cell instead of N samples (~100x cheaper) and
yield temperature-independent probabilities renormalizable to any T. Worth
revisiting once a sampled surface exists to validate against: compare
logprob-derived P(MOVE) to the sampled surface on all 45 cells for one model
and arm; if they agree within sampling CIs, switch future sweeps to logprobs.

## 6. Harden the plain-endpoint MOVE/STAY parser (deferred 2026-08-13)

The substring rule silently misparses two ways — truncated enumeration
(`'MOVE\nor\nST'` -> recorded as a confident MOVE) and keyword-inside-a-word
(`MOVEMENT` -> MOVE). Full measurement and rates in NOTES.md
("Prefer GRAMMAR arms"). Grammar arms are structurally immune, so this is only
worth doing if plain-endpoint runs stay in the design.

The fix is small: match on word boundaries (`\bMOVE\b`), and treat a reply with
`finish_reason == "length"` that ends in a proper prefix of the *other* keyword
as AMBIGUOUS rather than decisive. Deferred because it changes effective rates
in the affected plain arms and would break comparability with every published
`prompt_comparison_*` / `ratio_comparison_*` figure. If done, it should be a
flag (`--strict-parse`) plus a re-run of one affected label (llama grid
completions, the 0.82% worst case) to quantify how far the curves actually
move — the expected shift is a small DOWNWARD correction to P(MOVE), since
nearly all silent misparses resolve to MOVE.

## 7. 20x20 grid simulation campaign (deferred 2026-08-23)

Scale the VF-driven simulations from 10x10/80 agents to 20x20 with 100 runs
per scenario. Assessed feasible (~10-15 min per scenario on 20 procs, ~1.5 h
for all six + mechanical reference); the value function transfers UNCHANGED
(Moore neighbourhood is grid-size independent, composition lookup applies
as-is, no resampling). Work list, in order:

1. **Generalize the DI tract map** (blocker): `TRACT_MAP` in
   `analysis_tools/dissimilarity_index_over_time.py` is a hardcoded 10x10
   3/4/3 9-tract layout and `compute_dissimilarity_from_int_grid` raises on
   any other shape. Derive it from the STORED grid's shape (20x20 -> 6/8/6,
   same topology, comparable index) — not from config, so old 10x10 batches
   stay analyzable.
2. **config.py**: GRID_SIZE=20, NUM_TYPE_A=NUM_TYPE_B=160 (keeps 80%
   density — an explicit choice; no CLI overrides exist). config is GLOBAL:
   while it says 20x20, do not launch or resume any 10x10 run, and the
   mechanical baseline must be regenerated at 20x20.
3. **--save-every-steps ~50** on all runner invocations: states are logged
   per agent decision (~1 MB/step at 320 agents) and save_states
   re-compresses the whole growing array every save — O(T^2) I/O without
   the flag. A final unconditional save exists, so nothing is lost.
4. **max_steps via smoke pilot** (2 runs x few hundred steps): the 5-zero-
   move-sweep convergence rule will likely never fire with 320 stochastic
   agents, so runs use the full budget; pick it from where metrics plateau
   (expected <150; budget 300-500).

Post-simulation analysis at the new size: the sufficiency check,
normality-gated significance tests, and rankings are grid-agnostic and rerun
as-is (redo the half-data check at the new size — metric sensitivity to VF
noise can differ). Metric COMPARABILITY across resolutions is not automatic:
switch_rate/mix_deviation/share/DI are intensive (fractions, comparable),
while clusters/ghetto_rate/distance are extensive (raw counts/cell units —
normalize per-agent or per-cell before any cross-resolution comparison).

100x100/1000-run notes (much further out): per-move state/move logging
becomes fatal (~640 MB RAM/step) and compute_distance's O(agents x cells)
Python loop becomes minutes/step — both need structural fixes (per-step
int8 snapshots, vectorized metrics) before that scale is reachable.
