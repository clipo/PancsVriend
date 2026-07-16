# Prompt refinement — MOVE/STAY decision prompt

Why the prompt in `context_scenarios.py` was re-examined, how the candidates were
scored, and how to reproduce it.

## The two problems

**1. Cost.** We now serve the raw `/v1/completions` endpoint on llama.cpp, which caches a
prompt **prefix** across calls. Only tokens *before the first varying token* are reusable.
`{context}` (the 3×3 grid) is the only thing that varies, so **every token placed after it is
recomputed on every call**. The current template puts the grid near the top and ~145 tokens of
legend + instructions below it — the worst possible layout. At ~60 tok/s prompt-eval on the
GB10 that is ~2.4 s per decision, paid hundreds of thousands of times in a production run.

> Design rule: **all fixed text above `{context}`; keep what follows it short.**

**2. Quality.** The current template also produces a decision that barely tracks the
neighbourhood. A prompt must be **cheap AND monotonic AND responsive**. Scoring on cost alone
would happily select a prompt the model ignores, so both axes are measured.

## Files

| file | what it is |
|---|---|
| `prompt_templates.py` | the candidate templates (`CANDIDATES` dict) — edit/add here |
| `evaluate_prompts.py` | the runner: scores every candidate on cost + quality, per model |
| `results/prompt_comparison_<label>.md` | results table per (model, arm) |
| `results/prompt_comparison_<label>.csv` | same, machine-readable |

## Candidates

- **`0_current`** — what `context_scenarios.py` ships today. Grid early, everything after it
  recomputed. The baseline to beat.
- **`A_briefing_map_ask`** — briefing → map → ask. All fixed text above the grid; only a short
  question below. The one-word rule is *anticipatory* ("You will be shown a map… after looking
  at the map, decide"), so it reads as task setup instead of arriving with no context, and the
  real question lands last.
- **`B_briefing_map_ask_rule`** — as A, but the format rule is restated *after* the question.
  Tests whether ending on a terse command changes the answer.
- **`C_legend_after_grid`** — the "figure, then key" convention (show symbols, then explain).
  Reads naturally, but the legend is the largest fixed block and putting it below the grid means
  paying for it on every call.

## What is measured

**Cost — `recomputed_tokens`**: tokens after `{context}`. Re-evaluated every call. Lower = faster.

**Quality** — each (candidate, neighbourhood) is **sampled** `--samples` times using the exact
payload `llm_runner.py` sends in production (same `SAMPLER_PARAMS`, `max_tokens=5`, **no
`stop`** — a leading newline is a plausible first token on a raw completion, and a `"\n"` stop
would truncate it to an empty string and burn a retry) and parsed with the same MOVE/STAY rule.
So the reported `move_rate` is what the agents literally do in the simulation.

Swept over a clean gradient of 0–8 out-group neighbours:

- `move_range` — spread of move_rate over the gradient. Higher = the decision actually responds.
- `spearman_monotonic` — does move_rate *rise* with out-group count? **Negative or ~0 means the
  agent does not want to leave a hostile neighbourhood** — unusable.
- `bad_parses_total` — ambiguous/unparseable replies; each one costs a retry in the real runner.

**Layout randomisation.** Every sample draws a **fresh random arrangement** of the O/S cells at
its density (`--seed`, default 0), and the same layout set is shared across all candidates
(paired design). Sampling one fixed layout per density would confound a density effect with a
position/adjacency effect; marginalising over geometry measures P(MOVE | density) proper. The
gradient grids contain **no empty cell (`E`)** on purpose: a vacancy is itself a strong MOVE
cue, and varying it alongside composition would confound the result. Density 0/8 and 8/8 have
only one arrangement, so they are unaffected.

Sampling resolution is ~1/`--samples`. At T=0.3 the sampler is near-deterministic, so a 0% row
is itself the finding: that prompt produces no movement at all. (First-token logprobs were
considered and rejected: the answer is not reliably a single first token — leading whitespace,
casing and multi-token splits — and logprobs measure the T=1 preference, not the sharpened
T=0.3 behaviour the simulation actually runs at.)

**`--grammar`** adds a permissive GBNF grammar (optional leading whitespace + any casing of
MOVE/STAY, nothing else) to each request. Measured effect (see results): where the model already
answers cleanly the move-rate curves are unchanged within noise; where it wasted tokens on junk,
the junk becomes decisions and `bad_parses` drops to ~0. It also halts generation at 2–3 tokens
instead of always burning `max_tokens`. A candidate production change — if adopted, adopt it in
`llm_runner.py` too and re-evaluate.

## Endpoints: completions vs chat

`--llm-url .../v1/completions` sends the raw prompt (no chat template);
`.../v1/chat/completions` wraps it as a user turn so the server applies the model's chat
template. Findings from the 2026-07 runs:

- **Llama-3.3-70B**: the chat template *suppresses movement* — three of four candidates go
  completely inert and A's range drops (0.92 → 0.72). Use **completions**.
- **Gemma-4-31B**: chat mode triggers the model's **reasoning channel** — 300–1000+ thinking
  tokens per decision, often never reaching an answer within budget. Unusable at simulation
  scale. With `llama-server --reasoning off`, chat is competitive with completions (A/B perfect,
  0 bad parses), but completions still wins on cost and cross-model consistency.

## Results (production payload, random layouts, 50 samples/cell, T=0.3)

**`A_briefing_map_ask` wins on both axes on both models**: cheapest (23–25 recomputed
tokens vs 145–150 for `0_current`) and the only candidate with near-full range AND clean
monotonicity everywhere. `0_current` is essentially inert (never moves even at 8/8 hostile).
Per-model tables and full move-rate gradients: `results/`. Labels: plain = completions,
`-chat` = chat endpoint (`-chat-noreason` = with `--reasoning off`), `-grammar` = GBNF arm.

## Reproducing

Serve each model with the CUDA `llama-server`, then point the runner at it:

```bash
# Llama-3.3-70B  (port 8082)
/srv/shared/schelling/llama.cpp/build/bin/llama-server \
  -m llms/Llama-3.3-70B-Instruct-Q4_K_M.gguf --alias Llama-3.3-70B-Instruct-Q4_K_M \
  -ngl 999 -fa on -np 8 -c 16384 --host 127.0.0.1 --port 8082 &

python prompt_refinement/evaluate_prompts.py \
  --llm-url http://localhost:8082/v1/completions \
  --model Llama-3.3-70B-Instruct-Q4_K_M \
  --label llama-3.3-70b-instruct-q4_k_m

# Gemma-4-31B  (port 8083; add --reasoning off for chat-endpoint runs)
/srv/shared/schelling/llama.cpp/build/bin/llama-server \
  -m llms/gemma-4-31B-it-Q5_K_M.gguf --alias gemma-4-31B-it-Q5_K_M \
  -ngl 999 -fa on -np 8 -c 16384 --host 127.0.0.1 --port 8083 &

python prompt_refinement/evaluate_prompts.py \
  --llm-url http://localhost:8083/v1/completions \
  --model gemma-4-31B-it-Q5_K_M \
  --label gemma-4-31b-it-q5_k_m
```

`--temperature` defaults to 0.3 to match the simulation. `--seed` fixes the random layout set
(paired across candidates within a run; keep it fixed to compare arms). `--grammar` enables the
GBNF-constrained arm. Add a candidate by editing `CANDIDATES` in `prompt_templates.py` and
re-running — nothing else needs to change.

## Caveats

- Adopting a new template changes the prompt for **all** scenarios in `context_scenarios.py`, so
  existing gemma/llama results are no longer token-identical and the baselines must be re-run
  for a clean comparison.
- The eval payload mirrors `llm_runner.py` **by hand** (`max_tokens=5`, no stop, SAMPLER_PARAMS
  import). If the production payload changes, update `sample_once()` to match — results are only
  meaningful when the request is byte-identical to production.
