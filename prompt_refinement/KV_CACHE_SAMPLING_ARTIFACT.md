# The KV-cache sampling artifact: cache-state-dependent P(MOVE) at transition cells

Found 2026-08-22 during the ±2 recalibration of the gemma R3 value functions.
Status: **verified; remediation EXECUTED 2026-08-22 (clean 540-cell resample,
±2 pp certificate, contamination map, bitwise audit reference)**. Concept background:
`LLAMA_CPP_SERVING_NOTES.md`; grammar is orthogonal (`GRAMMAR_GBNF_NOTES.md`).

## 1. Claim

With llama-server prompt caching enabled (`cache_prompt` default true,
`-np 4`), the sampled MOVE probability of a byte-identical prompt at fixed
parameters (T=0.3, same grammar, same model, same server flags) depends on
the server's KV-cache state — i.e. on *what the slots processed earlier*.
The effect is:

- **large** at transition cells (tens of percentage points),
- **stable within a sampling episode** (hundreds of consecutive samples hold
  one rate), which made every individual batch look internally clean,
- **absent** at saturated cells,
- **eliminated** by `cache_prompt: false`.

## 2. Evidence

All measurements: gemma-4-31B-it-Q5_K_M, chat+grammar, T=0.3, prompts
byte-verified identical across episodes (raw logs).

**Cell A — baseline red, 2-of-2 opposite** (canonical clean value ≈ 0.44):

| episode (chronological) | condition | MOVE rate |
|---|---|---|
| pilot sweep (server session 1) | cache on | 46/100 = 0.460 |
| ±5 top-up (session 2) | cache on | 6/285 = **0.021** |
| ±2 top-up (session 2, +1 h) | cache on | 243/989 = **0.246** (flat 0.19–0.30 in every 100-window) |
| live probe ×4 (session 2, later) | cache on, conc. 4 | 1, 4, 4, 1 /100 |
| live probe (conc. 1) | cache on | 0/100 |
| **cache-priming**: 40 long unrelated gens, then measure | cache on | primer A: 4/100 · primer B: 10/100 · primer A again: 9/100 |
| same, immediately after priming | **cache off** | 42/100 · 44/100 |
| repeated batches | **cache off** | 43, 43, 43 /100 |
| distinct seed blocks | **cache off** + seeds | 47, 47, 40 /100; block repeat 44 (binomial scatter, sd≈5 ✓) |

**Cell B — race_white_black red, 0-of-5 similar** (clean ≈ 0.165):

| episode | condition | MOVE rate |
|---|---|---|
| pilot | cache on | 12/100 = 0.120 |
| ±5 top-up | cache on | 26/145 = 0.179 |
| ±2 top-up | cache on | 571/1325 = **0.431** |
| clean probe ×2 | cache off | 20/100, 13/100 (pooled 0.165) |

**Saturated cell — baseline red, 8-of-8 similar:** 0/100 with cache on AND
off; across all stages of all sweeps, saturated cells never disagreed.

Reading: episodes lock onto different rates (0.46 → 0.02 → 0.25 → 0.02–0.10
for cell A) far beyond binomial noise (>5σ); the priming experiment moves the
cache-on rate with slot history while cache-off is immune to the same
priming; cache-off restores one stable value consistent with proper i.i.d.
binomial sampling. Note contamination direction varies by cell/episode (cell
A biased low post-pilot, cell B biased high in stage 3) — it is not a fixed
offset that could be corrected post hoc.

## 3. Mechanism

The transformer's next-token logits are computed by attending over the KV
cache. With prompt caching, part of that cache is *reused* from earlier
requests: (a) retained KV carries the floating-point history of the batches
that computed it (reduction order/tiling differ by batch composition —
[llama.cpp Discussion #10311](https://github.com/ggml-org/llama.cpp/discussions/10311)
documents that results are not reproducible under caching/multi-slot);
(b) partially matching caches are repositioned by an approximate cache-shift
(RoPE re-rotation). Both make the decision token's logits a function of slot
history. At T=0.3, log-odds shifts are amplified ×3.3 (log-odds = Δlogit/T),
so a modest logit perturbation moves a near-tied decision by tens of points
while leaving saturated decisions (multi-nat gaps) untouched. Gemma's
sliding-window attention adds known cache-reuse complications
([#21468](https://github.com/ggml-org/llama.cpp/issues/21468),
[#21831](https://github.com/ggml-org/llama.cpp/issues/21831)); a documented
sibling failure is prompt-cache reuse across different LoRA adapters
contaminating outputs
([#26207](https://github.com/ggml-org/llama.cpp/issues/26207)).

## 4. How it was caught (and why not earlier)

The ±2 recalibration re-check flagged pooled p̂ shifts >5σ against the pilot
CI. Stage decomposition **from the stored raw replies** showed irreconcilable
per-episode rates within one server process; within-episode rates were flat
and prompts byte-identical, ruling out drift and prompt bugs; live probes
reproduced episode-locking; the `cache_prompt:false` A/B in the same session
was the causal identification; seed-block scatter confirmed restored i.i.d.
behaviour. It evaded earlier detection because each episode is internally
self-consistent — only *independent remeasurement of the same quantity*
(the recalibration loop) exposed it. Third instance of the repo's recurring
failure class: components that fail into success-shaped states (silent
misparse 2026-08-13; silent no-op analysis step 2026-08-22a; this).

## 5. Scope of impact

- **Transition cells in all cache-on sampling runs** (every sweep to date):
  unreliable — pilot values matched clean values for both probed cells, but
  per-cell cleanliness cannot be certified retroactively.
- **Unanimous cells (0-of-n / n-of-n): unaffected** (raw logit gap far
  beyond the perturbation's reach). REFINEMENT from the full contamination
  map (2026-08-22): mere near-saturation is NOT immunity — a cell with clean
  p̂ = 0.992 (raw logit gap ≈ 1.4 at T=0.3) was measurably contaminated
  (archived 0.828). Immunity claims are stated for unanimous cells only.
- **Simulation-level conclusions**: the ALL-PASS sufficiency verdicts and the
  scenario ordering rest overwhelmingly on saturated structure; transition
  cells shift thresholds by at most one composition step. Qualitative
  findings stand; exact threshold values await clean remeasurement.
- **Historical style sweeps** (`ratio_comparison_*`, `prompt_comparison_*`):
  qualitative rankings and saturated-cell structure stand; transition-cell
  NUMERIC values and anything interpolated through them — most directly
  figR3's implied P=0.5 thresholds — carry unquantified episode bias (up to
  ~1 composition step). Caveat, don't re-run: they were comparative
  scaffolding for choosing R3, and that choice is robust.
- **Live-LLM production runs (A2/A3)**: prompts vary constantly, so cache
  state appears as extra decision noise rather than stable bias; noted as a
  caveat, not a retraction.

## 6. Clean measurement protocol

`cache_prompt: false` + explicit distinct `seed` per request (server default
is seed = -1 = random per request — [server
README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md);
per-build seed handling should be spot-verified once, cf. historical
[#7381](https://github.com/ggml-org/llama.cpp/issues/7381)). Cost: full
prompt eval per request (~2× slower at our lengths). Residual limits: exact
bitwise replay still requires single-slot serial serving (multi-slot FP
noise; see the reproducibility table in `LLAMA_CPP_SERVING_NOTES.md`).

## 7. Remediation plan — full resample of ALL cells (PLANNED, not executed)

User decision 2026-08-22: redo **all 540 (scenario × role × composition)
cells**, not only transition cells, under the clean protocol.

1. Protocol change in `sampling_common.sample_once`: accept
   `cache_prompt=False` and per-request `seed` (global running sample index),
   threaded from `build_value_function.py`; artifact `meta` gains
   `sampling_protocol: {cache_prompt: false, seeded: true}`.
2. Archive current artifacts + raws to
   `results/value_functions_cacheon_archive/` (they are evidence, not trash).
3. Clean pilot: N=100 per composition, all 6 scenarios (54,000 requests).
4. ±2 top-up + recalibration loop as in `SAMPLING_METHODOLOGY.md`
   (expected ~12–20 wide cells × ≤2,401).
5. Rebuild artifacts/figures; rerun evaluation batch + half-data sufficiency
   check on clean artifacts.
6. Byproduct analysis: per-cell clean-vs-archived comparison table — the
   first quantitative map of cache contamination across a full surface.

**EXECUTED 2026-08-22.** Clean pilot (54,000 samples) + two top-up passes to
a zero-deficit ±2 pp certificate. Contamination map result: **8 of 540 cells
CI-disjoint**, max |Δp̂| = 0.249; every flagged cell is a transition or
near-saturated composition; zero unanimous cells flagged (negative controls
clean). Map: `results/value_functions/vf_contamination_map.{csv,png}`.

Estimated cost at cache-off throughput (~2–3 req/s): pilot ~6 h, top-up
~3–5 h ⇒ **~10–14 h GPU, unattended**.
