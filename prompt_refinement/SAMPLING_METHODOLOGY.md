# Value-function sampling: the two-stage top-up design

How we decide how many LLM samples each neighbourhood composition gets, how
the top-up recalibrates after new data arrives, and why the loop is guaranteed
to terminate. Companion figure: `results/figures/sampling_requirements.png`
(regenerate with `python prompt_refinement/plot_sampling_requirements.py`).
Implementation: `build_value_function.py --top-up --precision <w>`
(`topup_deficits()`); outcome-level verification:
`analysis_tools/vf_simulation_evaluation.py` (half-vs-full check).

![Sample-size requirements](results/figures/sampling_requirements.png)

## 1. The measurement and its uncertainty

Each (role, composition) cell is a repeated binary trial: ask the model n
times, count k MOVEs. The estimate is the sample proportion p̂ = k/n, and its
uncertainty is summarised by a **95% confidence interval** computed with the
**Wilson score interval** (better behaved than the naive p̂ ± 1.96·√(p̂q̂/n)
near 0 and 1, where much of our surface lives). The interval's **half-width**
is the "±X points" quoted everywhere.

Key structural fact: the half-width scales with √(p(1−p))/√n. The p(1−p)
factor is maximal at p = 0.5 and vanishes at the extremes — so *the same
precision costs wildly different n depending on where the cell's true
probability sits* (panel A of the figure).

## 2. Stage 1 — uniform pilot

Every composition gets the same N (the scenario sweep used N = 100 per
composition, per user decision: 1-of-2 and 2-of-4 are sampled independently
and never split a budget). The pilot has two jobs: give every cell a usable
estimate, and reveal *where each cell's p̂ sits* so stage 2 can be priced.

## 3. Stage 2 — targeted top-up

For a precision target w (half-width, e.g. 0.05 or 0.02), each cell is
tested and, if needed, topped up:

1. **Test — the actual interval.** Compute the cell's current Wilson CI from
   its pooled counts. If half-width ≤ w, the cell is done: deficit 0. This is
   the criterion, not the formula below — a 0-of-100 cell has interval
   [0, 3.7%], half-width 1.85 pp, and already meets a ±2 pp target even
   though a plug-in formula evaluated at its upper bound would demand ~340
   samples. (This distinction was a real bug fixed 2026-08-22.)
2. **Size — the planning formula with a safeguard.** If the interval is too
   wide, buy the deficit implied by the standard proportion sample-size
   formula, n = 1.96² · p*(1−p*) / w², **evaluated at p\* = the current CI
   bound nearest 0.5** (and p\* = 0.5 outright if the CI straddles 0.5). Using
   the pessimistic bound rather than p̂ means an undershooting pilot estimate
   cannot cause an under-buy: we size for the worst p the data considers
   plausible.
3. **Merge.** New counts add to old — binomial counts are sufficient
   statistics, so pooling is exact updating, and every raw reply is kept for
   audit/rebuild.

## 4. Recalibration: what if p̂ shifts?

After merging, p̂ is re-estimated from the pooled counts and can move — e.g. a
cell piloted at p̂ = 0.12 whose true p is 0.25 will drift toward 0.25, and
p(1−p) grows with it, so the freshly recomputed requirement can exceed what
was bought. The answer is that **`--top-up` is a fixed-point iteration**: it
always recomputes deficits from *current* counts, so it is simply run again
(a `--dry-run` prints the residual table without sampling). The loop
"converged" = the dry-run shows zero deficits everywhere, i.e. every cell's
actual 95% CI half-width ≤ w.

**Termination is guaranteed** by a hard ceiling: the requirement is maximal
at p = 0.5, where n = 1.96²·0.25/w² (385 at ±5 pp; 2,401 at ±2 pp — see the
peaks in panel A). No cell can ever demand more than that ceiling no matter
where p̂ wanders, every iteration adds samples, and a cell at the ceiling has
deficit 0 forever. In practice pass 1 clears every cell whose pilot CI
covered the truth (~95% of them by construction), pass 2 mops up the rare
escapees, pass 3 is essentially never reached.

## 5. Reference table

n required for 95% CI half-width ≤ w (formula at p; panel A of the figure):

| true p | pilot half-width at n=100 | n for ±5 pp | n for ±2 pp |
|---|---|---|---|
| 0.50 | ±9.8 pp | 385 | 2,401 |
| 0.30 | ±9.0 | 323 | 2,017 |
| 0.20 | ±7.8 | 246 | 1,537 |
| 0.10 | ±5.9 | 139 | 865 |
| 0.05 | ±4.3 | 73 | 457 |
| 0.02 | ±2.7 | 31 | 189 |
| 0.00 (0 of n) | ±1.85 | 0 (met) | 0 (met) |

Cost scales with 1/w²: tightening ±5 → ±2 is 2.5× the precision at 6.25× the
price (panel B). The two-stage design pays those prices only at cells whose
p̂ actually sits mid-range — in the gemma R3 sweep, 11 of 540 cells.

## 6. Measurement protocol (2026-08-22)

All value-function sampling uses the CLEAN PROTOCOL: `cache_prompt: false`
(no server-side prompt-cache reuse — see KV_CACHE_SAMPLING_ARTIFACT.md for
why this is mandatory) plus a deterministic per-request sampling seed,
`request_seed(stage|scenario|style|role|sim|occ|i)` (crc32), recorded in every
raw record and in the artifact's `meta.sampling_protocol` together with the
server build fingerprint. Bitwise audit: `replication_kit.py`.

## 7. The outcome-level backstop

CI targets are input-side plans; the verdict that matters is outcome-side:
rebuild each value function from **half** its samples
(`build_value_function.py --rebuild-from-raw --keep even`), rerun the
simulations with **paired seeds**, and compare every metric (6 Metrics.py
metrics + dissimilarity index). PASS = the paired mean difference is inside
the full-data mean's own 95% CI, i.e. sampling noise moves outcomes less than
run-to-run noise at reporting precision. This check runs in
`run_vf_eval_sims.sh` and is re-run on the final artifacts after any top-up;
it, not the CI table, is what certifies a value function as production.

## Concurrency (added 2026-09-05)

Every probability or sampling measurement is taken with ONE request in flight.
With several requests batched (`-np 4`, the historical campaigns' setting) the
server returns batch-dependent probabilities — tens of points on transition
cells, cache on or off, flash attention on or off — so a sampled rate at
concurrency > 1 is an average over a family of distributions rather than an
estimate of the model's. Details, measurements and the probe script:
`KV_CACHE_SAMPLING_ARTIFACT.md` §8, `batch_numerics/batch_numerics_probe.py`. The exact
extraction (`logprob_value_function.py`) is sequential by default.
