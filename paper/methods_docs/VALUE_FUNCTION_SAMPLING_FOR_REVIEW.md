# How many samples per cell? The value-function sampling rule, for review

*Self-contained note for colleagues vetting the measurement design: what is
being estimated, the interval used, the rule deciding how many samples a cell
buys, why the loop terminates, how cells aggregate, and the assumptions the
design rests on. Implementation: `prompt_refinement/build_value_function.py`
(`topup_deficits()`, `build_rows()`); interval in `sampling_common.wilson_ci()`.*

---

## 1. What is being estimated

Instead of calling the LLM live inside each simulation, we measure the model
once and simulate from the measurement. The measured object is

> **p(role, composition, scenario) = P(the model answers MOVE | it is told it
> is a `role` resident of a `scenario`, facing a neighbourhood with
> `n_similar` like and `n_occupied − n_similar` unlike neighbours)**

A *composition* is a pair (n_similar, n_occupied) with 0 ≤ n_similar ≤
n_occupied ≤ 8 — the full Moore neighbourhood state space, **45
compositions**. With 2 roles × 6 scenarios that is **540 cells per model**,
each needing its own estimate.

Within a cell the n samples are treated as **i.i.d. Bernoulli(p)**, so the
MOVE count k is **Binomial(n, p)**. That is the entire statistical model:
p̂ = k/n is the maximum-likelihood estimate, and (k, n) are **sufficient
statistics** — which is what makes topping up exact, since new counts simply
add to old and the pooled estimate equals what one build at the larger n
would have produced.

The i.i.d. assumption was not free. Server-side prompt-cache reuse
(llama.cpp's default) made successive draws within a cell dependent and
*shifted the sampled probability itself* by tens of percentage points at
mid-range cells. All production sampling therefore runs with
`cache_prompt: false` and a deterministic per-request seed
`crc32(stage|scenario|style|role|n_sim|n_occ|i)`, recorded per sample
(`KV_CACHE_SAMPLING_ARTIFACT.md`).

Bad parses are structurally impossible under the grammar arm; rates are
*effective* rates k_move/(k_move + k_stay), matching what production does.

---

## 2. The uncertainty measure: Wilson score interval

For each cell we report a 95% confidence interval for p — the set of p values
not rejected at the 5% level by the observed (k, n) — using the **Wilson
score interval**:

$$\text{centre} = \frac{\hat p + z^2/2n}{1 + z^2/n}, \qquad
\text{half-width} = \frac{z\sqrt{\hat p(1-\hat p)/n + z^2/4n^2}}{1 + z^2/n},
\qquad z = 1.96$$

"±X points" always means that half-width.

Wilson rather than the textbook Wald interval p̂ ± 1.96·√(p̂(1−p̂)/n) because
most of our surface is *saturated*: the model answers MOVE ~0% or ~100% of
the time at the majority of compositions. At p̂ = 0 the Wald interval
collapses to the degenerate point [0, 0] with coverage 0, whereas Wilson —
which inverts the score test instead of plugging p̂ into the variance — gives
[0, 0.037] at n = 100 and keeps close-to-nominal coverage at the boundary.
Given where our data live, that is not cosmetic.

**The fact that drives the whole design:** the half-width scales like
√(p(1−p)/n), and p(1−p) is maximal at p = 0.5 and vanishes at the extremes.
The same precision therefore costs wildly different n depending on where the
cell's true probability sits — a unanimous cell is cheap, a torn cell is
expensive — so a uniform budget is the wrong instrument.

---

## 3. The two-stage design

**Stage 1 — uniform pilot.** All 540 cells get N = 100 samples (54,000
requests per model): a usable estimate everywhere, and a located p̂ so stage 2
can be priced.

**Stage 2 — targeted top-up to a precision target w** (production w = 0.02),
per cell, in this order:

1. **Test, on the realised interval.** Compute the cell's Wilson CI from
   current counts; if the half-width ≤ w the cell is done, deficit 0. The
   criterion is the realised interval, *not* the formula in step 2. This
   matters: a 0-of-100 cell has interval [0, 0.037], half-width 1.85 pp, and
   already meets ±2 pp — while a plug-in formula evaluated at its upper bound
   would demand ~340 samples it does not need.

2. **Size, at a pessimistic p.** If the interval is too wide,

   $$n_{\text{needed}} = \left\lceil \frac{z^2\,p^*(1-p^*)}{w^2} \right\rceil,
   \qquad z = 1.96$$

   with **p\* the endpoint of the current Wilson CI nearest 0.5** (or 0.5
   outright if the interval straddles it). Sizing from the pessimistic
   plausible p rather than from p̂ is a deliberate safeguard: an undershooting
   pilot — a cell whose true p is 0.25 but which piloted at p̂ = 0.12 — cannot
   use its own lucky estimate as an excuse to under-buy. The cost is
   over-buying where p̂ was accurate, which is the conservative direction.
   The purchase is **deficit = max(0, n_needed − n_current)**.

3. **Merge.** New counts add to old (exact, by sufficiency); every raw reply
   is streamed with its seed, so any artifact rebuilds from raw.

| cell | p̂ | Wilson 95% CI | half-width | p\* | bought |
|---|---|---|---|---|---|
| 0 of 100 | 0.00 | [0.000, 0.037] | 1.85 pp | — | **0** |
| 2 of 100 | 0.02 | [0.005, 0.070] | 3.23 pp | 0.070 | 526 |
| 12 of 100 | 0.12 | [0.070, 0.198] | 6.41 pp | 0.198 | 1,426 |
| 50 of 100 | 0.50 | [0.404, 0.596] | 9.62 pp | 0.500 | 2,301 |

---

## 4. Why it iterates, and why it stops

Buying samples changes p̂, and a cell drifting *toward* 0.5 gets **more**
expensive. The 12-of-100 cell buys 1,426 samples; if its true p is really
0.25 it lands at 382-of-1,526, whose half-width (2.17 pp) is still too wide
and re-prices at a further 379. Total deficit is therefore **not monotone**,
so the loop cannot stop on "no change since last pass" — it stops on
**residual deficit exactly zero**, every cell's realised half-width ≤ w. It
is a fixed-point iteration that always re-derives deficits from current
counts (`--top-up --dry-run` prints the residual table without spending).

**Termination is guaranteed** by a hard ceiling: the requirement is maximal
at p = 0.5, where n = 1.96²·0.25/w² = **2,401** at w = 0.02. No cell can
demand more wherever p̂ wanders, and at n = 2,401 the worst-case realised
half-width (at k/n = 0.5) is 0.019984 ≤ 0.02 — so a cell reaching the ceiling
is satisfied by construction and its deficit is 0 forever. Every pass adds
samples, so the ceiling is reached in finitely many passes; in practice pass
1 clears every cell whose pilot interval covered the truth, pass 2 mops up
the escapees, and pass 3 is essentially never needed.

The driver also separates failure modes a naive loop conflates: a missing
artifact aborts rather than reading as converged, and a pass that adds shards
but zero valid samples aborts — the signature of a dead server returning
unparseable text, which would otherwise loop until the budget with nothing
bought.

**Realised cost** (540 cells per model, pilot N = 100, target ±2 pp, all
converged):

| model | cells topped up | cells with p̂ in (0.05, 0.95) | total samples |
|---|---|---|---|
| gemma-4-31B | 11 / 540 | 8 | 71,469 |
| llama-3.3-70B | 39 / 540 | 21 | 100,927 |
| qwen3.6-27B | 127 / 540 | 70 | 203,682 |
| deepseek-v4-flash | 113 / 540 | 62 | 201,713 |

2–24% of cells absorb nearly all the extra cost; a uniform budget large
enough for the torn cells would have cost ~1.3M requests per model.

---

## 5. From cells to the reported curve

The simulation consumes the **composition-level** cells directly, so the
intervals above are the ones that matter for the results. The 23-ratio curve
in the figures is an aggregate — the equal-weight mean of a ratio family's
member compositions, with member variances propagated through the mean.
Equal weights, not count-pooling: after the top-up, members carry very
unequal n (up to 24x), so pooling counts would silently re-weight the
datapoint toward whichever member needed the most samples.

That propagation uses the plug-in (Wald) variance, exactly the quantity that
degenerates at the boundary, so when every member of a family is unanimous
the propagated interval has zero width (22 of 24 ratio datapoints for
gemma/baseline/red). That is an artifact of Wald propagation, not a claim of
infinite precision; the honest per-cell uncertainty is the composition-level
Wilson interval, which is what the artifact carries and what the heatmaps
print. Nothing downstream depends on the degenerate ratio interval. See
`03_VALUE_FUNCTION.md` for why the composition surface, not the ratio curve,
is the policy.

---

## 6. Assumptions worth challenging

1. **Per-cell, not simultaneous, coverage.** Each interval is nominally 95%
   for its own cell; across 540 cells there is no multiplicity adjustment, so
   under nominal coverage ~27 intervals would miss. The real exposure is
   smaller — a cell whose true p is 0 or 1 is covered by construction — and
   concentrates on the 8–70 interior cells per model, an expected ~0.4–3.5
   misses. If simultaneous coverage is wanted the fix is cheap and local:
   raise z to Φ⁻¹(1 − 0.05/1080) ≈ 3.48, multiplying the requirement by ~3.2
   **at interior cells only**. We did not, and §7 is why.

2. **A ±2 pp CI is an input-side plan, not the finding.** It bounds error in
   one input to a stochastic simulation, not error in any reported metric.

3. **Independence within a cell** rests on the serving protocol, which is
   verified by a bitwise replication kit rather than assumed. This assumption
   failed once, under the server default, and failed large — hence the
   emphasis.

4. **Fixed conditions.** Uncertainty over phrasing, temperature and endpoint
   arm is deliberately not in these intervals; those are swept separately.

5. **Sequential sampling.** The top-up decides how much more to buy from data
   in hand, so realised intervals are strictly post-stopping. The p\*
   safeguard pushes conservatively, and the rule targets *precision* rather
   than *significance* — there is no "sample until it is significant"
   behaviour, which is where optional stopping usually does its damage.

---

## 7. The check that certifies a value function

The CI target is a plan; the verdict is outcome-side. Every table is rebuilt
from **half** its samples (`--rebuild-from-raw --keep even`), all simulations
are re-run with **paired seeds**, and every metric is compared:

> **PASS** = the paired mean difference between half-data and full-data
> simulations lies inside the full-data mean's own 95% CI — i.e. halving the
> measurement effort moves the reported outcome less than ordinary
> run-to-run variation does.

Result on the production artifacts: **42/42 (scenario × metric) PASS**. This
is the standard we treat as certifying, because it is stated in the units the
paper reports. A single even/odd split is one draw from a distribution, so it
has since been generalised to a multi-split test
(`analysis_tools/vf_multisplit_check.py --label <label> --splits 32`).

---

## 8. Inspecting

```bash
# what the current artifacts still owe, without spending anything
python prompt_refinement/build_value_function.py \
    --config configs/value_function_scenarios_gemma.yaml \
    --top-up --precision 0.02 --dry-run

# top up repeatedly until every cell is within ±2 pp
python prompt_refinement/build_value_function.py \
    --config configs/value_function_scenarios_gemma.yaml \
    --calibrate --precision 0.02 --max-passes 4 --plot
```

Per-cell counts, intervals and provenance are in
`results/value_functions/vf_<label>__<scenario>__R3_dual_count.json`
(`compositions.{red,blue}` is ground truth); every individual reply, with its
seed and parse verdict, is in `results/value_functions/raw/*.jsonl.gz`.
