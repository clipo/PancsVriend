# 03 — The sampled value function: what is measured, artifacts, coupling

Pipeline: `value_functions/sampling/build_value_function.py` (config:
`configs/value_function_scenarios_<model>.yaml`), consumed by
`llm_runner.py --value-function`. The sample-size rule and its statistics are
in `VALUE_FUNCTION_SAMPLING_FOR_REVIEW.md`.

## 1. What is measured

Rather than calling the model live inside every simulation — slow, noisy,
unreproducible — the model is measured once:

> p(role, composition, scenario) = P(the model answers MOVE | it is told it is
> a `role` resident of a `scenario` facing `n_similar` like and
> `n_occupied − n_similar` unlike neighbours).

A composition is a pair (n_similar, n_occupied) with 0 ≤ n_similar ≤
n_occupied ≤ 8: **45 compositions**, so 540 cells per model across 2 roles ×
6 scenarios. Every composition is sampled independently and gets the full
budget — 1-of-2 and 2-of-4 are never pooled, even though they share the
reduced ratio 1/2.

Rates are **effective** rates n_move/(n_move + n_stay), the
retry-until-parseable quantity production uses (moot under grammar, where bad
parses are structurally impossible).

**Sample sizes** follow a two-stage, interval-targeted design: a uniform
pilot of N=100 per composition, then a targeted top-up until every cell's
Wilson 95% CI half-width is ≤ 0.02, iterated to zero residual deficit and
bounded by a hard ceiling of 2,401 samples at p = 0.5. Saturated cells — most
of every surface — need nothing. Sufficiency is then demonstrated
outcome-side rather than assumed: rebuilding every table from half its
samples and re-running all simulations with paired seeds moves no metric
beyond its own error bar (42/42 checks). All sampling uses the clean serving
protocol of 04.

## 2. The artifact (schema vf-1)

One JSON per (scenario, style), under
`value_functions/results/sampled/vf_<label>__<scenario>__R3_dual_count.json`:

- `meta` — model, endpoint arm, temperature, sampler params, grammar hash,
  scenario and identity labels, role→type map, sampling stages, and
  `sampling_protocol` (cache off, seed scheme, server build fingerprint).
- `compositions.{red,blue}` — 45 rows of raw counts (n_move/n_stay/n_bad),
  effective rate and Wilson CI. These are sufficient statistics: everything
  downstream is recomputable and top-ups merge by exact count addition.
- `ratios.{red,blue}` — the 23 reachable other-group fractions plus the
  no-neighbours state, each the equal-weight mean of its member compositions.
  **A visualization and summary object only**, not the policy.

Every individual reply (prompt context, text, finish reason, parse verdict,
seed) is kept in `results/value_functions/raw/*.jsonl.gz`.

## 3. Composition-level lookup, and why

The simulation reads the **composition** surface — the agent's exact
(n_similar, n_occupied) cell rate, with no aggregation. Ratio-level lookup
was removed because the decision is demonstrably not a function of the
fraction alone.

The discriminating test holds the ratio CONSTANT while varying neighbourhood
scale (`analysis_tools/vf_scale_dependence_probe.py`): the members of a ratio
family (1/2 = 1-of-2, 2-of-4, 3-of-6, 4-of-8) share a fraction but differ in
absolute counts, so a pure ratio rule predicts identical rates within every
family. Across 72 multi-member families for gemma: **57 flat, 15 rising, 0
falling** — and the rises are jumps from 0.00 to ~1.00 within a single
fraction, not drift. The all-opposite family rises in 11 of 12 scenario-role
combinations: one opposite neighbour and nothing else means STAY, two or more
means FLEE. The pattern is not model-specific — llama-3.3-70B gives 17/72 and
qwen3.6-27B 26/72 rising families, with zero falling families in any of the
three, so where scale matters more opposite neighbours at an identical ratio
always raises P(MOVE).

Two consequences. The ratio-axis line plot is non-monotonic *by projection*,
because adjacent datapoints interleave families whose members disagree, so
its sawtooth is an averaging artifact rather than model noise. And the ratio
abstraction the classical Schelling model assumes is only an approximation of
how these agents behave. The heatmaps show the surface the simulation
actually consumes.

## 4. Simulation coupling

`llm_runner.py --value-function <file | '{scenario}' template | directory>`
loads the artifact, maps roles via `meta.role_to_type`, and at each decision
draws MOVE with the rate stored for the agent's exact composition cell. All
45 cells per role must carry a defined rate, validated at load time, so a
mid-run lookup failure cannot occur; a scenario-mismatch guard refuses an
artifact measured under a different social framing. Decisions cost
microseconds — a 100-run × 200-step batch takes seconds, fully seeded.

The YAML orchestrator (`configs/vf_run_<model>_r3.yaml`) runs the simulations
and the analysis suite as one command, and **freezes** the resolved
artifacts, their figures, a composition-heatmap grid and a
`TABLE_HASHES.json` into `<run_dir>/value_functions/`, pointing the
simulation at the frozen copies. Every run folder therefore carries the exact
decision tables it simulated from, and two runs used the same tables iff
their hashes match.

Figures: `vfS_*` per-scenario curves (both roles, Wilson whiskers, marker
area ∝ per-datapoint N) and `vfH_*` per-composition heatmaps with per-cell N
printed — sample sizes are visible on every plot by design.
