# 01 — The Schelling simulation: model, parameters, metrics, outputs

Implementation: `base_simulation.py` (engine), `Agent.py` (mechanical agent),
`baseline_runner.py` / `llm_runner.py` (batches), `config.py` (parameters),
`Metrics.py` + `DissimilarityIndex.py` (metrics).

## 1. World

| parameter | value | source |
|---|---|---|
| grid | 20 × 20, hard boundary (no wrap) | run YAML `grid_size: 20` (`config.GRID_SIZE` default is 10) |
| agents | 160 type-A + 160 type-B (80 cells empty) | run YAML `num_type_a/b: 160` |
| runs | 10,000 per model × scenario (9 × 6 × 10,000 = 540,000) | run YAML `runs` |
| initial placement | uniform random shuffle of agents + empties | `base_simulation.populate_grid` |
| neighbourhood | Moore (8-cell); off-grid and empty cells excluded from ratios | `Agent._unlike_ratio` |

## 2. Dynamics

Each **step**, agents are visited in a random order (reshuffled each step) and
decide stay/move from their local view; movers relocate to a **uniformly
random empty cell anywhere on the grid**. Relocation is global and
non-strategic, and identical for mechanical, live-LLM and value-function
agents — the agent types differ ONLY in the stay/move decision.

**Mechanical decision** (`Agent.random_response`): binary utility u = 1 if
similar_ratio ≥ `SIMILARITY_THRESHOLD` (0.5) else 0, over OCCUPIED neighbours
only; an agent with no neighbours is satisfied. It moves iff u ≤
`AGENT_SATISFACTION_THRESHOLD` (0) — **MOVE iff strictly more than half of its
occupied neighbours are unlike it**. The LLM / value-function decision is 02
and 03.

A step is one full sweep: EVERY agent decides once per step (not one agent
per step).

**Stopping**: a run stops after `NO_MOVE_THRESHOLD` = 5 consecutive steps
with zero moves, or at `max_steps` = 1000. The reported `convergence_step` is
the FIRST step of that zero-move window, so `final_step = convergence_step +
4`. A run's **final** value of a metric is its value at `final_step`, the
last step simulated — one rule for converged and capped runs alike (no
window means). Capped runs are therefore not necessarily stationary; the
share of runs that meet the criterion is reported per model × scenario.

**Seeding**: every run is seeded with `random_seed = run_id`: numpy draws the
initial grid and the per-step visit order. In value-function runs the
accept/reject draw and the destination draw are `keyed_uniform(run_id, step,
agent_id, salt)` — a splitmix64 hash, so each draw is a pure function of its
key (`rng_scheme: keyed` in `config.json`; `VF_RNG_SCHEME=shared` reproduces
runs made before 2026-09-04). Run k of any two batches shares its initial
grid, so batches are paired run-for-run — the basis of the paired tests in
§3b — and every run is regenerable from its `run_id`.

## 3. Metrics

All seven are computed per step on the live grid by
`Metrics.calculate_all_metrics`:

| metric | definition | neighbourhood |
|---|---|---|
| `clusters` | number of connected same-type components (flood fill) | 4-conn |
| `switch_rate` | fraction of adjacent pairs with unlike types | 4-conn |
| `distance` | mean Manhattan distance from each agent to the nearest unlike agent | global |
| `mix_deviation` | mean over agents of \|0.5 − like-share\| | 8-conn |
| `share` | global like-neighbour share over all neighbour pairs | 8-conn |
| `ghetto_rate` | **count** of agents with zero unlike neighbours (a count, not a rate) | 8-conn |
| `dissimilarity_index` | ½ Σ_tracts \|a_t/A − b_t/B\| over a 3×3 partition into 9 tracts | tract |

The dissimilarity index is the headline metric and owns its module,
`DissimilarityIndex.py` — the tract partition, the index, the frozen 10×10
reference implementation used as the test oracle, and the reference values.

At 20×20 the tracts split rows and columns 6-8-6: four 6×6 corners (36
cells), four 6×8 edges (48) and one 8×8 centre (64).

The index is a property of the (data, partition) pair, so its floor moves
with the grid. At 20×20 with 160 + 160 agents, random allocation scores
**0.125 (sd 0.033; 5th–95th percentile 0.075–0.181; 20,000 draws)** and a
two-cluster city with a randomly angled boundary scores 0.80 (sd 0.07),
against a maximum of 1.0. At 10×10 the same quantities are 0.25 and 0.81:
the random floor falls as 1/√(agents per tract) while the ceiling stays
flat, so **DI levels do not compare across grid sizes**. `python
DissimilarityIndex.py` regenerates the tract maps and the reference table.

## 3b. Significance rules (decided 2026-09-05)

"Chance" is tested like a scenario: each run's `initial_<metric>` (frame 0, a
uniformly random allocation) is its own paired draw from the chance
distribution, and a scenario not significantly above its initial grids
"shows no segregation". Every statement in the cross-model ranking table and
bump charts is a **paired t-test** on per-run differences (final − initial
for chance; final − final for two scenarios, paired by `run_id`),
**Holm-corrected**, with no normality gate and no practical floor. At n =
10,000 a DI gap of ~0.002 is significant, so effect size is reported
alongside: bump-chart gap bars are binned on Cohen's d (mean gap / pooled SD
of the two scenarios' final values; 0.2 / 0.5 / 0.8). Details:
`analysis_tools/analysis_guide.md` §7b.

The paper's ordering table (`key_results_ordering.tex`) is produced by a
different script, `analysis_tools/paper/compute_scenario_orderings.py` (identical copy in
`analysis_tools/ccs2026_presentation/`),
which uses **independent Welch t-tests**, Holm-corrected over the three
headline pairs, AND an effect-size floor: `<` needs p < 0.01 and |Cohen's d|
≥ 0.2, `≤` needs p < 0.05 and |d| ≥ 0.2, and `≈` is anything with |d| < 0.2
or p ≥ 0.05. Every `≈` in the paper's table comes from the floor, so it is a
judgment of practical equality, not a significance result. The full note is
[06](06_STATISTICAL_METHODOLOGY.md). The two procedures should be reconciled
before submission — see 05 caveat 6.

Caveat: the 4- vs 8-neighbourhood split between metrics is historical.

## 4. Scenarios (social framing)

`scenarios_a2.py` `CONTEXT_SCENARIOS` — identical mechanics, only the agent
identity labels differ (red = type_a, blue = type_b):

| scenario | type_a | type_b |
|---|---|---|
| baseline | red team resident | blue team resident |
| race_white_black | white middle class family | Black family |
| ethnic_asian_hispanic | Asian American family | Hispanic/Latino family |
| income_high_low | high-income household | low-income household |
| political_liberal_conservative | politically liberal household | politically conservative household |
| green_yellow | green team resident | yellow team resident |

## 5. Per-experiment outputs

`<run_dir>/<experiment>/` holds `config.json` (full provenance, including the
value-function file), `metrics_history.csv.gz` (row per run × step, all seven
metrics) and `run_summary.csv` (`convergence_summary.csv` and `step_statistics.csv` were dropped on 2026-09-05 as re-derivable from these two)
(one row per run: convergence step plus every metric at the final step), and
the run record — per-step decision counts and one grid frame per step, or one
record and frame per agent decision for live-LLM runs, which is where the raw
replies are kept. Everything reads these through `run_files.py`, which
returns the same per-step tables for either format.

Caveat (historical; the file has not been written since 2026-09-05): `step_statistics.csv` pools per-step survivors — runs stop writing
rows once converged — so it is survivor-biased at late steps. Trajectory
analyses use `metrics_history.csv.gz` with forward-fill
(`analysis_tools/plot_style.step_stats_forward_filled`), which is the honest
continuation because a converged run's grid is frozen.
