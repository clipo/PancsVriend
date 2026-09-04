# 01 — The Schelling simulation: model, parameters, metrics, outputs

Implementation: `base_simulation.py` (engine), `Agent.py` (mechanical agent),
`baseline_runner.py` / `llm_runner.py` (batches), `config.py` (parameters),
`Metrics.py` + `DissimilarityIndex.py` (metrics).

## 1. World

| parameter | value | source |
|---|---|---|
| grid | 10 × 10, hard boundary (no wrap) | `config.GRID_SIZE` |
| agents | 40 type-A + 40 type-B (20 cells empty) | `config.NUM_TYPE_A/B` |
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

**Convergence**: a run stops after `NO_MOVE_THRESHOLD` = 5 consecutive steps
with zero moves, or at `--max-steps`. The reported `convergence_step` is the
FIRST step of that zero-move window, so `final_step = convergence_step + 4`.

**Seeding**: every run is seeded with `random_seed = run_id`, seeding both
numpy (grid init, visit order) and stdlib random (decisions, destinations).
Run k of any two batches therefore shares its initial grid and random stream,
so batches are pairable run-for-run and value-function runs are exactly
reproducible.

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

The index is a property of the (data, partition) pair, so its floor moves
with the grid: at 10×10 with this population, random allocation already
scores 0.25 (sd 0.07) and a two-cluster city with a randomly angled boundary
scores 0.81, against a maximum of 1.0. The random floor falls as 1/√(agents
per tract) — 0.13 at 20×20 — while the ceiling stays flat, so **DI levels do
not compare across grid sizes**. `python DissimilarityIndex.py` regenerates
the tract maps and the reference table.

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

`experiments/<name>/` holds `config.json` (full provenance, including the
value-function file), `metrics_history.csv.gz` (row per run × step, all seven
metrics), `convergence_summary.csv`, `step_statistics.csv`, `run_summary.csv`
(one row per run: convergence step plus every metric at the final step), and
the run record — per-step decision counts and one grid frame per step, or one
record and frame per agent decision for live-LLM runs, which is where the raw
replies are kept. Everything reads these through `run_files.py`, which
returns the same per-step tables for either format.

Caveat: `step_statistics.csv` pools per-step survivors — runs stop writing
rows once converged — so it is survivor-biased at late steps. Trajectory
analyses use `metrics_history.csv.gz` with forward-fill
(`analysis_tools/plot_style.step_stats_forward_filled`), which is the honest
continuation because a converged run's grid is frozen.
