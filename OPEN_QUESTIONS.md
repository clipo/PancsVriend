# Open questions

Decisions to take in a meeting rather than in a session. One entry per
question: the current state of the code, the numbers that make it a
question, the options, and what to bring. Resolved entries move to the
methods docs (`paper/methods_docs/`) or the relevant module docstring.

## Should the scenario-ordering tests carry a practical floor? (raised 2026-09-05)
Since 2026-09-05 every significance statement in the bump charts
(`cross_model_bump_dissimilarity.png`, `cross_model_bump_all_metrics.png`) and the per-model
ranking tables is a paired t-test on per-run differences with **no floor**:
a rung is solid whenever the Holm-corrected p < 0.05, and a scenario is
"at chance" only when it is not significantly above its own initial grids.
With 10,000 runs per (model, scenario) the standard error of a mean DI is
~0.0003, so a gap of ~0.002 DI — about one agent's tract move — is
significant; under this rule the DI ordering resolves 41 of 45 rungs (20
under the previous 0.01 floor) and 46 of 54 (model, scenario) DI cells are
"above chance", including mistral's baseline at +0.003. Every figure caption
therefore says "solid = statistically distinguishable, not necessarily
large". The morning rule (0.01 DI floor = roughly two agents' moves; 0.3 x
chance SD for the other metrics) survives only in `vf_rank_stability`'s
FLOOR-TIE, where it bounds GPU top-up quotes.
- Decide: keep "no floor" for the statistical statements (magnitude read from
  the violins and `cross_model_rankings.csv`), or reinstate a stated margin
  as a minimum effect size (and, if so, test against it rather than cut on
  the point estimate).
- Decide: does FLOOR-TIE stay in the certification stage regardless?
- Numbers to bring: `prompt_refinement/results/figures/cross_model_pairwise_tests.csv`
  (`mean_gap` vs `p_holm`) and `cross_model_chance_tests.csv` (`excess`).


## Should a null *simulation* accompany the frame-0 chance scenario? (raised 2026-09-05)
"Chance" is currently each run's own initial grid (frame 0, a uniform random
allocation), paired with that run's final value. A composition-blind
value function (e.g. p_move = 0.5 in every cell, or a model's average move
rate) run through the full simulation gives the SAME null distribution for
every final-state metric — a uniformly chosen agent moving to a uniformly
chosen empty cell is a symmetric transition, so the uniform distribution over
configurations is preserved at every step — so it adds nothing to the chance
test. Its value is as a PLACEBO through the whole pipeline (keyed RNG,
packing, run_summary, ordering tests, rank-stability) with a known answer:
every scenario at chance, every rung hollow, blank bump column.
- Decide: add a small placebo campaign (composition-blind, mobility-matched
  to a real model, ~1,000 runs x 200 steps; it never converges) as a
  validation stage, and say so in the methods?
- Decide: mobility-matched (a model's mean p_move) or p = 0.5?
- Related reference already in the repo: `mech_baseline` (classical
  threshold agents) — "known structure", not "no structure".
