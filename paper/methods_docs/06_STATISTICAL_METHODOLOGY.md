# 06 — Statistical methodology of the paper's ordering table (pointer)

The note itself is `STATISTICAL_METHODOLOGY.md` at the repo root (added by
doctor-duus in 1fe75fb, 2026-11-16); it is not copied here so there is one
version. The code it describes is `compute_scenario_orderings.py`, which
exists as two byte-identical copies: `analysis_tools/paper/` (new, same
commit) and `analysis_tools/ccs2026_presentation/` (the note still cites
this path).

> **Status (2026-11-16).** The note describes the script that writes the
> paper's ordering table, and matches that script. It is NOT the
> procedure of the cross-model stage (`cross_model_vf_comparison.py`: paired
> t, Holm over adjacent ranks, alpha 0.05, no effect-size floor, chance tested
> as a scenario) — see 01 §3b. Three points to settle with co-authors:
> (1) the note's premise that runs are independent is not right — run k
> starts from the same initial grid in every scenario, and finals correlate
> across scenarios (r ≈ 0 for Olmo, 0.3–0.7 for most models, 0.96–1.00 for
> Granite / Mistral / Phi); (2) the ordering table applies Holm over the 3
> headline pairs, the note says 15; (3) "≈" is an effect-size judgment
> (|d| < 0.2), not a significance result.
