#!/usr/bin/env python3
"""Census goodness of fit: does sampling agree with the exact tables? (paper result)

    python value_functions/comparison/census_goodness_of_fit.py
    python value_functions/comparison/census_goodness_of_fit.py --out-dir DIR --alpha 0.05

For every model with both an exact (-lp) table and a sequential census (-sanity,
540 cells x n = 100, one request in flight, pinned clock), ONE test:

  G  = binomial deviance of the census against the exact table, summed over
       all 540 cells — how far, in total, the sampled counts sit from what
       the exact probabilities predict (each cell's null is the exact table's
       mass-bound interval [lo, hi]);
  p  = share of 10,000 censuses SIMULATED FROM THE EXACT TABLE whose G is at
       least as large (parametric bootstrap; seed fixed, so the numbers
       reproduce exactly).
PASS if p > alpha: sampling is indistinguishable from the exact table.

This is the census result REPORTED IN THE PAPER. The per-cell fresh-draw check
in sanity_vs_exact.py is the internal sanity check (user decision 2026-11-16);
see census_stats.py for why the two exist. What the global test catches
(measured on hermes, 2026-11-16): every interior cell off by 0.02 -> caught 72%,
by 0.05 -> 100%; a near-0 cell at 0.10 instead of 0.001 -> 93%; a single interior
cell off by 0.2 -> only 26% (the per-cell check's job).

Writes census_goodness_of_fit.csv and census_goodness_of_fit.png (simulated null
distribution per model, observed G marked) into --out-dir, by default the
cross-model results folder. Run as the `census_gof` stage of
run_llm_probability_simulation_analysis.py, so it never needs running by hand.
Exit 0 always: a FAIL is a result, not an error, and so is having no census yet
(a fresh clone or a model whose census has not run): it says so and writes nothing.
"""
import argparse
import csv
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent
for _p in (str(THIS), str(REPO), str(REPO / "prompt_refinement")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                   # noqa: E402
from value_functions.paths import CROSS_MODEL_DIR, SANITY_TABLES_DIR  # noqa: E402
import census_stats as cs                            # noqa: E402
import sanity_vs_exact as sve                        # noqa: E402

SUFFIX = "-chat-grammar"


def display_names():
    """Model order and names from the cross-model registry, so the paper's
    tables agree; falls back to the label if the import fails."""
    try:
        from cross_model_vf_comparison import MODELS
        return [(key, name) for key, name, _ in MODELS]
    except Exception:                                 # noqa: BLE001 — names are cosmetic
        return []


def census_labels(style):
    """Labels with a census for every scenario AND an exact table for each."""
    found = sorted({p.name.split("-sanity__")[0][3:]
                    for p in SANITY_TABLES_DIR.glob(f"vf_*-sanity__*__{style}.json")})
    ok = []
    for lab in found:
        if all(sve.sanity_path(lab, s, style).exists() and sve.exact_path(lab, s, style).exists()
               for s in sve.SCENARIOS):
            ok.append(lab)
        else:
            print(f"[skip] {lab}: census or exact table missing for some scenario")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(CROSS_MODEL_DIR))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--style", default=sve.STYLE_DEFAULT)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    labels = census_labels(args.style)
    if not labels:
        print(f"[census-gof] no complete census under {SANITY_TABLES_DIR}; nothing to test")
        return 0
    order = display_names()
    rank = {key + SUFFIX: i for i, (key, _) in enumerate(order)}
    names = {key + SUFFIX: name for key, name in order}
    labels.sort(key=lambda lab: (rank.get(lab, 99), lab))

    results = []
    for lab in labels:
        rows = sve.stage1(lab, sve.SCENARIOS, args.style)
        K = [r["n_move"] for r in rows]; N = [r["n_samples"] for r in rows]
        P = [r["p_exact"] for r in rows]
        g = cs.deviance_omnibus(K, N, P, [r["exact_lo"] for r in rows],
                                [r["exact_hi"] for r in rows], return_null=True)
        diffs = [abs(r["p_exact"] - r["p_sanity"]) for r in rows]
        results.append({"model": names.get(lab, lab.replace(SUFFIX, "")), "label": lab,
                        "cells": len(rows), "census_draws": int(sum(N)),
                        "G": g["G"], "G_null_mean": g["G_null_mean"], "G_null_q95": g["G_null_q95"],
                        "p_value": round(g["p_mc"], 4), "p_mc_se": g["p_mc_se"],
                        "verdict": "PASS" if g["p_mc"] > args.alpha else "FAIL",
                        "median_abs_diff": round(float(np.median(diffs)), 5),
                        "bootstrap_reps": g["reps"], "seed": g["seed"], "_null": g["null"]})

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = [c for c in results[0] if not c.startswith("_")]
    with open(out / "census_goodness_of_fit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 3
    nrow = -(-len(results) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 3.0 * nrow), squeeze=False)
    for ax, r in zip(axes.flat, results):
        ax.hist(r["_null"], bins=50, color="#b8c4d6", edgecolor="none")
        ax.axvline(r["G_null_q95"], color="#666666", ls=":", lw=1)
        ax.axvline(r["G"], color="#1f4e79" if r["verdict"] == "PASS" else "#b2182b", lw=2)
        ax.set_title(f"{r['model']}  p = {r['p_value']:.2f}", fontsize=10)
        ax.set_yticks([])
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(results):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("deviance G (all 540 cells)", fontsize=8)
    fig.suptitle("Census vs exact table: observed G (line) against 10,000 censuses simulated "
                 "from the exact table\n(dotted: 95th percentile; past it the census would "
                 f"disagree at alpha = {args.alpha})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "census_goodness_of_fit.png", dpi=args.dpi)
    plt.close(fig)

    print(f"{'model':22s} {'G':>8s} {'null mean':>9s} {'null 95%':>9s} {'p':>6s}  verdict")
    for r in results:
        print(f"{r['model']:22s} {r['G']:8.1f} {r['G_null_mean']:9.1f} {r['G_null_q95']:9.1f} "
              f"{r['p_value']:6.3f}  {r['verdict']}")
    print(f"wrote {out / 'census_goodness_of_fit.csv'} and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
