#!/usr/bin/env python3
"""How does the multi-split ruler scale with value-function precision?

    python analysis_tools/vf_ruler_scaling.py [--labels ...] [--out-dir DIR]

Reads, per model, the keyed rulers measured at keep fractions 1/2, 3/4 and 7/8
(prompt_refinement/results/value_functions/multisplit_<label>_b32g20_keyed,
..._keyed_f075, ..._keyed_f0875) and fits the exponent alpha in

    s  ∝  (injected table error)^alpha,    error_f ∝ sqrt(1/f - 1)

so log s_f = const + alpha * log sqrt(1/f - 1). Two readings:

  alpha ≈ 1    LINEAR: the simulation's displacement is proportional to the
               table error. vf_sampling_plan's pricing (n ∝ 1/w^2, halving the
               ruler costs 4x the samples) is right.
  alpha ≈ 0.5  FLIP: displacement is dominated by discrete decision flips
               whose NUMBER scales with the error but whose SIZE is set by the
               dynamics (chaotic amplification). Halving the ruler then costs
               16x the samples and every sampling-plan quote is optimistic by
               k^2. Sampling is a weak lever; exact probabilities are the fix.

Quantities fitted (DI only, the decision metric; others written for reference):
  * per scenario: RMS(Δ) of the full-vs-reduced displacement (vf_multisplit_check.csv)
  * per adjacent scenario pair: s_obs = SD over splits of the pair displacement
    (the ruler vf_rank_stability actually uses), pairs ordered by the full
    arm's means.
Each has a relative SE of ~1/sqrt(2B) = 12.5 % at B = 32, so alpha from two
fractions carries ~±0.2; three fractions and pooling across pairs/models tighten
it. Writes ruler_scaling.csv (one row per model x metric x quantity) and
ruler_scaling.png (log-log, one line per model, pooled alpha in the title).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_THIS = Path(__file__).resolve().parent
STORE = _THIS.parent / "prompt_refinement" / "results" / "value_functions"
FRACTIONS = {"": 0.5, "_f075": 0.75, "_f0875": 0.875}
LABELS = ["qwen3.6-27b-chat-grammar", "deepseek-v4-flash-chat-grammar",
          "hermes-4.3-36b-chat-grammar", "llama-3.3-70b-chat-grammar",
          "gemma-4-31b-chat-grammar", "olmo-2-32b-chat-grammar",
          "granite-4.2-30b-chat-grammar", "mistral-small-4-119b-chat-grammar",
          "phi-4-14b-chat-grammar"]


def injected_sd(f):
    return np.sqrt(1.0 / f - 1.0)


def load_rulers(label):
    """{f: (check DataFrame, deltas DataFrame)} for the fractions on disk."""
    out = {}
    for suffix, f in FRACTIONS.items():
        d = STORE / f"multisplit_{label}_b32g20_keyed{suffix}"
        if (d / "vf_multisplit_deltas.csv").exists():
            out[f] = (pd.read_csv(d / "vf_multisplit_check.csv"),
                      pd.read_csv(d / "vf_multisplit_deltas.csv"))
    return out


def pair_rulers(chk, dl, metric):
    """s_obs per adjacent pair (ordered by the full arm's means, high first)."""
    c = chk[chk.metric == metric].set_index("scenario")
    order = list(c.full_mean.sort_values(ascending=False).index)
    piv = dl[dl.metric == metric].pivot(index="split", columns="scenario", values="delta")
    return {f"{a}>{b}": float((piv[a] - piv[b]).std(ddof=1))
            for a, b in zip(order[:-1], order[1:]) if a in piv and b in piv}


def fit_alpha(fs, ss):
    """Slope of log s on log injected_sd; None with fewer than two points."""
    fs, ss = np.asarray(fs, float), np.asarray(ss, float)
    ok = ss > 0
    if ok.sum() < 2:
        return None
    x, y = np.log(injected_sd(fs[ok])), np.log(ss[ok])
    return float(np.polyfit(x, y, 1)[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", nargs="*", default=LABELS)
    ap.add_argument("--metric", default="dissimilarity_index")
    ap.add_argument("--out-dir", default=str(_THIS.parent / "prompt_refinement" / "results" / "figures"))
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    rows, curves = [], {}
    for label in args.labels:
        rulers = load_rulers(label)
        if len(rulers) < 2:
            print(f"  {label}: {len(rulers)} fraction(s) on disk — skipped (need >= 2)")
            continue
        fs = sorted(rulers)
        # per-scenario RMS
        for sc in rulers[0.5][0][rulers[0.5][0].metric == args.metric].scenario:
            ss = [float(rulers[f][0][(rulers[f][0].metric == args.metric)
                                     & (rulers[f][0].scenario == sc)].delta_rms.iloc[0]) for f in fs]
            rows.append({"label": label, "metric": args.metric, "quantity": "scenario_rms",
                         "item": sc, **{f"s_f{f:g}": s for f, s in zip(fs, ss)},
                         "alpha": fit_alpha(fs, ss)})
        # per-pair s_obs
        pr = {f: pair_rulers(*rulers[f], args.metric) for f in fs}
        for pair in pr[0.5]:
            ss = [pr[f].get(pair, np.nan) for f in fs]
            rows.append({"label": label, "metric": args.metric, "quantity": "pair_s_obs",
                         "item": pair, **{f"s_f{f:g}": s for f, s in zip(fs, ss)},
                         "alpha": fit_alpha(fs, ss)})
        # pooled per model: geometric mean of pair rulers at each fraction
        gm = [float(np.exp(np.nanmean(np.log([v for v in pr[f].values() if v > 0])))) for f in fs]
        a = fit_alpha(fs, gm)
        curves[label] = (fs, gm, a)
        rows.append({"label": label, "metric": args.metric, "quantity": "pooled_pairs_geomean",
                     "item": "all", **{f"s_f{f:g}": s for f, s in zip(fs, gm)}, "alpha": a})
        print(f"  {label:36s} fractions={fs}  pooled alpha = {a:+.2f}"
              + "   (1 = linear pricing right; 0.5 = flip model, 16x per halving)")

    if not rows:
        print("nothing to fit"); return 2
    tab = pd.DataFrame(rows)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "ruler_scaling.csv", index=False)
    pooled = tab[tab.quantity == "pooled_pairs_geomean"].alpha.dropna()
    print(f"\n  models: {len(pooled)}   median pooled alpha = {pooled.median():+.2f}   "
          f"IQR [{pooled.quantile(.25):+.2f}, {pooled.quantile(.75):+.2f}]")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for label, (fs, gm, a) in curves.items():
        ax.plot(injected_sd(np.array(fs)), gm, marker="o", lw=1.5,
                label=f"{label.split('-chat')[0]}  α={a:+.2f}")
    xs = np.array([injected_sd(0.875), injected_sd(0.5)])
    ref = np.median([c[1][-1] for c in curves.values()])
    for a, ls, name in ((1.0, "--", "linear α=1"), (0.5, ":", "flip α=0.5")):
        ax.plot(xs, ref * (xs / xs[-1]) ** a, ls, color="k", lw=1, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("injected table error, sqrt(1/f − 1)   (f = 7/8, 3/4, 1/2)")
    ax.set_ylabel(f"adjacent-pair ruler s ({args.metric.replace('_', ' ')})")
    ax.set_title(f"Ruler scaling with value-function precision — median α = {pooled.median():+.2f}  "
                 f"(reference slopes: α = 1 linear pricing, α = 0.5 flip model)", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(); fig.savefig(out / "ruler_scaling.png", dpi=args.dpi); plt.close(fig)
    print(f"wrote {out}/ruler_scaling.csv and ruler_scaling.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
