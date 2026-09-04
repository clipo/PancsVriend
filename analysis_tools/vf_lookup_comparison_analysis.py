#!/usr/bin/env python3
"""Does ratio-level aggregation change simulation results? — the A/B verdict.

The value function offers two lookup granularities (llm_runner --vf-lookup):

  composition  the agent's exact (n_similar, n_occupied) cell rate — the
               heatmap surface, aggregation-free ground truth;
  ratio        the 23-ratio datapoint = EQUAL-WEIGHT mean over the member
               compositions that alias to the same reduced ratio (the line
               plot). Aliasing is lossy where members genuinely differ
               (e.g. gemma baseline red at all-opposite: 1-of-1 p=0.00,
               2-of-2 p=0.46, 3..8-of-n p=1.00 -> mean 0.81).

This script consumes the manifest written by run_vf_lookup_comparison.sh
(paired batches, run k seeded identically in both modes) and answers, per
(scenario, metric): does the lookup choice shift the final-step mean by more
than the composition batch's own run-to-run 95% CI half-width?

    python analysis_tools/vf_lookup_comparison_analysis.py --manifest <m.json>

Manifest: {"composition": {scenario: dir}, "ratio": {scenario: dir},
           "max_steps": 200}
Outputs (next to the manifest, i.e. inside the self-contained folder):
  vf_lookup_check.{png,csv}     paired-comparison grid + verdict table
                                (reuses vf_simulation_evaluation.fig_half_check).
  vf_lookup_ordering.{png,csv}  scenario-ORDERING check: per metric, do the
                                two lookup modes rank the scenarios the same
                                way? Slopegraph of ranks + Kendall's tau
                                (rank-correlation: 1 = identical order) and
                                pairwise inversions; an inversion is
                                SIGNIFICANT only when the pair is separated
                                beyond both modes' 95% CIs (1.96*sqrt(SE_a^2
                                + SE_b^2)) with opposite signs — level shifts
                                that preserve ordering leave scenario
                                comparisons intact, significant inversions
                                would flip a paper conclusion.
"""
import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
for p in (_THIS, _THIS.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vf_simulation_evaluation import (  # noqa: E402
    ALL_METRICS, METRIC_LABELS, SCENARIO_ORDER, fig_half_check, load_batch,
)

LABELS = ("composition", "ratio-avg")
TITLE = ("Lookup-granularity check: exact composition cells vs "
         "equal-weight ratio datapoints")

SCENARIO_COLORS = {
    "baseline": "#666666", "race_white_black": "#e69f00",
    "ethnic_asian_hispanic": "#56b4e9", "income_high_low": "#009e73",
    "political_liberal_conservative": "#cc79a7", "green_yellow": "#b8860b",
}


def _finals(batch, metric):
    """Per-run final-step values (NaNs dropped) for one scenario batch."""
    import numpy as np
    v = batch[metric][:, -1]
    return v[~np.isnan(v)]


def fig_ordering_check(comp, ratio, scenarios, out_path, csv_path, dpi):
    """Does the lookup mode change how the SCENARIOS rank on each metric?

    Per metric: rank scenarios by their mean final-step value under each
    mode, plot as a slopegraph (rank 1 = highest mean; a crossing line = a
    reordering), and report Kendall's tau — the rank correlation
    2*(concordant - discordant pairs)/(n*(n-1)), 1 = same order, -1 =
    reversed. Every INVERTED pair goes to the CSV; it is flagged
    "significant" only when the two scenarios are separated beyond noise in
    BOTH modes (|mean_a - mean_b| > 1.96*sqrt(SE_a^2 + SE_b^2), the normal-
    approximation 95% criterion for a difference of independent means) with
    opposite signs. Inversions inside the noise band are expected churn.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    stats = {}   # (metric, scenario) -> {mode: (mean, se)}
    for sc in scenarios:
        for m in ALL_METRICS:
            row = {}
            for mode, batches in (("composition", comp), ("ratio", ratio)):
                v = _finals(batches[sc], m)
                row[mode] = (float(np.mean(v)),
                             float(np.std(v, ddof=1) / np.sqrt(len(v))))
            stats[(m, sc)] = row

    pair_rows, tau_by_metric = [], {}
    for m in ALL_METRICS:
        order = {mode: sorted(scenarios,
                              key=lambda sc: -stats[(m, sc)][mode][0])
                 for mode in ("composition", "ratio")}
        conc = disc = 0
        for i in range(len(scenarios)):
            for j in range(i + 1, len(scenarios)):
                a, b = scenarios[i], scenarios[j]
                d = {}
                for mode in ("composition", "ratio"):
                    (ma, sa), (mb, sb) = stats[(m, a)][mode], stats[(m, b)][mode]
                    d[mode] = {"diff": ma - mb,
                               "sep": abs(ma - mb) > 1.96 * np.sqrt(sa**2 + sb**2)}
                inverted = (np.sign(d["composition"]["diff"])
                            != np.sign(d["ratio"]["diff"]))
                if inverted:
                    disc += 1
                else:
                    conc += 1
                if inverted:
                    pair_rows.append({
                        "metric": m, "scenario_a": a, "scenario_b": b,
                        "composition_diff": d["composition"]["diff"],
                        "ratio_diff": d["ratio"]["diff"],
                        "significant": bool(d["composition"]["sep"]
                                            and d["ratio"]["sep"]),
                    })
        npairs = conc + disc
        tau_by_metric[m] = {"tau": (conc - disc) / npairs,
                            "inversions": disc, "order": order}

    pairs = pd.DataFrame(pair_rows, columns=[
        "metric", "scenario_a", "scenario_b",
        "composition_diff", "ratio_diff", "significant"])
    pairs.to_csv(csv_path, index=False)

    ncols = len(ALL_METRICS)
    fig, axes = plt.subplots(1, ncols, figsize=(2.05 * ncols + 1.2, 4.6),
                             squeeze=False)
    for k, m in enumerate(ALL_METRICS):
        ax = axes[0][k]
        info = tau_by_metric[m]
        rank = {mode: {sc: r for r, sc in enumerate(info["order"][mode])}
                for mode in ("composition", "ratio")}
        sig_pairs = {frozenset((r.scenario_a, r.scenario_b))
                     for r in pairs.itertuples()
                     if r.metric == m and r.significant}
        sig_scen = set().union(*sig_pairs) if sig_pairs else set()
        for sc in scenarios:
            r0, r1 = rank["composition"][sc], rank["ratio"][sc]
            crossed = r0 != r1
            hot = sc in sig_scen and crossed
            ax.plot([0, 1], [r0, r1], color=SCENARIO_COLORS[sc],
                    lw=2.4 if hot else 1.4,
                    ls="-" if hot or not crossed else "--",
                    marker="o", ms=4, alpha=1.0 if hot else 0.75)
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(len(scenarios) - 0.5, -0.5)      # rank 1 on top
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["composition", "ratio-avg"], fontsize=7)
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels([f"#{r + 1}" for r in range(len(scenarios))],
                           fontsize=7)
        n_sig = len(sig_pairs)
        ax.set_title(f"{METRIC_LABELS[m]}\nτ = {info['tau']:.2f}, "
                     f"{info['inversions']} inv. ({n_sig} signif.)",
                     fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    handles = [plt.Line2D([], [], color=SCENARIO_COLORS[sc], lw=2, marker="o",
                          ms=4, label=sc) for sc in scenarios]
    handles += [
        plt.Line2D([], [], color="#333", lw=2.4, label="rank change, pair separated beyond 95% CIs in BOTH modes"),
        plt.Line2D([], [], color="#333", lw=1.4, ls="--", label="rank change within noise (CIs overlap)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
               frameon=False)
    fig.suptitle("Scenario ORDERING under the two lookup granularities — "
                 "rank 1 = highest final-step mean; τ = Kendall rank correlation",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.13, 1, 0.9))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return pairs, tau_by_metric


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    man_path = Path(args.manifest)
    man = json.loads(man_path.read_text())
    max_steps = int(man.get("max_steps", 200))
    out_dir = man_path.parent

    scenarios = sorted(man["composition"],
                       key=lambda s: (SCENARIO_ORDER.index(s)
                                      if s in SCENARIO_ORDER else 99, s))
    print(f"loading {len(scenarios)} composition batches ...")
    comp = {sc: load_batch(d, max_steps) for sc, d in man["composition"].items()}
    print(f"loading {len(scenarios)} ratio batches ...")
    ratio = {sc: load_batch(d, max_steps) for sc, d in man["ratio"].items()}

    tab = fig_half_check(comp, ratio, scenarios,
                         out_dir / "vf_lookup_check.png",
                         out_dir / "vf_lookup_check.csv",
                         args.dpi, labels=LABELS, title=TITLE)
    print(f"wrote {out_dir}/vf_lookup_check.png and .csv")

    pairs, taus = fig_ordering_check(comp, ratio, scenarios,
                                     out_dir / "vf_lookup_ordering.png",
                                     out_dir / "vf_lookup_ordering.csv",
                                     args.dpi)
    print(f"wrote {out_dir}/vf_lookup_ordering.png and .csv")
    print("\n=== scenario-ordering check (Kendall tau per metric) ===")
    for m, info in taus.items():
        print(f"  {m:20s} tau = {info['tau']:+.2f}   "
              f"inversions: {info['inversions']}")
    n_sig = int(pairs.significant.sum()) if len(pairs) else 0
    if n_sig:
        print(f"\n{n_sig} SIGNIFICANT inversion(s) — the lookup mode changes "
              "which scenario ranks higher:")
        for r in pairs[pairs.significant].itertuples():
            print(f"  {r.metric}: {r.scenario_a} vs {r.scenario_b}  "
                  f"(comp diff {r.composition_diff:+.3f}, "
                  f"ratio diff {r.ratio_diff:+.3f})")
    else:
        print("\nNo significant inversions: all rank changes are within "
              "run-to-run noise — scenario ORDERINGS are robust to the "
              "lookup choice.")

    import pandas as pd
    print("\n=== lookup-granularity verdicts "
          "(|paired Δmean| vs composition-mean 95% CI) ===")
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(tab[["scenario", "metric", f"{LABELS[0]}_mean",
                   f"{LABELS[1]}_mean", "paired_diff_mean",
                   f"{LABELS[0]}_mean_ci95", "verdict"]].to_string(index=False))
    n_flag = int((tab.verdict == "FLAG").sum())
    print(f"\n{'NO MEANINGFUL DIFFERENCE' if n_flag == 0 else f'{n_flag} FLAG(s)'} "
          f"across {len(tab)} (scenario, metric) checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
