#!/usr/bin/env python3
"""Cross-model comparison of value-function-driven Schelling simulations.

Compares the five campaign models on the SAME pipeline (R3_dual_count value
functions, composition-level lookup, 100 runs x 6 scenarios each):

    python analysis_tools/cross_model_vf_comparison.py
    python analysis_tools/cross_model_vf_comparison.py --run-root <dir>

Input: the newest analysis/combined_final_metrics.csv per model under
<run-root>/run_*_<model>-vf-r3/ (final-step value per run).

Outputs (prompt_refinement/results/figures/ by default):
  cross_model_metrics.png       7 panels (one per metric): x = scenario,
                                one series per model, 95% CI whiskers.
  cross_model_dissimilarity.png BUMP CHART of the scenario ORDERING per
                                model (rank 1 = most segregating). Colour
                                encodes SCENARIO here — absolute DI levels
                                (colour = model) are the first panel of
                                cross_model_metrics.png, so the two figures
                                never use one palette for two meanings. A
                                horizontal line means the models agree on
                                where that context sits; crossings are
                                ordering disagreements. A marker is hollow
                                when that scenario is NOT significantly
                                different from the next-ranked one. The test
                                is PAIRED (run k uses the same seed, hence
                                the same initial grid, in every scenario) and
                                NORMALITY-GATED on the per-run DIFFERENCES —
                                which is the paired test's actual assumption,
                                not the normality of each scenario's levels:
                                Shapiro-Wilk pass -> paired t-test, fail ->
                                Wilcoxon signed-rank. Holm-corrected across
                                each model's adjacent pairs. Hollow = that
                                rung of the ladder is not resolved.
  cross_model_metric_rankings.png  the SAME ordering test applied to the
                                other six metrics, one bump-chart subplot
                                each. Rank 1 = MOST SEGREGATING in every
                                panel: clusters and switch_rate are inverted
                                (higher value = less segregation) via
                                SEGREGATION_DIRECTION, so all panels read the
                                same way.
  cross_model_pairwise_tests.csv  every adjacent comparison, ALL metrics, with
                                Shapiro p on the differences, the test used,
                                raw and Holm-adjusted p, and the verdict.
  cross_model_rankings.csv      per model, scenarios ranked by mean DI, with
                                Kendall tau vs every other model (does the
                                social-context ordering agree across models?)
"""
import argparse
import glob
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_THIS = Path(__file__).resolve().parent
REPO = _THIS.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis_tools.normality_tests import holm_correction  # noqa: E402

MODELS = [
    ("gemma-4-31b", "gemma-4-31B", "#1b9e77"),
    ("llama-3.3-70b", "llama-3.3-70B", "#d95f02"),
    ("qwen3.6-27b", "qwen3.6-27B", "#7570b3"),
    ("mistral-small-4-119b", "mistral-small-4-119B", "#e7298a"),
    ("deepseek-v4-flash", "deepseek-v4-flash", "#66a61e"),
]
METRICS = ["dissimilarity_index", "clusters", "switch_rate", "distance",
           "mix_deviation", "share", "ghetto_rate"]
# Which way does each metric point? +1 = higher value means MORE segregation,
# -1 = higher value means LESS. Verified against Metrics.py (2026-08-28):
#   clusters      count of connected same-type components -> more components
#                 means a more broken-up grid, i.e. LESS segregation
#   switch_rate   fraction of adjacent-neighbour transitions that change type
#                 -> more switching means more interleaving, LESS segregation
#   distance      mean distance to the nearest UNLIKE agent  -> more = more
#   mix_deviation |0.5 - like/total| per agent               -> more = more
#   share         like-neighbour fraction                    -> more = more
#   ghetto_rate   agents with no unlike neighbour            -> more = more
#   dissimilarity standard index                             -> more = more
SEGREGATION_DIRECTION = {
    "dissimilarity_index": +1, "clusters": -1, "switch_rate": -1,
    "distance": +1, "mix_deviation": +1, "share": +1, "ghetto_rate": +1,
}
METRIC_LABELS = {
    "dissimilarity_index": "dissimilarity index", "clusters": "clusters",
    "switch_rate": "switch rate", "distance": "distance",
    "mix_deviation": "mix deviation", "share": "share",
    "ghetto_rate": "ghetto count",
}
SCENARIO_ORDER = ["llm_baseline", "green_yellow", "income_high_low",
                  "political_liberal_conservative", "race_white_black",
                  "ethnic_asian_hispanic"]
SCENARIO_LABELS = {
    "llm_baseline": "red/blue\n(baseline)", "green_yellow": "green/yellow",
    "income_high_low": "income\nhigh/low",
    "political_liberal_conservative": "political\nlib/cons",
    "race_white_black": "racial\nwhite/black",
    "ethnic_asian_hispanic": "ethnic\nasian/hispanic",
}
MECH_SCENARIOS = {"mech_baseline", "baseline_mechanical", "mechanical"}


def load_models(run_root):
    """{model_key: DataFrame} from the newest run per model."""
    out, missing = {}, []
    for key, _, _ in MODELS:
        hits = sorted(glob.glob(str(Path(run_root) /
                                    f"run_*_{key}-vf-r3" / "analysis" /
                                    "combined_final_metrics.csv")))
        if not hits:
            missing.append(key)
            continue
        df = pd.read_csv(hits[-1])
        df["_source"] = hits[-1]
        out[key] = df
        print(f"  {key:22s} {len(df):4d} rows  <- {Path(hits[-1]).parents[1].name}")
    if missing:
        print(f"  WARNING: no run found for {missing}")
    if not out:
        sys.exit(f"no model runs found under {run_root}")
    return out


def mean_ci(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan
    if len(v) == 1:
        return float(v[0]), 0.0
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


def mech_reference(data, metric):
    """Mean mechanical-baseline value if any run carries one."""
    vals = []
    for df in data.values():
        m = df[df.scenario.isin(MECH_SCENARIOS)]
        if len(m):
            vals.append(m[metric].mean())
    return float(np.mean(vals)) if vals else None


def fig_metrics(data, scenarios, out_path, dpi):
    n = len(METRICS)
    fig, axes = plt.subplots(n, 1, figsize=(9.5, 2.45 * n), sharex=True)
    x = np.arange(len(scenarios))
    for ax, metric in zip(axes, METRICS):
        for k, (key, label, color) in enumerate(MODELS):
            if key not in data:
                continue
            df = data[key]
            means, errs = [], []
            for sc in scenarios:
                m, e = mean_ci(df[df.scenario == sc][metric])
                means.append(m); errs.append(e)
            off = (k - (len(MODELS) - 1) / 2) * 0.13
            ax.errorbar(x + off, means, yerr=errs, marker="o", ms=4.5, lw=1.5,
                        capsize=2.5, color=color, label=label, alpha=0.9)
        ref = mech_reference(data, metric)
        if ref is not None:
            ax.axhline(ref, color="black", ls="--", lw=1, alpha=0.6,
                       label="mechanical baseline")
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios],
                             fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Cross-model comparison — value-function Schelling simulations\n"
                 "final-step means, 100 runs per (model, scenario); "
                 "whiskers = 95% CI", fontsize=12)
    fig.tight_layout(rect=(0, 0.045, 1, 0.962))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out_path}")


def rank_and_test(data, scenarios, metric):
    """Per model: scenario ranks by SEGREGATION on `metric` (rank 1 = most
    segregating, using SEGREGATION_DIRECTION so inverted metrics like
    clusters and switch_rate are reversed), plus whether each rank is
    statistically distinguishable from the next one down.

    The comparison is PAIRED — run k uses the same seed, hence the same
    initial grid, in every scenario, so testing the per-run differences
    removes the shared initial-configuration variance. The test is
    NORMALITY-GATED on those DIFFERENCES (the paired test's actual
    assumption; the normality of each scenario's levels is a different
    question and does not justify it): Shapiro-Wilk pass -> paired t-test,
    fail -> Wilcoxon signed-rank. p-values are Holm-corrected across each
    model's adjacent comparisons.

    Returns (models_present, ranks, tied, rows).
    """
    models_present = [(k, lab) for k, lab, _ in MODELS if k in data]
    ranks, tied, rows = {}, {}, []
    for key, _ in models_present:
        df = data[key]
        series = {sc: df[df.scenario == sc].set_index("run_id")[metric]
                  for sc in scenarios}
        # Rank 1 = MOST SEGREGATING. For metrics where a higher value means
        # less segregation (clusters, switch_rate) the order is reversed, so
        # every panel reads the same way regardless of the metric's polarity.
        direction = SEGREGATION_DIRECTION.get(metric, +1)
        ordered = sorted(scenarios,
                         key=lambda sc: -direction * series[sc].mean())
        ranks[key] = {sc: r for r, sc in enumerate(ordered)}

        pvals, pairs = [], []
        for r, sc in enumerate(ordered[:-1]):
            nxt = ordered[r + 1]
            a, b = series[sc], series[nxt]
            common = a.index.intersection(b.index)
            d = (a.loc[common] - b.loc[common]).to_numpy(dtype=float)
            d = d[~np.isnan(d)]
            rec = {"metric": metric, "model": key, "rank": r + 1,
                   "segregation_direction": direction,
                   "scenario": sc, "next_scenario": nxt,
                   "mean_gap": float(a.loc[common].mean() - b.loc[common].mean()),
                   "n_pairs": int(len(d))}
            if len(d) < 2 or np.allclose(d, 0.0):
                # identical per-run values: nothing to resolve, and both tests
                # are undefined on an all-zero difference vector
                pv, test, sh = 1.0, "degenerate (all differences zero)", np.nan
            else:
                sh = float(stats.shapiro(d).pvalue)
                if sh >= 0.05:
                    test = "paired t"
                    pv = float(stats.ttest_rel(a.loc[common], b.loc[common],
                                               nan_policy="omit").pvalue)
                else:
                    # Wilcoxon assumes the differences are symmetric about
                    # their centre; for strongly skewed cases it is itself an
                    # approximation (the sign test would be assumption-free).
                    test = "Wilcoxon signed-rank"
                    pv = float(stats.wilcoxon(d).pvalue)
            rec.update({"shapiro_p_differences": sh, "test": test, "p_raw": pv})
            rows.append(rec)
            pvals.append(pv); pairs.append(sc)

        adj = holm_correction(np.array(pvals)) if pvals else np.array([])
        for k_i, (sc, pa) in enumerate(zip(pairs, adj)):
            tied[(key, sc)] = bool(pa >= 0.05)
            rows[-len(pairs) + k_i]["p_holm"] = float(pa)
            rows[-len(pairs) + k_i]["distinguishable"] = bool(pa < 0.05)
        tied[(key, ordered[-1])] = False       # last rank has no successor
    return models_present, ranks, tied, rows


def draw_bump(ax, models_present, ranks, tied, scenarios, scen_color,
              label_right=True, tick_fontsize=8):
    """Bump chart: one line per scenario, y = its rank under each model."""
    xm = np.arange(len(models_present))
    for sc in scenarios:
        ys = [ranks[k][sc] for k, _ in models_present]
        ax.plot(xm, ys, lw=2.0, color=scen_color[sc], alpha=0.85, zorder=2)
        for xi, (k, _) in enumerate(models_present):
            hollow = tied[(k, sc)]
            ax.scatter([xi], [ranks[k][sc]], s=70, zorder=3,
                       color="white" if hollow else scen_color[sc],
                       edgecolors=scen_color[sc], linewidths=1.8)
        if label_right:
            ax.annotate(SCENARIO_LABELS.get(sc, sc).replace("\n", " "),
                        (len(models_present) - 1 + 0.12, ys[-1]), fontsize=7.5,
                        va="center", color=scen_color[sc])
    ax.set_xticks(xm)
    ax.set_xticklabels([lab for _, lab in models_present],
                       fontsize=tick_fontsize, rotation=20, ha="right")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([f"#{r+1}" for r in range(len(scenarios))],
                       fontsize=tick_fontsize)
    ax.set_ylim(len(scenarios) - 0.4, -0.6)            # rank 1 on top
    ax.set_xlim(-0.4, len(models_present) - 1 + (1.5 if label_right else 0.4))
    ax.grid(axis="y", alpha=0.25)


def scenario_palette(scenarios):
    cmap = plt.get_cmap("tab10")
    return {sc: cmap(k % 10) for k, sc in enumerate(scenarios)}


def fig_dissimilarity(data, scenarios, out_path, dpi):
    """Scenario ordering per model for the dissimilarity index (bump chart).

    The absolute-level panel that used to sit beside this was dropped
    (2026-08-28): it duplicated the dissimilarity panel of
    cross_model_metrics.png, and it coloured by MODEL while this panel
    colours by SCENARIO — one figure, two palettes, two meanings.
    """
    models_present, ranks, tied, rows = rank_and_test(
        data, scenarios, "dissimilarity_index")
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.0))
    draw_bump(ax, models_present, ranks, tied, scenarios,
              scenario_palette(scenarios))
    ax.set_ylabel("rank")
    ax.set_title("crossing = models disagree on the ordering;  hollow marker = "
                 "not distinguishable from the next rank\n(paired test on "
                 "per-run differences, normality-gated t / Wilcoxon, "
                 "Holm-corrected within each model)", fontsize=9)
    fig.suptitle(f"Which social context segregates most? Scenario ordering by model\n"
                 f"dissimilarity index, final step, 100 runs per (model, scenario);"
                 f"  rank #1 = most segregated, #{len(scenarios)} = least",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out_path}")
    return rows


def fig_metric_rankings(data, scenarios, metrics, out_path, dpi):
    """One bump-chart subplot per metric, same ordering test as the DI figure.

    Rank 1 = MOST SEGREGATING in every panel. Two metrics run the other way
    (clusters and switch_rate: a higher value means a more broken-up, more
    interleaved grid), so their rankings are inverted via
    SEGREGATION_DIRECTION and each panel states its polarity in the title.
    Without that, half the panels would silently mean the opposite of the
    other half.
    """
    scen_color = scenario_palette(scenarios)
    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 3.6 * nrows),
                             squeeze=False)
    all_rows = []
    for idx, metric in enumerate(metrics):
        ax = axes[idx // ncols][idx % ncols]
        models_present, ranks, tied, rows = rank_and_test(data, scenarios, metric)
        all_rows.extend(rows)
        draw_bump(ax, models_present, ranks, tied, scenarios, scen_color,
                  label_right=False, tick_fontsize=7)
        n_tied = sum(1 for r in rows if not r["distinguishable"])
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)}\n"
                     f"{n_tied} of {len(rows)} rungs unresolved", fontsize=9.5)
        ax.set_ylabel("rank", fontsize=8)
    for idx in range(len(metrics), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    handles = [plt.Line2D([], [], color=scen_color[sc], lw=2.5, marker="o", ms=6,
                          label=SCENARIO_LABELS.get(sc, sc).replace("\n", " "))
               for sc in scenarios]
    handles += [plt.Line2D([], [], color="#555", lw=0, marker="o", ms=7,
                           markerfacecolor="white", markeredgecolor="#555",
                           label="hollow = not distinguishable from next rank")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=False)
    fig.suptitle(f"Scenario ordering per model — the other six metrics\n"
                 f"rank #1 = most segregated, #{len(scenarios)} = least;  "
                 f"crossing = models disagree;\n"
                 f"paired normality-gated test (t / Wilcoxon), Holm-corrected",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0.075, 1, 0.93))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out_path}")
    return all_rows


def rankings(data, scenarios, csv_path):
    """Per-model DI ranking + pairwise Kendall tau on the scenario ordering."""
    rows, order = [], {}
    for key, label, _ in MODELS:
        if key not in data:
            continue
        df = data[key]
        means = {sc: mean_ci(df[df.scenario == sc]["dissimilarity_index"])[0]
                 for sc in scenarios}
        ranked = sorted(means, key=lambda s: -means[s])
        order[key] = ranked
        for rank, sc in enumerate(ranked, 1):
            rows.append({"model": label, "scenario": sc, "rank": rank,
                         "mean_dissimilarity": round(means[sc], 4)})
    tab = pd.DataFrame(rows)
    tab.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    print("\n=== DI ranking per model (most -> least segregating) ===")
    for key, label, _ in MODELS:
        if key in order:
            print(f"  {label:22s} " + " > ".join(
                s.replace("llm_baseline", "baseline") for s in order[key]))

    print("\n=== ordering agreement (Kendall tau over scenario ranks) ===")
    taus = []
    for a, b in combinations([k for k, _, _ in MODELS if k in order], 2):
        ra = {s: i for i, s in enumerate(order[a])}
        rb = {s: i for i, s in enumerate(order[b])}
        conc = disc = 0
        for s1, s2 in combinations(scenarios, 2):
            same = (ra[s1] - ra[s2]) * (rb[s1] - rb[s2])
            conc += same > 0
            disc += same < 0
        tau = (conc - disc) / (conc + disc) if (conc + disc) else np.nan
        taus.append(tau)
        print(f"  {a:22s} vs {b:22s} tau = {tau:+.2f}")
    if taus:
        print(f"\n  mean pairwise tau = {np.mean(taus):+.2f} "
              f"(1.0 = every model ranks the six contexts identically)")
    return tab


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-root", default=str(REPO / "experiments_with_llama_cpp"))
    ap.add_argument("--out-dir",
                    default=str(REPO / "prompt_refinement" / "results" / "figures"))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    print("loading model runs:")
    data = load_models(args.run_root)
    present = set().union(*[set(df.scenario.unique()) for df in data.values()])
    scenarios = [s for s in SCENARIO_ORDER if s in present]
    extra = sorted(present - set(SCENARIO_ORDER) - MECH_SCENARIOS)
    scenarios += extra
    print(f"scenarios: {scenarios}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fig_metrics(data, scenarios, out / "cross_model_metrics.png", args.dpi)
    rows = fig_dissimilarity(data, scenarios,
                             out / "cross_model_dissimilarity.png", args.dpi)
    others = [m for m in METRICS if m != "dissimilarity_index"]
    rows += fig_metric_rankings(data, scenarios, others,
                                out / "cross_model_metric_rankings.png", args.dpi)

    tests = pd.DataFrame(rows)
    tests_path = out / "cross_model_pairwise_tests.csv"
    tests.to_csv(tests_path, index=False)
    print(f"wrote {tests_path}")
    print("\n=== adjacent-rank comparisons, all metrics ===")
    print(f"  tests used: {tests['test'].value_counts().to_dict()}")
    print(f"  differences normal (Shapiro p >= 0.05): "
          f"{int((tests['shapiro_p_differences'] >= 0.05).sum())} of "
          f"{int(tests['shapiro_p_differences'].notna().sum())}")
    per_metric = (tests.groupby('metric')['distinguishable']
                  .agg(['sum', 'count']))
    for m, r in per_metric.iterrows():
        print(f"  {m:22s} {int(r['sum'])}/{int(r['count'])} rungs resolved")

    rankings(data, scenarios, out / "cross_model_rankings.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
