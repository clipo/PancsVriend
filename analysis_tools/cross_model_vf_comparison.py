#!/usr/bin/env python3
"""Cross-model comparison of value-function-driven Schelling simulations.

Compares the nine campaign models on the SAME pipeline (R3_dual_count value
functions, composition-level lookup, 6 scenarios each; the run count is read from the data):

    python analysis_tools/cross_model_vf_comparison.py                 # exact logprob tables (default)
    python analysis_tools/cross_model_vf_comparison.py --family r3     # sampled tables (superseded)
    python analysis_tools/cross_model_vf_comparison.py --run-root <dir>

Input: the newest FULL analysis/run_summary_by_run.csv per model under
<run-root>/run_*_<model>-vf-<family>/ (final-step value per run; pick_run).
The two table FAMILIES are never mixed in one figure. `lp` (DEFAULT, the
result): the exact token-probability tables (llm_model suffix -vf-lp,
prompt_refinement/logprob_value_function.py). `r3`: the sampled R3_dual_count
tables (suffix -vf-r3) — SUPERSEDED 2026-09-06: sampled at concurrency > 1
they carry the llama-server batch-numerics artifact, so the user does not
consider them a meaningful output or a ground truth; they are kept only for
the artifact write-up, written under cross_model_sampled_* with a title that
says so, and the orchestrator's cross_model stage does not regenerate them.

Outputs (prompt_refinement/results/figures/ by default):
  cross_model_level_all_metrics.png  7 level-chart panels (one per metric), same grid
  cross_model_bump_all_metrics.png  7 bump-chart panels (one per metric): scenario
                                ordering per model (was absolute levels until 2026-09-05);
                                one series per model, 95% CI whiskers.
  cross_model_bump_<metric>.png  BUMP CHART of the scenario ORDERING per model, one per
                                metric ('dissimilarity' for DI, else the metric key)
  cross_model_level_<metric>.png  the LEVEL chart: same axes and markers with y = the
                                mean value itself (no gap bars), showing how large the gaps are
  (cross_model_bump_dissimilarity.png) BUMP CHART of the scenario ORDERING per
                                model (rank 1 = most segregating). Colour
                                encodes SCENARIO here — absolute DI levels
                                (colour = model) are the first panel of
                                cross_model_bump_all_metrics.png, so the two figures
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
                                paired t-test on the per-run differences
                                (2026-09-05: no normality gate; see
                                rank_and_test). Holm-corrected across
                                each model's adjacent pairs. Hollow = that
                                rung of the ladder is not resolved — OR
                                (2026-09-05, user decisions) the DI gap is
                                below the practical-significance floor
                                (0.01, vf_rank_stability --gap-floor), OR
                                the scenario sits at the random-allocation
                                DI for the board (no segregation; marked
                                the same as a tie). Both rules are read
                                from each model's newest
                                analysis/rank_stability/rank_status.json
                                so this figure and the notes agree.
  cross_model_metric_rankings.png  the SAME ordering test applied to the
                                other six metrics, one bump-chart subplot
                                each. Rank 1 = MOST SEGREGATING in every
                                panel: clusters and switch_rate are inverted
                                (higher value = less segregation) via
                                SEGREGATION_DIRECTION, so all panels read the
                                same way.
  cross_model_pairwise_tests.csv  every adjacent comparison, ALL metrics: the
                                mean gap, both scenarios' SDs, the pooled SD
                                and Cohen's d (what the gap bars are binned
                                on), the test used, raw and Holm-adjusted p,
                                and the verdict.
  cross_model_rankings.csv      per model, scenarios ranked by mean DI, with
                                Kendall tau vs every other model (does the
                                social-context ordering agree across models?)
"""
import argparse
import glob
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_THIS = Path(__file__).resolve().parent
REPO = _THIS.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis_tools.normality_tests import holm_correction  # noqa: E402

# (run-folder key, display label, colour). Nine models since 2026-09-05; the
# first five keep their original Dark2 colours so older figures stay readable.
MODELS = [
    ("gemma-4-31b", "gemma-4-31B", "#1b9e77"),
    ("qwen3.6-27b", "qwen3.6-27B", "#7570b3"),
    ("llama-3.3-70b", "llama-3.3-70B", "#d95f02"),
    ("hermes-4.3-36b", "hermes-4.3-36B", "#e6ab02"),
    ("deepseek-v4-flash", "deepseek-v4-flash", "#66a61e"),
    ("mistral-small-4-119b", "mistral-small-4-119B", "#e7298a"),
    ("granite-4.2-30b", "granite-4.2-30B", "#1f78b4"),
    ("phi-4-14b", "phi-4-14B", "#666666"),
    ("olmo-2-32b", "olmo-2-32B", "#a6761d"),
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

def _canon(sc):
    """The roll-up's scenario_key names the control 'llm_baseline'; the
    rank-stability status (from run_summary.csv) names it 'baseline'."""
    return sc[4:] if sc.startswith("llm_") else sc


# Table family -> (llm_model suffix, output-name infix, title note)
FAMILIES = {
    "lp": ("vf-lp", "", "exact logprob tables"),
    "r3": ("vf-r3", "sampled_",
           "SAMPLED tables — superseded (batch-numerics artifact), not a result"),
}
DEFAULT_FAMILY = "lp"
FAMILY_NOTE = FAMILIES[DEFAULT_FAMILY][2]      # set by main(); read by the suptitles


def _row_count(csv_path) -> int:
    """Data rows of a CSV without parsing it (header excluded)."""
    with open(csv_path, "rb") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def pick_run(hits):
    """The newest of a model's LARGEST runs.

    The orchestrator's cross_model stage (2026-09-06) runs after every
    pipeline, including smoke tests (2 runs x 2 steps under the same model
    slug), so "newest run" alone would let a smoke test displace the 10k
    production run in the comparison. Runs are ranked by row count first,
    timestamp second; a rerun of the same size supersedes, a smaller one
    never does.
    """
    hits = sorted(hits)
    if not hits:
        return None, []
    sizes = {h: _row_count(h) for h in hits}
    best = max(sizes.values())
    chosen = [h for h in hits if sizes[h] == best][-1]
    skipped = [(h, sizes[h]) for h in hits if h != chosen]
    return chosen, skipped


def load_models(run_root, family=DEFAULT_FAMILY):
    """{model_key: DataFrame} from the newest full run per model (pick_run)
    of one table family (FAMILIES)."""
    suffix = FAMILIES[family][0]
    out, missing = {}, []
    for key, _, _ in MODELS:
        hits = glob.glob(str(Path(run_root) / f"run_*_{key}-{suffix}" / "analysis" /
                             "run_summary_by_run.csv"))
        chosen, skipped = pick_run(hits)
        if chosen is None:
            missing.append(key)
            continue
        df = pd.read_csv(chosen)
        if "scenario_key" in df.columns:          # analysis key, not config.json's
            df = df.drop(columns=["scenario"]).rename(columns={"scenario_key": "scenario"})
        df["_source"] = chosen
        out[key] = df
        has_initial = any(c.startswith("initial_") for c in df.columns)
        print(f"  {key:22s} {len(df):5d} rows  <- {Path(chosen).parents[1].name}"
              + ("" if has_initial else "  (no initial_* columns: chance tests unavailable; rebuild run_summary)"))
        for h, n in skipped:
            print(f"  {'':22s} {n:5d} rows     skipped {Path(h).parents[1].name}")
    if missing:
        print(f"  WARNING: no {suffix} run found for {missing}")
    if not out:
        sys.exit(f"no {suffix} model runs found under {run_root}")
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
    """One bump-chart subplot per metric — the scenario ORDERING each model
    produces, with the same paired-t, Holm-corrected test as
    the dissimilarity figure. Until 2026-09-05 this figure plotted absolute
    final-step means with 95% CIs per model; the levels are in
    cross_model_rankings.csv, and the ordering is the comparison that
    matters, so the figure now shows that (and the separate six-metric
    ranking figure it duplicated is gone).

    Rank 1 = MOST SEGREGATING in every panel. Two metrics run the other way
    (clusters and switch_rate: a higher value means a more broken-up, more
    interleaved grid), so their rankings are inverted via
    SEGREGATION_DIRECTION and each panel states its polarity in the title.
    Without that, half the panels would silently mean the opposite of the
    other half. A model whose scenarios all sit at the metric's chance level
    gets a blank column (see draw_bump).
    """
    scen_color = scenario_palette(scenarios)
    ncols = 2
    nrows = int(np.ceil(len(METRICS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 3.6 * nrows),
                             squeeze=False)
    all_rows, all_chance = [], []
    for idx, metric in enumerate(METRICS):
        ax = axes[idx // ncols][idx % ncols]
        models_present, ranks, tied, rows = rank_and_test(data, scenarios, metric)
        all_rows.extend(rows)
        at_chance, chance_rows = chance_test(data, scenarios, metric)
        all_chance.extend(chance_rows)
        blank = {k for k, _ in models_present if all_at_chance(at_chance, k, scenarios)}
        draw_bump(ax, models_present, ranks, tied, scenarios, scen_color,
                  label_right=False, tick_fontsize=7, blank=blank, at_chance=at_chance,
                  gaps=gap_classes(rows, metric))
        n_tied = sum(1 for r in rows if not r["distinguishable"])
        polarity = ("higher = less segregated, ranking inverted"
                    if SEGREGATION_DIRECTION.get(metric, +1) < 0 else "higher = more segregated")
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)}  ({polarity})\n"
                     f"{n_tied} of {len(rows)} rungs unresolved", fontsize=9.5)
        ax.set_ylabel("rank", fontsize=8)
    for idx in range(len(METRICS), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    handles = [plt.Line2D([], [], color=scen_color[sc], lw=2.5, marker="o", ms=6,
                          label=SCENARIO_LABELS.get(sc, sc).replace("\n", " "))
               for sc in scenarios]
    handles += [plt.Line2D([], [], color="#555", lw=0, marker="o", ms=7,
                           markerfacecolor="white", markeredgecolor="#555",
                           label="hollow = not distinguishable from the next rank (paired t, Holm)"),
                plt.Line2D([], [], color="#555", lw=0, marker="$\\circ$", ms=7,
                           markerfacecolor="white", markeredgecolor="#555",
                           label="dotted centre (°) = not above its own random initial grids (no segregation)")]
    handles += [plt.Line2D([], [], color="#333333",
                           label=f"bar: gap Cohen's d {GAP_LABELS[cls]} ({GAP_NAMES[cls]})",
                           **GAP_STYLES[cls]) for cls in range(4)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=False)
    n_runs = runs_per_group(data, scenarios)
    fig.suptitle(f"Scenario ordering per model, all seven metrics — {FAMILY_NOTE}, final step, "
                 f"{n_runs:,} runs per (model, scenario)\n"
                 f"rank #1 = most segregated, #{len(scenarios)} = least;  "
                 f"{B('crossing')} = models disagree;  {B('blank column')} = no scenario above chance;\n"
                 f"paired t on per-run differences (runs share seeds), Holm-corrected — "
                 f"{B('solid')} = statistically distinguishable, not necessarily large;\n"
                 f"{B('hollow rungs are always dotted')} (a gap that cannot be signed at this n is negligible); "
                 f"a solid dotted rung is a signed but trivial gap",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0.075, 1, 0.92))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return all_rows, all_chance


def _paired_t(a, b):
    """(test name, p) for the paired t on a - b; degenerate when all equal."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2 or np.allclose(d, 0.0):
        return "degenerate (all differences zero)", 1.0
    return "paired t", float(stats.ttest_rel(a, b, nan_policy="omit").pvalue)


def chance_test(data, scenarios, metric):
    """Per model and scenario: is the final value MORE segregated than the
    run's own initial (random-allocation) grid? Paired t on final - initial
    across runs, Holm-corrected within the model over its scenarios.

    Chance is thereby a scenario like any other, tested exactly the way the
    rungs are (user decision 2026-09-05; no practical floor). A scenario is
    "at chance" when the test does not show it above its initial grids —
    which at n = 10,000 means a mean excess of about one agent's move or
    less. Returns ({model: set of at-chance scenarios}, rows).
    """
    at_chance, rows = {}, []
    direction = SEGREGATION_DIRECTION.get(metric, +1)
    init_col = f"initial_{metric}"
    for key, label in [(k, lab) for k, lab, _ in MODELS if k in data]:
        df = data[key]
        at_chance[key] = set()
        if init_col not in df.columns:
            continue
        recs, pvals = [], []
        for sc in scenarios:
            g = df[df.scenario == sc]
            a, b = g[metric].to_numpy(dtype=float), g[init_col].to_numpy(dtype=float)
            ok = ~np.isnan(a) & ~np.isnan(b)
            a, b = a[ok], b[ok]
            test, p = _paired_t(a, b)
            recs.append({"metric": metric, "model": key, "scenario": sc,
                         "segregation_direction": direction, "n_pairs": int(len(a)),
                         "mean_final": float(a.mean()) if len(a) else np.nan,
                         "mean_initial": float(b.mean()) if len(b) else np.nan,
                         "excess": float(direction * (a - b).mean()) if len(a) else np.nan,
                         "test": test, "p_raw": p})
            pvals.append(p)
        adj = holm_correction(np.array(pvals)) if pvals else np.array([])
        for rec, pa in zip(recs, adj):
            rec["p_holm"] = float(pa)
            rec["above_chance"] = bool(pa < 0.05 and rec["excess"] > 0)
            if not rec["above_chance"]:
                at_chance[key].add(rec["scenario"])
            rows.append(rec)
    return at_chance, rows


def rank_and_test(data, scenarios, metric):
    """Per model: scenario ranks by SEGREGATION on `metric` (rank 1 = most
    segregating, using SEGREGATION_DIRECTION so inverted metrics like
    clusters and switch_rate are reversed), plus whether each rank is
    statistically distinguishable from the next one down.

    The comparison is PAIRED — run k uses the same seed, hence the same
    initial grid, in every scenario, so testing the per-run differences
    removes the shared initial-configuration variance. The test is a paired
    t-test on the mean difference, Holm-corrected across each model's
    adjacent comparisons. It used to be gated on Shapiro-Wilk of the
    differences (pass -> paired t, fail -> Wilcoxon); at n = 10,000 the gate
    rejected on 282 of 301 rungs for deviations the t-test does not care
    about (its statistic is normal by the CLT at this n) and every verdict
    was the same under either test, so the gate was dropped (2026-09-05).
    The practical gap floor that also used to mark rungs hollow was dropped
    the same day: a rung is resolved iff the paired test resolves it. Solid
    therefore means "statistically distinguishable", not "large" — the
    levels are in cross_model_rankings.csv and the per-model violins.

    Each row also carries the gap's size as Cohen's d — the mean gap over
    the pooled SD of the two scenarios' final-step values (see pooled_sd) —
    which is what the bump charts' gap bars are binned on (gap_class).

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
            gap = float(a.loc[common].mean() - b.loc[common].mean())
            pooled = pooled_sd(a.loc[common], b.loc[common])
            test, pv = _paired_t(a.loc[common], b.loc[common])
            rows.append({"metric": metric, "model": key, "rank": r + 1,
                         "segregation_direction": direction,
                         "scenario": sc, "next_scenario": nxt,
                         "mean_gap": gap, "n_pairs": int(len(common)),
                         "sd_scenario": float(a.loc[common].std(ddof=1)),
                         "sd_next": float(b.loc[common].std(ddof=1)),
                         "pooled_sd": pooled,
                         "cohen_d": gap / pooled if pooled > 0 else 0.0,
                         "test": test, "p_raw": pv})
            pvals.append(pv); pairs.append(sc)

        adj = holm_correction(np.array(pvals)) if pvals else np.array([])
        for k_i, (sc, pa) in enumerate(zip(pairs, adj)):
            rec = rows[-len(pairs) + k_i]
            rec["p_holm"] = float(pa)
            rec["distinguishable"] = bool(pa < 0.05)
            tied[(key, sc)] = not rec["distinguishable"]
        tied[(key, ordered[-1])] = False          # nothing below the last rank
    return models_present, ranks, tied, rows


DEFAULT_BOARD = {"grid_size": 20, "num_type_a": 160, "num_type_b": 160}


def chance_stats(metric, board=None):
    """(mean, sd) of the metric under random allocation: DI from
    DissimilarityIndex.random_baseline (the published chance DI), the others
    from the cached 20k-draw metric_null."""
    board = board or DEFAULT_BOARD
    if metric == "dissimilarity_index":
        from DissimilarityIndex import random_baseline
        rb = random_baseline(board["grid_size"], board["num_type_a"], board["num_type_b"])
        return float(rb["mean"]), float(rb["sd"])
    try:
        from vf_rank_stability import metric_null
    except ImportError:
        from analysis_tools.vf_rank_stability import metric_null
    m = metric_null(board)["metrics"][metric]
    return float(m["mean"]), float(m["sd"])


def figure_stem(metric):
    """cross_model_<stem>: 'dissimilarity' for DI (the historical name), else the metric key."""
    return "dissimilarity" if metric == "dissimilarity_index" else metric


def gap_classes(rows, metric):
    """{(model, scenario): gap class} from rank_and_test rows, binned on
    the pair's Cohen's d (gap_class)."""
    return {(r["model"], r["scenario"]): gap_class(r["cohen_d"])
            for r in rows if r["metric"] == metric}


def all_at_chance(at_chance, key, scenarios):
    """True when every scenario of this model is at chance: no ordering to show."""
    return bool(scenarios) and at_chance.get(key, set()) >= set(scenarios)


def series_label(ax, text, xy, color, fontsize=8.5):
    """Right-hand series label in a darkened shade of the series colour —
    no outline or shadow (both blur the glyphs); the darker ink keeps the
    light colours (yellow, grey) legible on white while the hue still
    matches the line."""
    ax.annotate(text, xy, fontsize=fontsize, va="center", color=darken(color, 0.4),
                annotation_clip=False)


def darken(color, amount):
    """Blend an RGB(A) colour toward black by `amount` (0 = unchanged, 1 = black)."""
    r, g, b = mcolors.to_rgb(color)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))


def B(term):
    """Bold a caption term with mathtext (the only bold available inside a
    single title string); spaces and symbols mathtext rejects are escaped."""
    degree = term.endswith(" (°)")               # mathtext has no °: keep it outside
    if degree:
        term = term[:-4]
    out = f"$\\bf{{{term.replace(' ', chr(92) + ' ')}}}$"
    return out + " (°)" if degree else out


# Room to the right of the last model column for the series labels, in
# column widths; the single-metric figures are sized so the text fits
# inside the axes box (fig_ordering / fig_levels).
LABEL_MARGIN = 2.3

AT_CHANCE_FILL = "#777777"     # the centre dot of an at-chance marker (white face + this dot = °)
AT_CHANCE_DOT = 14


# Gap between successive ranks as Cohen's d — the mean gap divided by the
# pooled SD of the two scenarios' final-step values — so the bars mean the
# same thing on every metric and for every model: how far apart the two
# scenario distributions are, in units of their own spread. The connector
# drawn between two markers in a column gets thicker with |d|; below the
# first bin it is a faint dotted line (2026-09-07; nothing at all before,
# which left the near-chance models' columns looking unmeasured rather than
# negligible — "no mark" now means only "nothing to compare": the last rank
# and blank columns). Bins are Cohen's conventional small / medium / large
# (0.2 / 0.5 / 0.8).
# p-values cannot carry this gradation at n = 10,000 (38 of 45 DI rungs
# have Holm p < 0.001); the effect size can.
#
# Until 2026-09-07 the ruler was the metric's chance SD (the spread of
# random initial grids) with bins 0.15 / 0.6 / 1.5 back-derived from DI
# thresholds of 0.005 / 0.02 / 0.05 and applied to every metric's own null
# SD. That denominator is the same for every model, so it overstated the
# separation two- to three-fold for the segregating models (whose finals
# spread far wider than random grids) and silently transplanted a
# DI-based judgement onto six unrelated measures.
GAP_BINS_D = (0.2, 0.5, 0.8)
GAP_STYLES = [dict(lw=0.8, alpha=0.35, ls=(0, (1.5, 2.5))), dict(lw=1.0, alpha=0.5),
              dict(lw=2.4, alpha=0.7), dict(lw=4.2, alpha=0.9)]
GAP_LABELS = ["< 0.2", "0.2–0.5", "0.5–0.8", "≥ 0.8"]
GAP_NAMES = ["negligible", "small", "medium", "large"]


def pooled_sd(a, b):
    """Root-mean-square of the two groups' sample SDs — Cohen's pooled SD
    for equal-sized groups (paired here, so n is the same by construction)."""
    return float(np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0))


def gap_class(cohen_d):
    """0..3: which GAP_BINS_D band |d| falls in (0 = dotted connector)."""
    if not np.isfinite(cohen_d):
        return 0
    return int(np.searchsorted(GAP_BINS_D, abs(cohen_d), side="right"))


def draw_bump(ax, models_present, ranks, tied, scenarios, scen_color,
              label_right=True, tick_fontsize=8, blank=(), at_chance=None, gaps=None):
    """Bump chart: one line per scenario, y = its rank under each model.

    `gaps` = {(model, scenario): gap class 0..3} for the rung below that
    scenario; a grey vertical connector of that class is drawn between the
    two markers (dotted for class 0, thicker as the class rises).

    Markers carry two separate facts. HOLLOW = the scenario is not
    distinguishable from the next rank down (paired test, floor); the last
    rank has nothing below it and is therefore never hollow. CENTRE DOT (°) =
    the scenario sits at the metric's chance level (the ° of the
    rank-stability tables) — it does not segregate, so its rank is not a
    finding, wherever it lands. Until 2026-09-05 both were drawn hollow,
    which made the last rank look inconsistently marked.

    The model order is the fixed MODELS order in every chart. Models in
    `blank` (every scenario at chance) keep their column, empty, with
    "(no segregation)" in the tick label; each scenario's line crosses the
    empty column at low opacity, straight from the last ranked model to the
    next, so the rank change across it can still be read.
    """
    blank = set(blank)
    at_chance = at_chance or {}
    xm = np.arange(len(models_present))
    shown_x = [xi for xi, (k, _) in enumerate(models_present) if k not in blank]
    x_label = len(models_present) - 1 + 0.12
    draw_index = 0
    for sc in scenarios:
        ys = np.array([np.nan if k in blank else ranks[k][sc] for k, _ in models_present], dtype=float)
        for x0, x1 in zip(shown_x, shown_x[1:]):
            spans_blank = x1 - x0 > 1
            ax.plot([x0, x1], [ys[x0], ys[x1]], lw=2.0, color=scen_color[sc],
                    alpha=0.25 if spans_blank else 0.85, zorder=1 if spans_blank else 2)
        for xi, (k, _) in enumerate(models_present):
            if k in blank:
                continue
            chance_here = sc in at_chance.get(k, set())
            face = "white" if (chance_here or tied[(k, sc)]) else scen_color[sc]
            # Same marker as the levels chart: white halo directly beneath
            # each disc (in drawing order), coloured edge, optional ° dot.
            z = 3 + 0.01 * draw_index
            ax.scatter([xi], [ranks[k][sc]], s=125, color="white", zorder=z, linewidths=0)
            ax.scatter([xi], [ranks[k][sc]], s=70, zorder=z + 0.004, color=face,
                       edgecolors=scen_color[sc], linewidths=1.8)
            if chance_here:                     # the ° of the rank-stability tables
                ax.scatter([xi], [ranks[k][sc]], s=AT_CHANCE_DOT, zorder=z + 0.008,
                           color=AT_CHANCE_FILL, linewidths=0)
            draw_index += 1
        if label_right and shown_x:
            y_last = ys[shown_x[-1]]
            if shown_x[-1] < len(models_present) - 1:          # blank column(s) at the end
                ax.plot([shown_x[-1], x_label], [y_last, y_last], ls=":", lw=1.2,
                        color=scen_color[sc], alpha=0.45, zorder=1)
            series_label(ax, SCENARIO_LABELS.get(sc, sc).replace("\n", " "),
                         (x_label, y_last), scen_color[sc])
    if gaps:
        for xi, (k, _) in enumerate(models_present):
            if k in blank:
                continue
            for sc in scenarios:
                cls = gaps.get((k, sc), 0)
                if ranks[k][sc] < len(scenarios) - 1:
                    y0 = ranks[k][sc]
                    ax.plot([xi, xi], [y0 + 0.16, y0 + 0.84], color="#333333", solid_capstyle="butt",
                            zorder=1.5, **GAP_STYLES[cls])
    ax.set_xticks(xm)
    ax.set_xticklabels([lab if k not in blank else f"{lab}\n(no segregation)"
                        for k, lab in models_present],
                       fontsize=tick_fontsize, rotation=20, ha="right")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([f"#{r+1}" for r in range(len(scenarios))],
                       fontsize=tick_fontsize)
    ax.set_ylim(len(scenarios) - 0.4, -0.6)            # rank 1 on top
    ax.set_xlim(-0.4, len(models_present) - 1 + (LABEL_MARGIN if label_right else 0.4))
    ax.grid(axis="y", alpha=0.25)


def scenario_palette(scenarios):
    """The house scenario colours (experiment_list_for_analysis.SCENARIO_COLORS,
    the palette every per-campaign figure uses), so a scenario is the same
    colour here as in the violins; tab10 only for a key missing there."""
    try:
        from experiment_list_for_analysis import SCENARIO_COLORS
    except ImportError:
        SCENARIO_COLORS = {}
    cmap = plt.get_cmap("tab10")
    return {sc: SCENARIO_COLORS.get(sc, cmap(k % 10)) for k, sc in enumerate(scenarios)}


def _spread_labels(ys, min_gap):
    """Nudge label y-positions apart so none are closer than min_gap."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for i in range(1, len(order)):
        lo, hi = order[i - 1], order[i]
        if out[hi] - out[lo] < min_gap:
            out[hi] = out[lo] + min_gap
    return out


def fig_levels(data, scenarios, metric, out_path, dpi, dodge_width=0.0):
    """Companion to the bump chart (the LEVELS chart): same axes and markers,
    y = the mean final-step value itself instead of its rank, so the size of every gap is
    read directly (no gap bars). Whiskers are 95% CIs of the mean — with
    10,000 runs they are ±0.001 and vanish inside the markers, which is the
    point. Marker semantics as in draw_bump: hollow = not distinguishable
    from the next rank, centre dot (°) = not above its own initial grids; blank
    column = no scenario above chance. The chance level is a tick on the y
    axis (no line across the data). One continuous axis (a broken axis for
    olmo's high values was tried and rejected, 2026-09-05).

    Markers of one model are dodged horizontally in that model's value order
    (lowest leftmost) and sit on a white halo, so scenarios with nearly equal
    means (the at-chance models: all six within 0.01) stay individually
    visible as a staircase; the offsets carry no information beyond that.
    """
    models_present, ranks, tied, _ = rank_and_test(data, scenarios, metric)
    at_chance, _ = chance_test(data, scenarios, metric)
    blank = {k for k, _ in models_present if all_at_chance(at_chance, k, scenarios)}
    scen_color = scenario_palette(scenarios)
    xm = np.arange(len(models_present))
    shown_x = [xi for xi, (k, _) in enumerate(models_present) if k not in blank]
    # Sideways offsets follow each model's own value order (lowest leftmost),
    # so wherever scenarios stack they form a monotone staircase instead of a
    # zigzag. A fixed per-scenario order was tried: the best of the 720 still
    # ran against the value order in 9 of the 57 stacked pairs (|gap| < 0.01
    # DI, mostly the at-chance models, where all six stack); this gives 0.
    means = {k: {sc: mean_ci(data[k][data[k].scenario == sc][metric])
                 for sc in scenarios} for k, _ in models_present}
    # dodge_width = 0 (default since 2026-09-06) stacks the markers on the
    # model's x; each marker's own white halo keeps overlapping discs apart.
    slots = np.linspace(-dodge_width, dodge_width, len(scenarios))
    dodge = {}
    for k, _ in models_present:
        for pos, sc in enumerate(sorted(scenarios, key=lambda sc: means[k][sc][0])):
            dodge[(k, sc)] = slots[pos]

    fig, ax = plt.subplots(1, 1, figsize=(11.0, 6.4))
    last_y = {}
    draw_index = 0
    for sc in scenarios:
        ys, es = [], []
        for k, _ in models_present:
            m, e = means[k][sc]
            ys.append(np.nan if k in blank else m); es.append(e)
        ys = np.array(ys, dtype=float)
        xs = np.array([xi + dodge[(k, sc)] for xi, (k, _) in enumerate(models_present)])
        for x0, x1 in zip(shown_x, shown_x[1:]):
            spans_blank = x1 - x0 > 1
            ax.plot([xs[x0], xs[x1]], [ys[x0], ys[x1]], lw=2.0, color=scen_color[sc],
                    alpha=0.25 if spans_blank else 0.85, zorder=1 if spans_blank else 2)
        for xi, (k, _) in enumerate(models_present):
            if k in blank:
                continue
            chance_here = sc in at_chance.get(k, set())
            face = "white" if (chance_here or tied[(k, sc)]) else scen_color[sc]
            ax.errorbar([xs[xi]], [ys[xi]], yerr=[es[xi]], fmt="none", ecolor=scen_color[sc],
                        elinewidth=1.0, capsize=2, zorder=2.5)
            # Each marker gets its own halo directly beneath it in drawing
            # order (not one shared level under all markers), so a disc that
            # overlaps another still shows a white rim against it.
            z = 3 + 0.01 * draw_index
            ax.scatter([xs[xi]], [ys[xi]], s=125, color="white", zorder=z, linewidths=0)  # halo
            ax.scatter([xs[xi]], [ys[xi]], s=70, zorder=z + 0.004, color=face,
                       edgecolors=scen_color[sc], linewidths=1.8)
            if chance_here:
                ax.scatter([xs[xi]], [ys[xi]], s=AT_CHANCE_DOT, zorder=z + 0.008, color=AT_CHANCE_FILL,
                           linewidths=0)
            draw_index += 1
        if shown_x:
            last_y[sc] = ys[shown_x[-1]]
    if last_y:
        labels = list(last_y)
        lo, hi = ax.get_ylim()
        placed = _spread_labels([last_y[sc] for sc in labels], 0.035 * (hi - lo))
        pad = 0.03 * (hi - lo)                    # keep spread labels inside the frame
        if max(placed) + pad > hi or min(placed) - pad < lo:
            ax.set_ylim(min(lo, min(placed) - pad), max(hi, max(placed) + pad))
        x_label = len(models_present) - 1 + 0.30
        for sc, y_txt in zip(labels, placed):
            series_label(ax, SCENARIO_LABELS.get(sc, sc).replace("\n", " "),
                         (x_label, y_txt), scen_color[sc])
    try:
        from plot_style import mark_chance_on_axis
    except ImportError:
        from analysis_tools.plot_style import mark_chance_on_axis
    mark_chance_on_axis(ax, chance_stats(metric)[0])
    ax.set_xticks(xm)
    ax.set_xticklabels([lab if k not in blank else f"{lab}\n(no segregation)"
                        for k, lab in models_present], fontsize=8, rotation=20, ha="right")
    ax.set_xlim(-0.4, len(models_present) - 1 + LABEL_MARGIN)
    ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)} (mean of final step)")
    ax.grid(axis="y", alpha=0.25)
    if SEGREGATION_DIRECTION.get(metric, +1) < 0:
        # clusters, switch rate: a lower value is MORE segregated, so the axis
        # is flipped to keep "up = more segregated" across all level charts.
        ax.invert_yaxis()
    ax.set_title(f"same markers as the ordering chart: {B('hollow')} = not distinguishable from the next rank "
                 f"(paired t, Holm);\n{B('centre dot (°)')} = not above its own random initial grids;  "
                 f"{B('whiskers')} = 95% CI of the mean (inside the markers at n = 10,000);\n"
                 f"overlapping markers are separated by their white rims", fontsize=9)
    n_runs = runs_per_group(data, scenarios)
    polarity = ("\n(y axis inverted: a lower value is MORE segregated, so up = more segregated as in the other charts)"
                if SEGREGATION_DIRECTION.get(metric, +1) < 0 else "")
    fig.suptitle(f"{METRIC_LABELS.get(metric, metric).capitalize()} by model and social context\n"
                 f"{FAMILY_NOTE}, final step, {n_runs:,} runs per (model, scenario){polarity}",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_dissimilarity_levels(data, scenarios, out_path, dpi, dodge_width=0.0):
    return fig_levels(data, scenarios, "dissimilarity_index", out_path, dpi, dodge_width)


def draw_levels(ax, data, models_present, scenarios, metric, tied, at_chance, blank, scen_color,
                tick_fontsize=8):
    """One level panel (the body of fig_levels) on a given axes."""
    xm = np.arange(len(models_present))
    shown_x = [xi for xi, (k, _) in enumerate(models_present) if k not in blank]
    draw_index = 0
    for sc in scenarios:
        ys = np.array([np.nan if k in blank else mean_ci(data[k][data[k].scenario == sc][metric])[0]
                       for k, _ in models_present], dtype=float)
        for x0, x1 in zip(shown_x, shown_x[1:]):
            spans_blank = x1 - x0 > 1
            ax.plot([x0, x1], [ys[x0], ys[x1]], lw=1.6, color=scen_color[sc],
                    alpha=0.25 if spans_blank else 0.85, zorder=1 if spans_blank else 2)
        for xi, (k, _) in enumerate(models_present):
            if k in blank:
                continue
            chance_here = sc in at_chance.get(k, set())
            face = "white" if (chance_here or tied[(k, sc)]) else scen_color[sc]
            z = 3 + 0.01 * draw_index
            ax.scatter([xi], [ys[xi]], s=90, color="white", zorder=z, linewidths=0)
            ax.scatter([xi], [ys[xi]], s=50, zorder=z + 0.004, color=face,
                       edgecolors=scen_color[sc], linewidths=1.5)
            if chance_here:
                ax.scatter([xi], [ys[xi]], s=AT_CHANCE_DOT * 0.7, zorder=z + 0.008,
                           color=AT_CHANCE_FILL, linewidths=0)
            draw_index += 1
    try:
        from plot_style import mark_chance_on_axis
    except ImportError:
        from analysis_tools.plot_style import mark_chance_on_axis
    mark_chance_on_axis(ax, chance_stats(metric)[0], fontsize=6.5)
    ax.set_xticks(xm)
    ax.set_xticklabels([lab if k not in blank else f"{lab}\n(no segr.)" for k, lab in models_present],
                       fontsize=tick_fontsize, rotation=20, ha="right")
    ax.set_xlim(-0.5, len(models_present) - 0.5)
    ax.grid(axis="y", alpha=0.25)
    if SEGREGATION_DIRECTION.get(metric, +1) < 0:
        ax.invert_yaxis()


def fig_levels_grid(data, scenarios, out_path, dpi):
    """The level-chart counterpart of fig_metrics: one level panel per metric
    in the same 2 x 4 grid, y inverted for clusters and switch rate so "up =
    more segregated" holds in every panel."""
    scen_color = scenario_palette(scenarios)
    ncols = 2
    nrows = int(np.ceil(len(METRICS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 3.6 * nrows), squeeze=False)
    for idx, metric in enumerate(METRICS):
        ax = axes[idx // ncols][idx % ncols]
        models_present, ranks, tied, _ = rank_and_test(data, scenarios, metric)
        at_chance, _ = chance_test(data, scenarios, metric)
        blank = {k for k, _ in models_present if all_at_chance(at_chance, k, scenarios)}
        draw_levels(ax, data, models_present, scenarios, metric, tied, at_chance, blank, scen_color,
                    tick_fontsize=7)
        inverted = SEGREGATION_DIRECTION.get(metric, +1) < 0
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)}"
                     + ("  (y inverted: lower = more segregated)" if inverted else ""), fontsize=9.5)
        ax.set_ylabel("mean of final step", fontsize=8)
    for idx in range(len(METRICS), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    handles = [plt.Line2D([], [], color=scen_color[sc], lw=2.5, marker="o", ms=6,
                          label=SCENARIO_LABELS.get(sc, sc).replace("\n", " "))
               for sc in scenarios]
    handles += [plt.Line2D([], [], color="#555", lw=0, marker="o", ms=7,
                           markerfacecolor="white", markeredgecolor="#555",
                           label="hollow = not distinguishable from the next rank (paired t, Holm)"),
                plt.Line2D([], [], color="#555", lw=0, marker="$\\circ$", ms=7,
                           markerfacecolor="white", markeredgecolor="#555",
                           label="centre dot (°) = not above its own random initial grids"),
                plt.Line2D([], [], color="#444", lw=0, marker=">", ms=5,
                           label="triangle on the y axis = chance level (random allocation)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    n_runs = runs_per_group(data, scenarios)
    fig.suptitle(f"Metric levels per model and social context, all seven metrics — {FAMILY_NOTE}, "
                 f"final step, {n_runs:,} runs per (model, scenario)\n"
                 f"up = more segregated in every panel;  whiskers (95% CI of the mean) are inside "
                 f"the markers at this n and omitted", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.075, 1, 0.965))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def runs_per_group(data, scenarios):
    """Median run count per (model, scenario), for the captions."""
    counts = [len(df[df.scenario == sc]) for df in data.values() for sc in scenarios]
    return int(np.median(counts)) if counts else 0


def fig_ordering(data, scenarios, metric, out_path, dpi):
    """Scenario ordering per model for one metric (bump chart).

    One of these per metric (2026-09-06; DI only before). Rank 1 = most
    segregating under SEGREGATION_DIRECTION, so clusters and switch rate,
    where a higher value means LESS segregation, are inverted and say so in
    the caption. Gap bars are binned on Cohen's d (pooled SD of the two
    scenarios' final values), so they read the same on every metric.
    """
    models_present, ranks, tied, rows = rank_and_test(data, scenarios, metric)
    at_chance, _ = chance_test(data, scenarios, metric)
    blank = {k for k, _ in models_present if all_at_chance(at_chance, k, scenarios)}
    fig, ax = plt.subplots(1, 1, figsize=(11.0, 6.0))
    draw_bump(ax, models_present, ranks, tied, scenarios,
              scenario_palette(scenarios), blank=blank, at_chance=at_chance,
              gaps=gap_classes(rows, metric))
    for cls in range(4):
        ax.plot([], [], color="#333333",
                label=f"gap: Cohen's d {GAP_LABELS[cls]} ({GAP_NAMES[cls]})",
                **GAP_STYLES[cls])
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.40), ncol=4, fontsize=7.5, frameon=False)
    ax.set_ylabel("rank")
    polarity = ("\n(a higher value of this metric means LESS segregation, so the ranking is inverted)"
                if SEGREGATION_DIRECTION.get(metric, +1) < 0 else "")
    ax.set_title(f"{B('crossing')} = models disagree on the ordering;  {B('hollow')} = not distinguishable "
                 f"from the next rank (paired t, Holm);\n{B('centre dot (°)')} = not above its own random "
                 f"initial grids (no segregation);  {B('empty column')} = no scenario above chance;\n"
                 f"{B('bar between two markers')} = size of the gap as Cohen's d, the mean gap over the pooled SD "
                 f"of the two scenarios' final values (dotted when |d| < {GAP_BINS_D[0]}, thicker = larger);\n"
                 f"{B('hollow rungs are always dotted')}: a gap that cannot be signed at this n is necessarily "
                 f"negligible; a solid dotted rung is a signed but trivial gap", fontsize=9)
    n_runs = runs_per_group(data, scenarios)
    fig.suptitle(f"Which social context segregates most? Scenario ordering by model\n"
                 f"{METRIC_LABELS.get(metric, metric)}, {FAMILY_NOTE}, final step, {n_runs:,} runs per (model, scenario);"
                 f"  rank #1 = most segregated, #{len(scenarios)} = least{polarity}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965 if polarity else 0.98))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return rows


def fig_dissimilarity(data, scenarios, out_path, dpi):
    return fig_ordering(data, scenarios, "dissimilarity_index", out_path, dpi)


def tied_groups(ordered, tied_flags, at_chance):
    """{scenario: group id} — adjacent rungs left unresolved merge into one
    group; scenarios at chance form one group at the bottom regardless."""
    grp, gid = {}, 0
    for i, sc in enumerate(ordered):
        grp[sc] = gid
        if i < len(ordered) - 1 and not tied_flags[sc]:
            gid += 1
    if at_chance:
        bottom = max(grp.values()) + 1
        for sc in at_chance:
            grp[sc] = bottom
    return grp


def rankings(data, scenarios, csv_path):
    """Per-model DI ranking WITH TIES + pairwise Kendall tau-b.

    Adjacent rungs the DI test leaves unresolved merge into one tied group, so a
    model's ordering is a sequence of groups, e.g. politics > {baseline,
    green, income} > race > ethnic. Agreement between models is Kendall's
    tau-b, which treats a pair tied in either model as neither concordant nor
    discordant instead of counting it as a disagreement (plain tau did, and
    penalised models for honestly reporting ties):
        tau_b = (C - D) / sqrt((n0 - n1)(n0 - n2)),
    n0 = all scenario pairs, n1 / n2 = pairs tied in model a / b.
    """
    models_present, ranks, tied, _ = rank_and_test(data, scenarios, "dissimilarity_index")
    chance_by_model, _ = chance_test(data, scenarios, "dissimilarity_index")
    rows, groups, chance_sets = [], {}, {}
    for key, label in models_present:
        df = data[key]
        ordered = sorted(scenarios, key=lambda sc: ranks[key][sc])
        means = {sc: mean_ci(df[df.scenario == sc]["dissimilarity_index"])[0]
                 for sc in scenarios}
        at_chance = set(chance_by_model.get(key, set()))
        chance_sets[key] = at_chance
        grp = tied_groups(ordered, {sc: tied[(key, sc)] for sc in ordered}, at_chance)
        groups[key] = grp
        for rank, sc in enumerate(ordered, 1):
            rows.append({"model": label, "scenario": sc, "rank": rank,
                         "group": grp[sc], "at_chance": sc in at_chance,
                         "mean_dissimilarity": round(means[sc], 4)})
    tab = pd.DataFrame(rows)
    tab.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    print("\n=== DI ranking per model (most -> least segregating; {..} = tied, ° = at chance) ===")
    for key, label in models_present:
        grp, at_chance = groups[key], chance_sets[key]
        ordered = sorted(scenarios, key=lambda sc: ranks[key][sc])
        chain = []
        for g in sorted(set(grp.values())):
            names = [s.replace("llm_baseline", "baseline") + ("°" if s in at_chance else "")
                     for s in ordered if grp[s] == g]
            chain.append(names[0] if len(names) == 1 else "{" + ", ".join(names) + "}")
        print(f"  {label:22s} " + " > ".join(chain))

    print("\n=== ordering agreement (Kendall tau-b over scenario groups; ties neutral) ===")
    taus = []
    n0 = len(scenarios) * (len(scenarios) - 1) / 2
    for (a, _), (b, _) in combinations(models_present, 2):
        ga, gb = groups[a], groups[b]
        conc = disc = ta = tb = 0
        for s1, s2 in combinations(scenarios, 2):
            da, db = ga[s1] - ga[s2], gb[s1] - gb[s2]
            ta += da == 0
            tb += db == 0
            if da and db:
                conc += (da * db) > 0
                disc += (da * db) < 0
        denom = np.sqrt((n0 - ta) * (n0 - tb))
        tau = (conc - disc) / denom if denom else np.nan
        taus.append(tau)
        print(f"  {a:22s} vs {b:22s} tau_b = {tau:+.2f}   (tied pairs: {ta}/{tb} of {int(n0)})")
    finite = [t for t in taus if np.isfinite(t)]
    if finite:
        print(f"\n  mean pairwise tau_b = {np.mean(finite):+.2f} over {len(finite)} model pairs "
              f"(1.0 = identical group orderings; a model with every scenario tied is undefined)")
    return tab


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-root", default=str(REPO / "experiments_with_llama_cpp"))
    ap.add_argument("--out-dir",
                    default=str(REPO / "prompt_refinement" / "results" / "figures"))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--family", choices=sorted(FAMILIES), default=DEFAULT_FAMILY,
                    help="table family to compare: lp = exact logprob tables (run_*-vf-lp, "
                         "the result, default); r3 = sampled tables (run_*-vf-r3, superseded, "
                         "written as cross_model_sampled_*); never mixed")
    args = ap.parse_args()
    global FAMILY_NOTE
    _, infix, FAMILY_NOTE = FAMILIES[args.family]
    P = f"cross_model_{infix}"                  # output-name prefix

    print(f"loading model runs ({FAMILY_NOTE}):")
    data = load_models(args.run_root, args.family)
    present = set().union(*[set(df.scenario.unique()) for df in data.values()])
    scenarios = [s for s in SCENARIO_ORDER if s in present]
    extra = sorted(present - set(SCENARIO_ORDER) - MECH_SCENARIOS)
    scenarios += extra
    print(f"scenarios: {scenarios}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows, chance_rows = fig_metrics(data, scenarios, out / f"{P}bump_all_metrics.png", args.dpi)
    fig_levels_grid(data, scenarios, out / f"{P}level_all_metrics.png", args.dpi)
    for metric in METRICS:                     # bump + level chart per metric
        stem = figure_stem(metric)
        fig_ordering(data, scenarios, metric, out / f"{P}bump_{stem}.png", args.dpi)
        fig_levels(data, scenarios, metric, out / f"{P}level_{stem}.png", args.dpi)
    # Names before 2026-09-06 (cross_model_metrics / cross_model_<metric>[_levels]),
    # and the transitional cross_model_lp_* of the morning of 2026-09-06, when
    # the exact tables were not yet the unprefixed default.
    if args.family == DEFAULT_FAMILY:
        stale = ["cross_model_metrics.png", "cross_model_metric_rankings.png"] + \
                [f"cross_model_{figure_stem(m)}{suffix}.png" for m in METRICS for suffix in ("", "_levels")]
        stale += [q.name for q in out.glob("cross_model_lp_*")]
        for old in stale:
            if (out / old).exists():
                (out / old).unlink()
    chance_path = out / f"{P}chance_tests.csv"
    pd.DataFrame(chance_rows).to_csv(chance_path, index=False)
    print(f"wrote {chance_path}")

    tests = pd.DataFrame(rows)
    tests_path = out / f"{P}pairwise_tests.csv"
    tests.to_csv(tests_path, index=False)
    print(f"wrote {tests_path}")
    print("\n=== adjacent-rank comparisons, all metrics ===")
    print(f"  tests used: {tests['test'].value_counts().to_dict()}")
    per_metric = (tests.groupby('metric')['distinguishable']
                  .agg(['sum', 'count']))
    for m, r in per_metric.iterrows():
        print(f"  {m:22s} {int(r['sum'])}/{int(r['count'])} rungs resolved")

    rankings(data, scenarios, out / f"{P}rankings.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
