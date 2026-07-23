"""Normality tests + plots per metric, and normality-gated significance tests
between social-context scenarios.

Why: the choice of between-scenario significance test (parametric ANOVA /
Welch t vs non-parametric Kruskal-Wallis / Mann-Whitney) must be justified by
whether each metric's per-run final values are normal within each scenario.
This module tests it (Shapiro-Wilk), PLOTS it (per-metric Q-Q + histogram
grids), and then runs the appropriate omnibus + pairwise tests with Holm
correction.

Pipeline mode (registered in run_all_scenario_analysis.py, after
combined_final_metrics):
    run_from_combined_csv()        # reads <reports>/combined_final_metrics.csv

Standalone mode against an (in-progress) llama.cpp production run:
    .venv/bin/python analysis_tools/normality_tests.py \
        --run-dir experiments_with_llama_cpp/run_<...>

Outputs (into the reports dir, or <run_dir>/plots in --run-dir mode):
    normality_tests.csv            metric x scenario Shapiro-Wilk table
    normality/normality_<metric>.png   per-scenario Q-Q plot + histogram w/ fit
                                   (in a dedicated normality/ subfolder)
    significance_tests.csv         omnibus test per metric (test chosen by
                                   normality) + Holm-corrected pairwise tests
"""
import argparse
import itertools
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from analysis_tools.experiment_list_for_analysis import SCENARIO_COLORS  # noqa: E402

ALPHA = 0.05
METRICS_DEFAULT = ["clusters", "switch_rate", "distance", "mix_deviation",
                   "share", "ghetto_rate"]


def _color(scenario, index):
    return (SCENARIO_COLORS.get(scenario)
            or SCENARIO_COLORS.get(f"llm_{scenario}")
            or plt.get_cmap("tab10")(index % 10))


# --------------------------------------------------------------------------
# Core statistics
# --------------------------------------------------------------------------

def shapiro_table(values_by_scenario, metric):
    """Shapiro-Wilk per scenario -> list of row dicts."""
    rows = []
    for scenario, vals in values_by_scenario.items():
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        row = {"metric": metric, "scenario": scenario, "n": len(vals)}
        if len(vals) >= 3 and np.ptp(vals) > 0:
            w, p = stats.shapiro(vals)
            row.update({"shapiro_W": w, "shapiro_p": p,
                        "normal_at_0.05": bool(p >= ALPHA)})
        else:
            row.update({"shapiro_W": np.nan, "shapiro_p": np.nan,
                        "normal_at_0.05": None})
        rows.append(row)
    return rows


def holm_correction(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, min(adj, 1.0))
        adjusted[idx] = running_max
    return adjusted


def significance_tests(values_by_scenario, metric, all_normal):
    """Omnibus + Holm-corrected pairwise tests; test family chosen by normality."""
    groups = {s: np.asarray(v, dtype=float)[~np.isnan(np.asarray(v, dtype=float))]
              for s, v in values_by_scenario.items()}
    groups = {s: v for s, v in groups.items() if len(v) >= 2}
    rows = []
    if len(groups) < 2:
        return rows
    names = list(groups)
    if all_normal:
        omni_name = "one-way ANOVA"
        stat, p = stats.f_oneway(*groups.values())
        pair_name = "Welch t-test"
        pair = lambda a, b: stats.ttest_ind(a, b, equal_var=False)
    else:
        omni_name = "Kruskal-Wallis"
        stat, p = stats.kruskal(*groups.values())
        pair_name = "Mann-Whitney U"
        pair = lambda a, b: stats.mannwhitneyu(a, b, alternative="two-sided")
    rows.append({"metric": metric, "comparison": "omnibus", "test": omni_name,
                 "statistic": stat, "p_raw": p, "p_holm": p,
                 "significant_at_0.05": bool(p < ALPHA)})
    pairs = list(itertools.combinations(names, 2))
    if pairs:
        raw = []
        for a, b in pairs:
            s, pv = pair(groups[a], groups[b])
            raw.append((a, b, s, pv))
        adj = holm_correction(np.array([r[3] for r in raw]))
        for (a, b, s, pv), ph in zip(raw, adj):
            rows.append({"metric": metric, "comparison": f"{a} vs {b}",
                         "test": pair_name, "statistic": s, "p_raw": pv,
                         "p_holm": ph, "significant_at_0.05": bool(ph < ALPHA)})
    return rows


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_metric_normality(metric, values_by_scenario, shapiro_rows, out_path):
    """One figure per metric: per-scenario Q-Q plot (top) + histogram with
    fitted normal (bottom), annotated with Shapiro-Wilk W and p."""
    scenarios = list(values_by_scenario)
    n = len(scenarios)
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n + 1, 6.4), squeeze=False)
    stats_by_scenario = {r["scenario"]: r for r in shapiro_rows}
    for col, scenario in enumerate(scenarios):
        vals = np.asarray(values_by_scenario[scenario], dtype=float)
        vals = vals[~np.isnan(vals)]
        color = _color(scenario, col)
        row = stats_by_scenario[scenario]
        verdict = ("normal" if row["normal_at_0.05"]
                   else "NOT normal" if row["normal_at_0.05"] is not None else "n/a")

        ax_qq = axes[0][col]
        if len(vals) >= 3:
            (osm, osr), (slope, intercept, _r) = stats.probplot(vals, dist="norm")
            ax_qq.scatter(osm, osr, s=18, color=color, alpha=0.85, zorder=3)
            ax_qq.plot(osm, slope * np.asarray(osm) + intercept,
                       color="black", lw=1.0, zorder=2)
        ax_qq.set_title(f"{scenario} (n={row['n']})", fontsize=9)
        ax_qq.set_xlabel("theoretical quantiles", fontsize=8)
        if col == 0:
            ax_qq.set_ylabel("sample quantiles", fontsize=8)
        ax_qq.grid(alpha=0.3)
        if np.isfinite(row.get("shapiro_p", np.nan)):
            ax_qq.text(0.04, 0.96,
                       f"W={row['shapiro_W']:.3f}\np={row['shapiro_p']:.3g}\n{verdict}",
                       transform=ax_qq.transAxes, va="top", fontsize=8,
                       bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.85))

        ax_h = axes[1][col]
        if len(vals):
            ax_h.hist(vals, bins="auto", density=True, color=color, alpha=0.5,
                      edgecolor=color)
            if np.std(vals) > 0:
                xs = np.linspace(vals.min(), vals.max(), 200)
                ax_h.plot(xs, stats.norm.pdf(xs, np.mean(vals), np.std(vals, ddof=1)),
                          color="black", lw=1.2, label="normal fit")
        ax_h.set_xlabel(metric, fontsize=8)
        if col == 0:
            ax_h.set_ylabel("density", fontsize=8)
        ax_h.grid(alpha=0.3)
    fig.suptitle(f"Normality — {metric} (final-step value per run; Shapiro-Wilk, "
                 f"alpha={ALPHA})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------

def analyze(values_by_metric, out_dir, csv_prefix=""):
    """values_by_metric: {metric: {scenario: 1D array of per-run final values}}."""
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "normality")
    os.makedirs(plots_dir, exist_ok=True)
    normality_rows, significance_rows = [], []
    for metric, values_by_scenario in values_by_metric.items():
        rows = shapiro_table(values_by_scenario, metric)
        normality_rows.extend(rows)
        plot_metric_normality(metric, values_by_scenario, rows,
                              os.path.join(plots_dir, f"normality_{metric}.png"))
        decided = [r["normal_at_0.05"] for r in rows if r["normal_at_0.05"] is not None]
        all_normal = bool(decided) and all(decided)
        significance_rows.extend(
            significance_tests(values_by_scenario, metric, all_normal))
    norm_csv = os.path.join(out_dir, f"{csv_prefix}normality_tests.csv")
    sig_csv = os.path.join(out_dir, f"{csv_prefix}significance_tests.csv")
    pd.DataFrame(normality_rows).to_csv(norm_csv, index=False)
    pd.DataFrame(significance_rows).to_csv(sig_csv, index=False)
    print(f"[normality] wrote {norm_csv}")
    print(f"[normality] wrote {sig_csv} "
          f"({sum(r['significant_at_0.05'] for r in significance_rows)} significant rows)")
    return pd.DataFrame(normality_rows), pd.DataFrame(significance_rows)


def run_from_combined_csv(csv_path=None, out_dir=None):
    """Pipeline mode: consume combined_final_metrics.csv from the reports dir."""
    from analysis_tools.output_paths import get_reports_dir
    reports = get_reports_dir()
    csv_path = csv_path or os.path.join(reports, "combined_final_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"[normality] SKIP: {csv_path} not found (run combined_final_metrics first)")
        return None, None
    df = pd.read_csv(csv_path)
    metrics = [m for m in METRICS_DEFAULT + ["dissimilarity_index"] if m in df.columns]
    values_by_metric = {
        m: {s: g[m].to_numpy() for s, g in df.groupby("scenario")}
        for m in metrics
    }
    return analyze(values_by_metric, out_dir or str(reports))


def run_from_run_dir(run_dir, out_dir=None):
    """Standalone mode: compute final-step values straight from a run dir
    (works mid-campaign), reusing the preview loaders."""
    from analysis_tools.plot_run_preview import (
        collect_experiment, compute_series, _final_values, METRIC_KEYS)
    import glob
    values_by_metric = {m: {} for m in METRIC_KEYS + ["move_rate"]}
    for exp_dir in sorted(glob.glob(os.path.join(run_dir, "experiments", "llm_*"))):
        scenario = os.path.basename(exp_dir).replace("llm_", "").rsplit("_", 2)[0]
        runs = collect_experiment(exp_dir)
        if not runs:
            print(f"[normality] skip {scenario}: no completed runs")
            continue
        series = compute_series(runs)
        for m in values_by_metric:
            values_by_metric[m][scenario] = _final_values(series[m])
    values_by_metric = {m: v for m, v in values_by_metric.items() if len(v) >= 1}
    return analyze(values_by_metric, out_dir or os.path.join(run_dir, "plots"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="run_<timestamp>_<model> dir (mid-campaign OK)")
    src.add_argument("--combined-csv", help="explicit combined_final_metrics.csv path")
    ap.add_argument("--out", default=None, help="output dir override")
    args = ap.parse_args()
    if args.run_dir:
        run_from_run_dir(args.run_dir, args.out)
    else:
        run_from_combined_csv(args.combined_csv, args.out)


if __name__ == "__main__":
    main()
