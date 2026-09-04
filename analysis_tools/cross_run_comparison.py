"""Compare finished production runs against each other: model x prompt set x endpoint.

Every run under experiments_with_llama_cpp/ already ships
``analysis/combined_final_metrics.csv`` (one row per simulation run, final-step
values, including dissimilarity_index and scenario). This script concatenates
those, labels each row with the variant it came from, and draws the
cross-variant comparisons. Nothing is recomputed from states/ or move_logs/, so
it is cheap and can be re-run any time a queue entry finishes.

    .venv/bin/python analysis_tools/cross_run_comparison.py
    .venv/bin/python analysis_tools/cross_run_comparison.py --out-dir /tmp/figs
    .venv/bin/python analysis_tools/cross_run_comparison.py --include-legacy

A "variant" is (model, prompt set, endpoint), parsed from the run directory
name, e.g. run_20260727_162043_Qwen3.6-27B-Q5_K_M-a2-chat
-> Qwen3.6-27B / a2 / chat. Runs whose name carries no -a<N> prompt-set tag are
pre-campaign ("legacy") and are skipped unless --include-legacy.

Outputs (into --out-dir, default reports/cross_run/):
    cross_run_dissimilarity_index.png  DI per scenario, one box+points per variant
    cross_run_final_boxplots.png       all 6 metrics + DI, pooled over scenarios
    cross_run_dissimilarity_index.csv  per variant x scenario: n/mean/median/sd
    cross_run_final_metrics.csv        per variant x metric:   n/mean/median/sd
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from analysis_tools.output_paths import get_reports_dir  # noqa: E402
from analysis_tools.plot_style import overlay_run_points  # noqa: E402
from analysis_tools.experiment_list_for_analysis import SCENARIO_COLORS  # noqa: E402

METRIC_KEYS = ["clusters", "switch_rate", "distance", "mix_deviation",
               "share", "ghetto_rate", "dissimilarity_index"]
METRIC_LABELS = {
    "clusters": "Number of Clusters",
    "switch_rate": "Switch Rate",
    "distance": "Average Distance",
    "mix_deviation": "Mix Deviation",
    "share": "Segregation Share",
    "ghetto_rate": "Ghetto Formation Rate",
    "dissimilarity_index": "Dissimilarity Index",
}

# Model families, in the order they should appear on every x axis.
MODEL_PATTERNS = [
    ("Llama-3.3-70B", "llama"),
    ("Gemma-4-31B", "gemma"),
    ("Qwen3.6-27B", "qwen"),
]
MODEL_BASE_COLOR = {
    "Llama-3.3-70B": "#1f77b4",
    "Gemma-4-31B": "#2ca02c",
    "Qwen3.6-27B": "#d62728",
}
# Tick labels have to fit 8+ groups across one panel, so the axis gets these
# short codes and the legend carries the full model names.
MODEL_SHORT = {
    "Llama-3.3-70B": "Llama70B",
    "Gemma-4-31B": "Gemma31B",
    "Qwen3.6-27B": "Qwen27B",
}
ENDPOINT_SHORT = {"completions": "comp", "chat": "chat"}
FALLBACK_COLOR = "#7f7f7f"
ENDPOINT_ORDER = {"completions": 0, "chat": 1}
# A run needs more than the smoke-test footprint (2 runs of 1 scenario) to be
# worth comparing; anything smaller is a smoke dir sharing the run_* namespace.
MIN_SCENARIOS = 2


def _lighten(hex_color, amount=0.45):
    """Blend a hex colour toward white — chat arms get the lighter shade."""
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(rgb + (1.0 - rgb) * amount)


def parse_variant(run_dirname):
    """run_<date>_<time>_<model>[-a<N>][-chat] -> dict, or None if unparseable."""
    m = re.match(r"^run_\d{8}_\d{6}_(?P<rest>.+)$", run_dirname)
    if not m:
        return None
    rest = m.group("rest")

    tag = re.search(r"-(?P<set>a\d)(?:-(?P<chat>chat))?$", rest)
    if tag:
        prompt_set = tag.group("set")
        endpoint = "chat" if tag.group("chat") else "completions"
        model_token = rest[:tag.start()]
    else:
        prompt_set = "legacy"
        endpoint = "completions"
        model_token = rest

    model = next((name for name, pat in MODEL_PATTERNS
                  if pat in model_token.lower()), model_token)
    return {
        "run_dir": run_dirname,
        "model": model,
        "prompt_set": prompt_set,
        "endpoint": endpoint,
        # `variant` is the join key / legend text; `tick` is the compact axis form.
        "variant": f"{model} {prompt_set} · {endpoint}",
        "tick": (f"{MODEL_SHORT.get(model, model)}\n"
                 f"{prompt_set} · {ENDPOINT_SHORT.get(endpoint, endpoint)}"),
    }


def variant_sort_key(row):
    model_order = [name for name, _ in MODEL_PATTERNS]
    return (
        model_order.index(row["model"]) if row["model"] in model_order else len(model_order),
        row["model"],
        row["prompt_set"],
        ENDPOINT_ORDER.get(row["endpoint"], 9),
    )


def variant_color(model, endpoint):
    base = MODEL_BASE_COLOR.get(model, FALLBACK_COLOR)
    return _lighten(base) if endpoint == "chat" else base


def collect(runs_root, include_legacy=False):
    """-> (tidy DataFrame, ordered list of variant dicts, list of skip messages)."""
    frames, variants, skipped = [], [], []
    pattern = os.path.join(runs_root, "run_*", "analysis", "combined_final_metrics.csv")
    for csv_path in sorted(glob.glob(pattern)):
        run_dirname = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
        info = parse_variant(run_dirname)
        if info is None:
            skipped.append(f"{run_dirname}: name does not match run_<date>_<time>_<model>")
            continue

        df = pd.read_csv(csv_path)
        df["scenario"] = df["scenario"].str.replace("^llm_", "", regex=True)
        n_scen = df["scenario"].nunique()
        if n_scen < MIN_SCENARIOS:
            skipped.append(f"{run_dirname}: only {n_scen} scenario(s) — smoke run")
            continue
        if info["prompt_set"] == "legacy" and not include_legacy:
            skipped.append(f"{run_dirname}: pre-campaign run, no -a<N> tag "
                           f"(use --include-legacy to include)")
            continue

        for key, value in info.items():
            df[key] = value
        frames.append(df)
        variants.append({**info, "n_rows": len(df), "n_scenarios": n_scen})

    if not frames:
        return pd.DataFrame(), [], skipped

    variants.sort(key=variant_sort_key)
    return pd.concat(frames, ignore_index=True), variants, skipped


def _draw_variant_axis(ax, data_by_variant, variants, ylabel, title):
    """House violin + box + individual-run points, one group per variant."""
    positions = np.arange(len(variants))
    plot_data = [data_by_variant[v["variant"]] for v in variants]
    # violinplot chokes on an all-identical or empty group; guard both.
    drawable = [i for i, d in enumerate(plot_data) if len(d) > 1 and np.ptp(d) > 0]
    if drawable:
        parts = ax.violinplot([plot_data[i] for i in drawable],
                              positions=[positions[i] for i in drawable],
                              showmeans=False, showmedians=False, showextrema=False)
        for slot, pc in zip(drawable, parts["bodies"]):
            col = variant_color(variants[slot]["model"], variants[slot]["endpoint"])
            pc.set_facecolor(col); pc.set_edgecolor(col)
            pc.set_alpha(0.35); pc.set_linewidth(1.0)

    non_empty = [i for i, d in enumerate(plot_data) if len(d)]
    if non_empty:
        bp = ax.boxplot([plot_data[i] for i in non_empty],
                        positions=[positions[i] for i in non_empty],
                        widths=0.18, patch_artist=True, showfliers=False)
        for slot, patch in zip(non_empty, bp["boxes"]):
            col = variant_color(variants[slot]["model"], variants[slot]["endpoint"])
            patch.set_facecolor(col); patch.set_edgecolor(col)
            patch.set_alpha(0.65); patch.set_linewidth(1.0)
        for med in bp["medians"]:
            med.set_color("black"); med.set_linewidth(1.2); med.set_zorder(4)
        for wl in bp["whiskers"]:
            wl.set_color("#777777"); wl.set_linewidth(1.0)
        for cap in bp["caps"]:
            cap.set_color("#777777"); cap.set_linewidth(1.0)

    overlay_run_points(ax, plot_data, positions)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{v['tick']}\n(n={len(data_by_variant[v['variant']])})"
                        for v in variants], fontsize=7)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3, axis="y")


def _variant_legend(fig, variants):
    handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=8,
                      markerfacecolor=variant_color(v["model"], v["endpoint"]),
                      markeredgecolor="none", label=v["variant"])
               for v in variants]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))


def plot_dissimilarity_by_scenario(df, variants, out_path):
    """One panel per scenario; every individual run's DI shown as a point."""
    scenarios = sorted(df["scenario"].unique())
    n_cols = 3
    n_rows = int(np.ceil(len(scenarios) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 4.6 * n_rows),
                             squeeze=False)
    for ax, scenario in zip(axes.flat, scenarios):
        sub = df[df["scenario"] == scenario]
        data = {v["variant"]: sub.loc[sub["variant"] == v["variant"],
                                      "dissimilarity_index"].dropna().values
                for v in variants}
        _draw_variant_axis(ax, data, variants, METRIC_LABELS["dissimilarity_index"],
                           scenario)
    for extra_ax in axes.flat[len(scenarios):]:
        extra_ax.axis("off")
    fig.suptitle("Dissimilarity Index by scenario — model × prompt set × endpoint",
                 fontsize=15)
    _variant_legend(fig, variants)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_final_boxplots(df, variants, out_path):
    """All final-step metrics, pooled over scenarios, one group per variant."""
    n_cols = 4
    n_rows = int(np.ceil(len(METRIC_KEYS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 4.6 * n_rows),
                             squeeze=False)
    for ax, metric in zip(axes.flat, METRIC_KEYS):
        data = {v["variant"]: df.loc[df["variant"] == v["variant"], metric].dropna().values
                for v in variants}
        _draw_variant_axis(ax, data, variants, METRIC_LABELS[metric],
                           METRIC_LABELS[metric])
    for extra_ax in axes.flat[len(METRIC_KEYS):]:
        extra_ax.axis("off")
    fig.suptitle("Final-step metrics pooled across all scenarios — "
                 "model × prompt set × endpoint", fontsize=15)
    _variant_legend(fig, variants)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _scenario_color(scenario, index):
    """House scenario color; tab10 fallback (same rule as plot_run_preview)."""
    return (SCENARIO_COLORS.get(scenario)
            or SCENARIO_COLORS.get(f"llm_{scenario}")
            or SCENARIO_PALETTE_TAB10(index % 10))


SCENARIO_PALETTE_TAB10 = plt.get_cmap("tab10")
SCENARIO_SHORT = {
    "baseline": "baseline",
    "race_white_black": "race",
    "ethnic_asian_hispanic": "ethnic",
    "income_high_low": "income",
    "political_liberal_conservative": "political",
    "green_yellow": "green/yellow",
}


def plot_scenarios_by_variant(df, variants, metric, out_path, n_cols=4):
    """One subplot PER VARIANT (model x prompt set x endpoint); within each,
    the metric's distribution across social contexts — scenarios on the x axis,
    house scenario colors, individual runs as points. No cross-model mixing
    inside a panel; the shared y axis is what makes panels comparable."""
    scenarios = sorted(df["scenario"].unique())
    positions = np.arange(1, len(scenarios) + 1)
    n_rows = int(np.ceil(len(variants) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.4 * n_cols, 3.6 * n_rows),
                             squeeze=False, sharey=True)
    for ax, v in zip(axes.flat, variants):
        sub = df[df["variant"] == v["variant"]]
        plot_data = [sub.loc[sub["scenario"] == s, metric].dropna().values
                     for s in scenarios]
        drawable = [i for i, d in enumerate(plot_data) if len(d) > 1 and np.ptp(d) > 0]
        if drawable:
            parts = ax.violinplot([plot_data[i] for i in drawable],
                                  positions=[positions[i] for i in drawable],
                                  showmeans=False, showmedians=False,
                                  showextrema=False)
            for slot, pc in zip(drawable, parts["bodies"]):
                col = _scenario_color(scenarios[slot], slot)
                pc.set_facecolor(col); pc.set_edgecolor(col)
                pc.set_alpha(0.35); pc.set_linewidth(1.0)
        non_empty = [i for i, d in enumerate(plot_data) if len(d)]
        if non_empty:
            bp = ax.boxplot([plot_data[i] for i in non_empty],
                            positions=[positions[i] for i in non_empty],
                            widths=0.18, patch_artist=True, showfliers=False)
            for slot, patch in zip(non_empty, bp["boxes"]):
                col = _scenario_color(scenarios[slot], slot)
                patch.set_facecolor(col); patch.set_edgecolor(col)
                patch.set_alpha(0.65); patch.set_linewidth(1.0)
            for med in bp["medians"]:
                med.set_color("black"); med.set_linewidth(1.2); med.set_zorder(4)
            for wl in bp["whiskers"]:
                wl.set_color("#777777"); wl.set_linewidth(1.0)
            for cap in bp["caps"]:
                cap.set_color("#777777"); cap.set_linewidth(1.0)
        overlay_run_points(ax, plot_data, positions)
        n_runs = int(sub.groupby("scenario").size().max()) if len(sub) else 0
        ax.set_xticks(positions)
        ax.set_xticklabels([SCENARIO_SHORT.get(s, s) for s in scenarios],
                           fontsize=7, rotation=30, ha="right")
        ax.set_title(f"{v['variant']}  (n={n_runs}/scenario)", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    for ax in axes[:, 0]:
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
    for extra_ax in axes.flat[len(variants):]:
        extra_ax.axis("off")
    fig.suptitle(f"{METRIC_LABELS.get(metric, metric)} across social contexts — "
                 "one panel per model × prompt set × endpoint", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_di_by_step(runs_root, run_dir):
    """Per-step DI for one run dir (analysis stage output), or None if absent."""
    p = os.path.join(runs_root, run_dir, "analysis", "dissimilarity_index",
                     "dissimilarity_by_step_all.csv.gz")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["scenario"] = df["scenario"].str.replace("^llm_", "", regex=True)
    return df


def _mean_ci_by_step(sub):
    """(run_id, step, dissimilarity_index) rows -> steps, mean, ci half-width.

    Converged runs stop early, freezing their grid — so each run's series is
    forward-filled to the longest step count before averaging (a plateau, not
    missing data). Keeps the mean/CI from jumping when short runs drop out.
    """
    wide = (sub.pivot_table(index="step", columns="run_id",
                            values="dissimilarity_index", aggfunc="last")
            .sort_index().ffill())
    mean = wide.mean(axis=1)
    n = wide.notna().sum(axis=1)
    sd = wide.std(axis=1, ddof=1)
    hw = 1.96 * sd / np.sqrt(n.clip(lower=1))
    return wide.index.values, mean.values, hw.fillna(0).values


def plot_di_timeseries_by_variant(variants, runs_root, out_path, n_cols=4):
    """One subplot per variant; DI vs step, one line per social context
    (mean across runs, shaded 95% CI). Same panel layout and scenario colors
    as plot_scenarios_by_variant."""
    n_rows = int(np.ceil(len(variants) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.4 * n_cols, 3.4 * n_rows),
                             squeeze=False, sharey=True)
    scenario_order, drew_any = [], False
    for ax, v in zip(axes.flat, variants):
        df = load_di_by_step(runs_root, v["run_dir"])
        if df is None:
            ax.text(0.5, 0.5, "no per-step DI", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#888888")
            ax.set_title(v["variant"], fontsize=10)
            continue
        scenarios = sorted(df["scenario"].unique())
        scenario_order = scenario_order or scenarios
        for i, s in enumerate(scenarios):
            steps, mean, hw = _mean_ci_by_step(df[df["scenario"] == s])
            col = _scenario_color(s, i)
            ax.plot(steps, mean, color=col, lw=1.6,
                    label=SCENARIO_SHORT.get(s, s))
            ax.fill_between(steps, mean - hw, mean + hw, color=col, alpha=0.15)
        drew_any = True
        n_runs = df.groupby("scenario")["run_id"].nunique().max()
        ax.set_title(f"{v['variant']}  (n={n_runs}/scenario)", fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel(METRIC_LABELS["dissimilarity_index"], fontsize=9)
    for extra_ax in axes.flat[len(variants):]:
        extra_ax.axis("off")
    if drew_any:
        handles, labels = next(ax.get_legend_handles_labels()
                               for ax in axes.flat if ax.get_legend_handles_labels()[0])
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   fontsize=9, frameon=False)
    fig.suptitle("Dissimilarity Index by step — mean ± 95% CI across runs, "
                 "one panel per model × prompt set × endpoint", fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarise(df, group_cols, value_cols):
    rows = []
    for keys, grp in df.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for metric in value_cols:
            vals = grp[metric].dropna()
            rows.append({
                **dict(zip(group_cols, keys)),
                "metric": metric,
                "n": len(vals),
                "mean": vals.mean(),
                "median": vals.median(),
                "sd": vals.std(ddof=1) if len(vals) > 1 else np.nan,
                "min": vals.min(),
                "max": vals.max(),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", default="experiments_with_llama_cpp",
                    help="directory holding the run_<timestamp>_<model> dirs")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default <reports>/cross_run)")
    ap.add_argument("--include-legacy", action="store_true",
                    help="also include pre-campaign runs with no -a<N> tag")
    ap.add_argument("--per-variant-metrics", default="dissimilarity_index",
                    type=lambda s: METRIC_KEYS if s == "all" else s.split(","),
                    help="metrics for the per-variant scenario-comparison figure "
                         "(comma list or 'all'; default dissimilarity_index)")
    args = ap.parse_args()
    bad = [m for m in args.per_variant_metrics if m not in METRIC_KEYS]
    if bad:
        ap.error(f"unknown metric(s) {bad}; known: {METRIC_KEYS}")

    out_dir = args.out_dir or str(get_reports_dir() / "cross_run")
    os.makedirs(out_dir, exist_ok=True)

    df, variants, skipped = collect(args.runs_root, include_legacy=args.include_legacy)
    for msg in skipped:                       # never drop runs silently
        print(f"[skip] {msg}")
    if df.empty:
        sys.exit(f"no comparable runs found under {args.runs_root}")

    print(f"\n[compare] {len(variants)} variants, {len(df)} simulation runs total")
    for v in variants:
        print(f"  {v['model']:15s} {v['prompt_set']:6s} {v['endpoint']:12s} "
              f"n={v['n_rows']:4d} over {v['n_scenarios']} scenarios  ({v['run_dir']})")

    di_path = os.path.join(out_dir, "cross_run_dissimilarity_index.png")
    box_path = os.path.join(out_dir, "cross_run_final_boxplots.png")
    plot_dissimilarity_by_scenario(df, variants, di_path)
    plot_final_boxplots(df, variants, box_path)

    # Per-variant view: scenarios compared WITHIN each (model, set, endpoint).
    pv_paths = []
    for metric in args.per_variant_metrics:
        p = os.path.join(out_dir, f"cross_run_by_variant_{metric}.png")
        plot_scenarios_by_variant(df, variants, metric, p)
        pv_paths.append(p)
    ts_path = os.path.join(out_dir, "cross_run_by_variant_di_by_step.png")
    plot_di_timeseries_by_variant(variants, args.runs_root, ts_path)
    pv_paths.append(ts_path)

    di_csv = os.path.join(out_dir, "cross_run_dissimilarity_index.csv")
    met_csv = os.path.join(out_dir, "cross_run_final_metrics.csv")
    summarise(df, ["model", "prompt_set", "endpoint", "scenario"],
              ["dissimilarity_index"]).to_csv(di_csv, index=False)
    summarise(df, ["model", "prompt_set", "endpoint"],
              METRIC_KEYS).to_csv(met_csv, index=False)

    print("\nWrote:")
    for p in (di_path, box_path, *pv_paths, di_csv, met_csv):
        print(f"  {p}")


if __name__ == "__main__":
    main()
