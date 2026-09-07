#!/usr/bin/env python3
"""Shared metric-loading helpers, plus the RETIRED E3 sufficiency check.

RETIRED 2026-09-01 — the E3 CLI (`main()` below). Use instead:

    python value_functions/sampling/vf_multisplit_check.py --label <label> --splits 32

E3 compared the full artifact against ONE half-data artifact built from the
even-indexed raw samples. That is exactly the multi-split statistic at B=1 with
a deterministic partition instead of a random one, so it was strictly dominated
by a check that costs no GPU. One draw carries ~70% relative error on the
displacement, enough to invert the ranking outright: on qwen, E3 named
political/share the worst check at 1.049 (truly 0.563) while
race_white_black/share read 0.012 (truly 0.838).

Retiring it also removed the half arm from run_vf_eval_sims.sh — a full 100-run
x 1000-step x 6-scenario production batch per model that existed solely to feed
this one comparison.

THIS MODULE IS STILL IMPORTED as a library and must keep working:
  vf_multisplit_check.py         -> ALL_METRICS, METRIC_LABELS, SCENARIO_ORDER,
                                    load_batch
  vf_lookup_comparison_analysis.py -> the above plus fig_half_check
fig_half_check survives because the lookup-granularity comparison uses it for a
different purpose (composition vs ratio cells), not for sufficiency.

The manifest (written by run_vf_eval_sims.sh) maps experiment batches:

    {
      "full":       {"baseline": "experiments/llm_baseline_...", ...},
      "half":       {"baseline": "experiments/llm_baseline_...", ...},   # optional
      "mechanical": "experiments/baseline_...",                          # optional
      "max_steps":  200
    }

Output (durable, --dpi/--format aware):

  vfE3_half_check.*    the sampling-sufficiency verdict: full-data vs
                       half-data batch, PAIRED by run seed. Per (metric,
                       scenario): mean paired difference ± its 95% CI, next
                       to the full batch's own CI half-width. PASS when
                       |mean Δ| <= full-mean CI half-width. Also written as
                       vfE3_half_check.csv and printed as a table.

Metric matrices come from metrics_history.csv (NOT step_statistics — that file
pools survivors per step); the dissimilarity index is computed from the states
npz. Converged runs stop writing rows; their grids are frozen, so metrics are
FORWARD-FILLED to max_steps — the honest continuation, not an approximation.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
for p in (_THIS, _THIS.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dissimilarity_index_over_time import compute_dissimilarity_from_int_grid  # noqa: E402
from plot_run_preview import METRIC_KEYS  # noqa: E402
import run_files  # noqa: E402

ALL_METRICS = METRIC_KEYS + ["dissimilarity_index"]
METRIC_LABELS = {
    "clusters": "clusters", "switch_rate": "switch rate", "distance": "distance",
    "mix_deviation": "mix deviation", "share": "share",
    "ghetto_rate": "ghetto count", "dissimilarity_index": "dissimilarity index",
}
SCENARIO_ORDER = ["baseline", "race_white_black", "ethnic_asian_hispanic",
                  "income_high_low", "political_liberal_conservative",
                  "green_yellow"]


def load_batch(exp_dir, max_steps):
    """{metric: (n_runs, max_steps) ndarray}, runs ordered by run_id,
    columns steps 1..max_steps, forward-filled after convergence."""
    exp_dir = Path(exp_dir)
    mh = pd.read_csv(run_files.metrics_history_path(exp_dir))
    run_ids = sorted(mh.run_id.unique())
    out = {}
    for m in METRIC_KEYS:
        pv = (mh.pivot_table(index="run_id", columns="step", values=m)
                .reindex(index=run_ids, columns=range(1, max_steps + 1)))
        out[m] = pv.ffill(axis=1).to_numpy()

    # DI: FINAL value only, as a constant per-run row. The check consumes
    # finals ([:, -1]); for DI trajectories use the analysis suite's
    # dissimilarity_index_over_time. run_files hands back the final grid for
    # both run formats (per-step frames, or the older one-frame-per-move npz
    # whose frame index is NOT a step index — bug found 2026-08-22).
    di = np.full((len(run_ids), max_steps), np.nan)
    for i, rid in enumerate(run_ids):
        grid = run_files.load_final_grid(exp_dir, rid)
        if grid is not None:
            di[i, :] = compute_dissimilarity_from_int_grid(np.asarray(grid))
    out["dissimilarity_index"] = di
    return out


def fig_half_check(full, half, scenarios, out_path, csv_path, dpi,
                   labels=("full", "half"),
                   title="Sampling-sufficiency check: full-data vs half-data "
                         "value function"):
    """Paired A-vs-B comparison per (metric, scenario) + verdict table.

    `full`/`half` are the A and B batches (paired by run seed); `labels`
    names them in the x-ticks, legend, and CSV columns. The default is the
    sampling-sufficiency use; run_vf_lookup_comparison.py reuses it with
    labels=("composition", "ratio-avg")."""
    rows = []
    for sc in scenarios:
        for m in ALL_METRICS:
            f = full[sc][m][:, -1]
            h = half[sc][m][:, -1]
            n = min(len(f), len(h))
            f, h = f[:n], h[:n]                       # paired by run seed
            ok = ~(np.isnan(f) | np.isnan(h))
            f, h = f[ok], h[ok]
            d = f - h
            dmean = float(np.mean(d))
            dhw = float(1.96 * np.std(d, ddof=1) / np.sqrt(len(d)))
            fhw = float(1.96 * np.std(f, ddof=1) / np.sqrt(len(f)))
            rows.append({"scenario": sc, "metric": m, "n_pairs": len(d),
                         f"{labels[0]}_mean": float(np.mean(f)),
                         f"{labels[1]}_mean": float(np.mean(h)),
                         "paired_diff_mean": dmean, "paired_diff_ci95": dhw,
                         f"{labels[0]}_mean_ci95": fhw,
                         "verdict": "PASS" if abs(dmean) <= fhw else "FLAG"})
    tab = pd.DataFrame(rows)
    tab.to_csv(csv_path, index=False)

    nrows, ncols = len(ALL_METRICS), len(scenarios)
    # sharey="row": one common y-range per metric row, so magnitudes are
    # comparable ACROSS scenarios, not just within a panel.
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=True,
                             sharey="row",
                             figsize=(1.9 * ncols + 1.4, 1.15 * nrows + 1.6))
    for j, sc in enumerate(scenarios):
        for i, m in enumerate(ALL_METRICS):
            ax = axes[i][j]
            r = tab[(tab.scenario == sc) & (tab.metric == m)].iloc[0]
            # Point colours deliberately avoid red/blue: red is the FLAG
            # background (and elsewhere the red role), blue the blue role.
            ax.errorbar([0], [r[f"{labels[0]}_mean"]],
                        yerr=[r[f"{labels[0]}_mean_ci95"]],
                        fmt="o", ms=4, color="#404040", capsize=3)
            ax.errorbar([1], [r[f"{labels[1]}_mean"]],
                        yerr=[r.paired_diff_ci95],
                        fmt="s", ms=4, color="#7b3294", capsize=3)
            ax.set_xlim(-0.7, 1.7)
            ax.set_xticks([0, 1]); ax.set_xticklabels(list(labels), fontsize=6)
            ax.tick_params(axis="y", labelsize=6)
            verdict = r.verdict
            ax.set_facecolor("#f2fff2" if verdict == "PASS" else "#fff0f0")
            if i == 0:
                ax.set_title(sc, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(METRIC_LABELS[m], fontsize=7)
    n_flag = int((tab.verdict == "FLAG").sum())
    from matplotlib.patches import Patch
    handles = [
        plt.Line2D([], [], color="#404040", marker="o", ls="", ms=5,
                   label=f"{labels[0]} mean; whisker = its 95% CI (1.96·SE over runs)"),
        plt.Line2D([], [], color="#7b3294", marker="s", ls="", ms=5,
                   label=f"{labels[1]} mean; whisker = 95% CI of the PAIRED per-run difference"),
        Patch(facecolor="#f2fff2", edgecolor="#bbb",
              label=f"PASS: |paired Δmean| ≤ {labels[0]}-mean CI half-width"),
        Patch(facecolor="#fff0f0", edgecolor="#bbb",
              label="FLAG: the A/B difference exceeds run-to-run precision"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"{title} (paired seeds; {n_flag} flagged of {len(tab)})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.955))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return tab


def main() -> int:
    # Retired, not deleted: the module is still imported as a library, and an
    # old shell-history invocation must fail loudly with the replacement rather
    # than quietly emit a verdict we no longer trust. --i-know-e3-is-retired
    # runs it anyway, for reproducing a pre-2026-09-01 result.
    if "--i-know-e3-is-retired" not in sys.argv:
        print(__doc__.split("THIS MODULE")[0].strip(), file=sys.stderr)
        return 2
    sys.argv = [a for a in sys.argv if a != "--i-know-e3-is-retired"]
    ap = argparse.ArgumentParser(description="RETIRED E3 sufficiency check")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="default: <manifest dir>/vf_eval_figures")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text())
    max_steps = int(man.get("max_steps", 200))
    out_dir = Path(args.out_dir or (Path(args.manifest).parent / "vf_eval_figures"))
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = sorted(man["full"], key=lambda s: (SCENARIO_ORDER.index(s)
                                                   if s in SCENARIO_ORDER else 99, s))
    print(f"loading {len(scenarios)} full batches ...")
    full = {sc: load_batch(d, max_steps) for sc, d in man["full"].items()}
    ext = args.format

    if man.get("half"):
        print(f"loading {len(man['half'])} half batches ...")
        half = {sc: load_batch(d, max_steps) for sc, d in man["half"].items()}
        common = [sc for sc in scenarios if sc in half]
        tab = fig_half_check(full, half, common,
                             out_dir / f"vfE3_half_check.{ext}",
                             out_dir / "vfE3_half_check.csv", args.dpi)
        print(f"wrote {out_dir}/vfE3_half_check.{ext} and .csv")
        print("\n=== sufficiency verdicts (|paired Δmean| vs full-mean 95% CI) ===")
        with pd.option_context("display.width", 160, "display.max_rows", 200):
            print(tab[["scenario", "metric", "paired_diff_mean",
                       "full_mean_ci95", "verdict"]].to_string(index=False))
        n_flag = int((tab.verdict == "FLAG").sum())
        print(f"\n{'ALL PASS' if n_flag == 0 else f'{n_flag} FLAG(s)'} "
              f"across {len(tab)} (scenario, metric) checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
