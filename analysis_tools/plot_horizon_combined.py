"""One horizon figure for a model from several max_steps_horizon_check outputs.

Each scenario is drawn from the LONGEST-horizon output that contains it (the
outputs must come from the same run_ids, e.g. runs 0..999, so the curves agree
step for step where they overlap: keyed RNG makes a longer run a bit-identical
continuation). A scenario whose data stops before the figure's last step is
drawn solid up to its last simulated step and then as a DOTTED horizontal line
at its mean over the final --hold-window steps: a level held, not simulated.
Only do that for scenarios the equivalence test reported flat at the end.

Dashed segments keep the max_steps_horizon_check meaning: every run stopped
(the mean is pure carry-forward). Colours and labels are the paper's
(analysis_tools/paper/generate_convergence_plots.py).

    .venv/bin/python analysis_tools/plot_horizon_combined.py \\
        experiments_with_llama_cpp/horizon_check/long_10000 \\
        experiments_with_llama_cpp/horizon_check/long_2000 \\
        --out experiments_with_llama_cpp/horizon_check/combined_olmo_10000.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# paper palette and short labels (analysis_tools/paper/generate_convergence_plots.py)
SCENARIO_COLORS = {
    'baseline': '#7f7f7f',
    'green_yellow': '#2ca02c',
    'race_white_black': '#d62728',
    'ethnic_asian_hispanic': '#ff7f0e',
    'income_high_low': '#1f77b4',
    'political_liberal_conservative': '#9467bd',
}
SCENARIO_LABELS = {
    'baseline': 'Color(RvB)',
    'green_yellow': 'Color(GvY)',
    'race_white_black': 'Racial',
    'ethnic_asian_hispanic': 'Ethnic',
    'income_high_low': 'Economic',
    'political_liberal_conservative': 'Political',
}


def canon(sc: str) -> str:
    # the corrected race table ran under a temporary key before the rename
    return "race_white_black" if sc.startswith("race_white_black") else sc


def load(dirs, model=None):
    """{scenario: (curve DataFrame, n_runs)} from the longest output per scenario."""
    best = {}
    for d in dirs:
        d = Path(d)
        cur = pd.read_csv(d / "horizon_mean_curves.csv.gz")
        rows = pd.read_csv(d / "horizon_scenarios.csv")
        if model:
            cur = cur[cur["model"].str.startswith(model)]
            rows = rows[rows["model"].str.startswith(model)]
        for sc, c in cur.groupby("scenario"):
            n = int(rows.loc[rows["scenario"] == sc, "n_runs"].iloc[0])
            key = canon(sc)
            if key not in best or c["step"].max() > best[key][0]["step"].max():
                best[key] = (c.sort_values("step"), n, d.name)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", help="max_steps_horizon_check --out-dir folders")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=None, help="model prefix filter (e.g. olmo)")
    ap.add_argument("--zoom", type=int, default=500)
    ap.add_argument("--hold-window", type=int, default=500)
    ap.add_argument("--old-cap", type=int, default=1000)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load(a.dirs, a.model)
    T = max(int(c["step"].max()) + 1 for c, _, _ in data.values())
    # draw low-DI first so the highest curves sit on top (paper convention)
    order = sorted(data, key=lambda k: data[k][0]["mean"].iloc[-1])

    fig = plt.figure(figsize=(16, 6.2))
    g = fig.add_gridspec(2, 2, height_ratios=(4, 1), hspace=0.08, wspace=0.16,
                         width_ratios=(1.15, 1), left=0.06, right=0.82, top=0.88, bottom=0.1)
    lo_zoom = T - a.zoom
    handles = {}
    ends = sorted({int(data[k][0]["step"].max()) + 1 for k in data} - {T})
    for j, lo in enumerate((0, lo_zoom)):
        ax = fig.add_subplot(g[0, j])
        sx = fig.add_subplot(g[1, j], sharex=ax)
        for sc in order:
            c, n, _ = data[sc]
            col = SCENARIO_COLORS.get(sc, "#999999")
            last = int(c["step"].max())
            cc = c[c["step"] >= lo]
            if len(cc):
                x, y, ci = cc["step"].to_numpy(), cc["mean"].to_numpy(), cc["ci95"].to_numpy()
                live = cc["n_active"].to_numpy() > 0
                k = int(np.argmin(live)) if not live.all() else len(cc)
                ax.fill_between(x, y - ci, y + ci, color=col, alpha=0.15, lw=0)
                h, = ax.plot(x[:k], y[:k], color=col, lw=1.8)
                if k < len(cc):
                    ax.plot(x[max(k - 1, 0):], y[max(k - 1, 0):], color=col, lw=1.8, ls=(0, (4, 2)))
                sx.plot(x, 100 * cc["n_active"] / n, color=col, lw=1.3)
                handles.setdefault(sc, h)
            if last + 1 < T:                                   # held level, not simulated
                hold = float(c.loc[c["step"] > last - a.hold_window, "mean"].mean())
                x0 = max(lo, last)
                h2, = ax.plot([x0, T - 1], [hold, hold], color=col, lw=1.6, ls=(0, (1, 2)))
                handles.setdefault(sc, h2)
        for a_ in (ax, sx):
            a_.grid(alpha=0.25, lw=0.6)
            a_.spines[["top", "right"]].set_visible(False)
            for xv in [a.old_cap] + ends:
                if lo < xv < T:
                    a_.axvline(xv, color="#555555", lw=0.9, ls=":")
        if lo < a.old_cap < T:
            ax.text(a.old_cap, 0.30, f" previous cap ({a.old_cap:,})",
                    transform=ax.get_xaxis_transform(), va="bottom", ha="left", fontsize=8,
                    color="#555555")
        for e in ends:
            if lo < e < T:
                ax.text(e, 0.03, f" 4 scenarios simulated to {e:,}; dotted = level held",
                        transform=ax.get_xaxis_transform(), va="bottom", ha="left",
                        fontsize=8, color="#555555")
        ax.set_xlim(lo, T - 1)
        ax.tick_params(labelbottom=False)
        ax.set_ylabel("mean dissimilarity index")
        ax.set_title(f"steps 0-{T - 1:,}" if j == 0 else f"last {a.zoom} steps (zoom)",
                     fontsize=10, loc="left")
        sx.set_ylim(-5, 105)
        sx.set_ylabel("% runs\nstill moving", fontsize=8)
        sx.set_xlabel("step")
    labels = []
    for sc in reversed(order):                               # legend top = highest DI
        c, n, src = data[sc]
        last = int(c["step"].max()) + 1
        labels.append((handles[sc], f"{SCENARIO_LABELS.get(sc, sc)}  "
                       f"(n={n:,}, simulated to {last:,}"
                       + (", then dotted)" if last < T else ")")))
    fig.legend([h for h, _ in labels], [t for _, t in labels], loc="upper left",
               bbox_to_anchor=(0.83, 0.86), frameon=False, fontsize=9)
    model = a.model or "model"
    fig.suptitle(f"{model}: mean dissimilarity index per step (band = 95% CI; runs 0-"
                 f"{max(n for _, n, _ in data.values()) - 1:,}, the same runs in every scenario)",
                 fontsize=11, x=0.44)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"[horizon-combined] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
