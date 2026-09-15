#!/usr/bin/env python3
"""EXACT value functions vs the SEQUENTIAL sanity resample, every cell.

    python value_functions/logprob/plot_exact_vs_sanity.py
    python value_functions/logprob/plot_exact_vs_sanity.py --label qwen3.6-27b-chat-grammar

Writes value_functions/results/llm_logprob/seqcheck_plots/
  exact_vs_sanity_<label>.png    per model, cells coloured by scenario
  exact_vs_sanity_ALL_MODELS.png one panel per model

WHY THIS IS THE STRONGER CHECK (2026-09-09)
seqcheck_* re-samples 16-18 adversarially chosen cells at n=300 — about 3% of
the table. The sanity arm samples ALL 540 cells (45 compositions x 2 roles x 6
scenarios) at n=100, sequentially, one request in flight. So this compares the
whole value function, not a spot-check, and it costs nothing extra: the tables
were built for the simulation cross-check anyway.

The bands are Wilson 95% intervals on the sanity proportion (binomial n=100).
"cells inside CI" is the headline: with 540 independent 95% intervals, ~5% are
expected to miss even if the exact tables are perfect, so the number to compare
against is ~95%, NOT 100%.
"""
import argparse
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent
for _p in (str(THIS), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                              # noqa: E402
import matplotlib                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
from value_functions.paths import (LOGPROB_TABLES_DIR, SANITY_TABLES_DIR,  # noqa: E402
                                   SEQCHECK_PLOTS_DIR)

STYLE = "R3_dual_count"
SCEN_ORDER = ["baseline", "race_white_black", "ethnic_asian_hispanic",
              "income_high_low", "political_liberal_conservative", "green_yellow"]
# One hue per scenario; colour-blind-safe qualitative set, consistent ordering.
SCEN_COLOR = dict(zip(SCEN_ORDER, ["#4C6EF5", "#E8590C", "#2B8A3E",
                                   "#AE3EC9", "#F59F00", "#1098AD"]))
C_NEUTRAL, C_BAD = "#868E96", "#C92A2A"


def collect(label):
    """Matched cells: (exact p, sanity p, ci_lo, ci_hi, scenario, inside)."""
    rows = []
    for scen in SCEN_ORDER:
        pe = LOGPROB_TABLES_DIR / f"vf_{label}-lp__{scen}__{STYLE}.json"
        ps = SANITY_TABLES_DIR / f"vf_{label}-sanity__{scen}__{STYLE}.json"
        if not (pe.exists() and ps.exists()):
            continue
        e, s = json.loads(pe.read_text()), json.loads(ps.read_text())
        for role in e["compositions"]:
            lut = {(c["n_similar"], c["n_occupied"]): c for c in s["compositions"][role]}
            for c in e["compositions"][role]:
                k = (c["n_similar"], c["n_occupied"])
                if k not in lut:
                    continue
                d = lut[k]
                lo, hi = d.get("ci95") or (None, None)
                if lo is None or c["p_move_effective"] is None:
                    continue
                rows.append((c["p_move_effective"], d["p_move_effective"], lo, hi,
                             scen, lo <= c["p_move_effective"] <= hi))
    return rows


def draw(ax, rows, label, legend=False):
    x = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    lo = np.array([r[2] for r in rows]); hi = np.array([r[3] for r in rows])
    ax.errorbar(x, y, yerr=np.clip(np.vstack([y - lo, hi - y]), 0, None),
                fmt="none", ecolor=C_NEUTRAL, elinewidth=0.5, alpha=0.45, zorder=2)
    for scen in SCEN_ORDER:
        m = [i for i, r in enumerate(rows) if r[4] == scen]
        if m:
            ax.scatter(x[m], y[m], s=11, color=SCEN_COLOR[scen], alpha=0.85,
                       label=scen if legend else None, zorder=3, linewidths=0)
    out = [i for i, r in enumerate(rows) if not r[5]]
    if out:
        ax.scatter(x[out], y[out], s=46, facecolors="none", edgecolors=C_BAD,
                   lw=0.8, zorder=4, label="exact outside CI" if legend else None)
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", zorder=1)
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)
    ax.grid(alpha=0.2)
    inside = 100.0 * sum(r[5] for r in rows) / len(rows)
    mad = float(np.median(np.abs(x - y)))
    r = float(np.corrcoef(x, y)[0, 1])
    ax.set_title(f"{label.replace('-chat-grammar','')}\n"
                 f"{len(rows)} cells · {inside:.1f}% inside CI · med |Δ| {mad:.4f} · r {r:.4f}",
                 fontsize=9)
    return inside, mad, r, len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    labels = ([args.label] if args.label else
              sorted({p.name.split("-sanity__")[0][len("vf_"):]
                      for p in SANITY_TABLES_DIR.glob(f"vf_*-sanity__*__{STYLE}.json")}))
    data = [(l, collect(l)) for l in labels]
    data = [(l, r) for l, r in data if r]
    if not data:
        print("no matched exact/sanity table pairs yet")
        return 1
    SEQCHECK_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for label, rows in data:
        fig, ax = plt.subplots(figsize=(7.2, 6.4))
        draw(ax, rows, label, legend=True)
        ax.set_xlabel("P(MOVE) EXACT (enumerated over grammar token paths)")
        ax.set_ylabel("P(MOVE) SANITY resample (sequential, n=100/cell)")
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        fig.tight_layout()
        out = SEQCHECK_PLOTS_DIR / f"exact_vs_sanity_{label}.png"
        fig.savefig(out, dpi=args.dpi); plt.close(fig)
        print(f"wrote {out.relative_to(REPO)}")

    n = len(data); ncols = min(3, n); nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.4 * nrows), squeeze=False)
    tot_in = tot_n = 0
    for k, (label, rows) in enumerate(data):
        ax = axes[k // ncols][k % ncols]
        inside, mad, r, n_cells = draw(ax, rows, label, legend=(k == 0))
        tot_in += inside * n_cells / 100.0; tot_n += n_cells
        if k % ncols == 0:
            ax.set_ylabel("SANITY resample (sequential, n=100)", fontsize=8)
        if k // ncols == nrows - 1:
            ax.set_xlabel("EXACT (enumerated)", fontsize=8)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("EXACT value functions vs the SEQUENTIAL sanity resample — every cell\n"
                 f"{tot_n} cells across {n} model(s): {100.0*tot_in/tot_n:.1f}% of exact values inside the "
                 "sanity 95% CI (~95% expected if the exact tables are correct)",
                 fontsize=11.5, fontweight="bold", y=0.995)
    fig.text(0.5, 0.008,
             "Every cell of the value function, not the 3% spot-check in seqcheck_*: the sanity arm samples all 540 cells "
             "(45 compositions x 2 roles x 6 scenarios) at n=100 with ONE request in flight.\n"
             "Bars are Wilson 95% intervals on the sanity proportion. With 540 independent intervals ~5% are expected to miss "
             "by chance even when the exact tables are exactly right, so ~95% is the target, not 100%.",
             ha="center", va="bottom", fontsize=7.5, color="#495057", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.955))
    out = SEQCHECK_PLOTS_DIR / "exact_vs_sanity_ALL_MODELS.png"
    fig.savefig(out, dpi=args.dpi); plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}  ({tot_n} cells, {n} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
