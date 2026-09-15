#!/usr/bin/env python3
"""One figure: the sequential validation of the exact tables for EVERY model.

    python value_functions/logprob/plot_seqcheck_all.py

Writes value_functions/results/llm_logprob/seqcheck_plots/seqcheck_ALL_MODELS.png

The per-model figures (seqcheck_<label>.png) carry the detail; this is the
cross-model view — one panel per LLM, same axes, so "did the exact extraction
reproduce under ordinary sampling, for all of them" is a single glance.
Palette and conventions match plot_seqcheck.py so the two read as one family.
"""
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent
for _p in (str(THIS), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402
import matplotlib                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
from value_functions.paths import LOGPROB_VALIDATION_DIR, SEQCHECK_PLOTS_DIR  # noqa: E402

C_SEQ, C_NEUTRAL, C_BAD, C_OK = "#4C6EF5", "#868E96", "#C92A2A", "#2B8A3E"


def main():
    csvs = sorted(LOGPROB_VALIDATION_DIR.glob("seqcheck_*.csv"))
    if not csvs:
        print(f"no seqcheck_*.csv under {LOGPROB_VALIDATION_DIR}")
        return 1
    n = len(csvs)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 4.0 * nrows),
                             squeeze=False)
    n_pass = 0
    for k, csv in enumerate(csvs):
        label = csv.stem[len("seqcheck_"):]
        d = pd.read_csv(csv).sort_values("p_logprob")
        vj = LOGPROB_VALIDATION_DIR / f"validation_{label}.json"
        summ = json.loads(vj.read_text()) if vj.exists() else {}
        ax = axes[k // ncols][k % ncols]
        yerr = np.clip(np.vstack([d.p_sequential - d.seq_ci_low,
                                  d.seq_ci_high - d.p_sequential]), 0, None)
        ax.errorbar(d.p_logprob, d.p_sequential, yerr=yerr, fmt="o", ms=4.5,
                    color=C_SEQ, ecolor=C_SEQ, elinewidth=1.0, capsize=2, zorder=3)
        miss = d[~d.inside_seq_ci]
        if len(miss):
            ax.scatter(miss.p_logprob, miss.p_sequential, s=130, facecolors="none",
                       edgecolors=C_BAD, lw=1.8, zorder=4)
        ax.plot([0, 1], [0, 1], color=C_NEUTRAL, lw=1, ls="--", zorder=1)
        ok = summ.get("pass", False)
        n_pass += bool(ok)
        med = summ.get("seq_check_median_abs_diff")
        ins = summ.get("seq_check_inside_ci")
        ax.set_title(f"{label.replace('-chat-grammar','')}\n"
                     f"{'PASS' if ok else 'FAIL'} — {ins*100:.0f}% of {len(d)} cells in CI"
                     + (f", med |Δ| {med:.4f}" if med is not None else ""),
                     fontsize=9, color=C_OK if ok else C_BAD)
        ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
        if k % ncols == 0:
            ax.set_ylabel("P(MOVE) re-sampled sequentially", fontsize=8)
        if k // ncols == nrows - 1:
            ax.set_xlabel("P(MOVE) exact (grammar-masked logprobs)", fontsize=8)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.suptitle(f"Sequential validation of the EXACT value functions — {n_pass}/{n} models PASS",
                 fontsize=13, fontweight="bold", color=C_OK if n_pass == n else C_BAD, y=0.995)
    fig.text(0.5, 0.008,
             "Per model the cells where the exact tables and the old concurrency-4 campaign disagree MOST are re-sampled with ONE request in "
             "flight (n=300, escalated to 900 on a first-stage CI miss).\nA point on the dashed line means the enumerated probability reproduces "
             "under ordinary sampling; the bar is the Wilson 95% CI of the re-sample. PASS = the exact value falls inside the CI for EVERY checked cell.",
             ha="center", va="bottom", fontsize=7.5, color="#495057", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    SEQCHECK_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = SEQCHECK_PLOTS_DIR / "seqcheck_ALL_MODELS.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}  ({n} models, {n_pass} pass)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
