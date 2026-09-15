#!/usr/bin/env python3
"""Plot the SEQUENTIAL validation of the exact value functions — the real test.

    python value_functions/logprob/plot_seqcheck.py            # all models
    python value_functions/logprob/plot_seqcheck.py --label qwen3.6-27b-chat-grammar

Data:    value_functions/results/llm_logprob/validation_data/
Figures: value_functions/results/llm_logprob/seqcheck_plots/seqcheck_<label>.png

The extractor (logprob_value_function.validate) draws this figure itself as
soon as the sequential check has been written, so a fresh extraction never
lacks it; this script REDRAWS from the data on disk (after a styling change,
or for tables validated before the plot existed).

WHY THIS EXISTS (2026-09-07)
The artifact map (OUTDATED_artifact_samples_vs_exact_*.png) plots the exact
tables against the OUTDATED concurrency-4 campaign, so it is full of large
disagreements that are the batch-numerics artifact, not extraction error. It
was being read as though it were the validation. The actual pass/fail test —
the worst-disagreeing cells RE-SAMPLED SEQUENTIALLY, n=300, escalated 3x on a
first-stage miss — had no plot at all, only seqcheck_<label>.csv.

SEQUENTIAL ONLY (user decision 2026-09-07). The outdated concurrency-4
campaign is not drawn: those numbers are wrong and are not used going forward.
The one-off artifact maps that did show them (OUTDATED_artifact_samples_vs_exact_*)
and the code that drew them were deleted 2026-09-15.

  left   exact (x) vs sequential re-sample (y, Wilson 95% CI) against the
         identity line.
  right  the residual sequential - exact with the same CI. Zero inside every
         interval IS the pass criterion, so it is plotted directly rather than
         left for the reader to difference.

PASS = every checked cell's exact value inside its final sequential CI.
"""
import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent
for _p in (str(THIS), str(REPO), str(REPO / "prompt_refinement")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json                                    # noqa: E402
import numpy as np                             # noqa: E402
import pandas as pd                            # noqa: E402
import matplotlib                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                # noqa: E402

from value_functions.paths import LOGPROB_DIR, LOGPROB_VALIDATION_DIR, SEQCHECK_PLOTS_DIR  # noqa: E402

OUT = SEQCHECK_PLOTS_DIR
DATA = LOGPROB_VALIDATION_DIR
# Same two categorical hues the artifact map uses, plus a neutral for the
# identity line, so the two figures read as one family.
C_SEQ, C_CAMP, C_NEUTRAL, C_BAD = "#4C6EF5", "#E8590C", "#868E96", "#C92A2A"


def plot_one(label):
    csv = DATA / f"seqcheck_{label}.csv"
    if not csv.exists():
        print(f"{label}: no seqcheck csv"); return False
    d = pd.read_csv(csv).sort_values("p_logprob").reset_index(drop=True)
    vj = DATA / f"validation_{label}.json"
    summary = json.loads(vj.read_text()) if vj.exists() else {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))

    # SEQUENTIAL ONLY. The outdated concurrency-4 campaign is deliberately NOT
    # drawn here (user decision 2026-09-07): those numbers are wrong and are
    # not used going forward. This plot is the fresh work.
    ax = axes[0]
    yerr = np.clip(np.vstack([d.p_sequential - d.seq_ci_low,
                              d.seq_ci_high - d.p_sequential]), 0, None)
    ax.errorbar(d.p_logprob, d.p_sequential, yerr=yerr, fmt="o", ms=6, color=C_SEQ,
                ecolor=C_SEQ, elinewidth=1.2, capsize=3,
                label="sequential re-sample (Wilson 95% CI)", zorder=4)
    miss = d[~d.inside_seq_ci]
    if len(miss):
        ax.scatter(miss.p_logprob, miss.p_sequential, s=160, facecolors="none",
                   edgecolors=C_BAD, lw=2, label="exact outside CI (FAIL)", zorder=5)
    ax.plot([0, 1], [0, 1], color=C_NEUTRAL, lw=1, ls="--", zorder=1,
            label="exact = sequential")
    ax.set_xlabel("P(MOVE) from grammar-masked logprobs (exact)")
    ax.set_ylabel("P(MOVE) re-sampled sequentially")
    ax.set_title("every cell on the identity line, within CI", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)

    # Residual with its own CI: zero inside every interval IS the pass criterion,
    # so plot it directly rather than making the reader difference two series.
    ax = axes[1]
    x = np.arange(len(d))
    resid = d.p_sequential - d.p_logprob
    ax.axhline(0, color=C_NEUTRAL, lw=1.2, ls="--", label="exact")
    ax.errorbar(x, resid, yerr=yerr, fmt="o", ms=5, color=C_SEQ, ecolor=C_SEQ,
                elinewidth=1.2, capsize=3, label="sequential - exact, 95% CI")
    if len(miss):
        mi = [i for i, ok in enumerate(d.inside_seq_ci) if not ok]
        ax.scatter(mi, resid.iloc[mi], s=160, facecolors="none", edgecolors=C_BAD,
                   lw=2, label="outside CI (FAIL)")
    lim = float(np.abs(np.concatenate([resid.values + yerr[1], resid.values - yerr[0]])).max())
    ax.set_ylim(-1.25 * lim, 1.25 * lim)
    ax.set_xlabel("checked cell (sorted by exact P(MOVE))")
    ax.set_ylabel("sequential - exact")
    ax.set_title("residuals straddle zero", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2)

    inside = summary.get("seq_check_inside_ci")
    med = summary.get("seq_check_median_abs_diff")
    esc = summary.get("seq_check_escalated")
    verdict = "PASS" if summary.get("pass") else "FAIL"
    stats = (f"{inside*100:.0f}% of {len(d)} cells inside CI  ·  "
             f"median |exact - sequential| = {med:.4f}  ·  {esc} escalated"
             if inside is not None else "")
    fig.suptitle(f"{label} — SEQUENTIAL validation of the exact tables: {verdict}",
                 fontsize=12, fontweight="bold",
                 color="#2B8A3E" if summary.get("pass") else C_BAD, y=0.975)
    if stats:
        fig.text(0.5, 0.925, stats, ha="center", va="top", fontsize=9, color="#495057")
    fig.text(0.5, 0.015,
             "The pass/fail test for the log-probability extraction. Cells are chosen ADVERSARIALLY (the hardest ones) and "
             "re-sampled with ONE request in flight:\n"
             "n=300, and a first-stage CI miss is escalated to n=900 and re-tested, which separates a chance 5% miss from a "
             "real error. Sequential sampling is bit-reproducible on this server.",
             ha="center", va="bottom", fontsize=7.5, color="#495057", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.085, 1, 0.905))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"seqcheck_{label}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    try:
        shown = path.relative_to(REPO)
    except ValueError:                       # --out-dir outside the repo
        shown = path
    print(f"{label}: {shown}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", help="one model label (default: all with a seqcheck csv)")
    args = ap.parse_args()
    labels = ([args.label] if args.label
              else sorted(p.stem[len("seqcheck_"):] for p in DATA.glob("seqcheck_*.csv")))
    if not labels:
        print(f"no seqcheck_*.csv under {DATA}"); return 1
    ok = all([plot_one(l) for l in labels])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
