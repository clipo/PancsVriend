#!/usr/bin/env python3
"""Companion figure for SAMPLING_METHODOLOGY.md: what per-composition sample
size the CI target implies, and why extreme cells are cheap.

    python value_functions/sampling/plot_sampling_requirements.py
        -> results/figures/sampling_requirements.png

Panel A: n required for a 95% CI half-width <= w, as a function of the true
         move probability p (n = z^2 p(1-p) / w^2). The inverted-U is the
         whole economics of the two-stage design: p=0.5 costs 385 (w=5pp) or
         2,401 (w=2pp) samples, while saturated cells cost almost nothing.
Panel B: the same law seen from the other side — Wilson CI half-width vs n at
         several fixed p, with the +-5pp and +-2pp targets as horizontal
         lines. Shows the 1/sqrt(n) decay and where each p-level crosses each
         target.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from value_functions.paths import FIGURES_DIR, add_import_paths  # noqa: E402
add_import_paths()
from sampling_common import wilson_ci  # noqa: E402

OUT = FIGURES_DIR / "sampling_requirements.png"

# House palette (plot_results.py categorical slots)
S1_BLUE, S2_AQUA, S3_YELLOW, S4_GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"

Z2 = 1.96 ** 2


def n_required(p, w):
    return Z2 * p * (1 - p) / (w * w)


def main():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=300)

    # --- Panel A: n(p) for the two targets -----------------------------------
    p = np.linspace(0.001, 0.999, 500)
    for w, col in ((0.05, S1_BLUE), (0.02, S3_YELLOW)):
        ax_a.plot(p, n_required(p, w), color=col, lw=2.2,
                  label=f"target ±{w:.0%} (peak n = {int(round(n_required(0.5, w)))})")
    for w in (0.05, 0.02):
        n_peak = n_required(0.5, w)
        ax_a.annotate(f"{int(round(n_peak))}", (0.5, n_peak), ha="center",
                      va="bottom", fontsize=9, color=INK2)
    ax_a.axhline(100, color=MUTED, ls=":", lw=1.2)
    ax_a.annotate("pilot N = 100 per composition", (0.985, 130), ha="right",
                  fontsize=8.5, color=MUTED)
    ax_a.set_xlabel("true move probability p")
    ax_a.set_ylabel("samples n required for 95% CI half-width ≤ target")
    ax_a.set_title("A.  Required n peaks at p = 0.5 and vanishes at the extremes\n"
                   "n = 1.96² · p(1−p) / w²", fontsize=11)
    ax_a.legend(frameon=False, fontsize=9)
    ax_a.grid(alpha=0.3)

    # --- Panel B: Wilson half-width vs n at fixed p --------------------------
    ns = np.unique(np.round(np.geomspace(20, 4000, 220)).astype(int))
    for p_true, col in ((0.5, S1_BLUE), (0.2, S2_AQUA),
                        (0.05, S4_GREEN), (0.0, MUTED)):
        hw = []
        for n in ns:
            lo, hi = wilson_ci(int(round(p_true * n)), int(n))
            hw.append(100 * (hi - lo) / 2)
        ax_b.plot(ns, hw, color=col, lw=2.0,
                  label=f"p̂ = {p_true:g}" + ("  (0 moves observed)" if p_true == 0 else ""))
    for w_pp, txt in ((5, "±5 pp target"), (2, "±2 pp target")):
        ax_b.axhline(w_pp, color=INK, ls="--", lw=1.0, alpha=0.55)
        ax_b.annotate(txt, (22, w_pp * 1.06), fontsize=8.5, color=INK2)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("samples n (log scale)")
    ax_b.set_ylabel("Wilson 95% CI half-width (percentage points)")
    ax_b.set_title("B.  Halving the interval quadruples the price (∝ 1/√n);\n"
                   "a 0-of-n cell is inside ±2 pp by n ≈ 95", fontsize=11)
    ax_b.legend(frameon=False, fontsize=9)
    ax_b.grid(alpha=0.3, which="both")

    fig.suptitle("Sample-size requirements for the value-function two-stage design",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
