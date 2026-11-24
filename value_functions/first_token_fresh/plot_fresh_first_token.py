#!/usr/bin/env python3
"""Figures for the fresh first-token validation (vector PDF + 300 dpi PNG each).

    python value_functions/first_token_fresh/plot_fresh_first_token.py \
        value_functions/first_token_fresh/results/fresh_<label>_cases.csv

Reads the per-case CSV written by fresh_first_token.py and writes, next to it:
  <stem>_scatter.{pdf,png}             fresh vs stored P(MOVE), with y = x
  <stem>_absdiff_hist.{pdf,png}        absolute differences, log axis
  <stem>_choice_function_<role>.{pdf,png}
                                       P(MOVE) against the number of similar neighbours, one
                                       panel per (scenario, number of occupied neighbours),
                                       fresh and stored overlaid
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STORED, FRESH, WORD = "#1f4e79", "#d9730d", "#7a7a7a"      # stored path sum, fresh first token, whole-word read
SCENARIO_NAMES = {"baseline": "Red / blue", "green_yellow": "Green / yellow", "income_high_low": "Income",
                  "political_liberal_conservative": "Political", "race_white_black": "Racial",
                  "ethnic_asian_hispanic": "Ethnic"}
ZERO_FLOOR = 1e-18      # exact zeros are drawn at this value on log axes and labelled "0"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight"})


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        for k, v in r.items():
            if k in ("model", "scenario", "role", "prompt_sha256"):
                continue
            r[k] = math.nan if v in ("", "nan") else float(v)
        rows.append(r)
    return rows


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")


def scatter(rows, stem, label):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    panels = [("stored_ft_mask", "Stored first-step table, masked $P(\\mathrm{MOVE})$", "d_mask_vs_stored_first"),
              ("stored_path_sum", "Stored path-sum $P(\\mathrm{MOVE})$", "d_mask_vs_path")]
    for ax, (xcol, xlabel, dcol) in zip(axes, panels):
        sat = [r for r in rows if not r["unsaturated"]]
        uns = [r for r in rows if r["unsaturated"]]
        ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, zorder=1, label="$y = x$")
        ax.scatter([r[xcol] for r in sat], [r["fresh_ft_mask"] for r in sat], s=10, facecolor="none",
                   edgecolor=STORED, lw=0.5, zorder=2, label=f"saturated ({len(sat)})")
        ax.scatter([r[xcol] for r in uns], [r["fresh_ft_mask"] for r in uns], s=14, color=FRESH,
                   edgecolor="white", lw=0.3, zorder=3, label=f"unsaturated ({len(uns)})")
        dmax = max(r[dcol] for r in rows)
        ax.text(0.97, 0.05, f"largest $|\\Delta|$ = {dmax:.2g}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
    axes[0].set_ylabel("Fresh first token, masked $P(\\mathrm{MOVE})$")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle(f"{label}: fresh first-token query against the stored record ({len(rows)} cases)", fontsize=9.5, y=1.0)
    fig.tight_layout()
    save(fig, f"{stem}_scatter")


def absdiff_hist(rows, stem, label):
    series = [("d_mask_vs_stored_first", "fresh masked first token vs stored first-step table", FRESH),
              ("d_mask_vs_path", "fresh masked first token vs stored path sum", STORED),
              ("d_word_vs_path", "fresh whole-word read vs stored path sum", WORD)]
    fig, axes = plt.subplots(len(series), 1, figsize=(6.4, 5.6), sharex=True)
    edges = np.concatenate([[ZERO_FLOOR / 10, ZERO_FLOOR * 10], np.logspace(-17, 0, 35)])
    for ax, (col, title, colour) in zip(axes, series):
        for subset, alpha, name in ((rows, 0.35, "all cases"), ([r for r in rows if r["unsaturated"]], 0.95, "unsaturated")):
            vals = [r[col] for r in subset if not math.isnan(r[col])]
            ax.hist([max(v, ZERO_FLOOR) for v in vals], bins=edges, color=colour, alpha=alpha,
                    label=f"{name} (n = {len(vals)})")
        ax.axvline(0.01, color="#b00020", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylabel("Cases")
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2)
    axes[0].text(0.01, axes[0].get_ylim()[1], " 0.01", color="#b00020", fontsize=7.5, va="top")
    ticks = [ZERO_FLOOR] + [10.0 ** e for e in range(-15, 1, 3)]
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(["0"] + [f"$10^{{{e}}}$" for e in range(-15, 1, 3)])
    axes[-1].set_xlabel("Absolute difference in $P(\\mathrm{MOVE})$ (exact zeros shown at the left edge)")
    fig.suptitle(f"{label}: absolute differences, fresh vs stored", fontsize=9.5, y=0.995)
    fig.tight_layout()
    save(fig, f"{stem}_absdiff_hist")


def choice_function(rows, stem, label, role):
    rs = [r for r in rows if r["role"] == role]
    scenarios = [s for s in SCENARIO_NAMES if any(r["scenario"] == s for r in rs)]
    occupied = sorted({int(r["n_occupied"]) for r in rs})
    fig, axes = plt.subplots(len(scenarios), len(occupied), figsize=(1.25 * len(occupied) + 0.9, 1.25 * len(scenarios) + 0.9),
                             sharey=True, squeeze=False)
    for i, sc in enumerate(scenarios):
        for j, occ in enumerate(occupied):
            ax = axes[i][j]
            cell = sorted((r for r in rs if r["scenario"] == sc and int(r["n_occupied"]) == occ), key=lambda r: r["n_similar"])
            x = [r["n_similar"] for r in cell]
            ax.plot(x, [r["stored_path_sum"] for r in cell], color=STORED, lw=1.0, marker="o", ms=4.2,
                    mfc="white", mew=0.9, zorder=2, label="stored path sum")
            ax.plot(x, [r["fresh_ft_mask"] for r in cell], color=FRESH, lw=0, marker="x", ms=3.6, mew=0.9,
                    zorder=3, label="fresh first token (masked)")
            ax.set_ylim(-0.06, 1.06)
            ax.set_xlim(-0.6, max(occupied) + 0.6)
            ax.set_xticks([0, 4, 8])
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(length=2, labelsize=7)
            ax.grid(True, lw=0.3, color="#dddddd")
            if i == 0:
                ax.set_title(f"{occ} occupied", fontsize=8)
            if j == 0:
                ax.set_ylabel(SCENARIO_NAMES[sc], fontsize=8)
            if i < len(scenarios) - 1:
                ax.tick_params(labelbottom=False)
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.985))
    fig.supxlabel("Number of similar neighbours", fontsize=9)
    fig.supylabel("$P(\\mathrm{MOVE})$", fontsize=9)
    fig.suptitle(f"{label}: choice function, role “{role}”", fontsize=9.5, x=0.02, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, f"{stem}_choice_function_{role}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cases_csv")
    args = ap.parse_args()
    rows = load(args.cases_csv)
    stem = str(Path(args.cases_csv).with_suffix("")).removesuffix("_cases")
    label = rows[0]["model"]
    scatter(rows, stem, label)
    absdiff_hist(rows, stem, label)
    for role in sorted({r["role"] for r in rows}, reverse=True):
        choice_function(rows, stem, label, role)


if __name__ == "__main__":
    main()
