#!/usr/bin/env python3
"""Plot the prompt-comparison results in results/ as move-rate curves + parse-error bars.

    python prompt_refinement/plot_results.py            # writes results/figures/*.png
    python prompt_refinement/plot_results.py --out-dir /tmp/figs

Two figures, both in the same layout (rows = models, line colours = the four
endpoint x grammar arms, right-most column = bad-parse bars). Any model or arm
whose CSV does not exist yet is skipped, so re-running refreshes the figures as
new results land.

  fig1_endpoint_grammar.png   original candidates (0 / A / B / C), every model.
  fig2_a_family.png           the A-refinement family (A frozen baseline + A1-A4
                              single-change variants), every model. Replaces the
                              old fig2/fig3/fig4 (superseded 2026-07-15).

Arms: completions / chat / completions+grammar / chat+grammar. "chat" for Gemma,
Qwen and DeepSeek means the server ran with --reasoning off (their chat templates
otherwise open a reasoning channel; raw completions is unaffected by the flag).

The dashed step in every curve panel is the mechanical Agent.py reference: with
SIMILARITY_THRESHOLD = 0.5 the agent wants to move iff >0.5 of its 8 neighbours
are out-group, i.e. at 5..8 of 8 (see Agent.utility / best_response).

Samples/cell: original-candidate arms for Llama/Gemma were run at 50; everything
else at 100. The per-row bar panel states its own denominator.

Bad parses = AMBIGUOUS + UNPARSEABLE replies over the whole gradient; each one
costs a retry in the production runner. Per-sample raw replies for every run are
in results/raw/ (one .jsonl.gz per label) for post-hoc analysis.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"

# --- palette (validated categorical slots, light mode) ---
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, BASELINE = "#898781", "#e1e0d9", "#c3c2b7"
S1_BLUE, S2_AQUA, S3_YELLOW, S4_GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"

ARM_ORDER = ["completions", "chat", "compl+grammar", "chat+grammar"]
ARM_COLORS = {"completions": S1_BLUE, "chat": S2_AQUA,
              "compl+grammar": S3_YELLOW, "chat+grammar": S4_GREEN}

X = list(range(9))
MECHANICAL = [0, 0, 0, 0, 0, 1, 1, 1, 1]

# arm -> results label, per model. Loader skips labels whose CSV is absent.
FIG1_MODELS = [
    ("Llama-3.3-70B", {
        "completions":   "llama-3.3-70b-instruct-q4_k_m",
        "chat":          "llama-3.3-70b-instruct-q4_k_m-chat",
        "compl+grammar": "llama-3.3-70b-instruct-q4_k_m-grammar",
        "chat+grammar":  "llama-3.3-70b-instruct-q4_k_m-chat-grammar",
    }),
    ("Gemma-4-31B", {
        "completions":   "gemma-4-31b-it-q5_k_m",
        "chat":          "gemma-4-31b-it-q5_k_m-chat-noreason",
        "compl+grammar": "gemma-4-31b-it-q5_k_m-grammar",
        "chat+grammar":  "gemma-4-31b-it-q5_k_m-chat-noreason-grammar",
    }),
    ("Qwen3.6-27B", {
        "completions":   "qwen3.6-27b-q5_k_m",
        "chat":          "qwen3.6-27b-q5_k_m-chat",
        "compl+grammar": "qwen3.6-27b-q5_k_m-grammar",
        "chat+grammar":  "qwen3.6-27b-q5_k_m-chat-grammar",
    }),
    ("Mistral-Small-4-119B", {
        "completions":   "mistral-small-4-119b-ud-q4_k_m",
        "chat":          "mistral-small-4-119b-ud-q4_k_m-chat",
        "compl+grammar": "mistral-small-4-119b-ud-q4_k_m-grammar",
        "chat+grammar":  "mistral-small-4-119b-ud-q4_k_m-chat-grammar",
    }),
    ("DeepSeek-V4-Flash", {
        "completions":   "deepseek-v4-flash-ud-iq3_xxs",
        "chat":          "deepseek-v4-flash-ud-iq3_xxs-chat",
        "compl+grammar": "deepseek-v4-flash-ud-iq3_xxs-grammar",
        "chat+grammar":  "deepseek-v4-flash-ud-iq3_xxs-chat-grammar",
    }),
]

# A-family sources. Llama/Gemma have dedicated -arefine* runs (A-family only,
# 100/cell); the newer models' full-suite CSVs contain all 8 candidates and are
# simply filtered down to the A-family when plotting.
FIG2_MODELS = [
    ("Llama-3.3-70B", {
        "completions":   "llama-3.3-70b-instruct-q4_k_m-arefine",
        "chat":          "llama-3.3-70b-instruct-q4_k_m-arefine-chat",
        "compl+grammar": "llama-3.3-70b-instruct-q4_k_m-arefine-grammar",
        "chat+grammar":  "llama-3.3-70b-instruct-q4_k_m-arefine-chat-grammar",
    }),
    ("Gemma-4-31B", {
        "completions":   "gemma-4-31b-it-q5_k_m-arefine",
        "chat":          "gemma-4-31b-it-q5_k_m-arefine-chat",
        "compl+grammar": "gemma-4-31b-it-q5_k_m-arefine-grammar",
        "chat+grammar":  "gemma-4-31b-it-q5_k_m-arefine-chat-grammar",
    }),
    ("Qwen3.6-27B", dict(FIG1_MODELS[2][1])),
    ("Mistral-Small-4-119B", dict(FIG1_MODELS[3][1])),
    ("DeepSeek-V4-Flash", dict(FIG1_MODELS[4][1])),
]

FIG1_CANDS = ["0_current", "A_briefing_map_ask", "B_briefing_map_ask_rule",
              "C_legend_after_grid"]
FIG1_TITLES = {"0_current": "0_current", "A_briefing_map_ask": "A (briefing→map→ask)",
               "B_briefing_map_ask_rule": "B (rule after ask)",
               "C_legend_after_grid": "C (legend after grid)"}
FIG1_SHORT = {"0_current": "0", "A_briefing_map_ask": "A",
              "B_briefing_map_ask_rule": "B", "C_legend_after_grid": "C"}

FIG2_CANDS = ["A_briefing_map_ask", "A1_min_tail", "A2_ask_above_grid",
              "A3_stay_or_move", "A4_no_persona"]
FIG2_TITLES = {"A_briefing_map_ask": "A — baseline (frozen)",
               "A1_min_tail": "A1 — minimal tail",
               "A2_ask_above_grid": "A2 — ask above grid",
               "A3_stay_or_move": "A3 — “stay or move?”",
               "A4_no_persona": "A4 — no persona"}
FIG2_SHORT = {"A_briefing_map_ask": "A", "A1_min_tail": "A1", "A2_ask_above_grid": "A2",
              "A3_stay_or_move": "A3", "A4_no_persona": "A4"}


def load(label):
    """{'curves': {cand: [9 rates]}, 'bad': {cand: int}, 'n': samples/cell} or None."""
    p = RESULTS / f"prompt_comparison_{label}.csv"
    if not p.exists():
        return None
    curves, bad, n = {}, {}, None
    with p.open() as f:
        for r in csv.DictReader(f):
            curves[r["candidate"]] = [float(r[f"move_rate_{k}of8"]) for k in X]
            bad[r["candidate"]] = int(r["bad_parses_total"])
            n = int(r["samples_per_cell"])
    return {"curves": curves, "bad": bad, "n": n}


def style_curve_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xticks(X)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.3, 8.3)


def style_bar_ax(ax, ymax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_ylim(0, ymax * 1.18 if ymax else 1)


def mech_ref(ax):
    ax.step(X, MECHANICAL, where="mid", color=MUTED, linewidth=1.2,
            linestyle=(0, (4, 3)), zorder=1)


def bad_bars(ax, groups, series, total):
    gw = 0.8
    bw = gw / max(len(series), 1)
    ymax = max((v for s in series.values() for v in s), default=0)
    style_bar_ax(ax, ymax)
    for j, (arm, vals) in enumerate(series.items()):
        xs = [i - gw / 2 + bw * (j + 0.5) for i in range(len(groups))]
        ax.bar(xs, vals, width=bw * 0.92, color=ARM_COLORS[arm], zorder=3)
        for x, v in zip(xs, vals):
            if v > 0:
                ax.annotate(str(v), (x, v), ha="center", va="bottom",
                            fontsize=6, color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=8, color=INK2)
    ax.set_ylabel(f"bad parses (of {total})", fontsize=8, color=INK2)


def model_grid(out_dir, filename, model_specs, cands, titles, short, suptitle):
    """Shared layout: rows = models, cols = candidates + bad-parse bars."""
    rows = []
    for model, arm_labels in model_specs:
        arms = {a: load(lbl) for a, lbl in arm_labels.items()}
        arms = {a: d for a, d in arms.items()
                if d and any(c in d["curves"] for c in cands)}
        if arms:
            rows.append((model, arms))
    if not rows:
        print(f"{filename} skipped: no data")
        return

    ncols = len(cands) + 1
    fig, axes = plt.subplots(len(rows), ncols,
                             figsize=(3.1 * len(cands) + 3.6, 3.1 * len(rows) + 0.9),
                             dpi=150, squeeze=False,
                             gridspec_kw={"width_ratios": [1] * len(cands) + [1.15]})
    fig.patch.set_facecolor(SURFACE)
    present = []
    for row, (model, arms) in enumerate(rows):
        for a in ARM_ORDER:
            if a in arms and a not in present:
                present.append(a)
        for col, cand in enumerate(cands):
            ax = axes[row][col]
            style_curve_ax(ax)
            mech_ref(ax)
            for arm in ARM_ORDER:
                d = arms.get(arm)
                if d and cand in d["curves"]:
                    ax.plot(X, d["curves"][cand], color=ARM_COLORS[arm], linewidth=2,
                            marker="o", markersize=4, zorder=3)
            if row == 0:
                ax.set_title(titles[cand], fontsize=9.5, color=INK, pad=8)
            if col == 0:
                ax.set_ylabel(f"{model}\nP(MOVE)", fontsize=9, color=INK2)
            if row == len(rows) - 1:
                ax.set_xlabel("out-group neighbours (of 8)", fontsize=8, color=INK2)
        axb = axes[row][len(cands)]
        n = next(iter(arms.values()))["n"]
        series = {arm: [arms[arm]["bad"].get(c, 0) for c in cands]
                  for arm in ARM_ORDER if arm in arms}
        bad_bars(axb, [short[c] for c in cands], series, total=9 * n)
        if row == 0:
            axb.set_title("bad parses", fontsize=9.5, color=INK, pad=8)
        if row == len(rows) - 1:
            axb.set_xlabel("candidate", fontsize=8, color=INK2)

    handles = [plt.Line2D([], [], color=ARM_COLORS[a], linewidth=2, marker="o",
                          markersize=4, label=a) for a in ARM_ORDER if a in present]
    handles.append(plt.Line2D([], [], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
                              label="mechanical agent (Agent.py)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, -0.002))
    fig.suptitle(suptitle, fontsize=11.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    path = out_dir / filename
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {path}  ({len(rows)} models; arms: {', '.join(present)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(RESULTS / "figures"),
                    help="output directory for the PNGs (default: results/figures/)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_grid(out_dir, "fig1_endpoint_grammar.png",
               FIG1_MODELS, FIG1_CANDS, FIG1_TITLES, FIG1_SHORT,
               "Original candidates — MOVE rate across the out-group gradient, by "
               "endpoint × grammar arm, and total bad parses\n"
               "(production payload, random paired layouts, T=0.3; Llama/Gemma arms "
               "50 samples/cell, newer models 100; chat = reasoning off where applicable)")

    model_grid(out_dir, "fig2_a_family.png",
               FIG2_MODELS, FIG2_CANDS, FIG2_TITLES, FIG2_SHORT,
               "A-refinement family — frozen baseline A vs single-change variants, by "
               "endpoint × grammar arm, and total bad parses\n"
               "(production payload, random paired layouts, 100 samples/cell, T=0.3; "
               "chat = reasoning off where applicable)")


if __name__ == "__main__":
    main()
