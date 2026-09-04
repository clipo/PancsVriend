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
  fig3_role_overlay_*.png     (--role-overlay) the same two candidate sets with
                              RED and BLUE agent roles on shared axes, for one or
                              more endpoint arms at once (--overlay-arm, default
                              both grammar arms). Role = colour, arm = line style.

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

# Output resolution/format, set from --dpi/--format in main(). 300 is the print
# default; these grids are dense (up to 7 cols x 6 rows), so 150 was soft on the
# tick labels and bar annotations when zoomed. Use --format pdf/svg for vector
# output, which is resolution-independent and usually the right choice for a
# figure this size (dpi is then only used for any rasterized elements).
DPI = 300
EXT = "png"

# --- palette (validated categorical slots, light mode) ---
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, BASELINE = "#898781", "#e1e0d9", "#c3c2b7"
S1_BLUE, S2_AQUA, S3_YELLOW, S4_GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"

ARM_ORDER = ["completions", "chat", "compl+grammar", "chat+grammar"]
ARM_COLORS = {"completions": S1_BLUE, "chat": S2_AQUA,
              "compl+grammar": S3_YELLOW, "chat+grammar": S4_GREEN}

X = list(range(9))
# Every neighbourhood in THIS family is fully occupied (grid_with places 8
# neighbours and no empty cell), so the out-group COUNT and the out-group SHARE
# are the same variable: k of 8 = 12.5*k %. XPCT is that relabelling. Unlike the
# ratio sweep there are no sub-8 occupancies here, so no additional shares exist
# to pool in — the nine points are all this family can ever supply.
XPCT = [100.0 * k / 8.0 for k in X]
MECHANICAL = [0, 0, 0, 0, 0, 1, 1, 1, 1]
XAXIS = "percent"      # set from --x-axis in main(); "percent" or "count"


def xvals():
    return XPCT if XAXIS == "percent" else X

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

# A-family sources. Every full-suite run contains all 8 candidates, so the
# A-family is a FILTER on the same files fig1 uses — same arm, same N=100, same
# seed (paired layouts), and since the migration to the unified schema the same
# file also carries both roles. That is why FIG2 is just FIG1's labels: the
# A-family role overlay used to read red from the -arefine runs and blue from the
# -blue runs, which crossed a run boundary that had nothing to do with role.
FIG2_MODELS = [(m, dict(arms)) for m, arms in FIG1_MODELS]

# The dedicated red-only A-family replicate (Llama/Gemma, also 100/cell, run
# 2026-07-15 before the full suite covered A1-A4). Retained as an independent
# reproducibility check, NOT as a figure source — select it with --a-family-source
# arefine. It has no blue counterpart, so a role overlay on it is red-only.
AREFINE_MODELS = [
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


def _load_unified(path, role):
    """Long schema (candidate, agent_role, n_out, n_move, n_stay, n_bad, ...) —
    one file per arm carrying BOTH roles, written by evaluate_prompts.py and by
    unify_role_csvs.py. Counts are exact, so the effective rate is always
    available and 'eff' is unconditionally True."""
    curves, bad, n = {}, {}, None
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["agent_role"] != role:
                continue
            c = curves.setdefault(r["candidate"], [float("nan")] * len(X))
            k = int(r["n_out"])
            # Blank effective rate = every reply in the cell failed to parse.
            # Production would retry there, so it stays NaN (a gap in the curve)
            # rather than collapsing to 0 = "never moves".
            c[k] = float(r["move_rate_effective"]) if r["move_rate_effective"] else float("nan")
            bad[r["candidate"]] = bad.get(r["candidate"], 0) + int(r["n_bad"])
            n = int(r["n_samples"])
    if not curves:
        return None
    return {"curves": curves, "bad": bad, "n": n, "eff": True}


def _load_legacy_wide(path):
    """Superseded wide schema: one file per (arm, ROLE), role in the FILENAME,
    one row per candidate with move_rate_{k}of8 columns. Kept so the archived
    files under results/legacy_wide_per_role/ (and any un-migrated run) still
    plot. Prefers the EFFECTIVE columns when present; legacy red-sweep CSVs carry
    only the RAW single-shot rate, flagged via 'eff' so figures can say which
    quantity they show."""
    curves, bad, n, has_eff = {}, {}, None, False
    with path.open() as f:
        for r in csv.DictReader(f):
            if f"move_rate_eff_{X[0]}of8" in r and r[f"move_rate_eff_{X[0]}of8"] != "":
                has_eff = True
                curves[r["candidate"]] = [
                    float(v) if (v := r[f"move_rate_eff_{k}of8"]) != "" else float("nan")
                    for k in X]
            else:
                curves[r["candidate"]] = [float(r[f"move_rate_{k}of8"]) for k in X]
            bad[r["candidate"]] = int(r["bad_parses_total"])
            n = int(r["samples_per_cell"])
    return {"curves": curves, "bad": bad, "n": n, "eff": has_eff} if curves else None


def load(label, role="red"):
    """{'curves': {cand: [9 rates]}, 'bad': {cand: int}, 'n': ..., 'eff': bool} or None.

    Role is DATA, not a filename: the unified CSV for an arm holds both roles and
    is selected by the agent_role column. The legacy '<label>-blue.csv' fallback
    exists only for un-migrated files (see unify_role_csvs.py) — it is tried
    second so a migrated arm is never read from a stale per-role file."""
    unified = RESULTS / f"prompt_comparison_{label}.csv"
    if unified.exists():
        with unified.open() as f:
            header = csv.DictReader(f).fieldnames or []
        if "agent_role" in header:
            return _load_unified(unified, role)
    for cand_dir in (RESULTS, RESULTS / "legacy_wide_per_role"):
        p = cand_dir / f"prompt_comparison_{label}{'-blue' if role == 'blue' else ''}.csv"
        if p.exists():
            return _load_legacy_wide(p)
    return None


def style_curve_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_ylim(-0.05, 1.05)
    if XAXIS == "percent":
        # This family is always fully occupied, so only the EIGHTHS are reachable
        # here — a subset of neighbourhood_fracs(8), aligned with the ratio figures.
        tpos, tlab = pct_ticks([k / 8 for k in range(9)])
        ax.set_xticks(tpos)
        ax.set_xticklabels(tlab, fontsize=6, rotation=90)
        ax.set_xlim(-4, 104)
    else:
        ax.set_xticks(X)
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
    """Dashed mechanical reference: MOVE iff the out-group share exceeds 0.5.

    On the PERCENT axis the threshold is drawn at exactly 50%, the rule's actual
    cut point and the same place figR4 puts it, so the two figure families line
    up. On the COUNT axis it is drawn between 4 and 5 of 8 (where='mid'), because
    with 8 neighbours that is the finest the axis can resolve — 50% itself (4 of
    8) is a STAY, and the first MOVE is at 5."""
    if XAXIS == "percent":
        ax.step([0, 50, 50, 100], [0, 0, 1, 1], where="post", color=MUTED,
                linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    else:
        ax.step(X, MECHANICAL, where="mid", color=MUTED, linewidth=1.2,
                linestyle=(0, (4, 3)), zorder=1)


def bad_bars(ax, groups, series, total, colors=None, hatches=None):
    """Grouped bad-parse bars. `series` maps a series key (arm, or role x arm in
    the overlay figures) to one count per candidate; `colors` maps the same keys
    to colours, defaulting to the endpoint-arm palette. `hatches` optionally maps
    the same keys to a hatch pattern, so a second dimension (the endpoint arm)
    can share the colour axis with role without becoming unreadable."""
    colors = colors or ARM_COLORS
    hatches = hatches or {}
    gw = 0.8
    bw = gw / max(len(series), 1)
    ymax = max((v for s in series.values() for v in s), default=0)
    style_bar_ax(ax, ymax)
    for j, (key, vals) in enumerate(series.items()):
        xs = [i - gw / 2 + bw * (j + 0.5) for i in range(len(groups))]
        ax.bar(xs, vals, width=bw * 0.92, color=colors[key], zorder=3,
               hatch=hatches.get(key), edgecolor=SURFACE, linewidth=0.4)
        for x, v in zip(xs, vals):
            if v > 0:
                ax.annotate(str(v), (x, v), ha="center", va="bottom",
                            fontsize=6, color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=8, color=INK2)
    ax.set_ylabel(f"bad parses (of {total})", fontsize=8, color=INK2)
    if not ymax:   # otherwise an all-grammar figure shows a silently blank panel
        ax.annotate("no bad parses", (0.5, 0.5), xycoords="axes fraction",
                    ha="center", va="center", fontsize=8, color=MUTED)


def model_grid(out_dir, filename, model_specs, cands, titles, short, suptitle,
               rate="effective", role="red"):
    """Shared layout: rows = models, cols = candidates + bad-parse bars.

    rate="effective": plot ONLY arms whose CSV carries effective columns;
        arms without them are OMITTED and listed on the figure — never mixed.
    rate="raw_legacy": the complement — ONLY the arms lacking effective data,
        plotted as raw single-shot rates, in their own clearly-labeled figure.
    """
    rows, omitted = [], []
    for model, arm_labels in model_specs:
        arms = {a: load(lbl, role) for a, lbl in arm_labels.items()}
        arms = {a: d for a, d in arms.items()
                if d and any(c in d["curves"] for c in cands)}
        if rate == "effective":
            omitted += [f"{model}: {a}" for a, d in arms.items() if not d["eff"]]
            arms = {a: d for a, d in arms.items() if d["eff"]}
        else:
            arms = {a: d for a, d in arms.items() if not d["eff"]}
        if arms:
            rows.append((model, arms))
    if not rows:
        print(f"{filename} skipped: no data for rate={rate}")
        return

    ncols = len(cands) + 1
    fig, axes = plt.subplots(len(rows), ncols,
                             figsize=(3.1 * len(cands) + 3.6, 3.1 * len(rows) + 0.9),
                             dpi=DPI, squeeze=False,
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
                    ax.plot(xvals(), d["curves"][cand], color=ARM_COLORS[arm], linewidth=2,
                            marker="o", markersize=4, zorder=3)
            if row == 0:
                ax.set_title(titles[cand], fontsize=9.5, color=INK, pad=8)
            if col == 0:
                ax.set_ylabel(f"{model}\nP(MOVE)", fontsize=9, color=INK2)
            if row == len(rows) - 1:
                ax.set_xlabel("% of neighbours out-group (of 8)" if XAXIS == "percent"
                              else "out-group neighbours (of 8)", fontsize=8, color=INK2)
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
    rate_note = ("EFFECTIVE move rate n_move/(n_move+n_stay) — retry-equivalent, "
                 "matches production" if rate == "effective" else
                 "RAW single-shot move rate (bad parses in denominator) — legacy arms "
                 "without raw logs; NOT comparable to the effective figures")
    fig.suptitle(suptitle + "\n[rate shown: " + rate_note + "]",
                 fontsize=11.5, color=INK, y=0.995)
    if rate == "effective" and omitted:
        fig.text(0.5, 0.001,
                 "omitted (no effective data; see *_rawlegacy figure): " + "; ".join(omitted),
                 ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    path = out_dir / filename
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", dpi=DPI)
    print(f"wrote {path}  ({len(rows)} models; arms: {', '.join(present)})")


# Role encoding is deliberately literal (the teams ARE red and blue) and matches
# plot_ratio_results.py's figR4/figR6, so the two prompt families can be read
# side by side without relearning the colours.
ROLE_COLORS = {"red": "#d62728", "blue": "#1f77b4"}

# In the overlay figures ROLE owns the colour axis and the ENDPOINT ARM owns the
# line style / marker / hatch. That keeps role — the dominant effect (NOTES.md:
# "role asymmetry dominates everything") — as the most salient visual channel
# while still letting one panel carry more than one arm.
ARM_STYLES = {
    "completions":   {"ls": (0, (1, 1.1)),      "marker": "^", "hatch": ".."},
    "chat":          {"ls": (0, (5, 1, 1, 1)),  "marker": "v", "hatch": "xx"},
    "compl+grammar": {"ls": "-",                "marker": "o", "hatch": None},
    "chat+grammar":  {"ls": (0, (2.4, 1.4)),    "marker": "s", "hatch": "///"},
}


def neighbourhood_fracs(max_n=8):
    """Every distinct neighbour fraction reachable with at most `max_n` neighbours.

        fracs = [0]
        for x in range(1, 9):          # same-type neighbours
            for y in range(x, 9):      # total neighbours
                ...append x/y if new

    i.e. {0} u {x/y : 1 <= x <= y <= max_n} — the Farey sequence of order max_n.
    23 values for max_n=8. The set is symmetric under f -> 1-f, so it is the same
    whether the axis carries the SAME-type or the OUT-GROUP share, and it is
    exactly the set of shares the ratio sweep samples (verified against the
    loaded data, not assumed).

    These are the tick positions: marking them says where measurements can exist
    at all, and makes the uneven spacing of the reachable ratios visible instead
    of implying a uniform grid.
    """
    fracs = [0.0]
    for x in range(1, max_n + 1):
        for y in range(x, max_n + 1):
            f = x / y
            if f not in fracs:
                fracs.append(f)
    fracs.sort()
    return fracs


def pct_ticks(fracs, min_gap=0.0):
    """(positions in %, labels) — integer percents lose the decimal point.

    A TICK MARK is drawn at every reachable fraction. `min_gap` only suppresses
    the LABEL of a tick closer than that many percentage points to the previously
    labelled one: the reachable fractions cluster hard near 0 and 1 (12.5, 14.3,
    16.7 sit within four points of each other), so on a linear axis their labels
    overlap into an unreadable smear. Blanking a label loses nothing — the tick
    is still there and the position is still exact — whereas rescaling the axis
    to space them evenly would destroy the linear ratio metric that makes the 50%
    threshold meaningful.
    """
    pos = [100.0 * f for f in fracs]
    lab, last = [], None
    for v in pos:
        text = f"{v:.0f}%" if abs(v - round(v)) < 1e-9 else f"{v:.1f}%"
        if last is None or v - last >= min_gap:
            lab.append(text)
            last = v
        else:
            lab.append("")
    return pos, lab


def arm_ls(arm, arms):
    """Line style for an arm — solid when it is the only arm on the axes, so it
    does not compete with the dashed mechanical reference. See the ratio script's
    twin of this function; the two figure families stay visually in step."""
    return "-" if len(arms) == 1 else ARM_STYLES[arm]["ls"]


def arms_slug(arms):
    """Filename fragment for a set of arms — figures for different arm sets must
    not overwrite each other (an unsuffixed name silently clobbered the previous
    arm's figure once already, see the per-role naming note in the docstring)."""
    return "_".join(a.replace("+", "_") for a in arms)


def role_overlay(out_dir, filename, model_specs, cands, titles, short,
                 suptitle, arms):
    """Red vs blue role on the SAME axes, one panel per (model, candidate), for
    one or more endpoint arms at once.

    Roles are separate value functions and are never pooled — this figure only
    puts them on shared axes so the asymmetry is directly readable. Role is the
    colour, arm is the line style (see ARM_STYLES), so role x arm interactions
    are visible without confounding the two: e.g. whether the red-role freeze
    survives the switch from the completions to the chat endpoint.

    Only effective-rate data is plotted: a raw single-shot curve and an effective
    curve are different quantities, so overlaying them would fake a role effect
    out of a parse-health difference.
    """
    # Draw/legend order: arm-major, role-minor, so the two roles of one arm sit
    # next to each other in the legend and in each bar group.
    keys = [(role, arm) for arm in arms for role in ("red", "blue")]
    rows = []
    for model, arm_labels in model_specs:
        per_key = {}
        for role, arm in keys:
            d = load(arm_labels[arm], role) if arm in arm_labels else None
            if d and d["eff"] and any(c in d["curves"] for c in cands):
                per_key[(role, arm)] = d
        if per_key:
            rows.append((model, per_key))
    if not rows:
        print(f"{filename} skipped: no effective-rate data for arms={arms}")
        return

    ncols = len(cands) + 1
    fig, axes = plt.subplots(len(rows), ncols,
                             figsize=(3.1 * len(cands) + 3.6, 3.1 * len(rows) + 0.9),
                             dpi=DPI, squeeze=False,
                             gridspec_kw={"width_ratios": [1] * len(cands) + [1.15]})
    fig.patch.set_facecolor(SURFACE)
    present = []
    for row, (model, per_key) in enumerate(rows):
        for k in keys:
            if k in per_key and k not in present:
                present.append(k)
        for col, cand in enumerate(cands):
            ax = axes[row][col]
            style_curve_ax(ax)
            mech_ref(ax)
            for role, arm in keys:
                d = per_key.get((role, arm))
                if not d or cand not in d["curves"]:
                    continue
                st = ARM_STYLES[arm]
                ax.plot(xvals(), d["curves"][cand], color=ROLE_COLORS[role], linewidth=1.8,
                        linestyle=arm_ls(arm, arms), marker=st["marker"], markersize=4,
                        zorder=3)
            if row == 0:
                ax.set_title(titles[cand], fontsize=9.5, color=INK, pad=8)
            if col == 0:
                ax.set_ylabel(f"{model}\nP(MOVE)", fontsize=9, color=INK2)
            if row == len(rows) - 1:
                ax.set_xlabel("% of neighbours out-group (of 8)" if XAXIS == "percent"
                              else "out-group neighbours (of 8)", fontsize=8, color=INK2)
        # Same bad-parse panel as fig1/fig2, but grouped by ROLE x ARM: a curve is
        # only interpretable next to the parse health of the run that produced it.
        axb = axes[row][len(cands)]
        series = {f"{role} / {arm}": [per_key[(role, arm)]["bad"].get(c, 0) for c in cands]
                  for role, arm in keys if (role, arm) in per_key}
        colors = {f"{role} / {arm}": ROLE_COLORS[role] for role, arm in keys}
        hatches = {f"{role} / {arm}": ARM_STYLES[arm]["hatch"] for role, arm in keys}
        totals = {9 * d["n"] for d in per_key.values()}
        bad_bars(axb, [short[c] for c in cands], series,
                 total=max(totals), colors=colors, hatches=hatches)
        if len(totals) > 1:   # arms/roles sampled at different N — don't imply one denominator
            axb.set_ylabel("bad parses (per-run N differs)", fontsize=8, color=INK2)
        if row == 0:
            axb.set_title("bad parses", fontsize=9.5, color=INK, pad=8)
        if row == len(rows) - 1:
            axb.set_xlabel("candidate", fontsize=8, color=INK2)

    handles = [plt.Line2D([], [], color=ROLE_COLORS[role], linewidth=1.8,
                          linestyle=arm_ls(arm, arms),
                          marker=ARM_STYLES[arm]["marker"], markersize=4,
                          label=f"{role} role — {arm}") for role, arm in present]
    handles.append(plt.Line2D([], [], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
                              label="mechanical agent (Agent.py)"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5),
               frameon=False, fontsize=9, labelcolor=INK2,
               bbox_to_anchor=(0.5, -0.002))
    fig.suptitle(suptitle + f"\n[arms: {' | '.join(arms)} — role = colour, arm = line "
                 "style | rate shown: EFFECTIVE move rate n_move/(n_move+n_stay) — "
                 "retry-equivalent, matches production]",
                 fontsize=11.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0.035 + 0.012 * (len(handles) > 5), 1, 0.955))
    path = out_dir / filename
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", dpi=DPI)
    print(f"wrote {path}  ({len(rows)} models; arms={', '.join(arms)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(RESULTS / "figures"),
                    help="output directory for the figures (default: results/figures/)")
    ap.add_argument("--role", default="red", choices=["red", "blue"],
                    help="agent role for the single-role fig1/fig2. Since the "
                         "unified-schema migration this selects the agent_role COLUMN "
                         "of one CSV per arm; it is no longer a filename suffix. "
                         "Replaces the old --label-suffix=-blue.")
    ap.add_argument("--a-family-source", default="full-suite",
                    choices=["full-suite", "arefine"],
                    help="which runs back the A-family figures. 'full-suite' (default) "
                         "reads the same files as fig1 and therefore has BOTH roles; "
                         "'arefine' reads the dedicated red-only A-family replicate.")
    ap.add_argument("--role-overlay", action="store_true",
                    help="instead of the single-role fig1/fig2, write fig3_role_overlay_* "
                         "putting the red and blue roles on shared axes. Ignores --role "
                         "(it draws both).")
    ap.add_argument("--x-axis", default="percent", choices=["percent", "count"],
                    help="x-axis units. 'percent' (default) shows the out-group SHARE, "
                         "matching the canonical Schelling value function and the ratio "
                         "figures; 'count' shows the raw 0-8 out-group count. Same nine "
                         "data points either way — this family is always fully occupied.")
    ap.add_argument("--overlay-arm", nargs="+", default=["chat+grammar"],
                    choices=ARM_ORDER, metavar="ARM",
                    help="endpoint arm(s) drawn in the role-overlay figures; each is "
                         "shown for both roles on the same axes. Default chat+grammar: "
                         "grammar removes parse artifacts, and the chat endpoint is the "
                         "only grammar channel free of the max_tokens truncation that "
                         "affects compl+grammar on Gemma (grammar_truncation.md). Pass "
                         "more than one arm to overlay them as different line styles.")
    ap.add_argument("--dpi", type=int, default=DPI,
                    help=f"raster resolution (default {DPI}). Ignored for vector formats "
                         "except for any rasterized elements.")
    ap.add_argument("--format", default=EXT, choices=["png", "pdf", "svg"],
                    help="output format. pdf/svg are vector (resolution-independent) "
                         "and preferable for these dense grids at print size.")
    args = ap.parse_args()
    globals()["DPI"] = args.dpi
    globals()["EXT"] = args.format
    globals()["XAXIS"] = args.x_axis
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    a_specs = FIG2_MODELS if args.a_family_source == "full-suite" else AREFINE_MODELS
    a_note = ("" if args.a_family_source == "full-suite"
              else " [source: dedicated -arefine replicate, red-only]")

    if args.role_overlay:
        arms = [a for a in ARM_ORDER if a in args.overlay_arm]   # canonical order
        slug = arms_slug(arms)
        role_overlay(out_dir, f"fig3_role_overlay_original_{slug}.{EXT}",
                     FIG1_MODELS, FIG1_CANDS, FIG1_TITLES, FIG1_SHORT,
                     "Original candidates — RED vs BLUE agent role on shared axes"
                     "\n(production payload, random paired layouts, T=0.3)", arms)
        role_overlay(out_dir, f"fig3_role_overlay_afamily_{slug}.{EXT}",
                     a_specs, FIG2_CANDS, FIG2_TITLES, FIG2_SHORT,
                     "A-refinement family — RED vs BLUE agent role on shared axes"
                     f"\n(production payload, random paired layouts, T=0.3){a_note}", arms)
        return

    role = args.role
    fig_sfx = "" if role == "red" else f"_{role}"
    role_note = f" — ROLE: {role} agent"
    fig1_title = ("Original candidates — MOVE rate across the out-group gradient, by "
                  f"endpoint × grammar arm, and total bad parses{role_note}\n"
                  "(production payload, random paired layouts, 100 samples/cell except "
                  "Gemma-chat at 50, T=0.3; chat = reasoning off where applicable)")
    fig2_title = ("A-refinement family — frozen baseline A vs single-change variants, by "
                  f"endpoint × grammar arm, and total bad parses{role_note}\n"
                  "(production payload, random paired layouts, 100 samples/cell, T=0.3; "
                  f"chat = reasoning off where applicable){a_note}")
    for name, specs, cands, titles, short, title in (
            ("fig1_endpoint_grammar", FIG1_MODELS, FIG1_CANDS, FIG1_TITLES, FIG1_SHORT, fig1_title),
            ("fig2_a_family", a_specs, FIG2_CANDS, FIG2_TITLES, FIG2_SHORT, fig2_title)):
        model_grid(out_dir, f"{name}{fig_sfx}.{EXT}", specs, cands, titles, short,
                   title, rate="effective", role=role)
        # The raw-legacy complement is empty once an arm is migrated (exact counts
        # give every cell an effective rate); it stays wired up for un-migrated runs.
        model_grid(out_dir, f"{name}{fig_sfx}_rawlegacy.{EXT}", specs, cands, titles,
                   short, title, rate="raw_legacy", role=role)


if __name__ == "__main__":
    main()
