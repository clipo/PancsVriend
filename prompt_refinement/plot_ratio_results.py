#!/usr/bin/env python3
"""Plot the ratio-prompt sweep results.

    python prompt_refinement/plot_ratio_results.py            # -> results/figures/
    python prompt_refinement/plot_ratio_results.py --out-dir /tmp/figs

Reads every results/ratio_comparison_<label>.csv (long format, written by
evaluate_ratio_prompts.py) and produces:

    figR1_occ8_gradient_<role>.png  rows = models, cols = candidates; P(MOVE)
                              along the classic n_occ=8 gradient, one line per
                              arm, dashed mechanical step (move iff >4 of 8
                              opposite). Backward-comparable with plot_results.py.
    figR2_surfaces_<role>.png P(MOVE | n_similar, n_occupied) heatmaps, one
                              panel per (model, candidate) at a chosen arm —
                              the sampled value function itself.
    figR3_thresholds_<role>.png  implied P=0.5 threshold at n_occ=8 per
                              (model, candidate, arm) vs the mechanical 4.5.
    figR4_ratio_collapse_<arms>.png  both roles overlaid on the opposite-neighbor
                              ratio axis (role IS the comparison here, so this
                              one file covers both and carries no role suffix);
                              the arm set is in the filename since --overlay-arm
                              can draw more than one.
    figR6_role_arm_gradient_<arms>.png  the figR1 gradient transposed: both roles
                              AND the selected arms on shared axes (role = colour,
                              arm = line style), with the mechanical step. This is
                              the figure for "how far apart are the two roles, and
                              does the endpoint arm move that gap".
    figR5_bad_parses_<role>.png  parse failures per candidate x arm — the
                              counterpart of fig1/fig2's bar panel in
                              plot_results.py, and the evidence behind the
                              voluntary/forced-choice verdicts.

R1-R3 are per-role and MUST carry the role in the filename: roles are separate
value functions that are never pooled, and an unsuffixed name meant a second
`--role` invocation silently overwrote the first role's figures.

The outcome-space DI comparison figure is added at M3 once the ratio-policy
simulations exist. Same visual conventions as plot_results.py: arms are line
colors, mechanical reference is a dashed black step, missing files are skipped
so the figures refresh as sweep results land.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ratio_consistency import (  # noqa: E402
    LABEL_RE, implied_threshold, load_all, occ8_gradient, surface,
)
from ratio_prompt_templates import RATIO_CANDIDATES, mechanical_move  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Output resolution/format, set from --dpi/--format in main(). See the matching
# block in plot_results.py — the two scripts stay in step so a figure set is
# rendered consistently. pdf/svg are vector and preferable at print size.
DPI = 300
EXT = "png"

MODEL_ORDER = ["llama-3.3-70b", "gemma-4-31b", "qwen3.6-27b",
               "mistral-small-4-119b", "deepseek-v4-flash"]

# Server-configuration probes that the loader sees as extra "models" because the
# label carries them, but which are NOT models: same weights, same sampler, same
# prompts, only the chat wrapping differs. Excluded from the figures by default
# (--include-probes brings them back); the CSVs and raw logs are kept, since they
# are the evidence for the equivalence.
#
# llama-3.3-70b-nojinja: --no-jinja drops the date/knowledge-cutoff system block
# that Llama's embedded template auto-inserts (src/llama-chat.cpp:485-493).
# Measured 2026-08-21 against the jinja default: per-cell mean |d| 0.0004-0.026
# over all styles/roles/chat arms, implied thresholds within 0.05 of a neighbour
# in 9 of 12 (style, role) pairs, 0.00% bad parses either way. Equivalent, so
# carrying it as a sixth row only cost space. See NOTES.md.
PROBE_MODELS = {"llama-3.3-70b-nojinja"}


def drop_probes(df, include_probes=False):
    """Filter server-config probe arms out of a loaded frame (see PROBE_MODELS).

    Applied once at load time in main() so EVERY figure gets the same model set —
    filtering per-figure is how a probe ends up in one panel grid and not another.
    """
    if include_probes or df is None:
        return df
    return df[~df.model.isin(PROBE_MODELS)]
ARM_ORDER = ["completions", "chat", "completions+grammar", "chat+grammar"]
ARM_COLORS = {"completions": "#1f77b4", "chat": "#d62728",
              "completions+grammar": "#2ca02c", "chat+grammar": "#ff7f0e"}
MECH_OCC8 = [mechanical_move(8 - k, 8) for k in range(9)]   # 0..8 opposite

# In the role-overlay figures (R4, R6) ROLE owns the colour axis and the ENDPOINT
# ARM owns the line style / marker / hatch — role asymmetry is the dominant effect
# (NOTES.md), so it gets the most salient channel. Mirrors ARM_STYLES in
# plot_results.py so the two prompt families read the same way.
ARM_STYLES = {
    "completions":         {"ls": (0, (1, 1.1)),     "marker": "^", "hatch": ".."},
    "chat":                {"ls": (0, (5, 1, 1, 1)), "marker": "v", "hatch": "xx"},
    "completions+grammar": {"ls": "-",               "marker": "o", "hatch": None},
    "chat+grammar":        {"ls": (0, (2.4, 1.4)),   "marker": "s", "hatch": "///"},
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
    """Line style for an arm. When only ONE arm is drawn there is no arm
    dimension to encode, so it goes solid — a dashed series would otherwise
    compete with the dashed mechanical reference for no information gain."""
    return "-" if len(arms) == 1 else ARM_STYLES[arm]["ls"]


def arms_slug(arms):
    """Filename fragment for a set of arms — figures for different arm sets must
    not overwrite each other (same rule as the per-role suffix above)."""
    return "_".join(a.replace("+", "_") for a in arms)


def _model_sort(models):
    return sorted(models, key=lambda m: (MODEL_ORDER.index(m)
                                         if m in MODEL_ORDER else 99, m))


SHORT_CAND = {"R1_count_opposite": "R1", "R2_count_similar": "R2",
              "R3_dual_count": "R3", "R4_percent_opposite": "R4",
              "R5_percent_similar": "R5", "G0_grid_anchor": "G0"}


def _bad_parse_panel(ax, df, model, cands, xlabel=False, title=False):
    """One model's parse failures per candidate x arm, as % of samples.

    Shared by the in-figure column of figR1 and the standalone figR5 so the two
    cannot drift apart. Percentage rather than raw count so arms stay comparable
    when N per cell differs; the 5% line is the voluntary/forced-choice cutoff
    used by analyze_ratio_consistency.py.
    """
    arms = [a for a in ARM_ORDER if not df[(df.model == model) & (df.arm == a)].empty]
    gw = 0.8
    bw = gw / max(len(arms), 1)
    for j, arm in enumerate(arms):
        pcts = []
        for cand in cands:
            g = df[(df.model == model) & (df.arm == arm) & (df.candidate == cand)]
            tot = float(g.n_samples.sum())
            pcts.append(100.0 * float(g.n_bad.sum()) / tot if tot else 0.0)
        xs = [k - gw / 2 + bw * (j + 0.5) for k in range(len(cands))]
        ax.bar(xs, pcts, width=bw * 0.92, color=ARM_COLORS[arm], zorder=3)
    ax.axhline(5.0, color="black", ls=":", lw=1.0, alpha=0.6, zorder=2)
    ax.set_xticks(range(len(cands)))
    ax.set_xticklabels([SHORT_CAND.get(c, c) for c in cands], fontsize=8)
    ax.set_ylabel("% unparseable", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    if title:
        ax.set_title("bad parses (dotted = 5%)", fontsize=9)
    if xlabel:
        ax.set_xlabel("candidate", fontsize=8)


def fig_occ8_gradient(df, out_path, role):
    models = _model_sort(df.model.unique())
    cands = [c for c in RATIO_CANDIDATES if c in set(df.candidate)]
    # Rightmost column is the bad-parse bar panel, matching plot_results.py's
    # fig1/fig2 layout: a P(MOVE) curve is only interpretable next to the parse
    # health of the run that produced it. sharey is deliberately NOT used — the
    # bar panel is a percentage axis, not a probability axis.
    ncols = len(cands) + 1
    fig, axes = plt.subplots(len(models), ncols,
                             figsize=(3.2 * len(cands) + 3.8, 2.6 * len(models)),
                             squeeze=False,
                             gridspec_kw={"width_ratios": [1] * len(cands) + [1.3]})
    x = np.arange(9)
    for i, model in enumerate(models):
        for j, cand in enumerate(cands):
            ax = axes[i][j]
            ax.step(x, MECH_OCC8, where="mid", color="black", ls="--",
                    lw=1.0, alpha=0.6)
            for arm in ARM_ORDER:
                grp = df[(df.model == model) & (df.candidate == cand)
                         & (df.arm == arm)]
                if grp.empty:
                    continue
                grad = occ8_gradient(surface(grp))
                ax.plot(x, grad, color=ARM_COLORS[arm], lw=1.6, marker="o",
                        ms=3, label=arm)
            if i == 0:
                ax.set_title(cand, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{model}\neffective P(MOVE)", fontsize=8)
            if i == len(models) - 1:
                ax.set_xlabel("opposite neighbors (of 8)", fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
        _bad_parse_panel(axes[i][len(cands)], df, model, cands,
                         xlabel=(i == len(models) - 1), title=(i == 0))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(ARM_ORDER),
               fontsize=9, frameon=False)
    fig.suptitle(f"ROLE: {role} agent — EFFECTIVE P(MOVE) = n_move/(n_move+n_stay) "
                 "along the fully-occupied gradient — dashed = mechanical",
                 fontsize=13, color=ROLE_COLORS.get(role, "black"))
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def fig_surfaces(df, arm, out_path, role):
    sub = df[df.arm == arm]
    if sub.empty:
        return False
    models = _model_sort(sub.model.unique())
    cands = [c for c in RATIO_CANDIDATES if c in set(sub.candidate)]
    # Color range 0..max OBSERVED (not 0..1): move-averse regimes live in a
    # narrow band and a fixed 0..1 scale hides all their structure.
    vmax = max(float(sub.move_rate_effective.max()), 1e-9)
    fig, axes = plt.subplots(len(models), len(cands),
                             figsize=(3.0 * len(cands), 2.8 * len(models)),
                             squeeze=False)
    for i, model in enumerate(models):
        for j, cand in enumerate(cands):
            ax = axes[i][j]
            grp = sub[(sub.model == model) & (sub.candidate == cand)]
            mat = np.full((9, 9), np.nan)          # rows n_occ, cols n_sim
            for r in grp.itertuples():
                mat[r.n_occupied, r.n_similar] = r.move_rate_effective
            im = ax.imshow(mat, origin="lower", vmin=0, vmax=vmax,
                           cmap="RdYlBu_r", aspect="equal")
            ax.set_xticks(range(0, 9, 2)); ax.set_yticks(range(0, 9, 2))
            if i == 0:
                ax.set_title(cand, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{model}\nn_occupied", fontsize=8)
            if i == len(models) - 1:
                ax.set_xlabel("n_similar", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.6, label=f"P(MOVE)  [0 .. {vmax:.2f} observed max]")
    fig.suptitle(f"ROLE: {role} agent — sampled value functions, EFFECTIVE "
                 f"P(MOVE | n_similar, n_occupied) — {arm} "
                 f"(color scaled to observed max {vmax:.2f})", fontsize=13,
                 color=ROLE_COLORS.get(role, "black"))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return True


ROLE_COLORS = {"red": "#d62728", "blue": "#1f77b4"}


def fig_ratio_collapse(df_by_role, arms, out_path):
    """P(MOVE) vs the OUT-GROUP SHARE (n_opposite / n_occupied, as a PERCENTAGE)
    — every sampled composition collapsed onto the canonical Schelling x-axis.

    This is the whole 45-cell surface projected onto one line, not the n_occ=8
    slice: compositions with the SAME out-group share pool together regardless of
    how many neighbours the agent has, which is the quantity the classic Schelling
    value function is drawn against. 4/8, 3/6, 2/4 and 1/2 are all one point at
    50%; 2/8 and 1/4 are one point at 25%. 44 non-empty cells -> 23 distinct
    shares. The zero-neighbour cell has no defined share and is dropped (STAY by
    construction — see mechanical_move).

    Pooling is by COUNTS, not by averaging per-cell rates:
    sum(n_move) / sum(n_move + n_stay) over the cells sharing a share. A mean of
    rates would give a cell with two valid replies the same weight as one with
    fifty; at 0% and 100% eight cells pool, so the difference is not academic.

    Dots are the individual cells behind each pooled point (size = occupancy), so
    the spread within one share stays visible: if 1/2 and 4/8 disagree, the share
    is not a sufficient statistic for this model and the pooled line is hiding it.
    Roles overlay in red/blue; each arm gets its own line style.
    """
    frames = [d.assign(role=role) for role, d in df_by_role.items() if d is not None]
    if not frames:
        return False
    sub = pd.concat(frames)[lambda d: d.arm.isin(arms)]
    if sub.empty:
        return False
    sub = sub[sub.n_occupied > 0].copy()
    sub["opp_pct"] = 100.0 * sub.n_opposite / sub.n_occupied
    keys = [(role, arm) for arm in arms for role in ("red", "blue")]
    keys = [k for k in keys if not sub[(sub.role == k[0]) & (sub.arm == k[1])].empty]
    show_dots = len(arms) == 1
    models = _model_sort(sub.model.unique())
    cands = [c for c in RATIO_CANDIDATES if c in set(sub.candidate)]
    fig, axes = plt.subplots(len(models), len(cands),
                             figsize=(3.9 * len(cands), 2.8 * len(models)),
                             squeeze=False, sharex=True, sharey=True)
    for i, model in enumerate(models):
        for j, cand in enumerate(cands):
            ax = axes[i][j]
            ax.step([0, 50, 50, 100], [0, 0, 1, 1], where="post",
                    color="black", ls="--", lw=1.0, alpha=0.5,
                    label="mechanical agent")
            grp_all = sub[(sub.model == model) & (sub.candidate == cand)]
            for role, arm in keys:
                grp = grp_all[(grp_all.role == role) & (grp_all.arm == arm)]
                if grp.empty:
                    continue
                col, st = ROLE_COLORS[role], ARM_STYLES[arm]
                ls = arm_ls(arm, arms)
                if show_dots:
                    ax.scatter(grp.opp_pct, grp.move_rate_effective,
                               s=8 + 3 * grp.n_occupied, c=[col], alpha=0.30,
                               edgecolors="none", zorder=2)
                g = grp.groupby("opp_pct")[["n_move", "n_stay"]].sum()
                valid = g.n_move + g.n_stay
                pooled = (g.n_move / valid.where(valid > 0)).dropna()
                ax.plot(pooled.index, pooled.values, color=col, lw=1.8, ls=ls,
                        marker=st["marker"], ms=3, zorder=3,
                        label=f"{role} role — {arm}")
            if i == 0:
                ax.set_title(cand, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{model}\neffective P(MOVE)", fontsize=8)
            if i == len(models) - 1:
                ax.set_xlabel("% of neighbours out-group", fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(-3, 103)
            tpos, tlab = pct_ticks(neighbourhood_fracs(8), min_gap=3.4)
            ax.set_xticks(tpos)
            ax.set_xticklabels(tlab, fontsize=6, rotation=90)
            # Gridlines only on the coarse landmarks: 23 vertical rules would read
            # as texture rather than reference.
            ax.grid(alpha=0.0)
            for v in (0, 25, 50, 75, 100):
                ax.axvline(v, color="0.85", lw=0.6, zorder=0)
            ax.grid(alpha=0.3, axis="y")
    handles, labels = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in labels:
                handles.append(hh); labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=min(max(len(labels), 1), 4),
               fontsize=9, frameon=False)
    dot_note = ("dots = individual cells, size = occupancy; " if show_dots else
                "per-cell dots omitted (multi-arm); ")
    fig.suptitle("EFFECTIVE P(MOVE) vs OUT-GROUP SHARE — all 44 non-empty compositions "
                 f"collapsed (23 distinct shares) — arms: {' | '.join(arms)}\n"
                 f"(equal shares pooled by counts: 4/8 = 3/6 = 2/4 = 1/2 = 50%; "
                 f"role = colour; {dot_note}dashed = mechanical threshold at 50%)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return True


def fig_bad_parses(df, out_path, role):
    """Bad parses per (model, candidate, arm) — the family-1 fig1/fig2 bar panel,
    which the ratio figures were missing entirely.

    This is the evidence behind the voluntary/forced-choice split in
    analyze_ratio_consistency.py: a plain-endpoint arm that fails to parse most
    of the time has a grammar counterpart that is protocol-constructed, not a
    reading of preference. Shown as a % of samples so the arms stay comparable
    when N per cell differs, with the raw count annotated.
    """
    models = _model_sort(df.model.unique())
    cands = [c for c in RATIO_CANDIDATES if c in set(df.candidate)]
    fig, axes = plt.subplots(len(models), 1, squeeze=False,
                            figsize=(1.7 * len(cands) + 3.0, 2.5 * len(models)))
    gw = 0.8
    for i, model in enumerate(models):
        ax = axes[i][0]
        _bad_parse_panel(ax, df, model, cands,
                         xlabel=(i == len(models) - 1), title=False)
        # Standalone version is roomier than figR1's column, so it can afford
        # full style names and the raw counts on top of each bar.
        ax.set_xticklabels(cands, fontsize=8)
        ax.set_ylabel(f"{model}\n% samples unparseable", fontsize=8)
        arms = [a for a in ARM_ORDER
                if not df[(df.model == model) & (df.arm == a)].empty]
        bw = gw / max(len(arms), 1)
        for j, arm in enumerate(arms):
            for k, cand in enumerate(cands):
                g = df[(df.model == model) & (df.arm == arm) & (df.candidate == cand)]
                bad, tot = float(g.n_bad.sum()), float(g.n_samples.sum())
                pct = 100.0 * bad / tot if tot else 0.0
                if pct > 0:
                    ax.annotate(f"{int(bad)}", (k - gw / 2 + bw * (j + 0.5), pct),
                                ha="center", va="bottom", fontsize=6, color="#444444")
        if i == 0:
            ax.legend(handles=[plt.Line2D([], [], color=ARM_COLORS[a], lw=6, label=a)
                               for a in arms], fontsize=8, frameon=False, ncol=len(arms))
    fig.suptitle(f"ROLE: {role} agent — parse failures by candidate x arm "
                 "(bar = % of samples, label = raw count; dotted = 5% "
                 "voluntary/forced-choice line)", fontsize=12,
                 color=ROLE_COLORS.get(role, "black"))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return True


def _role_arm_bad_panel(ax, sub, model, cands, keys, xlabel=False, title=False):
    """Parse failures per candidate x (role, arm), as % of samples.

    Role x arm counterpart of _bad_parse_panel, for the overlay figures: a curve
    is only interpretable next to the parse health of the run that produced it,
    and here a panel carries two roles and two arms at once.
    """
    gw = 0.8
    bw = gw / max(len(keys), 1)
    any_bad = False
    for j, (role, arm) in enumerate(keys):
        pcts = []
        for cand in cands:
            g = sub[(sub.model == model) & (sub.role == role) & (sub.arm == arm)
                    & (sub.candidate == cand)]
            tot = float(g.n_samples.sum())
            pcts.append(100.0 * float(g.n_bad.sum()) / tot if tot else 0.0)
        any_bad = any_bad or any(v > 0 for v in pcts)
        xs = [k - gw / 2 + bw * (j + 0.5) for k in range(len(cands))]
        ax.bar(xs, pcts, width=bw * 0.92, color=ROLE_COLORS[role], zorder=3,
               hatch=ARM_STYLES[arm]["hatch"], edgecolor="white", linewidth=0.4)
    ax.axhline(5.0, color="black", ls=":", lw=1.0, alpha=0.6, zorder=2)
    ax.set_xticks(range(len(cands)))
    ax.set_xticklabels([SHORT_CAND.get(c, c) for c in cands], fontsize=8)
    ax.set_ylabel("% unparseable", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    if not any_bad:   # otherwise an all-grammar figure shows a silently blank panel
        ax.annotate("no bad parses", (0.5, 0.5), xycoords="axes fraction",
                    ha="center", va="center", fontsize=8, color="#898781")
    if title:
        ax.set_title("bad parses (dotted = 5%)", fontsize=9)
    if xlabel:
        ax.set_xlabel("candidate", fontsize=8)


def fig_role_arm_gradient(df_by_role, arms, out_path):
    """figR6 — the figR1 gradient with BOTH roles and one or more arms on shared
    axes: rows = models, cols = candidates, role = colour, arm = line style.

    figR1 fixes the role and spends colour on the arm; this is the transpose, and
    is the figure to read when the question is how far apart the two roles' value
    functions sit and whether the endpoint arm moves that gap. Roles remain
    separate value functions — shared axes are for reading the asymmetry, never
    for pooling.
    """
    frames = [d.assign(role=r) for r, d in df_by_role.items() if d is not None]
    if not frames:
        return False
    sub = pd.concat(frames, ignore_index=True)
    sub = sub[sub.arm.isin(arms)]
    if sub.empty:
        return False
    keys = [(role, arm) for arm in arms for role in ("red", "blue")]
    keys = [k for k in keys if not sub[(sub.role == k[0]) & (sub.arm == k[1])].empty]
    models = _model_sort(sub.model.unique())
    cands = [c for c in RATIO_CANDIDATES if c in set(sub.candidate)]
    ncols = len(cands) + 1
    fig, axes = plt.subplots(len(models), ncols,
                             figsize=(3.2 * len(cands) + 3.8, 2.6 * len(models)),
                             squeeze=False,
                             gridspec_kw={"width_ratios": [1] * len(cands) + [1.3]})
    # x is the OUT-GROUP SHARE, not the count: this slice fixes n_occ=8, so the
    # nine points are k/8 = 0, 12.5, ... 100%. Same axis units as figR4, so the
    # full-occupancy slice and the all-occupancy collapse can be read together.
    x = np.arange(9) * 100.0 / 8.0
    for i, model in enumerate(models):
        for j, cand in enumerate(cands):
            ax = axes[i][j]
            # Threshold drawn at exactly 50% — the rule's cut point, and the same
            # place figR4 and plot_results.py's mech_ref put it, so every
            # percentage-axis figure in the set reads identically. (figR1 keeps
            # the between-samples 'mid' step because its axis is a raw count.)
            ax.step([0, 50, 50, 100], [0, 0, 1, 1], where="post", color="black",
                    ls="--", lw=1.0, alpha=0.6, label="mechanical")
            for role, arm in keys:
                grp = sub[(sub.model == model) & (sub.candidate == cand)
                          & (sub.arm == arm) & (sub.role == role)]
                if grp.empty:
                    continue
                st = ARM_STYLES[arm]
                ax.plot(x, occ8_gradient(surface(grp)), color=ROLE_COLORS[role],
                        lw=1.6, ls=arm_ls(arm, arms), marker=st["marker"], ms=3,
                        label=f"{role} role — {arm}")
            if i == 0:
                ax.set_title(cand, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{model}\neffective P(MOVE)", fontsize=8)
            if i == len(models) - 1:
                ax.set_xlabel("% of neighbours out-group (of 8)", fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            # This figure fixes n_occ=8, so only the EIGHTHS are sampled — a
            # subset of the full reachable set (they align with figR4's ticks).
            tpos, tlab = pct_ticks([k / 8 for k in range(9)])
            ax.set_xticks(tpos)
            ax.set_xticklabels(tlab, fontsize=6, rotation=90)
            ax.grid(alpha=0.3)
        _role_arm_bad_panel(axes[i][len(cands)], sub, model, cands, keys,
                            xlabel=(i == len(models) - 1), title=(i == 0))
    handles, labels = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in labels:
                handles.append(hh); labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
               fontsize=9, frameon=False)
    fig.suptitle("RED vs BLUE agent role on shared axes — EFFECTIVE P(MOVE) = "
                 "n_move/(n_move+n_stay) vs out-group share, FULL OCCUPANCY ONLY "
                 "(n_occ=8; see figR4 for all occupancies collapsed)\n"
                 f"arms: {' | '.join(arms)} — role = colour, arm = line style; "
                 "dashed step = mechanical", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return True


def fig_thresholds(df, out_path, role):
    rows = []
    for (model, arm, cand), grp in df.groupby(["model", "arm", "candidate"]):
        thr = implied_threshold(surface(grp), 8)
        rows.append({"model": model, "arm": arm, "candidate": cand, "thr": thr})
    t = pd.DataFrame(rows).dropna(subset=["thr"])
    if t.empty:
        return False
    models = _model_sort(t.model.unique())
    cands = [c for c in RATIO_CANDIDATES if c in set(t.candidate)]
    fig, ax = plt.subplots(figsize=(2.0 + 1.6 * len(models), 4.5))
    width = 0.8 / max(len(cands), 1)
    for j, cand in enumerate(cands):
        for a_i, arm in enumerate(ARM_ORDER):
            sub = t[(t.candidate == cand) & (t.arm == arm)]
            xs = [models.index(m) + (j - len(cands) / 2 + 0.5) * width
                  for m in sub.model]
            ax.scatter(xs, sub.thr, color=ARM_COLORS[arm], s=28,
                       marker="osD^"[a_i % 4], alpha=0.85,
                       label=f"{cand} / {arm}" if models and a_i == 0 and False else None)
    ax.axhline(4.5, color="black", ls="--", lw=1.0, alpha=0.7,
               label="mechanical (4.5)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("implied MOVE threshold (opposite of 8)")
    ax.set_title(f"ROLE: {role} agent — implied P=0.5 threshold at full occupancy "
                 "— arms = colors, styles = x-offset", fontsize=11,
                 color=ROLE_COLORS.get(role, "black"))
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--role", default="both", choices=["red", "blue", "both"],
                    help="which agent role's value function to plot. 'both' "
                         "(default) writes the per-role R1-R3 set for each role; "
                         "R4 always overlays the two.")
    ap.add_argument("--surface-arm", default="completions+grammar",
                    help="arm shown in the surface-heatmap figure (figR2)")
    ap.add_argument("--overlay-arm", nargs="+", metavar="ARM",
                    default=["chat+grammar"], choices=ARM_ORDER,
                    help="arm(s) drawn in the role-overlay figures (figR4, figR6); "
                         "each is shown for BOTH roles on the same axes. Default "
                         "chat+grammar: grammar removes parse artifacts, and the chat "
                         "endpoint is the only grammar channel with zero max_tokens "
                         "truncation (see NOTES.md / grammar_truncation.md). Pass more "
                         "than one arm to overlay them as different line styles.")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR / "figures"))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--include-probes", action="store_true",
                    help="also plot the server-configuration probe arms "
                         f"({', '.join(sorted(PROBE_MODELS))}), which are excluded by "
                         "default because they measured equivalent to their real "
                         "counterpart. They are not separate models.")
    ap.add_argument("--dpi", type=int, default=DPI,
                    help=f"raster resolution (default {DPI})")
    ap.add_argument("--format", default=EXT, choices=["png", "pdf", "svg"],
                    help="output format; pdf/svg are vector (resolution-independent)")
    args = ap.parse_args()
    globals()["DPI"] = args.dpi
    globals()["EXT"] = args.format
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)

    roles = ["red", "blue"] if args.role == "both" else [args.role]
    frames = {}
    for role in roles:
        frames[role] = df = drop_probes(load_all(results_dir, role=role),
                                        args.include_probes)
        r1 = out_dir / f"figR1_occ8_gradient_{role}.{EXT}"
        fig_occ8_gradient(df, r1, role)
        print(f"wrote {r1}")
        r2 = out_dir / f"figR2_surfaces_{role}.{EXT}"
        if fig_surfaces(df, args.surface_arm, r2, role):
            print(f"wrote {r2}")
        r3 = out_dir / f"figR3_thresholds_{role}.{EXT}"
        if fig_thresholds(df, r3, role):
            print(f"wrote {r3}")
        r5 = out_dir / f"figR5_bad_parses_{role}.{EXT}"
        if fig_bad_parses(df, r5, role):
            print(f"wrote {r5}")

    # R4 overlays the roles, so it needs whichever role the loop did not load.
    for role in ("red", "blue"):
        if role not in frames:
            try:
                frames[role] = drop_probes(load_all(results_dir, role=role),
                                           args.include_probes)
            except SystemExit:
                frames[role] = None
    overlay_arms = [a for a in ARM_ORDER if a in args.overlay_arm]   # canonical order
    slug = arms_slug(overlay_arms)
    r4 = out_dir / f"figR4_ratio_collapse_{slug}.{EXT}"
    if fig_ratio_collapse(frames, overlay_arms, r4):
        print(f"wrote {r4}")
    r6 = out_dir / f"figR6_role_arm_gradient_{slug}.{EXT}"
    if fig_role_arm_gradient(frames, overlay_arms, r6):
        print(f"wrote {r6}")


if __name__ == "__main__":
    main()
