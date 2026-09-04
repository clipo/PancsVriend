#!/usr/bin/env python3
"""Compare value functions ACROSS SCENARIOS: one panel per scenario, both roles.

    python prompt_refinement/plot_value_functions.py \
        --label gemma-4-31b-chat-grammar --style R1_count_opposite

Reads every results/value_functions/vf_<label>__<scenario>__<style>.json (the
vf-1 artifacts written by build_value_function.py) and lays them out like the
per-style figures, except the varying dimension is the SOCIAL CONTEXT: same
model, same composition sentences, same endpoint arm — only the identity
labels differ ("red team resident" vs "white middle class family" ...). Panel
titles carry each scenario's actual identity labels so the curves can be read
without a decoder ring; red/blue keep the figR4 role colours (red = type_a).

Single-panel drawing is shared with build_value_function.py's per-artifact
plot via draw_value_function_axes(), so the two figures cannot drift.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sampling_common import NO_NEIGHBORS_KEY, RESULTS_DIR  # noqa: E402

VF_DIR = RESULTS_DIR / "value_functions"
ROLE_COLORS = {"red": "#d62728", "blue": "#1f77b4"}
# Canonical panel order: control first, then the loaded contexts as they
# appear in scenarios_a2.py, then anything unexpected alphabetically.
SCENARIO_ORDER = ["baseline", "race_white_black", "ethnic_asian_hispanic",
                  "income_high_low", "political_liberal_conservative",
                  "green_yellow"]


def draw_value_function_axes(ax, vf, legend_roles=True):
    """The vf-1 panel: 23 ratio datapoints per role with 95% CIs (variance-propagated equal-weight member mean)
    (fixed-size markers), the no-neighbours diamond at x=-0.06, and the
    mechanical step. Sample sizes are NOT encoded in marker area — they get
    their own bar panel (draw_n_bars), per user decision 2026-08-23."""
    ax.step([0, 0.5, 0.5, 1.0], [0, 0, 1, 1], where="post",
            color="black", ls="--", lw=1.0, alpha=0.6)
    for role, rows in vf["ratios"].items():
        pts = [(r["ratio_float"], r["p_move_effective"], r["ci95"])
               for r in rows if r["ratio_float"] is not None
               and r["p_move_effective"] is not None]
        pts.sort()
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        lo = [p[1] - p[2][0] for p in pts]; hi = [p[2][1] - p[1] for p in pts]
        col = ROLE_COLORS.get(role, "black")
        ax.errorbar(xs, ys, yerr=[lo, hi], color=col, lw=1.6, marker="o",
                    ms=3.5, capsize=2, elinewidth=0.9,
                    label=f"{role} role" if legend_roles else None)
        none_row = next(r for r in rows if r["ratio"] == NO_NEIGHBORS_KEY)
        if none_row["p_move_effective"] is not None:
            ax.scatter([-0.06], [none_row["p_move_effective"]], color=col,
                       marker="D", s=25, zorder=3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.12, 1.02)
    ax.grid(alpha=0.3)


def draw_n_bars(ax, vf):
    """Companion bar panel: pooled sample count per ratio datapoint, one thin
    bar per role at each ratio position (+ the no-neighbours point at -0.06).
    Shares the x-axis with the curve panel above it."""
    w = 0.011
    for k, (role, rows) in enumerate(sorted(vf["ratios"].items(),
                                            key=lambda kv: kv[0] != "red")):
        off = (k - 0.5) * w
        xs, ns = [], []
        for r in rows:
            x = -0.06 if r["ratio"] == NO_NEIGHBORS_KEY else r["ratio_float"]
            xs.append(x + off); ns.append(r["n_samples"])
        ax.bar(xs, ns, width=w, color=ROLE_COLORS.get(role, "black"),
               alpha=0.85, linewidth=0)
    ax.set_xlim(-0.12, 1.02)
    ax.set_ylabel("N", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.25, axis="y")
    ax.set_axisbelow(True)


def fig_heatmaps(found, scenarios, out_path, dpi=300):
    """P(MOVE | n_similar, n_occupied) heatmaps: rows = scenarios, cols = roles.

    The composition-level counts the artifacts keep as ground truth, drawn in
    figR2_surfaces' conventions (x = n_similar, y = n_occupied, RdYlBu_r,
    colour scaled to the max observed across the WHOLE figure so panels are
    directly comparable; the triangle above the diagonal is unreachable).
    Small black dots mark cells where the mechanical rule says MOVE, so each
    panel carries its own reference without a second colour scale.
    """
    import numpy as np

    roles = sorted({r for vf in found.values() for r in vf["compositions"]},
                   key=lambda r: (r != "red", r))   # red first, matches the line plots
    vmax = max((c["p_move_effective"] or 0.0)
               for vf in found.values()
               for rows in vf["compositions"].values() for c in rows)
    vmax = max(vmax, 1e-9)
    # constrained_layout: the one manager that spaces suptitle, panel titles
    # and a figure-level colorbar without manual rect tuning at any row count.
    fig, axes = plt.subplots(len(scenarios), len(roles), squeeze=False,
                             figsize=(3.4 * len(roles) + 1.6, 3.2 * len(scenarios) + 1.6),
                             constrained_layout=True)
    im = None
    for i, sc in enumerate(scenarios):
        vf = found[sc]
        labels = vf["meta"].get("role_labels", {})
        for j, role in enumerate(roles):
            ax = axes[i][j]
            mat = np.full((9, 9), np.nan)
            mech = []
            for c in vf["compositions"][role]:
                if c["p_move_effective"] is not None:
                    mat[c["n_occupied"], c["n_similar"]] = c["p_move_effective"]
                if c["mechanical"]:
                    mech.append((c["n_similar"], c["n_occupied"]))
            im = ax.imshow(mat, origin="lower", vmin=0, vmax=vmax,
                           cmap="RdYlBu_r", aspect="equal")
            if mech:
                ax.scatter(*zip(*mech), s=4, c="black", alpha=0.55, zorder=3)
            # Per-cell sample count (user requirement 2026-08-21: final N must
            # be visible per datapoint). Text colour flips against the RdYlBu_r
            # extremes (dark blue / dark red) and stays black on the light mid.
            # N and the achieved CI half-width. The half-width is the number
            # that says whether the cell is calibrated; N alone does not, since
            # the samples needed depend on p̂ (a saturated cell reaches ±2pp at
            # n=100 while one near p=0.5 needs 2401). Cells outside the target
            # are outlined so a shortfall is findable at a glance rather than by
            # reading 81 numbers.
            target = ((vf.get("meta") or {}).get("calibration") or {}).get("requested_w")
            for c in vf["compositions"][role]:
                n = c["n_samples"]
                if n == 0:
                    continue
                v = (c["p_move_effective"] or 0.0) / vmax
                txt_col = "white" if (v < 0.2 or v > 0.8) else "black"
                ci = c.get("ci95")
                hw = (ci[1] - ci[0]) / 2 if ci else None
                # Both labels sit in the LOWER half of the cell (the mechanical
                # dot occupies the centre) and inside its ±0.5 bounds — at
                # va="top" the second line used to overflow the cell edge.
                ax.annotate(str(n), (c["n_similar"], c["n_occupied"] - 0.18),
                            ha="center", va="center", fontsize=3.6, color=txt_col)
                if hw is not None:
                    ax.annotate(f"±{hw:.3f}".replace("0.", "."),
                                (c["n_similar"], c["n_occupied"] - 0.36),
                                ha="center", va="center", fontsize=3.0, color=txt_col)
                    if target and hw > target:
                        ax.add_patch(plt.Rectangle(
                            (c["n_similar"] - 0.5, c["n_occupied"] - 0.5), 1, 1,
                            fill=False, edgecolor="#000000", lw=0.7, zorder=4))
            ax.set_xticks(range(0, 9, 2)); ax.set_yticks(range(0, 9, 2))
            ax.set_title(f"{sc} — {role}\n({labels.get(role, '?')})", fontsize=8)
            if j == 0:
                ax.set_ylabel("n_occupied", fontsize=8)
            if i == len(scenarios) - 1:
                ax.set_xlabel("n_similar", fontsize=8)
    m = next(iter(found.values()))["meta"]
    fig.colorbar(im, ax=axes, shrink=0.5,
                 label=f"effective P(MOVE)  [0 .. {vmax:.2f} observed max]")
    fig.suptitle(f"Sampled value-function SURFACES by scenario — {m['model']} / "
                 f"{m['style']} / {m['arm']}\n(per-composition ground truth; "
                 "dots = mechanical agent would MOVE; T="
                 f"{m['temperature']})", fontsize=11)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True, help="artifact label, e.g. gemma-4-31b-chat-grammar")
    ap.add_argument("--style", default="R3_dual_count")
    ap.add_argument("--vf-dir", default=str(VF_DIR))
    ap.add_argument("--out", default=None,
                    help="output path (default figures/vfS_<label>__<style>.<fmt>)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--ncols", type=int, default=3)
    ap.add_argument("--no-heatmaps", action="store_true",
                    help="skip the vfH_* composition-surface heatmap figure")
    args = ap.parse_args()

    vf_dir = Path(args.vf_dir)
    found = {}
    for p in sorted(vf_dir.glob(f"vf_{args.label}__*__{args.style}.json")):
        vf = json.loads(p.read_text())
        found[vf["meta"]["scenario"]] = vf
    if not found:
        sys.exit(f"no vf_{args.label}__<scenario>__{args.style}.json under {vf_dir}")
    scenarios = sorted(found, key=lambda s: (SCENARIO_ORDER.index(s)
                                             if s in SCENARIO_ORDER else 99, s))

    ncols = min(args.ncols, len(scenarios))
    nblocks = -(-len(scenarios) // ncols)
    # Each scenario is a BLOCK of two stacked axes sharing x: the value-function
    # curve (tall) and its per-datapoint N bar chart (short).
    # Rows per block: curve (3), N bars (1), spacer (0.6) — the spacer keeps
    # the next block's two-line title clear of the bar panel above it.
    ratios = ([3, 1, 0.6] * nblocks)[:-1]
    fig = plt.figure(figsize=(4.6 * ncols, 5.0 * nblocks))
    gs = fig.add_gridspec(len(ratios), ncols, height_ratios=ratios,
                          hspace=0.25, wspace=0.22)
    for i, sc in enumerate(scenarios):
        br, bc = divmod(i, ncols)
        ax = fig.add_subplot(gs[3 * br, bc])
        axn = fig.add_subplot(gs[3 * br + 1, bc], sharex=ax)
        vf = found[sc]
        draw_value_function_axes(ax, vf, legend_roles=(i == 0))
        draw_n_bars(axn, vf)
        labels = vf["meta"].get("role_labels", {})
        ax.set_title(f"{sc}\nred = {labels.get('red', '?')}  |  "
                     f"blue = {labels.get('blue', '?')}", fontsize=8.5)
        ax.tick_params(labelbottom=False, labelsize=7)
        if bc == 0:
            ax.set_ylabel("effective P(MOVE)", fontsize=9)
        if br == nblocks - 1:
            axn.set_xlabel("opposite / occupied neighbors", fontsize=9)

    m = next(iter(found.values()))["meta"]
    ns = [r["n_samples"] for vf in found.values()
          for rows in vf["ratios"].values() for r in rows]
    n_txt = f"{min(ns)}" if min(ns) == max(ns) else f"{min(ns)}–{max(ns)}"
    handles = [plt.Line2D([], [], color=ROLE_COLORS[r], lw=1.6, marker="o",
                          ms=3.5, label=f"{r} role") for r in ("red", "blue")]
    handles.append(plt.Line2D([], [], color="black", ls="--", lw=1,
                              label="mechanical (>0.5 moves)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=False)
    fig.suptitle(f"Value functions by SCENARIO — {m['model']} / {m['style']} / {m['arm']}\n"
                 f"(same composition sentences, only identity labels differ; T={m['temperature']};\n"
                 f"whiskers = 95% CI (variance-propagated over members); diamond = no neighbors; "
                 f"N per datapoint [{n_txt}] in the bar panels)",
                 fontsize=10.5, y=0.985)
    fig.subplots_adjust(top=0.885, bottom=0.06, left=0.06, right=0.985)

    out = Path(args.out) if args.out else (
        RESULTS_DIR / "figures" / f"vfS_{args.label}__{args.style}.{args.format}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}  ({len(scenarios)} scenarios: {', '.join(scenarios)})")

    if not args.no_heatmaps:
        out_h = out.with_name(out.name.replace("vfS_", "vfH_", 1))
        fig_heatmaps(found, scenarios, out_h, dpi=args.dpi)
        print(f"wrote {out_h}  (composition-surface heatmaps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
