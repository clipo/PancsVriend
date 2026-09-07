#!/usr/bin/env python3
"""Scale-dependence probe: at a FIXED opposite/total ratio, does P(MOVE)
still depend on the NUMBER of opposite neighbours? Works on any vf-1 label.

Motivated by the sawtooth in the gemma political red line plot. The first
version of this probe held allies fixed and varied the opposite count — but
that also varies the ratio, so monotonicity there is expected under a pure
ratio rule too and discriminates nothing (caught by the user, 2026-08-25).

The DISCRIMINATING test holds the ratio CONSTANT and varies scale. A ratio
family is the set of compositions sharing one reduced ratio:
    1/2 -> (1,2), (2,4), (3,6), (4,8)   [opposite counts 1, 2, 3, 4]
    1/1 -> (0,1) ... (0,8)              [opposite counts 1 ... 8]
Under a pure RATIO rule every member of a family has the same P(MOVE) (flat
line). Under a COUNT rule P(MOVE) rises with the opposite count within the
family. That is the hypothesis this script tests.

Standalone by design — NOT part of any pipeline or workflow; run by hand.

    python analysis_tools/vf_scale_dependence_probe.py                  # gemma (default)
    python analysis_tools/vf_scale_dependence_probe.py --label llama-3.3-70b-chat-grammar
    python analysis_tools/vf_scale_dependence_probe.py --label qwen3.6-27b-chat-grammar

Outputs:
    value_functions/results/figures/scale_dependence/vf_<label>_scale_dependence.png
        Family-strip view (lines would overlap: most families sit flat at
        0.00). 6 scenarios x 2 roles; each ROW is one ratio family, x = the
        number of opposite neighbours, cell colour = P(MOVE). A flat row
        (uniform colour) is ratio-consistent; a row that changes colour
        left-to-right is scale-dependent and is boxed + flagged in the label.
    printed audit:
      - per family: flat vs rising, and whether the rise is CI-significant
        (last member's CI above the first member's CI, non-overlapping)
      - global tallies: how many families are flat / rising / falling
"""
import json
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))
from value_functions.paths import SAMPLED_DIR, SCALE_DEPENDENCE_DIR, add_import_paths  # noqa: E402
add_import_paths()

VF_DIR = SAMPLED_DIR
FIG_DIR = SCALE_DEPENDENCE_DIR
LABEL = "gemma-4-31b-chat-grammar"      # --label overrides
OUT = FIG_DIR / "vf_gemma-4-31b-chat-grammar_scale_dependence.png"
STYLE = "R3_dual_count"
SCENARIOS = ["baseline", "race_white_black", "ethnic_asian_hispanic",
             "income_high_low", "political_liberal_conservative", "green_yellow"]
ROLE_NAMES = {"red": "red role (type_a)", "blue": "blue role (type_b)"}


def load(scenario):
    """{role: {ratio: [(n_opposite, p, (lo, hi)), ...] sorted}} — multi-member
    ratio families only (single-member ratios cannot test scale)."""
    vf = json.loads((VF_DIR / f"vf_{LABEL}__{scenario}__{STYLE}.json").read_text())
    out = {}
    for role in ("red", "blue"):
        fams = defaultdict(list)
        for c in vf["compositions"][role]:
            n_sim, n_occ = c["n_similar"], c["n_occupied"]
            n_opp = n_occ - n_sim
            if n_occ == 0 or n_opp == 0:
                continue                      # no-neighbours / all-similar: ratio 0
            p = c["p_move_effective"]
            lo, hi = c["ci95"]
            fams[Fraction(n_opp, n_occ)].append(
                (n_opp, p, (min(lo, p), max(hi, p))))
        out[role] = {f: sorted(v) for f, v in fams.items() if len(v) >= 2}
    return out


def classify(members):
    """'rising' / 'falling' / 'flat' by first-vs-last CI overlap, plus the span."""
    (_, p0, ci0), (_, pl, cil) = members[0], members[-1]
    span = pl - p0
    if ci0[1] < cil[0]:
        return "rising", span
    if cil[1] < ci0[0]:
        return "falling", span
    return "flat", span


def main():
    global LABEL, OUT
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", default=LABEL,
                    help="vf-1 label, e.g. llama-3.3-70b-chat-grammar")
    args = ap.parse_args()
    LABEL = args.label
    OUT = FIG_DIR / f"vf_{LABEL}_scale_dependence.png"
    missing = [sc for sc in SCENARIOS
               if not (VF_DIR / f"vf_{LABEL}__{sc}__{STYLE}.json").exists()]
    if missing:
        sys.exit(f"label {LABEL!r}: missing artifacts for {missing}")
    print(f"label: {LABEL}")

    data = {sc: load(sc) for sc in SCENARIOS}
    all_fams = sorted({f for sc in SCENARIOS for role in ("red", "blue")
                       for f in data[sc][role]})
    cmap = plt.get_cmap("turbo")
    fam_color = {f: cmap(i / max(1, len(all_fams) - 1))
                 for i, f in enumerate(all_fams)}

    tally = defaultdict(int)
    rows = []
    verdicts = {}      # (scenario, role, family) -> verdict
    for sc in SCENARIOS:
        for role in ("red", "blue"):
            for f, members in data[sc][role].items():
                verdict, span = classify(members)
                tally[verdict] += 1
                verdicts[(sc, role, f)] = verdict
                rows.append((sc, role, str(f), len(members), members[0][1],
                             members[-1][1], span, verdict))

    # ---- family-strip figure: one row per ratio family, colour = P(MOVE) ----
    fams = all_fams                      # ascending ratio
    ypos = {f: k for k, f in enumerate(fams)}
    cmap = plt.get_cmap("RdYlBu_r")
    fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(11.5, 15),
                             sharex=True, sharey=True)
    for i, sc in enumerate(SCENARIOS):
        for j, role in enumerate(("red", "blue")):
            ax = axes[i][j]
            for f in fams:
                members = data[sc][role].get(f)
                y = ypos[f]
                if not members:
                    continue
                rising = verdicts[(sc, role, f)] == "rising"
                xs = [m[0] for m in members]
                ps = [m[1] for m in members]
                # connect the family so its scale ordering is visible
                ax.plot(xs, [y] * len(xs), color="#bbbbbb", lw=0.8, zorder=1)
                ax.scatter(xs, [y] * len(xs), c=ps, cmap=cmap, vmin=0, vmax=1,
                           s=105, marker="s", edgecolors="black" if rising else "#999",
                           linewidths=1.5 if rising else 0.4, zorder=3)
                for x, pv in zip(xs, ps):
                    ax.text(x, y, f"{pv:.2f}"[1:], ha="center", va="center",
                            fontsize=5.2, zorder=4,
                            color="white" if pv > 0.75 or pv < 0.12 else "black")
                if rising:
                    ax.annotate("scale-dependent", (max(xs) + 0.35, y),
                                fontsize=5.5, va="center", color="#b2182b")
            ax.set_yticks(range(len(fams)))
            ax.set_yticklabels([str(f) for f in fams], fontsize=7)
            ax.set_xticks(range(1, 9))
            ax.set_xlim(0.4, 9.6)
            ax.set_ylim(-0.6, len(fams) - 0.4)
            ax.grid(axis="y", alpha=0.25)
            if i == 0:
                ax.set_title(ROLE_NAMES[role], fontsize=10)
            if j == 0:
                ax.set_ylabel(sc.replace("_", "\n"), fontsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("number of OPPOSITE-type neighbours (scale within the family)",
                      fontsize=8.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("P(MOVE)", fontsize=9)
    n_rise = tally["rising"]; n_tot = sum(tally.values())
    fig.suptitle(
        f"{LABEL} / {STYLE}: does P(MOVE) depend on SCALE at a FIXED ratio?\n"
        "each row holds opposite/occupied constant; a row of uniform colour is "
        "ratio-consistent, a colour change across the row is scale-dependent "
        f"(black-edged, {n_rise} of {n_tot} families)", fontsize=11)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")

    print("\n=== within-ratio scale dependence (the discriminating test) ===")
    print("ratio family members share the SAME opposite/occupied fraction;")
    print("'flat' = consistent with a pure ratio rule, 'rising' = count/scale matters\n")
    for sc, role, f, n, p0, pl, span, verdict in rows:
        mark = "" if verdict == "flat" else "   <<<"
        print(f"  {sc:32s} {role:4s} ratio {f:>4} ({n} members): "
              f"{p0:.2f} -> {pl:.2f}  (Δ{span:+.2f})  {verdict}{mark}")
    tot = sum(tally.values())
    print(f"\n  rising: {tally['rising']}/{tot}   flat: {tally['flat']}/{tot}   "
          f"falling: {tally['falling']}/{tot}")
    print("\n  A pure ratio rule predicts ALL families flat. Any rising family "
          "is direct\n  evidence that the opposite COUNT (scale), not just the "
          "fraction, drives the decision.")


if __name__ == "__main__":
    main()
