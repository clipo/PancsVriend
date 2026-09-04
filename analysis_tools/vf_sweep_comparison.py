#!/usr/bin/env python3
"""Validate the clean value function against the Phase-A ratio sweep (R3, baseline).

    python analysis_tools/vf_sweep_comparison.py

The Phase-A style sweep (ratio_comparison_ratio-gemma-4-31b-chat-grammar.csv,
2026-08-02, N=50/composition, cache-on era) and the clean-protocol value
function (vf_gemma-4-31b-chat-grammar__baseline__R3_dual_count.json,
2026-08-22/23, N>=100, cache off + seeds) measured the SAME quantity three
weeks apart with different sample sizes and different serving protocols —
their agreement is a replication check, and their disagreements should be
exactly the documented cache-contaminated cells.

Figure (vf_sweep_comparison.png):
  left  — per-composition scatter, sweep vs clean, both roles, y=x line,
          CI-disjoint cells annotated;
  right — ratio-axis curves overlaid per role, UNWEIGHTED mean-of-members on
          BOTH sides (the sweep's uniform N makes its pooled = unweighted;
          using the clean pooled would confound the comparison with top-up
          weighting — see the 2026-08-23 aggregation discussion).
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve().parent
for p in (_THIS, _THIS.parent, _THIS.parent / "prompt_refinement"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sampling_common import wilson_ci, ratio_of  # noqa: E402

PR = _THIS.parent / "prompt_refinement" / "results"
SWEEP_CSV = PR / "ratio_comparison_ratio-gemma-4-31b-chat-grammar.csv"
VF_JSON = PR / "value_functions" / "vf_gemma-4-31b-chat-grammar__baseline__R3_dual_count.json"
OUT = PR / "figures" / "vf_sweep_comparison.png"
ROLE_COLORS = {"red": "#d62728", "blue": "#1f77b4"}


def main():
    sweep = {}
    for row in csv.DictReader(open(SWEEP_CSV)):
        if row["candidate"] != "R3_dual_count":
            continue
        m, s = int(row["n_move"]), int(row["n_stay"])
        sweep[(row["agent_role"], int(row["n_similar"]), int(row["n_occupied"]))] = (m, s)
    vf = json.loads(VF_JSON.read_text())

    rows = []
    for role in ("red", "blue"):
        for c in vf["compositions"][role]:
            k = (role, c["n_similar"], c["n_occupied"])
            if k not in sweep or c["p_move_effective"] is None:
                continue
            sm, ss = sweep[k]
            sv = sm + ss
            if sv == 0:
                continue
            p_s = sm / sv
            cv = c["n_move"] + c["n_stay"]
            lo_s, hi_s = wilson_ci(sm, sv)
            lo_c, hi_c = wilson_ci(c["n_move"], cv)
            rows.append({
                "role": role, "n_sim": c["n_similar"], "n_occ": c["n_occupied"],
                "p_sweep": p_s, "n_sweep": sv,
                "p_clean": c["p_move_effective"], "n_clean": cv,
                "delta": abs(p_s - c["p_move_effective"]),
                "disjoint": hi_s < lo_c or hi_c < lo_s,
            })

    deltas = np.array([r["delta"] for r in rows])
    n_dis = sum(r["disjoint"] for r in rows)
    corr = np.corrcoef([r["p_sweep"] for r in rows],
                       [r["p_clean"] for r in rows])[0, 1]
    print(f"cells compared: {len(rows)}   pearson r = {corr:.4f}")
    print(f"|Δ|: mean {deltas.mean():.3f}  median {np.median(deltas):.3f}  "
          f"max {deltas.max():.3f}   >0.15: {(deltas > 0.15).sum()}   "
          f"CI-disjoint: {n_dis}")
    for r in sorted(rows, key=lambda r: -r["delta"])[:5]:
        note = "  <-- CI-disjoint" if r["disjoint"] else ""
        print(f"  {r['role']:4s} sim={r['n_sim']} occ={r['n_occ']}: "
              f"sweep {r['p_sweep']:.2f} (n={r['n_sweep']}) vs "
              f"clean {r['p_clean']:.2f} (n={r['n_clean']})  |Δ|={r['delta']:.2f}{note}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    # left: scatter
    axL.plot([0, 1], [0, 1], color="#999", lw=1, ls="--", zorder=1)
    for role in ("red", "blue"):
        rr = [r for r in rows if r["role"] == role]
        axL.scatter([r["p_sweep"] for r in rr], [r["p_clean"] for r in rr],
                    s=22, color=ROLE_COLORS[role], alpha=0.75,
                    edgecolors="white", linewidths=0.4, label=f"{role} role")
    for r in rows:
        if r["disjoint"]:
            axL.annotate(f"({r['n_sim']},{r['n_occ']}) {r['role']}",
                         (r["p_sweep"], r["p_clean"]), fontsize=7,
                         xytext=(6, -8), textcoords="offset points")
            axL.scatter([r["p_sweep"]], [r["p_clean"]], marker="o", s=90,
                        facecolors="none", edgecolors="#111", linewidths=1.2)
    axL.set_xlabel("Phase-A sweep p̂  (N=50/composition, cache-on era, Aug 2)")
    axL.set_ylabel("clean value function p̂  (N≥100, cache off + seeds, Aug 22)")
    axL.set_title(f"Per-composition agreement — 90 cells, r = {corr:.3f}\n"
                  "(circled = Wilson CIs disjoint)", fontsize=10)
    axL.legend(fontsize=8, frameon=False)
    axL.grid(alpha=0.3)

    # right: ratio-axis curves, unweighted mean of members on both sides
    def ratio_curve(get_p):
        pts = {}
        for r in rows:
            f = ratio_of(r["n_sim"], r["n_occ"])
            if f is None:
                continue
            pts.setdefault((r["role"], float(f)), []).append(get_p(r))
        return pts

    for src, style, get in (("sweep", ":", lambda r: r["p_sweep"]),
                            ("clean", "-", lambda r: r["p_clean"])):
        pts = ratio_curve(get)
        for role in ("red", "blue"):
            xy = sorted((x, np.mean(v)) for (ro, x), v in pts.items() if ro == role)
            axR.plot([p[0] for p in xy], [p[1] for p in xy], ls=style, lw=1.8,
                     marker="o" if src == "clean" else "^", ms=3.5,
                     color=ROLE_COLORS[role], alpha=0.9 if src == "clean" else 0.65,
                     label=f"{role} {src}")
    axR.step([0, 0.5, 0.5, 1.0], [0, 0, 1, 1], where="post", color="black",
             ls="--", lw=1.0, alpha=0.5)
    axR.set_xlabel("opposite / occupied neighbors")
    axR.set_ylabel("P(MOVE), unweighted mean over member compositions")
    axR.set_title("Ratio-axis view — both sides aggregated identically\n"
                  "(unweighted member mean; dashed step = mechanical)", fontsize=10)
    axR.legend(fontsize=8, frameon=False, ncol=2)
    axR.grid(alpha=0.3)

    fig.suptitle("Replication check: Phase-A R3 dual-count sweep vs clean value function — gemma, baseline, chat+grammar",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
