#!/usr/bin/env python3
"""Quantify the KV-cache sampling artifact: clean vs archived value functions.

    python analysis_tools/vf_contamination_comparison.py \
        [--clean-dir ...] [--archive-dir ...] [--out-dir ...]

For every (scenario, role, composition) cell present in both the CLEAN
artifacts (cache_prompt=false protocol) and the ARCHIVED cache-on artifacts,
reports p̂_clean, p̂_archived, |Δ|, both Wilson CIs, and a DISJOINT flag
(CIs non-overlapping ⇒ the difference cannot be sampling noise ⇒ measured
contamination). Outputs:

    vf_contamination_map.csv      per-cell table
    vf_contamination_map.png      |Δp̂| heatmaps, scenarios × roles
    printed summary               flagged cells, max |Δ|, sanity checks

Sanity expectation (KV_CACHE_SAMPLING_ARTIFACT.md): cells UNANIMOUS in both
datasets (0-of-n or n-of-n on both sides, same direction) cannot be flagged —
a flag there means a bug in this comparison. Near-saturated cells (p̂ ~ 0.99)
CAN be genuinely contaminated: their raw logit gap (~|ln(p/(1-p))|·T) is
within the cache perturbation's reach at T=0.3.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
for p in (_THIS, _THIS.parent, _THIS.parent / "prompt_refinement"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sampling_common import wilson_ci  # noqa: E402

SCENARIO_ORDER = ["baseline", "race_white_black", "ethnic_asian_hispanic",
                  "income_high_low", "political_liberal_conservative",
                  "green_yellow"]
DEFAULT_CLEAN = _THIS.parent / "prompt_refinement" / "results" / "value_functions"
DEFAULT_ARCHIVE = _THIS.parent / "prompt_refinement" / "results" / "value_functions_cacheon_archive"


def cells(vf):
    for role, rows in vf["compositions"].items():
        for c in rows:
            yield role, c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clean-dir", default=str(DEFAULT_CLEAN))
    ap.add_argument("--archive-dir", default=str(DEFAULT_ARCHIVE))
    ap.add_argument("--out-dir", default=None, help="default: <clean-dir>")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    clean_dir, arch_dir = Path(args.clean_dir), Path(args.archive_dir)
    out_dir = Path(args.out_dir or clean_dir)

    rows = []
    for cp in sorted(clean_dir.glob("vf_*__*__*.json")):
        apath = arch_dir / cp.name
        if not apath.exists():
            print(f"[skip] no archived counterpart for {cp.name}")
            continue
        clean, arch = json.loads(cp.read_text()), json.loads(apath.read_text())
        sc = clean["meta"]["scenario"]
        a_ix = {(r, c["n_similar"], c["n_occupied"]): c for r, c in cells(arch)}
        for role, c in cells(clean):
            a = a_ix.get((role, c["n_similar"], c["n_occupied"]))
            if a is None:
                continue
            cv, av = c["n_move"] + c["n_stay"], a["n_move"] + a["n_stay"]
            if cv == 0 or av == 0:
                continue
            p_c, p_a = c["n_move"] / cv, a["n_move"] / av
            lo_c, hi_c = wilson_ci(c["n_move"], cv)
            lo_a, hi_a = wilson_ci(a["n_move"], av)
            rows.append({
                "scenario": sc, "role": role,
                "n_similar": c["n_similar"], "n_occupied": c["n_occupied"],
                "p_clean": round(p_c, 4), "n_clean": cv,
                "p_archived": round(p_a, 4), "n_archived": av,
                "abs_delta": round(abs(p_c - p_a), 4),
                "ci_clean": [round(lo_c, 4), round(hi_c, 4)],
                "ci_archived": [round(lo_a, 4), round(hi_a, 4)],
                "saturated_clean": p_c <= 0.01 or p_c >= 0.99,
                # Mechanistic immunity holds only for UNANIMOUS cells (raw
                # logit gap >> cache perturbation). A p̂=0.99 cell sits ~1.4
                # raw logits from the tie at T=0.3 and IS perturbable —
                # observed: political red (2,8) clean 0.992 vs archived 0.828.
                "unanimous_both": (c["n_move"] in (0, cv)) and (a["n_move"] in (0, av))
                                   and (c["n_move"] == 0) == (a["n_move"] == 0),
                "flag_disjoint": hi_c < lo_a or hi_a < lo_c,
            })
    if not rows:
        sys.exit(f"nothing to compare between {clean_dir} and {arch_dir}")
    df = pd.DataFrame(rows)
    csv_path = out_dir / "vf_contamination_map.csv"
    df.to_csv(csv_path, index=False)

    scenarios = sorted(df.scenario.unique(),
                       key=lambda s: (SCENARIO_ORDER.index(s)
                                      if s in SCENARIO_ORDER else 99, s))
    roles = sorted(df.role.unique(), key=lambda r: (r != "red", r))
    fig, axes = plt.subplots(len(scenarios), len(roles), squeeze=False,
                             figsize=(3.4 * len(roles) + 1.6,
                                      3.2 * len(scenarios) + 1.6),
                             constrained_layout=True)
    vmax = max(df.abs_delta.max(), 1e-9)
    im = None
    for i, sc in enumerate(scenarios):
        for j, role in enumerate(roles):
            ax = axes[i][j]
            sub = df[(df.scenario == sc) & (df.role == role)]
            mat = np.full((9, 9), np.nan)
            for r in sub.itertuples():
                mat[r.n_occupied, r.n_similar] = r.abs_delta
            im = ax.imshow(mat, origin="lower", vmin=0, vmax=vmax, cmap="magma_r")
            for r in sub[sub.flag_disjoint].itertuples():
                ax.scatter([r.n_similar], [r.n_occupied], marker="x", s=40,
                           c="#00c0ff", linewidths=1.4, zorder=3)
            ax.set_xticks(range(0, 9, 2)); ax.set_yticks(range(0, 9, 2))
            ax.set_title(f"{sc} — {role}", fontsize=8)
            if j == 0:
                ax.set_ylabel("n_occupied", fontsize=8)
            if i == len(scenarios) - 1:
                ax.set_xlabel("n_similar", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.5,
                 label=f"|Δ p̂|  clean vs cache-on archive  [0 .. {vmax:.2f}]")
    fig.suptitle("KV-cache contamination map — |p̂_clean − p̂_archived| per composition\n"
                 "(× = Wilson CIs disjoint: difference not attributable to sampling noise)",
                 fontsize=11)
    png_path = out_dir / "vf_contamination_map.png"
    fig.savefig(png_path, dpi=args.dpi)
    plt.close(fig)

    flagged = df[df.flag_disjoint]
    sat_flagged = flagged[flagged.unanimous_both]
    print(f"wrote {csv_path}\nwrote {png_path}")
    print(f"\ncells compared: {len(df)}   CI-disjoint (contaminated): {len(flagged)}   "
          f"max |Δ|: {df.abs_delta.max():.3f}")
    if len(flagged):
        with pd.option_context("display.width", 140):
            print(flagged[["scenario", "role", "n_similar", "n_occupied",
                           "p_clean", "p_archived", "abs_delta"]]
                  .sort_values("abs_delta", ascending=False).to_string(index=False))
    print(f"\nsanity: flagged UNANIMOUS-both cells (MUST be 0): {len(sat_flagged)}")
    near_sat = flagged[flagged.saturated_clean & ~flagged.unanimous_both]
    if len(near_sat):
        print(f"note: {len(near_sat)} flagged cell(s) are NEAR-saturated (p̂>=0.99 "
              f"but not unanimous) — genuine contamination within logit reach, "
              f"not a sanity failure")
    return 1 if len(sat_flagged) else 0


if __name__ == "__main__":
    sys.exit(main())
