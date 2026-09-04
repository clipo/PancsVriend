#!/usr/bin/env python3
"""Consistency metrics for the ratio-prompt sweep: are sampled value functions
consistent across prompt styles within a (model, arm)?

Reads every results/ratio_comparison_<label>.csv (long format, one row per
(candidate, agent_role, n_similar, n_occupied)) produced by
evaluate_ratio_prompts.py and writes:

    results/ratio_consistency_summary.csv   one row per (model, arm, candidate)
                                            with coherence + distance metrics
    results/ratio_consistency_pairs.csv     one row per (model, arm, style pair)
                                            with D_inf and mean |dP|
    results/ratio_consistency_summary.md    human summary incl. winner ranking

Metrics (see plans/wondrous-bouncing-ladybug.md):
- coherence filter: a (model, arm, style) is USABLE iff Spearman rho >= 0.9
  along the n_occ=8 gradient AND overall dynamic range >= 0.5. Degenerate flat
  arms (e.g. the known Llama-chat ~0-move pathology) fail here and are excluded
  from consistency scoring instead of faking perfect agreement.
- pairwise D_inf = max over the 45 cells |P_s - P_t|, and mean |dP|.
- implied threshold per occupancy row: interpolated P=0.5 crossing along
  n_opposite at fixed n_occ (mechanical reference: n_occ/2).
- distance to the cell-wise median surface computed WITHIN each model over its
  usable (style, arm) surfaces — central behaviour is per model, across arms,
  NEVER pooled across models (each model gets its own value function; pooling
  would reward genre artifacts that make different models look alike). The
  Phase-B winner is chosen PER MODEL: lowest mean distance to that model's
  median over the arms where the style is usable (>= 2 arms; ties: fewest bad
  parses). Percent styles may not win unless they agree with the count forms
  (numeric-anchoring guard) — flagged, not silently enforced.

The outcome-space check (simulate each style's surface as a policy and compare
final dissimilarity index) is run at M3 via the ratio-policy simulation path,
not here — this script is pure CSV arithmetic.
"""
import argparse
import glob
import itertools
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_prompts import spearman  # noqa: E402
from ratio_prompt_templates import ALL_COMPOSITIONS  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

RHO_MIN = 0.9      # coherence: monotonicity along the n_occ=8 gradient
RANGE_MIN = 0.5    # coherence: overall dynamic range of the surface

# label form: ratio-<model>[-chat][-grammar]  (Phase B uses ratio2d-;
# the M1 pilot used pilot- / pilot-roleswap- / pilot-blue- prefixes)
LABEL_RE = re.compile(
    r"^(?:ratio(?:2d)?|pilot(?:-roleswap|-blue)?)-(?P<model>.+?)(?P<chat>-chat)?(?P<gram>-grammar)?$")


def parse_label(label):
    m = LABEL_RE.match(label)
    if not m:
        return None
    endpoint = "chat" if m.group("chat") else "completions"
    arm = endpoint + ("+grammar" if m.group("gram") else "")
    return m.group("model"), arm


def load_all(results_dir, role="red"):
    """-> tidy df: model, arm, candidate, n_similar, n_occupied, move_rate, n_bad.

    role=None keeps every role (agent_role column preserved) — the analysis
    NEVER pools behaviour across roles; they are separate value functions."""
    frames = []
    for path in sorted(glob.glob(str(results_dir / "ratio_comparison_*.csv"))):
        label = os.path.basename(path)[len("ratio_comparison_"):-len(".csv")]
        parsed = parse_label(label)
        if parsed is None:
            print(f"[skip] {label}: label does not parse")
            continue
        model, arm = parsed
        df = pd.read_csv(path)
        if role is not None:
            df = df[df["agent_role"] == role]
        df["model"], df["arm"] = model, arm
        # Normalize both schemas (old files have a bare ambiguous "move_rate" =
        # raw single-shot; new files ship move_rate_raw/move_rate_effective) to
        # ONLY the two explicit names. There is deliberately no "move_rate"
        # column after loading: the EFFECTIVE rate n_move/(n_move+n_stay) is
        # the production-faithful quantity (production retries on bad parses),
        # and anything still referencing the ambiguous name must fail loudly
        # rather than silently use probabilities that count unparseable replies.
        valid = df["n_move"] + df["n_stay"]
        if "move_rate_raw" not in df.columns:
            df["move_rate_raw"] = df["move_rate"]
        df["move_rate_effective"] = np.where(
            valid > 0, df["n_move"] / valid.clip(lower=1), np.nan)
        df = df.drop(columns=[c for c in ("move_rate",) if c in df.columns])
        frames.append(df)
    if not frames:
        sys.exit(f"no ratio_comparison_*.csv under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def surface(df_group):
    """df rows for one (model, arm, candidate) -> {(n_sim, n_occ): move_rate}.

    Uses move_rate_effective (see load_all); cells where every sample failed
    to parse have no defined rate and are dropped."""
    return {(r.n_similar, r.n_occupied): r.move_rate_effective
            for r in df_group.itertuples()
            if r.move_rate_effective == r.move_rate_effective}


def occ8_gradient(surf):
    """P(move) along the classic gradient, indexed by n_opposite 0..8."""
    return [surf.get((8 - k, 8), np.nan) for k in range(9)]


def implied_threshold(surf, n_occ):
    """Interpolated n_opposite where P crosses 0.5 along fixed n_occ, or nan."""
    ps = [surf.get((n_occ - k, n_occ), np.nan) for k in range(n_occ + 1)]
    for i in range(len(ps) - 1):
        a, b = ps[i], ps[i + 1]
        if np.isnan(a) or np.isnan(b):
            continue
        if (a - 0.5) * (b - 0.5) <= 0 and a != b:
            return i + (0.5 - a) / (b - a)
    return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    results_dir = Path(args.results_dir)

    # All roles in one pass; every aggregation below is grouped by role.
    df = load_all(results_dir, role=None)
    cells = ALL_COMPOSITIONS

    summary_rows, pair_rows = [], []
    surfaces = {}   # (model, arm, role, candidate) -> surface dict
    for (model, arm, role, cand), grp in df.groupby(["model", "arm", "agent_role", "candidate"]):
        surf = surface(grp)
        surfaces[(model, arm, role, cand)] = surf
        grad = occ8_gradient(surf)
        vals = [surf[c] for c in cells if c in surf]
        rho = spearman(list(range(9)), grad) if not any(map(math.isnan, grad)) else float("nan")
        rng = (max(vals) - min(vals)) if vals else float("nan")
        usable = (not math.isnan(rho)) and rho >= RHO_MIN and rng >= RANGE_MIN
        summary_rows.append({
            "model": model, "arm": arm, "role": role, "candidate": cand,
            "n_cells": len(vals), "spearman_occ8": round(rho, 3),
            "range": round(rng, 3), "usable": usable,
            "bad_parses": int(grp["n_bad"].sum()),
            "thr_occ8": round(implied_threshold(surf, 8), 2),
            "thr_occ4": round(implied_threshold(surf, 4), 2),
        })

    # Pairwise distances stay per (model, arm): they diagnose style disagreement
    # under identical conditions.
    summary = pd.DataFrame(summary_rows)
    for (model, arm, role), grp in summary.groupby(["model", "arm", "role"]):
        cands = sorted(grp.loc[grp.usable, "candidate"])
        for s, t in itertools.combinations(cands, 2):
            ss, tt = surfaces[(model, arm, role, s)], surfaces[(model, arm, role, t)]
            diffs = [abs(ss[c] - tt[c]) for c in cells if c in ss and c in tt]
            pair_rows.append({
                "model": model, "arm": arm, "role": role, "style_a": s, "style_b": t,
                "d_inf": round(max(diffs), 3) if diffs else float("nan"),
                "mean_abs_dp": round(float(np.mean(diffs)), 3) if diffs else float("nan"),
            })

    # Central behaviour is defined WITHIN a (model, role), across its usable
    # (style, arm) surfaces. Pooling happens ONLY over presentation dimensions
    # (prompt style, endpoint/grammar arm) — NEVER across models, NEVER across
    # agent roles, and (when contexts are added) NEVER across social contexts:
    # those are the measured objects, each its own value function (user
    # decisions 2026-07-31). dist_to_median = max-cell distance of a
    # (style, arm) surface to its own (model, role) median surface.
    for (model, role), grp in summary.groupby(["model", "role"]):
        usable_pairs = [(r.arm, r.candidate) for r in grp.itertuples() if r.usable]
        if len(usable_pairs) < 2:
            continue
        med = {c: float(np.median([surfaces[(model, a, role, s)][c]
                                   for a, s in usable_pairs
                                   if c in surfaces[(model, a, role, s)]]))
               for c in cells}
        for a, s in usable_pairs:
            ss = surfaces[(model, a, role, s)]
            d = max(abs(ss[c] - med[c]) for c in cells if c in ss)
            summary.loc[(summary.model == model) & (summary.arm == a)
                        & (summary.role == role)
                        & (summary.candidate == s), "dist_to_median"] = round(d, 3)

    pairs = pd.DataFrame(pair_rows)
    summary_path = results_dir / "ratio_consistency_summary.csv"
    pairs_path = results_dir / "ratio_consistency_pairs.csv"
    summary.to_csv(summary_path, index=False)
    pairs.to_csv(pairs_path, index=False)

    # ---- Protocol-validity table -----------------------------------------
    # Per (model, role, endpoint): is the grammar arm's value function
    # VOLUNTARY behaviour (plain arm parses cleanly and agrees) or a
    # FORCED-CHOICE construction (grammar renormalising a model that wasn't
    # answering)? plain bad-parse % + mean |effective_plain - grammar| gap.
    BAD_MAX, GAP_MAX = 0.05, 0.05
    validity_rows = []
    for (model, role), _ in df.groupby(["model", "agent_role"]):
        for endpoint in ("completions", "chat"):
            dplain = df[(df.model == model) & (df.agent_role == role)
                        & (df.arm == endpoint)]
            dgram = df[(df.model == model) & (df.agent_role == role)
                       & (df.arm == f"{endpoint}+grammar")]
            if dplain.empty or dgram.empty:
                continue
            bad = dplain.n_bad.sum() / max(dplain.n_samples.sum(), 1)
            key = ["candidate", "n_similar", "n_occupied"]
            m = dplain.merge(dgram, on=key, suffixes=("_p", "_g"))
            m = m.dropna(subset=["move_rate_effective_p", "move_rate_effective_g"])
            gap = (float((m.move_rate_effective_p - m.move_rate_effective_g).abs().mean())
                   if len(m) else float("nan"))
            n_dead = int((dplain.n_move + dplain.n_stay == 0).sum())
            verdict = ("voluntary" if bad < BAD_MAX and gap == gap and gap < GAP_MAX
                       else "forced-choice" if bad >= BAD_MAX
                       else "inconsistent (investigate)")
            validity_rows.append({"model": model, "role": role, "endpoint": endpoint,
                                  "plain_bad_pct": round(100 * bad, 2),
                                  "plain_vs_grammar_gap": round(gap, 3) if gap == gap else None,
                                  "cells_all_unparseable": n_dead,
                                  "verdict": verdict})
    validity = pd.DataFrame(validity_rows)
    validity_path = results_dir / "ratio_validity.csv"
    if len(validity):
        validity.to_csv(validity_path, index=False)

    # PER-(MODEL, ROLE) winner ranking: the ratio style (G0 is an anchor,
    # never a winner) with the lowest mean distance to that (model, role)
    # median, over the arms where it is usable (>= ARMS_MIN arms).
    ARMS_MIN = 2
    ratio_styles = summary[~summary.candidate.str.startswith("G0")]
    md = ["# Ratio-prompt consistency summary — per-(model, role) winners", "",
          f"- coherence: rho>={RHO_MIN} (occ8) and range>={RANGE_MIN}",
          "- central behaviour = median surface WITHIN each (model, role) over its "
          "usable (style, arm) surfaces. Pooled ONLY over presentation (styles, "
          "endpoint arms) — never across models, roles, or social contexts.",
          f"- inputs: {summary[['model', 'arm', 'role']].drop_duplicates().shape[0]} (model, arm, role) sets",
          "- all rates are EFFECTIVE move rates n_move/(n_move+n_stay) — the "
          "production-faithful quantity (production retries until parseable).", ""]
    if len(validity):
        md += ["## Protocol validity — voluntary vs forced-choice value functions", "",
               "| model | role | endpoint | plain bad % | plain-vs-grammar gap | dead cells | verdict |",
               "|---|---|---|---|---|---|---|"]
        for r in validity.itertuples():
            md.append(f"| {r.model} | {r.role} | {r.endpoint} | {r.plain_bad_pct}% | "
                      f"{r.plain_vs_grammar_gap} | {r.cells_all_unparseable} | **{r.verdict}** |")
        md += ["", "voluntary = plain arm parses cleanly (<5% bad) AND agrees with the "
               "grammar arm (mean |dP| < 0.05): grammar readings reflect natural behaviour. "
               "forced-choice = the model often does not answer unconstrained; grammar "
               "readings are protocol-constructed — interpret as behaviour UNDER the "
               "production protocol, not as preference.", ""]
    md += ["## Winner ranking (lower dist better)", "",
          "| model | role | rank | candidate | usable arms | mean dist to median | bad parses |",
          "|---|---|---|---|---|---|---|"]
    winners = {}
    for (model, role), grp in ratio_styles[ratio_styles.usable].groupby(["model", "role"]):
        if "dist_to_median" not in grp:
            continue
        rows = []
        for cand, g in grp.groupby("candidate"):
            g = g.dropna(subset=["dist_to_median"])
            if g["arm"].nunique() >= ARMS_MIN:
                rows.append({"candidate": cand, "arms": g["arm"].nunique(),
                             "dist": g["dist_to_median"].mean(),
                             "bad": int(g["bad_parses"].sum())})
        rows.sort(key=lambda r: (r["dist"], r["bad"]))
        for rank, r in enumerate(rows, 1):
            md.append(f"| {model} | {role} | {rank} | `{r['candidate']}` | {r['arms']} | "
                      f"{r['dist']:.3f} | {r['bad']} |")
        if rows:
            winners[(model, role)] = rows[0]
            if "_percent" in rows[0]["candidate"]:
                md.append(f"| {model} | {role} |  | **WARNING: percent style leads — "
                          f"accept only if it agrees with the count forms (pairs file)** |  |  |  |")
    md.append("")
    split = [m for m in {mk for mk, _ in winners}
             if len({winners[k]["candidate"] for k in winners if k[0] == m}) > 1]
    if split:
        md.append(f"**NOTE:** red and blue winners differ for: {', '.join(sorted(split))} — "
                  "Phase B samples each (model, role)'s own winner; roles are separate "
                  "value functions and are never merged.")
    if winners and len({w["candidate"] for w in winners.values()}) > 1:
        md.append("**NOTE:** winners differ across (model, role) sets — no global "
                  "style is forced; Phase B follows the per-set winners.")
    md.append("\nPer-(model, arm, role, style) coherence: "
              f"`{summary_path.name}`; pairwise distances: `{pairs_path.name}`.")
    md_path = results_dir / "ratio_consistency_summary.md"
    md_path.write_text("\n".join(md) + "\n")

    print(f"wrote {summary_path}\nwrote {pairs_path}\nwrote {md_path}")
    for (model, role), w in sorted(winners.items()):
        print(f"  {model:24s} [{role:4s}] winner: {w['candidate']} "
              f"(dist {w['dist']:.3f}, {w['arms']} arms)")


if __name__ == "__main__":
    main()
