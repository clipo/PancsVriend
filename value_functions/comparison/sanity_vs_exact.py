#!/usr/bin/env python3
"""Check a SANITY (sequential sampled) value function against the EXACT one.

    # report only (no server needed)
    python value_functions/comparison/sanity_vs_exact.py --label qwen3.6-27b-chat-grammar

    # with escalation: re-sample the disagreeing cells against a live server
    python value_functions/comparison/sanity_vs_exact.py --label ... \
        --url http://127.0.0.1:8085/v1/chat/completions --escalate 300

WHAT THIS IS FOR
The sanity arm has no purpose on its own: it exists to be compared against the
exact log-probability tables, and it is always built AFTER them. So the exact
tables are a HARD REQUIREMENT here — a missing one is an error, not a silent
skip, because a sanity table with nothing to compare against is wasted GPU.

WHAT IT CHECKS
Every cell (45 compositions x 2 roles x 6 scenarios = 540), not the ~3%
spot-check in seqcheck_*. For each, the Wilson 95% interval on the sanity
proportion either contains the exact value or does not.

READING THE NUMBER — the part that is easy to get wrong
With 540 INDEPENDENT 95% intervals, about 5% (~27 cells) miss BY CHANCE even
when the exact tables are perfectly correct. So ~95% inside is the expectation
and 100% would be surprising. A stage-1 miss is therefore not evidence of an
error; it is a candidate.

ESCALATION (the same idea as seqcheck's, scaled up)
--escalate N buys N more sequential draws for each missing cell only, merges
them into that cell's counts (counts are sufficient statistics, so addition is
exact), and re-tests. A chance miss usually resolves as the interval tightens
around the true value; a real discrepancy persists. The verdict then compares
the RESIDUAL miss count against what chance still predicts for the escalated
subset (binomial upper tail), so the test is not "zero misses", which a correct
extractor would fail routinely.

Exit codes: 0 pass, 2 exact tables missing, 3 residual misses above chance.
"""
import argparse
import json
import math
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent
for _p in (str(THIS), str(REPO), str(REPO / "prompt_refinement"),
           str(REPO / "value_functions" / "sampling")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                    # noqa: E402
from value_functions.paths import (LOGPROB_TABLES_DIR, SANITY_DIR,  # noqa: E402
                                   SANITY_TABLES_DIR, SEQCHECK_PLOTS_DIR)
from sampling_common import wilson_ci                 # noqa: E402

STYLE_DEFAULT = "R3_dual_count"
# Saturated cells (k = 0 or k = n) have Wilson limits that are analytically
# exactly 0 / 1 but land ~1e-16 inside in floating point. Without this the
# containment test rejects an exact value of exactly 0.0 or 1.0 against its own
# limit. 1e-9 is ~7 orders below the smallest disagreement worth calling real.
CI_EPS = 1e-9
SCENARIOS = ["baseline", "race_white_black", "ethnic_asian_hispanic",
             "income_high_low", "political_liberal_conservative", "green_yellow"]


def exact_path(label, scen, style):
    return LOGPROB_TABLES_DIR / f"vf_{label}-lp__{scen}__{style}.json"


def sanity_path(label, scen, style):
    return SANITY_TABLES_DIR / f"vf_{label}-sanity__{scen}__{style}.json"


def require_exact(label, scenarios, style):
    """The load-bearing precondition: no exact table, no comparison."""
    missing = [s for s in scenarios if not exact_path(label, s, style).exists()]
    if missing:
        print(f"[error] sanity_vs_exact: no EXACT table for {label!r} in "
              f"{', '.join(missing)} (looked in {LOGPROB_TABLES_DIR}).", file=sys.stderr)
        print("        The sanity arm is a CHECK ON the exact tables and is always built "
              "after them; without them its tables cannot be interpreted.\n"
              "        Build them first:  python value_functions/logprob/logprob_value_function.py "
              f"--label {label}", file=sys.stderr)
        return False
    return True


def compare(label, scenarios, style):
    """One row per cell: exact p, sanity counts, Wilson CI, inside?"""
    rows = []
    for scen in scenarios:
        e = json.loads(exact_path(label, scen, style).read_text())
        sp = sanity_path(label, scen, style)
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        for role in e["compositions"]:
            lut = {(c["n_similar"], c["n_occupied"]): c for c in s["compositions"].get(role, [])}
            for c in e["compositions"][role]:
                key = (c["n_similar"], c["n_occupied"])
                d = lut.get(key)
                if d is None or c["p_move_effective"] is None:
                    continue
                n, mv = d.get("n_samples", 0), d.get("n_move", 0)
                lo, hi = wilson_ci(mv, n) if n else (0.0, 1.0)
                lo, hi = lo - CI_EPS, hi + CI_EPS
                rows.append({"scenario": scen, "role": role,
                             "n_similar": key[0], "n_occupied": key[1],
                             "p_exact": c["p_move_effective"],
                             "p_sanity": d.get("p_move_effective"),
                             "n_samples": n, "n_move": mv,
                             "ci_low": lo, "ci_high": hi,
                             "inside": bool(lo <= c["p_move_effective"] <= hi)})
    return rows


def binom_upper_p(k, n, p=0.05):
    """P(X >= k) for X ~ Binomial(n, p) — is the residual miss count above chance?"""
    from math import comb
    if n == 0:
        return 1.0
    return float(sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1)))


def draws_needed(p_hat, dev, z=1.96, frac=0.5):
    """Draws for the Wilson half-width at p_hat to be `frac` of the observed
    deviation — i.e. enough resolution to tell a real gap from noise.

    n = z^2 p(1-p) / (frac*dev)^2 . A cell whose exact and sanity values truly
    agree sees dev shrink as n grows, so this converges; a cell with a real gap
    keeps a finite target and gets resolved. Capped by --max-n, because a gap
    of 1e-4 would otherwise ask for millions of draws.

    Resolving by SAMPLING rather than by declaring a practical floor is the
    user's decision (2026-09-11): a floor would call a 0.004 disagreement
    acceptable by fiat, where more draws simply answer whether it is real.
    """
    target = max(frac * abs(dev), 1e-6)
    return int(math.ceil(z * z * max(p_hat * (1 - p_hat), 1e-6) / (target * target)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True, help="e.g. qwen3.6-27b-chat-grammar")
    ap.add_argument("--style", default=STYLE_DEFAULT)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--escalate", type=int, default=0,
                    help="extra sequential draws per disagreeing cell (needs --url). "
                         "With --adaptive the per-cell amount is sized from the observed "
                         "deviation instead and this becomes the MINIMUM.")
    ap.add_argument("--adaptive", action="store_true",
                    help="size each cell's extra draws from its deviation and repeat "
                         "until it resolves or hits --max-n (resolve by sampling, not a floor)")
    ap.add_argument("--max-n", type=int, default=4000, help="per-cell ceiling for --adaptive")
    ap.add_argument("--rounds", type=int, default=3, help="adaptive rounds")
    ap.add_argument("--url", default=None, help="live server for --escalate")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    scenarios = args.scenarios or SCENARIOS
    scenarios = [s for s in scenarios if sanity_path(args.label, s, args.style).exists()]
    if not scenarios:
        print(f"[error] no sanity tables for {args.label!r} under {SANITY_TABLES_DIR}",
              file=sys.stderr)
        return 2
    if not require_exact(args.label, scenarios, args.style):
        return 2

    rows = compare(args.label, scenarios, args.style)
    miss = [r for r in rows if not r["inside"]]
    n, k = len(rows), len(miss)
    exp = 0.05 * n
    print(f"{args.label}: {n} cells, {n-k} inside CI ({100.0*(n-k)/n:.1f}%), "
          f"{k} outside (chance expects ~{exp:.0f})")

    escalated = 0
    if args.escalate and miss:
        if not args.url:
            print("[warn] --escalate needs --url; skipping escalation", file=sys.stderr)
        else:
            rounds = args.rounds if args.adaptive else 1
            for rnd in range(rounds):
                if not miss:
                    break
                if args.adaptive:
                    todo = []
                    for r in miss:
                        want = min(draws_needed(r["p_sanity"],
                                                abs(r["p_exact"] - r["p_sanity"])), args.max_n)
                        extra = max(want - r["n_samples"], 0)
                        if extra >= max(args.escalate, 1):
                            todo.append((r, extra))
                    if not todo:
                        print(f"  round {rnd+1}: every remaining cell is at the --max-n "
                              f"ceiling ({args.max_n}); stopping")
                        break
                    print(f"  round {rnd+1}: {len(todo)} cell(s), "
                          f"{sum(e for _, e in todo):,} extra draws")
                    escalated += escalate(args, [r for r, _ in todo], rows,
                                          per_cell={(r["scenario"], r["role"],
                                                     r["n_similar"], r["n_occupied"]): e
                                                    for r, e in todo})
                else:
                    escalated = escalate(args, miss, rows)
                rows = compare(args.label, scenarios, args.style)
                miss = [r for r in rows if not r["inside"]]
                print(f"  after round {rnd+1}: {len(rows)-len(miss)} inside "
                      f"({100.0*(len(rows)-len(miss))/len(rows):.1f}%), {len(miss)} outside")

    # Verdict. Without escalation, judge stage 1 against chance directly.
    tested = escalated or n
    p_chance = binom_upper_p(len(miss), tested)
    verdict = "PASS" if p_chance > 0.05 else "DISCREPANCY"
    summary = {"label": args.label, "cells": n, "inside": n - k,
               "stage1_outside": k, "chance_expected": round(exp, 1),
               "escalated_cells": escalated, "escalate_draws": args.escalate,
               "residual_outside": len(miss),
               "p_residual_above_chance": round(p_chance, 4),
               "verdict": verdict,
               "median_abs_diff": round(float(np.median(
                   [abs(r["p_exact"] - r["p_sanity"]) for r in rows])), 6)}
    out_dir = Path(args.out_dir) if args.out_dir else SEQCHECK_PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"sanity_vs_exact_{args.label}.json").write_text(json.dumps(summary, indent=1))
    import csv
    with open(out_dir / f"sanity_vs_exact_{args.label}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary, indent=1))
    return 0 if verdict == "PASS" else 3


def escalate(args, miss, rows, per_cell=None):
    """Buy more SEQUENTIAL draws for the disagreeing cells only.

    per_cell maps (scenario, role, n_similar, n_occupied) -> extra draws; without
    it every listed cell gets --escalate.
    """
    from build_value_function import (sample_into_counts, _blank_counts,
                                      _assemble_artifact, build_rows)
    from sampling_common import RawWriter, role_keywords
    from ratio_prompt_templates import RATIO_CANDIDATES
    by_scen = {}
    for r in miss:
        by_scen.setdefault(r["scenario"], set()).add((r["role"], (r["n_similar"], r["n_occupied"])))
    done = 0
    for scen, cells in sorted(by_scen.items()):
        p = sanity_path(args.label, scen, args.style)
        vf = json.loads(p.read_text())
        roles = sorted(vf["compositions"])
        counts = _blank_counts(roles)
        for role in roles:                                  # seed from what is already there
            for c in vf["compositions"][role]:
                key = (role, (c["n_similar"], c["n_occupied"]))
                if key in counts:
                    counts[key] = {"move": c["n_move"], "stay": c["n_stay"],
                                   "bad": c.get("n_bad", 0), "samples": c["n_samples"]}
        want = {(role, cell) for role, cell in cells}
        tpl, fn = RATIO_CANDIDATES[args.style]
        m = vf["meta"]
        raw_path = SANITY_DIR / "raw" / f"vf_{args.label}-sanity__{scen}__{args.style}_escalate_raw.jsonl.gz"
        with RawWriter(raw_path, {**m, "stage": "escalate", "extra": args.escalate}) as raw:
            sample_into_counts(
                counts, lambda role, cell: (
                    (per_cell or {}).get((scen, role, cell[0], cell[1]), args.escalate)
                    if (role, cell) in want else 0),
                roles, role_keywords(scen, m.get("scenario_file")), args.style, tpl, fn,
                args.url, m["model"], m["temperature"], True, 1, m.get("seed") or 0,
                raw, f"escalate {scen}", None, seed_ctx=("escalate", scen))
        vf = _assemble_artifact(m, counts, roles)
        vf["meta"]["escalated_cells"] = sorted(f"{r}:{c}" for r, c in want)
        vf["meta"]["escalate_draws"] = args.escalate
        p.write_text(json.dumps(vf, indent=1))
        done += len(want)
        print(f"  escalated {len(want)} cell(s) in {scen}")
    return done


if __name__ == "__main__":
    raise SystemExit(main())
