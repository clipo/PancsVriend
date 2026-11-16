#!/usr/bin/env python3
"""Check a SANITY (sequential sampled) value function against the EXACT one.

    # report only (no server needed)
    python value_functions/comparison/sanity_vs_exact.py --label qwen3.6-27b-chat-grammar

    # with escalation: ONE round of fresh draws for the census misses
    python value_functions/comparison/sanity_vs_exact.py --label ... \
        --url http://127.0.0.1:8085/v1/chat/completions --escalate 300 --max-n 1200

WHAT THIS IS FOR
The sanity arm exists to be compared against the exact log-probability tables,
and is always built AFTER them, so a missing exact table is an error.

TWO CHECKS, TWO JOBS (user decision 2026-11-16)
  * GOODNESS OF FIT (step 2) is the census result REPORTED IN THE PAPER: one
    test per model, all cells, census draws only. The cross-model table and
    figure are census_goodness_of_fit.py (a pipeline stage).
  * The PER-CELL CHECK (steps 3-4) is the INTERNAL sanity check that the
    extracted table is a good cell-by-cell replacement for sampling. It is
    not a paper result: near-0/1 cells cannot be resolved by sampling at this
    scale (BEYOND_CAP), and it targets single-cell errors the global test is
    built to overlook.

THE DESIGN (replaced the pooled-escalation verdict 2026-11-16, reviewed twice)
1. Census only. Stage 1 reads the census draws from the census RAW file (the
   stored table may already hold escalation draws). Denominator = valid
   replies (move + stay). The exact side is the composite null p in [lo, hi]
   (the exact table's mass-bound ci95).
2. GLOBAL test over all cells: deviance G vs a parametric bootstrap from the
   exact table (census_stats.deviance_omnibus). No selection is involved.
3. Descriptive: census Wilson misses against their EXACT null expectation
   (Poisson-binomial). At n = 100, Wilson over-covers p = 0/1 cells and
   under-covers p ~ 1e-3, so "5% of 540 = 27" is not the expectation.
4. CELL test. The family F = census misses, each counted ONCE. Each gets ONE
   round of FRESH draws: never-issued seeds (the stage string carries the
   census id; indices start past every earlier one), n sized from the census
   alone for `--power` against an odds ratio of `--odds-ratio` in the census
   deviation's direction (floor --escalate, cap --max-n). Two-sided exact
   binomial test of the fresh draws ONLY, maximised over [lo, hi]; Holm across F.
   Cells not rejected whose power at their n is below --power are BEYOND_CAP:
   an instrument limit (near-saturated p), not a pass.
5. Verdicts, alpha pre-committed per model (--alpha, each check at alpha):
   goodness_of_fit PASS if its p > alpha, else FAIL;
   sanity_check DISCREPANCY if any Holm-adjusted cell p <= alpha, UNRESOLVED if
   some cell of F has no fresh draws, PASS otherwise. The top-level verdict
   (exit code) is PASS only when both pass.

WHY NOT POOL (the pre-2026-11-16 method)
Escalation used to merge the fresh draws into the census counts and re-test the
pool, over up to 3 adaptive rounds, judging the residual misses against
Binomial(escalated, 0.05). Three faults: (a) a cell chosen because its first
draws were extreme kept those draws, so under the null it stayed a miss about
half the time, not 5% of the time (hermes and olmo were flagged DISCREPANCY on
exactly that); (b) every round restarted its seed index at 0 and replayed the
previous round's draws, counted twice; (c) the denominator counted a cell once
per round. Further rounds are refused for the same reason as pooling: each
re-selects on the previous round's draws.

A census can be escalated ONCE. A second escalation of the same census is
refused — rebuild the census (value_functions/sampling/census.py) instead.

Exit codes: 0 PASS, 2 inputs missing, 3 DISCREPANCY, 4 UNRESOLVED or refused.
"""
import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent
for _p in (str(THIS), str(REPO), str(REPO / "prompt_refinement"),
           str(REPO / "value_functions" / "sampling")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                    # noqa: E402
from scipy.stats import binomtest                     # noqa: E402
from value_functions.paths import (LOGPROB_TABLES_DIR, SANITY_DIR,  # noqa: E402
                                   SANITY_TABLES_DIR, SEQCHECK_PLOTS_DIR)
from sampling_common import wilson_ci                 # noqa: E402
import census_stats as cs                             # noqa: E402

STYLE_DEFAULT = "R3_dual_count"
SCENARIOS = ["baseline", "race_white_black", "ethnic_asian_hispanic",
             "income_high_low", "political_liberal_conservative", "green_yellow"]


def exact_path(label, scen, style):
    return LOGPROB_TABLES_DIR / f"vf_{label}-lp__{scen}__{style}.json"


def sanity_path(label, scen, style):
    return SANITY_TABLES_DIR / f"vf_{label}-sanity__{scen}__{style}.json"


def census_raw_path(label, scen, style):
    return SANITY_DIR / "raw" / f"vf_{label}-sanity__{scen}__{style}_raw.jsonl.gz"


def escalate_raw_path(label, scen, style):
    return SANITY_DIR / "raw" / f"vf_{label}-sanity__{scen}__{style}_escalate_raw.jsonl.gz"


def cell_key(role, n_sim, n_occ):
    return f"{role}|{n_sim}|{n_occ}"


def census_id(meta):
    """Identifies one census build: its creation stamp (census.py rebuilds a
    census into a fresh table, so the stamp changes with every build)."""
    return str(meta.get("created") or "unknown")


def already_escalated(meta):
    """Legacy tables carry escalated_cells; new ones an escalations record."""
    cid = census_id(meta)
    return bool(meta.get("escalated_cells")) or any(
        e.get("census_id") == cid for e in meta.get("escalations", []))


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


def census_counts(label, scen, style, table):
    """{cell_key: [move, stay, bad]} from the census draws only.

    From the census raw file; if it is missing, the table is usable only when it
    was never escalated (then table == census)."""
    raw = census_raw_path(label, scen, style)
    out = defaultdict(lambda: [0, 0, 0])
    if raw.exists():
        with gzip.open(raw, "rt") as fh:
            for line in fh:
                d = json.loads(line)
                if d.get("_meta"):
                    continue
                c = out[cell_key(d["agent_role"], d["n_similar"], d["n_occupied"])]
                p = (d.get("parse") or "").upper()
                c[0 if p == "MOVE" else 1 if p == "STAY" else 2] += 1
        return out
    if already_escalated(table["meta"]):
        raise FileNotFoundError(f"{raw} missing and the table holds escalation draws")
    for role, rows in table["compositions"].items():
        for c in rows:
            out[cell_key(role, c["n_similar"], c["n_occupied"])] = [
                c.get("n_move", 0), c.get("n_stay", 0), c.get("n_bad", 0)]
    return out


def stage1(label, scenarios, style):
    """One row per cell: exact p and interval, census counts, Wilson CI, inside?"""
    rows = []
    for scen in scenarios:
        e = json.loads(exact_path(label, scen, style).read_text())
        table = json.loads(sanity_path(label, scen, style).read_text())
        cen = census_counts(label, scen, style, table)
        for role in e["compositions"]:
            for c in e["compositions"][role]:
                p = c["p_move_effective"]
                if p is None:
                    continue
                mv, st, _ = cen.get(cell_key(role, c["n_similar"], c["n_occupied"]), (0, 0, 0))
                n = mv + st
                if n == 0:
                    continue
                lo, hi = wilson_ci(mv, n)
                elo, ehi = (c.get("ci95") or [p, p])
                rows.append({"scenario": scen, "role": role,
                             "n_similar": c["n_similar"], "n_occupied": c["n_occupied"],
                             "p_exact": p, "exact_lo": min(elo, p), "exact_hi": max(ehi, p),
                             "p_sanity": mv / n, "n_samples": n, "n_move": mv,
                             "ci_low": lo - cs.CI_EPS, "ci_high": hi + cs.CI_EPS,
                             "inside": not cs.wilson_miss(mv, n, p)})
    return rows


def recorded_fresh(label, scenarios, style):
    """Fresh counts recorded by an earlier escalation of the CURRENT census."""
    out = {}
    for scen in scenarios:
        meta = json.loads(sanity_path(label, scen, style).read_text())["meta"]
        cid = census_id(meta)
        for e in meta.get("escalations", []):
            if e.get("census_id") == cid:
                for k, v in e["cells"].items():
                    out[(scen, k)] = (v["move"], v["stay"])
    return out


def cell_tests(F, fresh, args):
    """Fresh-draw test per family cell; Holm across the tested ones."""
    a_cells = args.alpha
    for r in F:
        f = fresh.get((r["scenario"], cell_key(r["role"], r["n_similar"], r["n_occupied"])))
        direction = 1 if r["p_sanity"] >= r["p_exact"] else -1
        orr = args.odds_ratio if direction > 0 else 1 / args.odds_ratio
        if f is None:
            r.update(fresh_move=None, fresh_n=None, p_fresh=None, status="NOT_TESTED")
            continue
        mv, st = f
        n = mv + st
        ci = binomtest(mv, n).proportion_ci(method="exact") if n else None
        r.update(fresh_move=mv, fresh_n=n,
                 p_fresh=cs.composite_binom_p(mv, n, r["exact_lo"], r["exact_hi"], r["p_exact"]),
                 fresh_ci_low=ci.low if ci else None, fresh_ci_high=ci.high if ci else None,
                 power=round(cs.power(n, r["p_exact"], orr, a_cells), 3))
    tested = [r for r in F if r["p_fresh"] is not None]
    for r, a in zip(tested, cs.holm([r["p_fresh"] for r in tested])):
        r["p_holm"] = a
        r["status"] = ("DISAGREES" if a <= a_cells else
                       "CONSISTENT" if r["power"] >= args.power else "BEYOND_CAP")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True, help="e.g. qwen3.6-27b-chat-grammar")
    ap.add_argument("--style", default=STYLE_DEFAULT)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--escalate", type=int, default=0,
                    help="escalate the census misses once (needs --url); the MINIMUM "
                         "fresh draws per cell")
    ap.add_argument("--max-n", type=int, default=1200, help="ceiling of fresh draws per cell")
    ap.add_argument("--odds-ratio", type=float, default=2.0,
                    help="alternative the fresh n is sized for (odds x or / this)")
    ap.add_argument("--power", type=float, default=0.8, help="target power for that alternative")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="per-model error rate, applied to each check separately")
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
    try:
        rows = stage1(args.label, scenarios, args.style)
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    F = [r for r in rows if not r["inside"]]
    fresh = recorded_fresh(args.label, scenarios, args.style)
    if args.escalate and F and args.url:
        scen_F = {r["scenario"] for r in F}
        done = [s for s in scen_F
                if already_escalated(json.loads(sanity_path(args.label, s, args.style).read_text())["meta"])]
        if done:
            print(f"[refused] the census for {', '.join(sorted(done))} was already escalated; a "
                  "second round would re-select on its draws. Rebuild the census to escalate again.",
                  file=sys.stderr)
            return 4
        a_cells = args.alpha
        per_cell = {}
        for r in F:
            n, pw = cs.size_fresh(r["p_exact"], 1 if r["p_sanity"] >= r["p_exact"] else -1,
                                  args.odds_ratio, args.power, a_cells, args.escalate, args.max_n)
            per_cell[(r["scenario"], r["role"], r["n_similar"], r["n_occupied"])] = n
        print(f"  escalating {len(F)} cell(s) once: {sum(per_cell.values()):,} fresh draws")
        fresh = escalate(args, F, per_cell)
    elif args.escalate and F:
        print("[warn] --escalate needs --url; skipping escalation", file=sys.stderr)
    verdict = judge_and_write(args, rows, fresh)
    return {"PASS": 0, "DISCREPANCY": 3}.get(verdict, 4)


def judge_and_write(args, rows, fresh):
    """Both checks on one model; writes the json and csv.

    goodness_of_fit — REPORTED IN THE PAPER: the global deviance test on the
                      census (PASS if p > alpha).
    sanity_check    — INTERNAL: the per-cell fresh-draw check of the census
                      misses (PASS / DISCREPANCY / UNRESOLVED).
    args needs: label, alpha, odds_ratio, power, out_dir. `fresh` maps
    (scenario, cell_key) -> (move, stay) of fresh draws. Returns the combined
    verdict ('verdict': PASS only if both pass), used for exit codes."""
    K = [r["n_move"] for r in rows]; N = [r["n_samples"] for r in rows]
    P = [r["p_exact"] for r in rows]
    exp_miss, tail = cs.expected_misses(N, P)
    omni = cs.deviance_omnibus(K, N, P, [r["exact_lo"] for r in rows],
                               [r["exact_hi"] for r in rows])
    gof_verdict = "PASS" if omni["p_mc"] > args.alpha else "FAIL"
    F = [r for r in rows if not r["inside"]]
    print(f"{args.label}: {len(rows)} cells, {sum(N):,} census draws")
    print(f"  GOODNESS OF FIT (paper): deviance G {omni['G']} vs null mean {omni['G_null_mean']} "
          f"(95% {omni['G_null_q95']}), p {omni['p_mc']:.3f} -> {gof_verdict}")
    cell_tests(F, fresh, args)
    status = defaultdict(int)
    for r in F:
        status[r["status"]] += 1
    if status["DISAGREES"]:
        sanity = "DISCREPANCY"
    elif status["NOT_TESTED"]:
        sanity = "UNRESOLVED"
    else:
        sanity = "PASS"
    print(f"  PER-CELL SANITY CHECK (internal): census misses {len(F)} (null expectation "
          f"{exp_miss:.1f}); fresh draws: {status['CONSISTENT']} consistent, "
          f"{status['BEYOND_CAP']} beyond cap, {status['DISAGREES']} disagree, "
          f"{status['NOT_TESTED']} not tested -> {sanity}")
    verdict = ("DISCREPANCY" if gof_verdict == "FAIL" or sanity == "DISCREPANCY"
               else "UNRESOLVED" if sanity == "UNRESOLVED" else "PASS")
    summary = {"label": args.label, "cells": len(rows), "census_draws": int(sum(N)),
               "goodness_of_fit": {"role": "reported in the paper", **omni,
                                   "alpha": args.alpha, "verdict": gof_verdict},
               "sanity_check": {
                   "role": "internal: is the extracted table a good replacement for sampling, "
                           "cell by cell? Not a paper result",
                   "design": "census misses (each once); one fresh round; exact binomial vs "
                             "composite null; Holm",
                   "alpha": args.alpha, "odds_ratio": args.odds_ratio, "power_target": args.power,
                   "census_misses": len(F), "census_misses_null_expected": round(exp_miss, 2),
                   "p_census_misses_ge_obs": round(tail(len(F)), 4),
                   "cells_disagree": status["DISAGREES"], "cells_consistent": status["CONSISTENT"],
                   "cells_beyond_cap": status["BEYOND_CAP"], "cells_not_tested": status["NOT_TESTED"],
                   "min_p_holm": round(min([r["p_holm"] for r in F if "p_holm" in r], default=1.0), 4),
                   "verdict": sanity},
               "verdict": verdict,
               "median_abs_diff": round(float(np.median([abs(r["p_exact"] - r["p_sanity"])
                                                        for r in rows])), 6)}
    out_dir = Path(args.out_dir) if args.out_dir else SEQCHECK_PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"sanity_vs_exact_{args.label}.json").write_text(json.dumps(summary, indent=1))
    cols = list(rows[0]) + ["fresh_move", "fresh_n", "fresh_ci_low", "fresh_ci_high",
                            "p_fresh", "p_holm", "power", "status"]
    with open(out_dir / f"sanity_vs_exact_{args.label}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=1))
    for r in F:
        print(f"  {r['scenario'][:14]:14s} {r['role']:4s} ({r['n_similar']},{r['n_occupied']}) "
              f"p_exact {r['p_exact']:.5f} census {r['n_move']}/{r['n_samples']} "
              f"fresh {r['fresh_move']}/{r['fresh_n']} p {r.get('p_fresh')} "
              f"holm {r.get('p_holm')} power {r.get('power')} -> {r['status']}")
    return verdict


def escalate(args, F, per_cell):
    """ONE round of fresh SEQUENTIAL draws for the family cells.

    Seeds are never-issued: the stage string carries the census id (a rebuilt
    census restarts n at 100, so an index alone could repeat an archived
    census's seeds) and indices start at the cell's current sample count
    (past every earlier index of this census). Fresh counts are kept apart
    from the census and recorded in the table's meta["escalations"]; the
    merged counts still go into the table, which is a data store — the verdict
    never reads them. Returns {(scenario, cell_key): (move, stay)}.
    """
    from build_value_function import sample_into_counts, _blank_counts, _assemble_artifact
    from sampling_common import RawWriter, role_keywords
    from ratio_prompt_templates import RATIO_CANDIDATES
    by_scen = defaultdict(set)
    for r in F:
        by_scen[r["scenario"]].add((r["role"], (r["n_similar"], r["n_occupied"])))
    fresh = {}
    for scen, want in sorted(by_scen.items()):
        p = sanity_path(args.label, scen, args.style)
        vf = json.loads(p.read_text())
        roles = sorted(vf["compositions"])
        counts = _blank_counts(roles)
        for role in roles:
            for c in vf["compositions"][role]:
                key = (role, (c["n_similar"], c["n_occupied"]))
                if key in counts:
                    counts[key] = {"move": c["n_move"], "stay": c["n_stay"],
                                   "bad": c.get("n_bad", 0), "samples": c["n_samples"]}
        before = {k: dict(counts[k]) for k in want}
        start = {key: c["samples"] for key, c in counts.items()}
        m = vf["meta"]
        cid = census_id(m)
        stage = f"escalate@{cid}"
        cells = sorted(f"{r}:{c}" for r, c in want)
        tpl, fn = RATIO_CANDIDATES[args.style]
        raw_meta = {**m, "stage": stage, "census_id": cid, "escalated_cells": cells,
                    "fresh_n": {f"{r}:{c}": per_cell[(scen, r, c[0], c[1])] for r, c in want},
                    "sample_index_start": "cell n_samples before this round"}
        raw_meta.pop("escalations", None)
        with RawWriter(escalate_raw_path(args.label, scen, args.style), raw_meta, append=True) as raw:
            sample_into_counts(
                counts, lambda role, cell: (per_cell[(scen, role, cell[0], cell[1])]
                                            if (role, cell) in want else 0),
                roles, role_keywords(scen, m.get("scenario_file")), args.style, tpl, fn,
                args.url, m["model"], m["temperature"], True, 1, m.get("seed") or 0,
                raw, f"escalate {scen}", None, seed_ctx=(stage, scen),
                index_offset_fn=lambda role, cell: start[(role, cell)])
        rec = {}
        for (role, cell) in want:
            d = {x: counts[(role, cell)][x] - before[(role, cell)][x] for x in ("move", "stay", "bad")}
            rec[cell_key(role, *cell)] = d
            fresh[(scen, cell_key(role, *cell))] = (d["move"], d["stay"])
        vf = _assemble_artifact(m, counts, roles)
        vf["meta"].setdefault("escalations", []).append(
            {"census_id": cid, "seed_stage": stage, "date": datetime.now().isoformat(timespec="seconds"),
             "cells": rec})
        vf["meta"]["escalated_cells"] = cells
        p.write_text(json.dumps(vf, indent=1))
        print(f"  escalated {len(want)} cell(s) in {scen}")
    return fresh


if __name__ == "__main__":
    raise SystemExit(main())
