#!/usr/bin/env python3
"""Build a TRUSTWORTHY exact value function from repeated extractions + sampling.

    python value_functions/comparison/build_consensus_table.py --label llama-3.3-70b-chat-grammar

THE PROBLEM (measured 2026-09-11)
A single "exact" extraction is not reproducible for a minority of cells. Two
independent passes of llama's 540 cells, identical in prompt, llama.cpp build
(a4ce259, binary unchanged since 2026-07-14), gguf, server flags, extraction
parameters and client concurrency, gave 461 cells identical to 1e-6 and 79
that differ — 25 by more than 0.01, 2 by more than 0.30. Values are perfectly
stable WITHIN a session (5 repeats, zero spread) and differ ACROSS sessions.
So one extraction is a draw, not a measurement, and "re-extract and replace"
would swap one unreliable draw for another.

HOW A CELL EARNS ITS VALUE
Every cell is classified from N extraction runs plus sequential sampling:

  STABLE      every run agrees within --tol. Deterministic; adopted.

  Otherwise the distinct values are CANDIDATES, and sampling has to pick one.
  The first version of this script asked which candidate lay inside the
  sanity Wilson CI. That was wrong twice over (user, 2026-09-11): it picked a
  winner from the same n=100 draws that defined the interval, and its power
  was never stated — on 5 of 19 "resolved" cells the data were less than 100x
  likelier under the winner (Bayes factor < 100, twice ~4), the loser simply
  sat a hair outside an interval boundary. adjudication_power.py measured
  that. The rule is now a PRE-COMMITTED likelihood-ratio test on FRESH draws:

  n_decisive  the sample size at which deciding by the likelihood ratio
              L(v_a)/L(v_b) — L(v) = v^k (1-v)^(n-k), the binomial likelihood
              of the draws under candidate v — makes the wrong call with
              probability <= --alpha and no call with probability <= --beta,
              whichever candidate is true. Exact enumeration, not a normal
              approximation, since most candidates sit near 0 or 1.

  BEYOND_CAP  the closest candidates need more than --n-cap draws to tell
              apart (near-saturated cells: 0.0001 vs 0.0002 needs millions).
              Sampling cannot adjudicate; NOT adopted; the spread is reported
              and the cell is listed with the n it would take. This is a
              statement about the instrument's resolving power at a stated
              budget, not a materiality floor — raise --n-cap to buy more.
  PENDING     fresh draws are not yet enough; lists n_decisive and what is
              banked so far. No tool buys them yet, and none has been needed:
              the cross-session disagreement this script was written for
              turned out to be the chat template's date (LLAMA_CPP_SERVING_
              NOTES.md §6), and with the clock pinned every re-extraction
              reproduces (reextract_run2_*: 540/540 within 1e-8 on the seven
              date-free models). If a cell ever needs it, write
              adjudication_<label>.json in the form load_adjudication reads —
              {"cells": {"[scenario, role, n_similar, n_occupied]": {"n_move": k,
              "n_samples": n}}} —
              from SEQUENTIAL draws (recheck_cells.py has the sampling loop).
  RESOLVED    fresh draws >= n_decisive, the likelihood favours one candidate
              by a factor >= --bf AND that candidate is not itself rejected by
              the draws (two-sided exact binomial p >= 0.05). Adopted.
  UNRESOLVED  enough fresh draws, but no decisive winner, or the winner is
              rejected — the truth may be a value NO extraction produced.
              NOT adopted; a real finding, listed.

Candidates the sampler cannot separate at the cap are clustered and one
representative (earliest run) stands for the cluster; if such a cluster wins,
its internal spread is recorded as cluster_spread. Fresh draws come ONLY from
value_functions/results/sampled_small/adjudication/ — never from the sanity
census, whose escalated cells were chosen for disagreeing with the stored
table (outcome-dependent), and whose n=100 already informed the plan.

Nothing is overwritten: consensus_<label>.csv is written alongside the
originals with per-cell provenance, so every adopted value traces to the
evidence that justified it.
"""
import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent
for _p in (str(THIS), str(REPO), str(REPO / "prompt_refinement")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adjudication_power import binom_two_sided_p, log_lr, n_decisive   # noqa: E402
from value_functions.paths import (ADJUDICATION_DIR, LOGPROB_RAW_DIR,   # noqa: E402
                                   SANITY_TABLES_DIR, SEQCHECK_PLOTS_DIR)

STYLE = "R3_dual_count"
SCENARIOS = ["baseline", "race_white_black", "ethnic_asian_hispanic",
             "income_high_low", "political_liberal_conservative", "green_yellow"]
KEY = ("scenario", "role", "n_similar", "n_occupied")
STATUSES = ("STABLE", "RESOLVED", "PENDING", "UNRESOLVED", "BEYOND_CAP", "MISSING")
FIT_ALPHA = 0.05


def run1_from_traces(label):
    """Run 1 = the values in the stored extraction traces."""
    out = {}
    for scen in SCENARIOS:
        gz = LOGPROB_RAW_DIR / f"vflp_{label}__{scen}__{STYLE}_states.jsonl.gz"
        if not gz.exists():
            continue
        for line in gzip.open(gz, "rt"):
            r = json.loads(line)
            if r.get("_meta"):
                continue
            out[(scen, r["role"], r["n_similar"], r["n_occupied"])] = r["p_move"]
    return out


def run_from_csv(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[(r["scenario"], r["role"], int(r["n_similar"]), int(r["n_occupied"]))] = \
                float(r["p_reextracted"])
    return out


def sanity_counts(label):
    """The n=100 census (plus its escalations): reported for context, never adjudicating."""
    out = {}
    for scen in SCENARIOS:
        p = SANITY_TABLES_DIR / f"vf_{label}-sanity__{scen}__{STYLE}.json"
        if not p.exists():
            continue
        vf = json.loads(p.read_text())
        for role, rows in vf["compositions"].items():
            for c in rows:
                out[(scen, role, c["n_similar"], c["n_occupied"])] = (c.get("n_move", 0), c.get("n_samples", 0))
    return out


def adjudication_path(label):
    return ADJUDICATION_DIR / f"adjudication_{label}.json"


def load_adjudication(label):
    """Fresh draws: {key: (n_move, n_samples)}; empty if none bought yet."""
    p = adjudication_path(label)
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {tuple(json.loads(k)): (c["n_move"], c["n_samples"]) for k, c in d["cells"].items()}


def cluster(cands, alpha, beta, log_k, cap):
    """Union candidates the sampler cannot separate within the cap.

    cands: [(value, source)] sorted by run order. Returns (reps, plan_n) where
    reps = [(value, sources, spread)] one per cluster and plan_n = the largest
    pairwise n_decisive among representatives (None if only one cluster).
    Transitive merging can in principle join A and C through a B between them
    even when A-C alone is decidable; with 2-3 runs that is rare and errs
    toward NOT adopting, which is the safe side.
    """
    parent = list(range(len(cands)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pair_n = {}
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            nd = n_decisive(cands[i][0], cands[j][0], alpha, beta, log_k, cap)
            pair_n[(i, j)] = nd
            if nd is None:
                parent[find(i)] = find(j)
    groups = {}
    for i, (v, src) in enumerate(cands):
        groups.setdefault(find(i), []).append((v, src))
    reps = []
    for members in groups.values():
        vals = [v for v, _ in members]
        reps.append((members[0][0], [s for _, s in members], max(vals) - min(vals)))
    reps.sort(key=lambda r: r[0])
    if len(reps) == 1:
        return reps, None
    # plan on the representatives only
    plan = 0
    for a in range(len(reps)):
        for b in range(a + 1, len(reps)):
            nd = n_decisive(reps[a][0], reps[b][0], alpha, beta, log_k, cap)
            plan = max(plan, nd if nd is not None else cap + 1)
    return reps, plan


def classify(vals, adj, args, log_k):
    """vals: [(run_name, value)] in run order. adj: (k, n) fresh draws or None."""
    vs = [v for _, v in vals]
    spread = max(vs) - min(vs)
    base = {"spread": spread, "n_candidates": 1, "n_clusters": 1, "adopted": None,
            "cluster_spread": 0.0, "adj_k": "", "adj_n": "", "log10_bf": "",
            "fit_p": "", "n_decisive": ""}
    if spread <= args.tol:
        return {**base, "status": "STABLE", "adopted": vs[0],
                "why": f"all {len(vs)} runs agree (spread {spread:.2e})"}

    seen, cands = {}, []
    for name, v in vals:
        r = round(v, 12)
        if r in seen:
            cands[seen[r]] = (cands[seen[r]][0], cands[seen[r]][1] + "+" + name)
        else:
            seen[r] = len(cands)
            cands.append((v, name))
    reps, plan = cluster(cands, args.alpha, args.beta, log_k, args.n_cap)
    base.update(n_candidates=len(cands), n_clusters=len(reps))

    if plan is None:
        return {**base, "status": "BEYOND_CAP", "n_decisive": f">{args.n_cap}",
                "why": (f"{len(cands)} values within {spread:.2e}: telling the closest "
                        f"pair apart needs more than {args.n_cap} draws")}
    k, n = adj if adj else (0, 0)
    if n < plan:
        return {**base, "status": "PENDING", "n_decisive": plan, "adj_k": k, "adj_n": n,
                "why": f"{len(reps)} separable value(s); need {plan} fresh draws, have {n}"}

    # the pre-committed test on the fresh draws only
    ll = [k * math.log(max(v, 1e-12)) + (n - k) * math.log(max(1 - v, 1e-12)) for v, _, _ in reps]
    order = sorted(range(len(reps)), key=lambda i: -ll[i])
    lead, second = order[0], order[1]
    log10_bf = (ll[lead] - ll[second]) / math.log(10)
    fit_p = binom_two_sided_p(k, n, reps[lead][0])
    v_lead, srcs, cspread = reps[lead]
    base.update(adj_k=k, adj_n=n, n_decisive=plan, log10_bf=round(log10_bf, 3),
                fit_p=round(fit_p, 4), cluster_spread=cspread)
    if log10_bf >= log_k / math.log(10) and fit_p >= FIT_ALPHA:
        return {**base, "status": "RESOLVED", "adopted": v_lead,
                "why": (f"fresh {k}/{n}: BF 10^{log10_bf:.1f} for {v_lead:.6f} "
                        f"({'+'.join(srcs)}) over {reps[second][0]:.6f}; fit p={fit_p:.3f}")}
    why = (f"fresh {k}/{n} (p_hat {k/n:.4f}): " +
           (f"leader {v_lead:.6f} rejected, fit p={fit_p:.4f}" if fit_p < FIT_ALPHA
            else f"BF only 10^{log10_bf:.2f} for {v_lead:.6f} over {reps[second][0]:.6f}"))
    return {**base, "status": "UNRESOLVED", "why": why}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--runs", nargs="*", default=None,
                    help="CSVs of additional runs (default: reextract_run*_<label>.csv)")
    ap.add_argument("--tol", type=float, default=1e-6, help="STABLE if all runs agree within this")
    ap.add_argument("--alpha", type=float, default=0.01, help="max P(wrong call) of the LR test")
    ap.add_argument("--beta", type=float, default=0.05, help="max P(no call) of the LR test")
    ap.add_argument("--bf", type=float, default=100.0, help="likelihood-ratio threshold K")
    ap.add_argument("--n-cap", type=int, default=50000,
                    help="draws per cell beyond which candidates are declared inseparable")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    log_k = math.log(args.bf)

    runs = [("run1_stored", run1_from_traces(args.label))]
    paths = args.runs or sorted(
        str(p) for p in SEQCHECK_PLOTS_DIR.glob(f"reextract_run*_{args.label}.csv"))
    for p in paths:
        runs.append((Path(p).stem.replace(f"_{args.label}", ""), run_from_csv(p)))
    if len(runs) < 2:
        print(f"[error] need at least 2 runs; found {len(runs)}", file=sys.stderr)
        return 2
    print(f"{args.label}: {len(runs)} extraction run(s): {', '.join(n for n, _ in runs)}")
    adj = load_adjudication(args.label)
    census = sanity_counts(args.label)
    print(f"fresh adjudication draws on {len(adj)} cell(s); "
          f"LR test alpha={args.alpha} beta={args.beta} K={args.bf:g} cap={args.n_cap}")

    rows, tally = [], {s: 0 for s in STATUSES}
    for k in sorted(runs[0][1]):
        vals = [(n, d[k]) for n, d in runs if k in d]
        if len(vals) < len(runs):
            tally["MISSING"] += 1
            continue
        c = classify(vals, adj.get(k), args, log_k)
        tally[c["status"]] += 1
        cm, cn = census.get(k, ("", ""))
        rows.append({"scenario": k[0], "role": k[1], "n_similar": k[2], "n_occupied": k[3],
                     **{f"p_{n}": v for n, v in vals},
                     "census_k": cm, "census_n": cn, **c})

    out_dir = Path(args.out_dir) if args.out_dir else SEQCHECK_PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"consensus_{args.label}.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    tot = sum(tally.values())
    desc = {"STABLE": "deterministic across runs", "RESOLVED": "fresh draws decided it",
            "PENDING": "awaiting fresh draws", "UNRESOLVED": "fresh draws could not decide",
            "BEYOND_CAP": f"inseparable within {args.n_cap} draws", "MISSING": "absent from some run"}
    print()
    for s in STATUSES:
        if tally[s] or s not in ("MISSING",):
            print(f"  {s:<11}{tally[s]:4d}  ({100.0*tally[s]/tot:5.1f}%)  {desc[s]}")
    pend = [r for r in rows if r["status"] == "PENDING"]
    need = sum(r["n_decisive"] - r["adj_n"] for r in pend)
    adoptable = tally["STABLE"] + tally["RESOLVED"]
    print(f"\n  adopted: {adoptable}/{tot} ({100.0*adoptable/tot:.1f}%)")
    if pend:
        print(f"  PENDING needs {need:,} more fresh draws (~{need*1.065/3600:.1f} h at 1.065 s/draw)")
    print(f"wrote {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
