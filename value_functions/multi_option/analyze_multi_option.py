"""Summarise Stage 1 (exact reads) and the census (tests) of the multi-option runs.

Usage (from the repo root):
  python value_functions/multi_option/analyze_multi_option.py --stage1 <stage1 file> [--census <census file> --plan <plan file>]

Writes <stage1 name>_summary.json and _summary.md next to the Stage 1 file.

What is reported (design v4, sections 6 and 9):
  Q1 with grammar:     per arm, the gap (TV) between the path sum and each
                       first-token read over unsaturated states (median, 90th
                       percentile, largest), the unresolved share, and the
                       number of bound violations (must be 0).
  Q2 grammar effect:   per arm, the gap between the with-grammar path sum and
                       the free on-format share, the off-format share, and the
                       largest log-odds shift of one option.
  Q3 without grammar:  per arm, the gap between the free on-format share and
                       the free first-token read.
  Census:              the three primary likelihood-ratio tests (Holm-adjusted)
                       and the validation deviance tests.
"""

import argparse  # command-line options
import gzip  # compressed input
import json  # records
import math  # logs and NaN checks
from collections import defaultdict  # grouping
from pathlib import Path  # file names

import numpy as np  # percentiles

import multi_option_census as MC  # the tests

UNSAT = 0.995  # a state is unsaturated when its most likely option is below this


def load(path):
    """Header and records of a gzip JSONL file."""
    with gzip.open(path, "rt") as fh:
        rows = [json.loads(l) for l in fh]
    return rows[0], rows[1:]


def tv(p, q):
    """Total variation distance."""
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def nan(v):
    """True if a read is undefined (contains NaN)."""
    return any(isinstance(x, float) and math.isnan(x) for x in v)


def stats(xs):
    """Median, 90th percentile and largest of a list (None if empty)."""
    if not xs:
        return None
    a = np.asarray(xs)
    return {"n": len(xs), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def log_odds_shift(p, q):
    """Largest |log(p_o / (1 - p_o)) - log(q_o / (1 - q_o))| over options with both inside (0, 1)."""
    best = 0.0
    for a, b in zip(p, q):
        if 1e-12 < a < 1 - 1e-12 and 1e-12 < b < 1 - 1e-12:
            best = max(best, abs(math.log(a / (1 - a)) - math.log(b / (1 - b))))
    return best


def summarise_stage1(rows):
    """The per-arm numbers for Q1, Q2 and Q3."""
    rows = [r for r in rows if r["kind"] == "state"]
    by = {(r["task"], r["arm"], r["mode"], r["reversed"], r["i"]): r for r in rows}
    out = defaultdict(dict)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["arm"], r["mode"], r["reversed"])].append(r)
    for (task, arm, mode, rev), rs in sorted(groups.items()):
        key = f"{task}/{arm}/{mode}/{'reversed' if rev else 'listed'}"
        uns = [r for r in rs if max(r["p"] if mode == "grammar" else r["free_conditional"]) < UNSAT]
        target = (lambda r: r["p"]) if mode == "grammar" else (lambda r: r["free_conditional"])
        g = {"states": len(rs), "unsaturated": len(uns),
             "requests_per_state": float(np.mean([r["n_requests"] for r in rs])),
             "max_bound": max(r["bound"] for r in rs),
             "m_off": stats([r["m_off"] for r in uns])}
        for read in ("ft_mask", "ft_mask_t1", "ft_split", "ft_split_t1", "depth1", "depth2", "depth3"):
            ok = [r for r in uns if not nan(r[read])]
            g[f"gap_{read}"] = stats([tv(target(r), r[read]) for r in ok])
            g[f"undefined_{read}"] = len(uns) - len(ok)
        if mode == "grammar":  # the bound must hold on every state (a violation is a bug)
            g["bound_violations"] = sum(
                1 for r in rs for d in (1, 2, 3)
                if not nan(r[f"depth{d}"]) and tv(r["p"], r[f"depth{d}"]) > r[f"m_off{d}"] + 2 * r["bound"] + 1e-6)
        else:  # Q2: compare with the same state read with grammar
            pairs = [(r, by.get((task, arm, "grammar", rev, r["i"]))) for r in rs]
            pairs = [(f, gr) for f, gr in pairs if gr is not None]
            if pairs:
                g["off_format"] = stats([f["off_format"] for f, _ in pairs])
                g["gap_grammar_vs_free_conditional"] = stats(
                    [tv(gr["p"], f["free_conditional"]) for f, gr in pairs if max(gr["p"]) < UNSAT])
                g["log_odds_shift"] = stats(
                    [log_odds_shift(gr["p"], f["free_conditional"]) for f, gr in pairs if max(gr["p"]) < UNSAT])
        out[key] = g
    return dict(out)


def run_tests(plan, census_rows, stage1_rows):
    """The census tests: primary likelihood-ratio tests (Holm) and validation deviance tests."""
    by = {(r["task"], r["arm"], r["mode"], r["reversed"], r["i"]): r for r in stage1_rows if r["kind"] == "state"}
    gram = {(r["task"], r["arm"], r["i"]): r for r in stage1_rows if r["kind"] == "state" and r["mode"] == "grammar"
            and not r["reversed"]}
    draws = defaultdict(list)
    for c in census_rows:
        if c["kind"] == "census":
            draws[c["plan"]].append(c)
    res = {}
    for p in plan["plans"]:
        cs = draws.get(p["name"], [])
        if not cs:
            continue
        rows = [by[tuple(c["key"])] for c in cs]
        if p["name"].startswith("validation"):  # does the census agree with the path sum? (K options + other)
            counts = [c["strict"] for c in cs]
            P = [r["p"] + [max(0.0, 1.0 - sum(r["p"]))] for r in rows]
            res[p["name"]] = MC.deviance_test(counts, P, n_boot=10000)
            continue
        if p["name"].startswith("Q1"):  # with grammar: options only
            counts = [c["strict"][:-1] for c in cs]
            P, R = [r["p"] for r in rows], [r["ft_split"] for r in rows]
        elif p["name"].startswith("Q2"):  # free census, loose parse, on-format part only
            counts = [c["loose"][:-1] for c in cs]
            P = [r["free_conditional"] for r in rows]
            R = [gram[(r["task"], r["arm"], r["i"])]["p"] for r in rows]
        else:  # Q3: free census, strict parse, on-format part only
            counts = [c["strict"][:-1] for c in cs]
            P, R = [r["free_conditional"] for r in rows], [r["ft_mask"] for r in rows]
        t = MC.lr_test(counts, P, R, n_boot=10000)
        t["draws"] = int(np.sum(counts))
        t["off_format_share"] = float(np.mean([c["strict"][-1] / max(1, c["n"]) for c in cs]))
        res[p["name"]] = t
    prim = [k for k in res if k[:2] in ("Q1", "Q2", "Q3") and "p_under_alt" in res[k]]
    if prim:  # Holm over the primary tests: p is the smaller one-sided p-value of the two
        ps = [min(res[k]["p_under_alt"], res[k]["p_under_path"]) for k in prim]
        for k, a in zip(prim, MC.holm(ps)):
            res[k]["holm_p"] = a
    return res


def to_markdown(summary, tests):
    """A short human-readable version."""
    lines = ["| Arm | States (unsat.) | Req./state | Max B | median m_off | gap FT-mask (med / max) | gap FT-split (med / max) | gap depth 2 (med / max) | bound violations |",
             "|---|---|---|---|---|---|---|---|---|"]

    def mm(s):
        return "–" if not s else f"{s['median']:.3g} / {s['max']:.3g}"
    for k, g in summary.items():
        lines.append(f"| {k} | {g['states']} ({g['unsaturated']}) | {g['requests_per_state']:.1f} | {g['max_bound']:.1e} | "
                     f"{'–' if not g['m_off'] else format(g['m_off']['median'], '.3g')} | {mm(g['gap_ft_mask'])} | "
                     f"{mm(g['gap_ft_split'])} | {mm(g['gap_depth2'])} | {g.get('bound_violations', '–')} |")
    lines += ["", "| Free arm | off-format (median) | gap grammar vs free (med / max) | log-odds shift (med / max) |", "|---|---|---|---|"]
    for k, g in summary.items():
        if "off_format" in g:
            lines.append(f"| {k} | {g['off_format']['median']:.3g} | {mm(g['gap_grammar_vs_free_conditional'])} | {mm(g['log_odds_shift'])} |")
    if tests:
        lines += ["", "| Test | Decision / p | Draws |", "|---|---|---|"]
        for k, t in tests.items():
            if "decision" in t:
                lines.append(f"| {k} | {t['decision']} (Holm p {t.get('holm_p', float('nan')):.3g}) | {t.get('draws', '')} |")
            else:
                lines.append(f"| {k} | G {t['G']:.1f} vs null mean {t['null_mean']:.1f}, p {t['p']:.3f} | |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--census")
    ap.add_argument("--plan")
    args = ap.parse_args()
    hdr, rows = load(args.stage1)
    summary = summarise_stage1(rows)
    tests = None
    if args.census and args.plan:
        _, crow = load(args.census)
        tests = run_tests(json.load(open(args.plan)), crow, rows)
    stem = Path(args.stage1).name.replace(".jsonl.gz", "")
    out = Path(args.stage1).with_name(stem + "_summary.json")
    json.dump({"stage1_header": hdr, "summary": summary, "tests": tests}, open(out, "w"), indent=1)
    md = out.with_suffix(".md")
    md.write_text(to_markdown(summary, tests))
    print(md.read_text())
    print(f"Written: {out} and {md}")


if __name__ == "__main__":
    main()
