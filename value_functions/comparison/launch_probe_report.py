#!/usr/bin/env python3
"""Report the multi-launch determinism probe (run_launch_determinism_probe.sh).

    python value_functions/comparison/launch_probe_report.py --label llama-3.3-70b-chat-grammar

Inputs: seqcheck_plots/launch_probe/probe_<variant>_L<k>.csv — the same
disagreeing cells re-extracted under K fresh server launches per variant
(baseline flags; GGML_CUDA_DISABLE_GRAPHS=1; CUBLAS_WORKSPACE_CONFIG=:4096:8).
Plus the three full runs (stored traces, reextract_run2/3) for the same cells.

QUESTIONS IT ANSWERS
  1. Within a variant, do K launches agree? (per-launch non-determinism)
  2. Does any variant make launches agree where baseline does not? (a fix)
  3. Across ALL sources for a cell, how many distinct values appear? Two
     recurring values = a two-state path selection; a new value every launch
     = continuous. This decides whether "which candidate" is even the right
     question for the adjudication sampling.
"""
import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import SEQCHECK_PLOTS_DIR  # noqa: E402

TOL = 1e-6


def load(path, col):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[(r["scenario"], r["role"], int(r["n_similar"]), int(r["n_occupied"]))] = float(r[col])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    pdir = SEQCHECK_PLOTS_DIR / "launch_probe"
    files = sorted(glob.glob(str(pdir / "probe_*_L*.csv")))
    if not files:
        sys.exit(f"no probe files in {pdir}")
    variants = {}
    for f in files:
        m = re.match(r"probe_(.+)_L(\d+)\.csv", os.path.basename(f))
        variants.setdefault(m.group(1), []).append((int(m.group(2)), load(f, "p_reextracted")))
    runs = {}
    first = next(iter(variants.values()))[0][1]
    runs["run1"] = load(files[0], "p_stored")           # every probe file carries the stored value
    for k in (2, 3):
        p = SEQCHECK_PLOTS_DIR / f"reextract_run{k}_{args.label}.csv"
        if p.exists():
            runs[f"run{k}"] = load(p, "p_reextracted")
    cells = sorted(first)

    print(f"{args.label}: {len(cells)} probed cells, variants: "
          + ", ".join(f"{v} x{len(ls)}" for v, ls in variants.items()) + "\n")
    print("1. within-variant agreement across launches")
    for v, ls in variants.items():
        ident = 0; worst = 0.0; worst_cell = None
        for c in cells:
            vals = [d[c] for _, d in ls if c in d]
            sp = max(vals) - min(vals) if vals else 0.0
            if sp <= TOL:
                ident += 1
            if sp > worst:
                worst, worst_cell = sp, c
        print(f"   {v:<28} {ident:>3}/{len(cells)} identical (<= {TOL:g});  max spread {worst:.4f}"
              + (f" at {worst_cell}" if worst_cell else ""))

    print("\n2. do probe launches reproduce a full-run value? (per variant: cells where every launch")
    print("   equals run1, run2 or run3 to 1e-6 — i.e. the value set is a small fixed menu)")
    for v, ls in variants.items():
        on_menu = 0
        for c in cells:
            menu = {round(r[c], 6) for r in runs.values() if c in r}
            if all(round(d[c], 6) in menu for _, d in ls if c in d):
                on_menu += 1
        print(f"   {v:<28} {on_menu:>3}/{len(cells)} cells stay on the run1/2/3 menu")

    print("\n3. distinct values per cell across ALL sources (runs + every probe launch)")
    hist = {}
    detail = []
    for c in cells:
        vals = [r[c] for r in runs.values() if c in r]
        for ls in variants.values():
            vals += [d[c] for _, d in ls if c in d]
        distinct = []
        for x in sorted(vals):
            if not distinct or x - distinct[-1] > TOL:
                distinct.append(x)
        hist[len(distinct)] = hist.get(len(distinct), 0) + 1
        detail.append((len(distinct), max(vals) - min(vals), c, len(vals)))
    n_src = len(runs) + sum(len(ls) for ls in variants.values())
    for k in sorted(hist):
        print(f"   {k:>2} distinct value(s): {hist[k]:>3} cells   (out of {n_src} sources each)")
    detail.sort(key=lambda t: (-t[0], -t[1]))
    print("\n   most variable cells:")
    for nd, sp, c, ns in detail[:12]:
        print(f"     {c[0]:<32}{c[1]:<6}{c[2]}/{c[3]}   {nd} distinct over {ns} sources, spread {sp:.4f}")

    print("\n4. per variant/launch: cells equal (1e-9) to today's baseline vs to the 09-06 stored values")
    base = variants.get("baseline", [(0, runs.get("run2", runs["run1"]))])[0][1]
    for v, ls in variants.items():
        for k, d in sorted(ls):
            eq_today = sum(1 for c in cells if c in d and abs(d[c] - base[c]) <= 1e-9)
            eq_r1 = sum(1 for c in cells if c in d and abs(d[c] - runs["run1"][c]) <= 1e-9)
            mx = max((abs(d[c] - base[c]) for c in cells if c in d), default=0.0)
            print(f"   {v+'_L'+str(k):<18} = today {eq_today:>3}/{len(cells)}   = 09-06 {eq_r1:>3}/{len(cells)}"
                  f"   max |delta vs today| {mx:.4f}")

    out = pdir / f"launch_probe_summary_{args.label}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["scenario", "role", "n_similar", "n_occupied"] + list(runs) + \
              [f"{v}_L{k}" for v, ls in variants.items() for k, _ in sorted(ls)] + ["n_distinct", "spread_all"]
        w.writerow(hdr)
        for nd, sp, c, _ in sorted(detail, key=lambda t: t[2]):
            row = list(c) + [runs[r].get(c, "") for r in runs]
            for v, ls in variants.items():
                row += [d.get(c, "") for _, d in sorted(ls)]
            w.writerow(row + [nd, round(sp, 6)])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
