#!/usr/bin/env python3
"""Does pinning the server's clock reproduce each day's llama values?

    python value_functions/comparison/date_pin_report.py --label llama-3.3-70b-chat-grammar

Inputs: launch_probe/probe_date_<clock>_L1.csv from run_tenant_probe.sh (the
date-pin probe): the 79 diagnostic cells re-extracted with the server's clock
pinned to 2026-09-06 / 09-09 / 09-11 and once unpinned.

Per clock it reports agreement (to 1e-9) with the original R1 traces
(2026-09-06, the p_stored column), with run 3 (2026-09-11), and — for 09-09,
which has no exact table — the two-sided exact binomial p of the census
counts drawn that day (sampled_small raw, n=100) under the pinned values,
i.e. whether the census could have been drawn from them.
"""
import argparse
import collections
import csv
import glob
import gzip
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import DATE_SENSITIVITY_DIR, LOGPROB_DIR, SEQCHECK_PLOTS_DIR  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adjudication_power import binom_two_sided_p  # noqa: E402

KEY = lambda r: (r["scenario"], r["role"], int(r["n_similar"]), int(r["n_occupied"]))  # noqa: E731


def load(path, col):
    return {KEY(r): float(r[col]) for r in csv.DictReader(open(path))}


def census_counts(label):
    c = collections.defaultdict(lambda: [0, 0])
    for f in glob.glob(str(DATE_SENSITIVITY_DIR / "llama" / "census_2026-09-09" / "**" / f"vf_{label}-sanity__*__R3_dual_count_raw.jsonl.gz"), recursive=True):
        scen = os.path.basename(f).split("__")[1]
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            if r.get("_meta"):
                continue
            p = (r.get("parse") or "").upper()
            k = (scen, r["agent_role"], r["n_similar"], r["n_occupied"])
            if p.startswith("MOVE"):
                c[k][0] += 1; c[k][1] += 1
            elif p.startswith("ST"):
                c[k][1] += 1
    return c


import json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    pdir = SEQCHECK_PLOTS_DIR / "launch_probe"
    probes = {os.path.basename(f)[len("probe_date_"):-len("_L1.csv")]: f
              for f in sorted(glob.glob(str(pdir / "probe_date_*_L1.csv")))}
    if not probes:
        sys.exit("no probe_date_*_L1.csv")
    # R1 = the 2026-09-06 traces, kept in the date-sensitivity study (llm_logprob/raw
    # holds the canonical table, so a probe file's p_stored column is not R1)
    r1 = {}
    for gz in glob.glob(str(DATE_SENSITIVITY_DIR / "llama" / "tables" / "2026-09-06" / "raw" / "vflp_*_states.jsonl.gz")):
        scen = os.path.basename(gz).split("__")[1]
        for line in gzip.open(gz, "rt"):
            r = json.loads(line)
            if not r.get("_meta"):
                r1[(scen, r["role"], r["n_similar"], r["n_occupied"])] = r["p_move"]
    any_f = next(iter(probes.values()))
    probe_cells = set(load(any_f, "p_stored"))
    r1 = {k: v for k, v in r1.items() if k in probe_cells}
    r3p = SEQCHECK_PLOTS_DIR / f"reextract_run3_{args.label}.csv"
    r3 = load(r3p, "p_reextracted") if r3p.exists() else {}
    cens = census_counts(args.label)
    cells = sorted(r1)
    print(f"{len(cells)} diagnostic cells; clocks: {', '.join(probes)}\n")
    print(f"{'clock':<12}{'= R1 (06 Sep)':>15}{'= run3 (11 Sep)':>17}{'census 09-09 fits (p>=.05)':>28}{'median |d - R1|':>17}{'median |d - run3|':>19}")
    for clock, f in probes.items():
        d = load(f, "p_reextracted")
        eq1 = sum(abs(d[c] - r1[c]) <= 1e-9 for c in cells)
        eq3 = sum(abs(d[c] - r3[c]) <= 1e-9 for c in cells if c in r3)
        fits = sum(1 for c in cells if c in cens and cens[c][1] and binom_two_sided_p(cens[c][0], cens[c][1], d[c]) >= 0.05)
        m1 = sorted(abs(d[c] - r1[c]) for c in cells)[len(cells) // 2]
        m3 = sorted(abs(d[c] - r3[c]) for c in cells if c in r3)[len(cells) // 2] if r3 else float("nan")
        print(f"  {clock:<10}{eq1:>10}/{len(cells):<4}{eq3:>12}/{len(cells):<4}{fits:>20}/{len(cells):<7}{m1:>17.4f}{m3:>19.4f}")
    print("\nreading: a clock that reproduces a day's extraction to 1e-9 on every cell IS that day's regime.")


if __name__ == "__main__":
    main()
