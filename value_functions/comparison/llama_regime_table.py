#!/usr/bin/env python3
"""Every sampling source for one model's disputed cells, side by side, by date.

    python value_functions/comparison/llama_regime_table.py --label llama-3.3-70b-chat-grammar

WHY (2026-09-11)
Llama's served numbers went through THREE self-consistent regimes in six days,
and pooled counts hid it. Sources, each with the regime it was drawn in:

  R1 exact      date_sensitivity/llama/tables/2026-09-06/raw     the 09-06 extraction
  R1 seq        its seqcheck (sequential, n=300-1200, same session)
  Sep9          date_sensitivity/llama/census_2026-09-09/*_raw    the census, n=100, 09-09
  Sep11_04h     .../census_2026-09-09/*_escalate_raw               escalation draws, 09-11 04-05h
  R3 exact      llm_logprob/raw/vflp_*      NOTE: now the CANONICAL (26 Jul 2024) table,
                                            not the 09-11 one (which is tables/2026-09-11/)
  R3 seq        its seqcheck
  fresh100      recheck_<label>.json (deleted 2026-09-15; column stays empty)

Since 2026-09-15 this is a historical tool: the regimes were the calendar
date (LLAMA_CPP_SERVING_NOTES.md §6) and date_pin_report.py is the proof.

Reading across a row: R1 exact agrees with R1 seq; the Sep9 census is a
different set of values again (ethnic red 0/8: 0.51 vs R1 0.19 vs R3 0.13);
the 09-11 04h escalation draws agree with R3 on every escalated cell. The
sanity table pools Sep9 + Sep11_04h into one count, which is why "the census
sides with the old table" on some cells was an artifact: income red 2/6
pooled (85+175)/400 = 0.65 = the old value, by coincidence.

Reads only; writes seqcheck_plots/regime_table_<label>.csv.
"""
import argparse
import collections
import csv
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import DATE_SENSITIVITY_DIR, LOGPROB_DIR, LOGPROB_RAW_DIR, LOGPROB_VALIDATION_DIR, SEQCHECK_PLOTS_DIR  # noqa: E402

STYLE = "R3_dual_count"


def rates(path):
    c = collections.defaultdict(lambda: [0, 0])
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        if r.get("_meta"):
            continue
        p = (r.get("parse") or "").upper()
        k = (r["agent_role"], r["n_similar"], r["n_occupied"])
        if p.startswith("MOVE"):
            c[k][0] += 1; c[k][1] += 1
        elif p.startswith("ST"):
            c[k][1] += 1
    return c


def scen_of(f):
    return os.path.basename(f).split("__")[1]


def traces(pattern):
    out = {}
    for gz in glob.glob(pattern):
        for line in gzip.open(gz, "rt"):
            r = json.loads(line)
            if not r.get("_meta"):
                out[(scen_of(gz), r["role"], r["n_similar"], r["n_occupied"])] = r["p_move"]
    return out


def seq(path):
    out = {}
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            out[(r["scenario"], r["role"], int(r["n_similar"]), int(r["n_occupied"]))] = \
                (int(r["seq_move"]), int(r["seq_n"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--superseded", default=None,
                    help="folder holding the earlier extraction (default superseded_*_<short>)")
    ap.add_argument("--min-diff", type=float, default=0.01)
    args = ap.parse_args()
    short = args.label.split("-")[0]
    sup = args.superseded or str(DATE_SENSITIVITY_DIR / short / "tables" / "2026-09-06")
    if not sup:
        sys.exit("no superseded extraction folder")
    S = str(DATE_SENSITIVITY_DIR / "llama" / "census_2026-09-09")
    r2a, r2b = {}, {}
    for f in glob.glob(f"{S}/**/vf_{args.label}-sanity__*__{STYLE}_raw.jsonl.gz", recursive=True):
        for k, v in rates(f).items():
            r2a[(scen_of(f),) + k] = v
    for f in glob.glob(f"{S}/**/vf_{args.label}-sanity__*__{STYLE}_escalate_raw.jsonl.gz", recursive=True):
        for k, v in rates(f).items():
            r2b[(scen_of(f),) + k] = v
    old = traces(f"{sup}/raw/vflp_{args.label}__*_states.jsonl.gz")
    new = traces(str(LOGPROB_RAW_DIR / f"vflp_{args.label}__*_states.jsonl.gz"))
    s1 = seq(f"{sup}/validation_data/seqcheck_{args.label}.csv")
    s3 = seq(str(LOGPROB_VALIDATION_DIR / f"seqcheck_{args.label}.csv"))
    rc = {}
    p = SEQCHECK_PLOTS_DIR / f"recheck_{args.label}.json"
    if p.exists():
        j = json.load(open(p))
        for r in (j if isinstance(j, list) else j.get("cells", [])):
            rc[(r["scenario"], r["role"], r["n_similar"], r["n_occupied"])] = (r["fresh_move"], r["fresh_n"])

    disp = sorted(k for k in old if k in new and abs(old[k] - new[k]) > args.min_diff)
    cols = ["R1_exact", "R1_seq", "Sep9", "Sep11_04h", "R3_exact", "R3_seq", "fresh100"]

    def fmt(v):
        return f"{v[0]/v[1]:.3f}({v[1]:>4})" if v and v[1] else "      -    "

    rows = []
    print(f"{'cell':<38}" + "".join(f"{c:>12}" for c in cols))
    for k in disp:
        vals = [old[k], s1.get(k), r2a.get(k), r2b.get(k), new[k], s3.get(k), rc.get(k)]
        print(f"{k[0][:20]+' '+k[1]+' '+str(k[2])+'/'+str(k[3]):<38}{old[k]:>12.3f}{fmt(vals[1]):>12}"
              f"{fmt(vals[2]):>12}{fmt(vals[3]):>12}{new[k]:>12.3f}{fmt(vals[5]):>12}{fmt(vals[6]):>12}")
        rows.append({"scenario": k[0], "role": k[1], "n_similar": k[2], "n_occupied": k[3],
                     **{c: (v if not isinstance(v, list) and not isinstance(v, tuple) else
                            (f"{v[0]}/{v[1]}" if v else "")) for c, v in zip(cols, vals)}})
    out = SEQCHECK_PLOTS_DIR / f"regime_table_{args.label}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{len(disp)} cells differ by > {args.min_diff} between the two extractions; wrote {out}")


if __name__ == "__main__":
    main()
