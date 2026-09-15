#!/usr/bin/env python3
"""Re-extract a model's EXACT value function and diff it against what is stored.

    python value_functions/comparison/reextract_diff.py --label llama-3.3-70b-chat-grammar \
        --url http://127.0.0.1:8085/v1/chat/completions

Reads NOTHING but the stored traces for the reference values, and WRITES NO
tables — the stored artifacts are evidence and stay untouched. Output is a per
cell diff CSV plus a summary.

WHY (2026-09-11)
Three llama cells were found whose stored exact value does not reproduce:
re-extraction gives a different number, and the new number agrees with
independent sequential sampling while the stored one does not. Prompt hash,
llama.cpp build, gguf, slot count and extraction parameters all match, and the
value is perfectly stable within a session (5 repeats, zero spread) — so the
difference is across sessions, for a minority of cells. 12 of 14 randomly
sampled cells reproduced bit-exactly, so this is not general corruption. This
script measures how many cells are actually affected instead of extrapolating
from the three the sanity arm happened to flag.

Extraction parameters are taken from each trace's own _meta header, so the
re-run matches the original rather than the current defaults.
"""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent
for _p in (str(THIS), str(REPO), str(REPO / "prompt_refinement"),
           str(REPO / "value_functions" / "logprob")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from logprob_value_function import Server, path_sum        # noqa: E402
from sampling_common import role_keywords                  # noqa: E402
from ratio_prompt_templates import RATIO_CANDIDATES        # noqa: E402
from evaluate_ratio_prompts import render_prompt           # noqa: E402
from value_functions.paths import LOGPROB_RAW_DIR, SEQCHECK_PLOTS_DIR  # noqa: E402

SCENARIOS = ["baseline", "race_white_black", "ethnic_asian_hispanic",
             "income_high_low", "political_liberal_conservative", "green_yellow"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--style", default="R3_dual_count")
    ap.add_argument("--scenarios", nargs="*", default=SCENARIOS)
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="|new-stored| above this counts as a difference")
    ap.add_argument("--big", type=float, default=0.01, help="threshold for a MATERIAL difference")
    ap.add_argument("--cells-from", default=None,
                    help="CSV with scenario,role,n_similar,n_occupied[,status]: re-extract only "
                         "these cells (rows with status STABLE are skipped) — for probes")
    ap.add_argument("--out", default=None, help="output CSV (default seqcheck_plots/reextract_diff_<label>.csv)")
    args = ap.parse_args()

    subset = None
    if args.cells_from:
        subset = set()
        with open(args.cells_from) as f:
            for r in csv.DictReader(f):
                if r.get("status", "") != "STABLE":
                    subset.add((r["scenario"], r["role"], int(r["n_similar"]), int(r["n_occupied"])))
        print(f"restricting to {len(subset)} cell(s) from {args.cells_from}")

    tpl, fn = RATIO_CANDIDATES[args.style]
    rows, n_diff, n_big = [], 0, 0
    for scen in args.scenarios:
        gz = LOGPROB_RAW_DIR / f"vflp_{args.label}__{scen}__{args.style}_states.jsonl.gz"
        if not gz.exists():
            print(f"  {scen}: no trace, skipping"); continue
        recs = [json.loads(l) for l in gzip.open(gz, "rt")]
        meta = next((r for r in recs if r.get("_meta")), {})
        cells = [r for r in recs if not r.get("_meta")]
        if subset is not None:
            cells = [r for r in cells
                     if (scen, r["role"], r["n_similar"], r["n_occupied"]) in subset]
            if not cells:
                continue
        n_probs = meta.get("n_probs", 64)
        max_depth = meta.get("max_depth", 8)
        mass_floor = meta.get("mass_floor", 1e-7)
        temp = meta.get("temperature", 0.3)
        srv = Server(args.url.rsplit("/v1/", 1)[0], None, temp, n_probs)
        srv.model = srv.props().get("model_path", "").split("/")[-1]
        kw = role_keywords(scen, meta.get("scenario_file") or "scenarios_a2.py")
        sd = sb = 0
        for c in cells:
            prompt = render_prompt(args.style, tpl, fn, c["n_similar"], c["n_occupied"],
                                   kw[c["role"]])[0]
            new = path_sum(srv, prompt, max_depth, mass_floor)
            d = abs(new["p_move"] - c["p_move"])
            rows.append({"scenario": scen, "role": c["role"],
                         "n_similar": c["n_similar"], "n_occupied": c["n_occupied"],
                         "p_stored": c["p_move"], "p_reextracted": new["p_move"],
                         "delta": d, "differs": d > args.tol, "material": d > args.big,
                         "mass_bound_stored": c["mass_bound"],
                         "mass_bound_new": new["mass_bound"]})
            if d > args.tol:
                sd += 1; n_diff += 1
            if d > args.big:
                sb += 1; n_big += 1
        print(f"  {scen}: {len(cells)} cells, {sd} differ (>{args.tol}), {sb} material (>{args.big})",
              flush=True)

    out = Path(args.out) if args.out else SEQCHECK_PLOTS_DIR / f"reextract_diff_{args.label}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{args.label}: {len(rows)} cells re-extracted")
    print(f"  differ  (>{args.tol}): {n_diff} ({100.0*n_diff/len(rows):.1f}%)")
    print(f"  MATERIAL(>{args.big}): {n_big} ({100.0*n_big/len(rows):.1f}%)")
    print(f"wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
