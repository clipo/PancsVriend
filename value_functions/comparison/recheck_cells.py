#!/usr/bin/env python3
"""Independently re-verify specific cells where exact and sanity disagree.

    python value_functions/comparison/recheck_cells.py --label llama-3.3-70b-chat-grammar \
        --url http://127.0.0.1:8085/v1/chat/completions --above 0.10

For each selected cell it does TWO independent things against a live server:

  1. RE-EXTRACTS the exact value (path_sum over the grammar's token paths).
     If the enumeration is deterministic the number must come back identical;
     a different number means the exact side is not reproducible, which would
     be a finding in itself.
  2. Draws a FRESH, INDEPENDENT n sample — NOT merged with the stored counts.
     The escalation in sanity_vs_exact.py adds draws to the original 100, so a
     stage-1 fluctuation keeps 25% weight and can hold a cell outside its
     interval on its own. A clean sample removes that coupling entirely.

A discrepancy that survives both is a real disagreement between enumeration and
sampling for that prompt, and the next step is to read the stored token paths.
"""
import argparse
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent
for _p in (str(THIS), str(REPO), str(REPO / "prompt_refinement"),
           str(REPO / "value_functions" / "logprob")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd                                        # noqa: E402
from logprob_value_function import Server, path_sum        # noqa: E402
from sampling_common import wilson_ci, role_keywords       # noqa: E402
from ratio_prompt_templates import RATIO_CANDIDATES        # noqa: E402
from evaluate_ratio_prompts import render_prompt           # noqa: E402
from value_functions.paths import SEQCHECK_PLOTS_DIR, LOGPROB_RAW_DIR  # noqa: E402
import gzip                                                # noqa: E402


def trace_params(label, scenario, style):
    """The extraction parameters the STORED table was built with.

    Never assume the CLI defaults match: the tables use n_probs=64 /
    mass_floor=1e-7, the defaults were 20 / 1e-6, and a re-extraction under
    different parameters is not a comparison.
    """
    gz = LOGPROB_RAW_DIR / f"vflp_{label}__{scenario}__{style}_states.jsonl.gz"
    if not gz.exists():
        return {}
    with gzip.open(gz, "rt") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("_meta"):
                return rec
            break
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--style", default="R3_dual_count")
    ap.add_argument("--scenario-file", default="scenarios_a2.py")
    ap.add_argument("--above", type=float, default=0.10,
                    help="re-check residual cells whose |exact-sanity| exceeds this")
    ap.add_argument("--samples", type=int, default=100, help="fresh independent draws per cell")
    ap.add_argument("--seed0", type=int, default=7_000_000)
    ap.add_argument("--n-probs", type=int, default=20)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--mass-floor", type=float, default=1e-6)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    csv = SEQCHECK_PLOTS_DIR / f"sanity_vs_exact_{args.label}.csv"
    if not csv.exists():
        print(f"[error] no comparison csv at {csv}", file=sys.stderr)
        return 2
    d = pd.read_csv(csv)
    d = d[~d.inside].copy()
    d["dev"] = (d.p_exact - d.p_sanity).abs()
    sel = d[d.dev > args.above].sort_values("dev", ascending=False)
    if not len(sel):
        print(f"no residual cells with |dev| > {args.above}")
        return 0
    print(f"{args.label}: re-checking {len(sel)} cell(s) with |dev| > {args.above}\n")

    tpl, fn = RATIO_CANDIDATES[args.style]
    servers = {}   # one per scenario: its own n_probs / temperature

    def server_for(scen):
        if scen not in servers:
            m = trace_params(args.label, scen, args.style)
            npb = m.get("n_probs", args.n_probs)
            temp = m.get("temperature", args.temperature)
            srv = Server(args.url.rsplit("/v1/", 1)[0], None, temp, npb)
            srv.model = srv.props().get("model_path", "").split("/")[-1] or None
            servers[scen] = (srv, m.get("max_depth", args.max_depth),
                             m.get("mass_floor", args.mass_floor))
        return servers[scen]

    out = []
    for i, (_, r) in enumerate(sel.iterrows()):
        kw = role_keywords(r.scenario, args.scenario_file)
        prompt = render_prompt(args.style, tpl, fn,
                              int(r.n_similar), int(r.n_occupied), kw[r.role])[0]
        server, mdepth, mfloor = server_for(r.scenario)
        res = path_sum(server, prompt, mdepth, mfloor)
        exact2 = res["p_move"]
        mass_bound, nreq = res["mass_bound"], res["n_requests"]
        mv, st = server.sample_sequential(prompt, args.samples, args.seed0 + 1000 * i)
        n = mv + st
        lo, hi = wilson_ci(mv, n) if n else (0.0, 1.0)
        rec = {"scenario": r.scenario, "role": r.role,
               "n_similar": int(r.n_similar), "n_occupied": int(r.n_occupied),
               "exact_stored": round(float(r.p_exact), 6),
               "exact_recomputed": round(float(exact2), 6),
               "exact_reproducible": abs(exact2 - r.p_exact) < 1e-6,
               "mass_bound": mass_bound, "n_requests": nreq,
               "sanity_stored": round(float(r.p_sanity), 6),
               "sanity_stored_n": int(r.n_samples),
               "fresh_move": mv, "fresh_n": n,
               "fresh_p": round(mv / n, 6) if n else None,
               "fresh_ci_low": round(lo, 6), "fresh_ci_high": round(hi, 6),
               "exact_inside_fresh_ci": bool(lo <= exact2 <= hi),
               "dev_fresh": round(abs(exact2 - (mv / n if n else 0)), 6)}
        out.append(rec)
        print(f"  {r.scenario}/{r.role} {int(r.n_similar)}/{int(r.n_occupied)}")
        print(f"    exact  stored {r.p_exact:.6f}  recomputed {exact2:.6f}  "
              f"{'REPRODUCIBLE' if rec['exact_reproducible'] else '*** DIFFERS ***'}")
        print(f"    sanity stored {r.p_sanity:.4f} (n={int(r.n_samples)})   "
              f"fresh {mv}/{n}={rec['fresh_p']:.4f} CI [{lo:.4f},{hi:.4f}]")
        print(f"    -> exact inside fresh CI: {rec['exact_inside_fresh_ci']}   "
              f"dev {rec['dev_fresh']:.4f}\n")

    p = SEQCHECK_PLOTS_DIR / f"recheck_{args.label}.json"
    p.write_text(json.dumps(out, indent=1))
    persist = [o for o in out if not o["exact_inside_fresh_ci"]]
    print(f"wrote {p.relative_to(REPO)}")
    print(f"\n{len(persist)} of {len(out)} discrepancies PERSIST under fresh independent sampling")
    print(f"{sum(1 for o in out if not o['exact_reproducible'])} cell(s) had a non-reproducible exact value")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
