#!/usr/bin/env python3
"""Per-artifact P(MOVE)-vs-ratio figures for the EXACT (log-probability) tables.

    python value_functions/logprob/plot_exact_vf_mapping.py            # all labels
    python value_functions/logprob/plot_exact_vf_mapping.py --label qwen3.6-27b-chat-grammar-lp

Writes value_functions/results/llm_logprob/vf_mapping_plots/, mirroring
sampled/vf_mapping_plots/ so both stores carry the same set of figures.

WHY A SEPARATE SCRIPT (2026-09-07)
The sampled route gets these for free: build_value_function.py --plot calls
plot_vf() as it writes each table. The exact route never runs that builder —
logprob_value_function.py extracts the tables instead — so the exact store had
tables and no per-artifact figures. plot_vf() itself is reused unchanged (it is
exact-aware: on a log-probability table it drops the N and precision panels,
whose n_samples slots are inherited from the sampled campaign, not measured
here), so the two stores' figures cannot drift.
"""
import argparse
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent
for _p in (str(THIS), str(REPO), str(REPO / "prompt_refinement"),
           str(REPO / "value_functions" / "sampling")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from value_functions.paths import LOGPROB_DIR, LOGPROB_TABLES_DIR  # noqa: E402
from build_value_function import plot_vf, vf_plot_path  # noqa: E402



def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", help="one label, e.g. qwen3.6-27b-chat-grammar-lp")
    ap.add_argument("--style", default="R3_dual_count")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    args = ap.parse_args()

    pat = (f"vf_{args.label}__*__{args.style}.json" if args.label
           else f"vf_*__*__{args.style}.json")
    tables = sorted(LOGPROB_TABLES_DIR.glob(pat))
    if not tables:
        print(f"no tables matching {pat} under {LOGPROB_TABLES_DIR}")
        return 1
    for t in tables:
        vf = json.loads(t.read_text())
        m = vf["meta"]
        out = vf_plot_path(LOGPROB_DIR, m["label"], m["scenario"], m["style"], args.format)
        plot_vf(vf, out, dpi=args.dpi)
        print(f"wrote {out.relative_to(REPO)}")
    print(f"\n{len(tables)} figure(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
