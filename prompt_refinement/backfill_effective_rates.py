#!/usr/bin/env python3
"""Backfill EFFECTIVE move-rate columns into legacy prompt_comparison CSVs.

The original sweep CSVs store only the RAW single-shot rate
(move_rate_{k}of8 = n_MOVE / N, bad parses in the denominator). Production
retries on bad parses, so its per-decision probability is the EFFECTIVE rate
n_MOVE / (n_MOVE + n_STAY). The per-cell counts needed for that are not in the
wide CSVs — but every run made after raw logging landed (2026-07-15) has a
per-reply log in results/raw/, from which this script rebuilds the counts and
writes move_rate_eff_{k}of8 columns back into the same CSV (idempotent;
re-running refreshes them).

CSVs whose runs predate raw logging (early llama/gemma arms) cannot be
backfilled and keep only the raw columns — plot_results.py labels those rows
accordingly ("MIXED"/"RAW" in the figure title).

    python prompt_refinement/backfill_effective_rates.py          # all CSVs
    python prompt_refinement/backfill_effective_rates.py --dry-run
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
X = list(range(9))


def effective_by_cell(raw_path):
    """raw jsonl.gz -> {(candidate, n_out): eff_rate or None if no valid replies}."""
    counts = defaultdict(lambda: [0, 0])
    with gzip.open(raw_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("_meta"):
                continue
            if r["parse"] == "MOVE":
                counts[(r["candidate"], r["n_out"])][0] += 1
            elif r["parse"] == "STAY":
                counts[(r["candidate"], r["n_out"])][1] += 1
    return {k: (m / (m + s) if (m + s) else None) for k, (m, s) in counts.items()}


def backfill(csv_path, dry_run=False):
    label = csv_path.stem[len("prompt_comparison_"):]
    raw_path = RESULTS / "raw" / f"prompt_comparison_{label}_raw.jsonl.gz"
    if not raw_path.exists():
        return None
    eff = effective_by_cell(raw_path)

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    max_delta, n_changed = 0.0, 0
    for r in rows:
        for k in X:
            e = eff.get((r["candidate"], k))
            r[f"move_rate_eff_{k}of8"] = round(e, 3) if e is not None else ""
            if e is not None:
                d = abs(e - float(r[f"move_rate_{k}of8"]))
                max_delta = max(max_delta, d)
                if d > 0.01:
                    n_changed += 1
    if not dry_run:
        # eff columns appended after the existing ones; rewrite preserves order
        fields = [c for c in rows[0].keys()]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return max_delta, n_changed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    done, skipped = [], []
    for p in sorted(RESULTS.glob("prompt_comparison_*.csv")):
        out = backfill(p, dry_run=args.dry_run)
        label = p.stem[len("prompt_comparison_"):]
        if out is None:
            skipped.append(label)
        else:
            done.append((label, *out))

    print(f"{'label':52s} {'max|eff-raw|':>12s} {'cells>0.01':>10s}")
    for label, md, nc in done:
        flag = "  <-- reevaluated" if md > 0.05 else ""
        print(f"{label:52s} {md:12.3f} {nc:10d}{flag}")
    print(f"\nbackfilled {len(done)}; no raw log (stay raw-only): {len(skipped)}")
    for label in skipped:
        print(f"  [raw-only] {label}")


if __name__ == "__main__":
    main()
