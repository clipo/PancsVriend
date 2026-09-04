#!/usr/bin/env python3
"""Migrate the prompt-comparison results to ONE FILE PER ARM CARRYING BOTH ROLES.

    python prompt_refinement/unify_role_csvs.py --dry-run   # show what would change
    python prompt_refinement/unify_role_csvs.py             # do it

Why this exists
---------------
The prompt-comparison family (`evaluate_prompts.py`) historically wrote one WIDE
CSV per (arm, role), with the role encoded in the FILENAME: `<label>.csv` = red,
`<label>-blue.csv` = blue. The ratio family (`evaluate_ratio_prompts.py`) already
writes one LONG CSV per arm with an `agent_role` COLUMN, both roles inside.

Two schemas meant every red-vs-blue figure had to open two files and trust that
they lined up, and it is why the A-family role overlay read red from the
dedicated `-arefine` runs and blue from the full-suite `-blue` runs. This script
converts the prompt family to the ratio family's schema so a role comparison is
a groupby, not a filename convention.

Source of truth
---------------
Rebuilt from `results/raw/*_raw.jsonl.gz` — the per-sample logs — NOT by
reshaping the wide CSVs. The raw logs carry the parse verdict of every single
reply, so n_move / n_stay / n_bad per cell are exact counts rather than integers
recovered from 3-decimal rounded rates. The raw `_meta` line carries the role, so
the role is read from the run itself instead of being inferred from the filename.

What it writes / moves
----------------------
  results/prompt_comparison_<arm>.csv   NEW long schema, both roles (see FIELDS)
  results/legacy_wide_per_role/         the superseded wide CSVs, moved not deleted

`<arm>` is the old red label with no `-blue` suffix. The per-run `.md` summaries
are left untouched: they are human summaries of one run each and are still
accurate. Per-candidate cost/monotonicity metadata (cached_prefix_tokens,
spearman_monotonic, ...) is per-CANDIDATE, not per-cell, so it is not
denormalised into the long CSV; it survives in the `.md` and in the archived
wide CSVs.

Arms that only ever ran one role (the `-arefine` A-family runs, `*-chat` for
Gemma) are still converted — they simply contain one role. That is the point of
the schema: role is data, not a filename.
"""
import argparse
import csv
import gzip
import json
import shutil
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
PREFIX = "prompt_comparison_"

# Mirrors ratio_comparison_*.csv, minus the occupancy coordinates that family
# varies: this sweep is always fully occupied, so the cell coordinate is n_out.
FIELDS = ["candidate", "agent_role", "n_out", "n_samples", "n_move", "n_stay",
          "n_bad", "move_rate_raw", "move_rate_effective"]


def read_raw(path):
    """-> (role, {(candidate, n_out): Counter-ish [n_move, n_stay, n_bad]}).

    Role comes from the run's own _meta line. Older raw logs predate the field;
    those fall back to the filename convention and are reported, so a silent
    mislabel is impossible."""
    role, cells = None, defaultdict(lambda: [0, 0, 0])
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("_meta"):
                role = r.get("role")
                continue
            c = cells[(r["candidate"], r["n_out"])]
            if r["parse"] == "MOVE":
                c[0] += 1
            elif r["parse"] == "STAY":
                c[1] += 1
            else:
                c[2] += 1
    return role, cells


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--archive-dir", default=None,
                    help="where the superseded wide CSVs are moved "
                         "(default: <results-dir>/legacy_wide_per_role)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written/moved, change nothing")
    args = ap.parse_args()
    results = Path(args.results_dir)
    archive = Path(args.archive_dir) if args.archive_dir else results / "legacy_wide_per_role"

    # Group raw logs by ARM (the label with any -blue suffix stripped).
    groups = defaultdict(dict)
    for p in sorted((results / "raw").glob(f"{PREFIX}*_raw.jsonl.gz")):
        label = p.name[len(PREFIX):-len("_raw.jsonl.gz")]
        arm = label[:-len("-blue")] if label.endswith("-blue") else label
        role, cells = read_raw(p)
        if role is None:
            role = "blue" if label.endswith("-blue") else "red"
            print(f"[warn] {label}: no role in _meta, inferred '{role}' from filename")
        if role in groups[arm]:
            print(f"[warn] {arm}: role '{role}' seen twice ({label}); keeping the first")
            continue
        groups[arm][role] = cells

    written = moved = 0
    for arm, by_role in sorted(groups.items()):
        rows = []
        for role in ("red", "blue"):                    # stable, red-first ordering
            cells = by_role.get(role)
            if not cells:
                continue
            for (cand, n_out), (mv, st, bad) in sorted(
                    cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                n = mv + st + bad
                valid = mv + st
                rows.append({
                    "candidate": cand, "agent_role": role, "n_out": n_out,
                    "n_samples": n, "n_move": mv, "n_stay": st, "n_bad": bad,
                    "move_rate_raw": round(mv / n, 6) if n else "",
                    # Blank, never 0, when every reply failed to parse: production
                    # would retry there, so "no measurement" must not read as "never moves".
                    "move_rate_effective": round(mv / valid, 6) if valid else "",
                })
        out = results / f"{PREFIX}{arm}.csv"
        legacy = [results / f"{PREFIX}{arm}.csv", results / f"{PREFIX}{arm}-blue.csv"]
        roles = "+".join(r for r in ("red", "blue") if r in by_role)
        print(f"{arm:56s} roles={roles:9s} rows={len(rows):5d}"
              f"{'  [dry-run]' if args.dry_run else ''}")
        if args.dry_run:
            continue
        archive.mkdir(parents=True, exist_ok=True)
        for src in legacy:
            if src.exists():
                shutil.move(str(src), str(archive / src.name))
                moved += 1
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        written += 1
    if args.dry_run:
        print(f"\n[dry-run] would write {len(groups)} unified CSVs")
    else:
        print(f"\nwrote {written} unified CSVs; archived {moved} wide CSVs -> {archive}")


if __name__ == "__main__":
    main()
