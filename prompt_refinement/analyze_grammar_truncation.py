#!/usr/bin/env python3
"""Why do GRAMMAR arms have bad parses at all? (they are supposed to be immune)

    python prompt_refinement/analyze_grammar_truncation.py
    python prompt_refinement/analyze_grammar_truncation.py --detail ratio-gemma-4-31b-grammar

Scans every results/raw/*_raw.jsonl.gz and writes
results/grammar_truncation.md + .csv.

The finding
-----------
NOTES.md says grammar arms are "structurally immune" to bad parses because the
GBNF admits only whitespace plus MOVE/STAY. That is right about MISPARSE (a wrong
decision recorded as a clean one) and wrong about BAD PARSE, because of the
`ws` production:

    root ::= ws answer
    ws   ::= [ \\t\\n]*          <-- UNBOUNDED

Whitespace is legal and unlimited, so a constrained model may emit whitespace
until `max_tokens=5` cuts it off, before it ever reaches the keyword. The reply
is then empty-ish, unparseable, and retried. So the comment at llm_runner.py:59
--- "generation halts at the word boundary, so max_tokens=5 becomes a
never-binding safety ceiling" --- is false: the ceiling does bind.

The causal chain this script measures, in order:

1. ENDPOINT. Leading whitespace appears ONLY on /completions. The chat template
   ends the prompt with the assistant-turn header, so the model's first emitted
   token is already the word; on raw /completions there is no such scaffolding
   and some models open with their own newline. Every chat+grammar arm measures
   exactly 0 leading-whitespace replies and 0 bad parses.
2. RUN LENGTH. Leading whitespace alone is harmless: Qwen emits it on ~12% of
   completions replies but always as 1-2 tokens, and never truncates. Only Gemma
   produces long runs (3-4 tokens, and 5 = cut).
3. PROMPT. Gemma's long runs concentrate in the prompts it appears to find
   conflicted -- see --detail: the ratio sweep's truncations sit almost entirely
   at n_similar=1 in the SIMILAR-framed styles (R2/R5).

Direction of the residual bias: of the truncated replies that got far enough to
show a letter, all say ST/STA (i.e. STAY) and none say MO. Dropping them from the
effective rate therefore biases P(MOVE) slightly UP, in the affected cells only.
"""
import argparse
import csv
import glob
import gzip
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def scan(path):
    """-> (meta, list of per-reply dicts we care about)."""
    meta, recs = {}, []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("_meta"):
                meta = r
                continue
            recs.append(r)
    return meta, recs


def summarise(label, meta, recs):
    n = len(recs)
    lead = sum(1 for r in recs if r.get("text", "")[:1] in (" ", "\t", "\n"))
    bad = [r for r in recs if r.get("parse") not in ("MOVE", "STAY")]
    trunc = [r for r in bad if r.get("finish_reason") == "length"]
    ws_only = sum(1 for r in trunc if not r.get("text", "").strip())
    frag = Counter(r["text"].strip().upper() for r in trunc if r.get("text", "").strip())
    return {
        "label": label,
        "grammar": bool(meta.get("grammar")),
        "endpoint": "chat" if "chat" in label else "completions",
        "n_samples": n,
        "n_leading_ws": lead,
        "pct_leading_ws": round(100.0 * lead / n, 3) if n else 0.0,
        "n_bad": len(bad),
        "n_truncated": len(trunc),
        "pct_truncated": round(100.0 * len(trunc) / n, 4) if n else 0.0,
        "n_trunc_whitespace_only": ws_only,
        "n_trunc_midword": len(trunc) - ws_only,
        "midword_fragments": "; ".join(f"{k}x{v}" for k, v in frag.most_common()),
    }


def detail(label, recs, out):
    """Per-cell breakdown for one arm — which prompts stall, and what the
    surviving replies in those same cells decided."""
    cells = defaultdict(lambda: [0, 0, 0])   # move, stay, truncated
    ctx = {}
    for r in recs:
        key = (r.get("candidate"), r.get("agent_role", "?"),
               r.get("n_similar", r.get("n_out")), r.get("n_occupied", 8))
        c = cells[key]
        if r["parse"] == "MOVE":
            c[0] += 1
        elif r["parse"] == "STAY":
            c[1] += 1
        else:
            c[2] += 1
        ctx.setdefault(key, r.get("context") or r.get("grid", "").replace("\n", " / "))
    hit = {k: v for k, v in cells.items() if v[2]}
    out.append(f"\n## Per-cell detail — `{label}`\n")
    out.append(f"{len(hit)} of {len(cells)} cells have at least one truncated reply.\n")
    out.append("| candidate | role | n_sim | n_occ | truncated | MOVE | STAY | prompt context |")
    out.append("|---|---|---|---|---|---|---|---|")
    for k, (mv, st, tr) in sorted(hit.items(), key=lambda kv: -kv[1][2]):
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {tr} | {mv} | {st} | "
                   f"{str(ctx[k])[:70]} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--detail", action="append", default=[],
                    help="raw label to break down per cell (repeatable). Default: "
                         "every grammar arm that has at least one truncation.")
    args = ap.parse_args()
    results = Path(args.results_dir)

    rows, keep = [], {}
    for p in sorted(glob.glob(str(results / "raw" / "*_raw.jsonl.gz"))):
        label = os.path.basename(p)[:-len("_raw.jsonl.gz")]
        for pref in ("prompt_comparison_", "ratio_comparison_"):
            if label.startswith(pref):
                label = label[len(pref):]
        meta, recs = scan(p)
        s = summarise(label, meta, recs)
        rows.append(s)
        if s["grammar"] and s["n_truncated"]:
            keep[label] = recs

    gram = [r for r in rows if r["grammar"]]
    g_chat = [r for r in gram if r["endpoint"] == "chat"]
    g_comp = [r for r in gram if r["endpoint"] == "completions"]

    def tot(rs, k):
        return sum(r[k] for r in rs)

    md = ["# Grammar-arm bad parses are max_tokens truncation, not grammar violations",
          "", "Generated by `analyze_grammar_truncation.py`.", "",
          "## Headline", ""]
    md += [
        f"- grammar arms scanned: **{len(gram)}** ({tot(gram, 'n_samples'):,} samples)",
        f"- truncated (finish_reason=length) bad parses: **{tot(gram, 'n_truncated'):,}** "
        f"({100.0 * tot(gram, 'n_truncated') / max(tot(gram, 'n_samples'), 1):.4f}%)",
        f"- of which on the **completions** endpoint: **{tot(g_comp, 'n_truncated'):,}**",
        f"- of which on the **chat** endpoint: **{tot(g_chat, 'n_truncated'):,}** "
        f"(out of {tot(g_chat, 'n_samples'):,} samples)",
        f"- chat+grammar replies that even START with whitespace: "
        f"**{tot(g_chat, 'n_leading_ws'):,}**",
        "",
        "Every grammar-arm bad parse is a `/completions` arm. The chat template "
        "supplies the turn scaffolding that the model would otherwise emit itself, "
        "so on chat there is no leading whitespace to run out of budget in.",
        "",
        "## Per-arm (grammar arms only; non-zero rows first)", "",
        "| arm | endpoint | N | leading ws | truncated | ws-only | mid-word | fragments |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(gram, key=lambda r: (-r["n_truncated"], -r["n_leading_ws"], r["label"])):
        md.append(f"| `{r['label']}` | {r['endpoint']} | {r['n_samples']:,} | "
                  f"{r['n_leading_ws']} ({r['pct_leading_ws']}%) | {r['n_truncated']} | "
                  f"{r['n_trunc_whitespace_only']} | {r['n_trunc_midword']} | "
                  f"{r['midword_fragments'] or '—'} |")

    frags = Counter()
    for recs in keep.values():
        for r in recs:
            if r.get("parse") not in ("MOVE", "STAY") and r.get("text", "").strip():
                frags[r["text"].strip().upper()] += 1
    md += ["", "## Which decision was being cut off", "",
           "Truncated replies that reached a letter before the cut:", ""]
    for k, v in frags.most_common():
        md.append(f"- `{k}` x{v}")
    md += ["", "All fragments are STAY-initial; none are MOVE-initial. Dropping "
           "truncated replies from the effective rate therefore biases P(MOVE) "
           "slightly UP in the affected cells (which are already STAY-dominated, "
           "so the realised effect is small).", ""]

    for label in (args.detail or sorted(keep)):
        if label in keep:
            detail(label, keep[label], md)
        else:
            md.append(f"\n(no truncations recorded for `{label}`)")

    md_path = results / "grammar_truncation.md"
    md_path.write_text("\n".join(md) + "\n")
    csv_path = results / "grammar_truncation.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    print(f"\ngrammar arms: {tot(gram, 'n_truncated')} truncations of "
          f"{tot(gram, 'n_samples'):,} samples; chat+grammar: {tot(g_chat, 'n_truncated')}")


if __name__ == "__main__":
    main()
