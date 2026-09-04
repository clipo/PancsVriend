#!/usr/bin/env python3
"""Sample P(MOVE | neighborhood composition) for the ratio-only prompt candidates.

Sibling of evaluate_prompts.py (which sweeps the 3x3-grid candidates over the
0..8 out-group gradient). This script sweeps the RATIO candidates — prompts
that state only how many neighbors the agent has and their composition — over
EVERY composition the simulation can visit: n_occupied 0..8 x n_similar
0..n_occupied, i.e. 45 cells. The n_occupied=8 slice reproduces the classic
9-point gradient for direct comparison against prompt_comparison_* results.

Run one model at a time (one llama.cpp server), one arm per invocation:

    python prompt_refinement/evaluate_ratio_prompts.py \
        --llm-url http://localhost:8085/v1/completions \
        --model Qwen3.6-27B-Q5_K_M --label ratio-qwen3.6-27b-grammar \
        --grammar --samples 100

    # render every prompt without a server (M0 gate):
    python prompt_refinement/evaluate_ratio_prompts.py --dry-run

Estimation is PURE SAMPLING at the production temperature (0.3), mirroring the
production payload exactly (same SAMPLER_PARAMS, max_tokens=5, no stop, same
MOVE/STAY parse rule) — token-logprob estimation was considered and rejected
until its accuracy is established (see FUTURE_EXPLORATIONS.md).

Outputs (schema is LONG-format — one row per (candidate, role, cell) — because
the 2D composition table does not fit the wide move_rate_{k}of8 schema):
    results/ratio_comparison_<label>.csv
        candidate, agent_role, n_occupied, n_similar, n_opposite, n_empty,
        n_samples, n_move, n_stay, n_bad, move_rate
    results/ratio_comparison_<label>.md          human summary
    results/raw/ratio_comparison_<label>_raw.jsonl.gz
        every reply verbatim (one JSON object per line, _meta header first)

Ratio prompts are deterministic per cell (no layout randomness), so resolution
is ~1/samples and near-determinism at T=0.3 applies unchanged — a 0% cell means
"this prompt produces no movement at that composition". The G0 grid anchor keeps
seeded random layouts per sample, marginalising geometry exactly like the parent
sweep.
"""
import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Shared harness plumbing lives in sampling_common (2026-08-21 dedup): the
# production-payload sampler, the parse rule, the role->keyword maps, slice
# pings, the batch executor and the raw-reply writer used to be duplicated
# between this file and evaluate_prompts.py. GRAMMAR is production's
# llm_runner.MOVE_STAY_GRAMMAR (byte-identical +grammar arms), re-exported.
from sampling_common import (  # noqa: E402
    GRAMMAR,
    RESULTS_DIR,
    ROLE_KW,
    init_slice_state,
    parse,
    sample_batch,
    slice_ping,
    spearman,
    write_raw_gz,
)
from ratio_prompt_templates import (  # noqa: E402
    ALL_COMPOSITIONS,
    RATIO_CANDIDATES,
    grid_context,
    mechanical_move,
    plural,
    with_article,
)


def render_prompt(name, template, context_fn, n_sim, n_occ, role_kw, rng=None):
    """Render the full prompt for one cell. G0's renderer takes the rng.

    The template gets grammar-aware derived keys on top of role_kw
    (a_agent_type = article + label, agent_type_plural, ...) so loaded-context
    identity labels render as 'an Asian American family' / 'Black families'
    rather than 'a Asian...' / 'familys'. For the baseline labels these
    reproduce the historical bytes exactly.
    """
    if context_fn is grid_context:
        ctx = context_fn(n_sim, n_occ, rng=rng, **role_kw)
    else:
        ctx = context_fn(n_sim, n_occ, **role_kw)
    fmt = dict(role_kw,
               a_agent_type=with_article(role_kw["agent_type"]),
               a_opposite_type=with_article(role_kw["opposite_type"]),
               agent_type_plural=plural(role_kw["agent_type"]),
               opposite_type_plural=plural(role_kw["opposite_type"]))
    return template.format(context=ctx, **fmt), ctx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm-url", help="e.g. http://localhost:8085/v1/completions")
    ap.add_argument("--model", help="model id as served (GET /v1/models)")
    ap.add_argument("--label", help="slug for the output filenames")
    ap.add_argument("--samples", type=int, default=100,
                    help="samples per (candidate, role, cell). Resolution ~1/samples.")
    ap.add_argument("--temperature", type=float, default=0.3,
                    help="MUST match the simulation's temperature (config default: 0.3)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="parallel in-flight requests; keep <= the server's -np slots")
    ap.add_argument("--candidates", default=None,
                    help="comma-separated candidate names (default: all in RATIO_CANDIDATES)")
    ap.add_argument("--roles", choices=["red", "blue", "both"], default="red",
                    help="agent role(s) to sample; Phase A uses red (parent-sweep parity), "
                         "the Phase-B policy pass uses both")
    ap.add_argument("--grammar", action="store_true",
                    help="constrain generation with the permissive MOVE/STAY GBNF grammar")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for G0's per-sample random layouts (paired across arms)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every rendered prompt context for every cell and exit "
                         "(no server needed)")
    args = ap.parse_args()

    if args.candidates:
        wanted = [c.strip() for c in args.candidates.split(",")]
        missing = [c for c in wanted if c not in RATIO_CANDIDATES]
        if missing:
            ap.error(f"unknown candidate(s): {missing}; known: {list(RATIO_CANDIDATES)}")
        candidates = {c: RATIO_CANDIDATES[c] for c in wanted}
    else:
        candidates = dict(RATIO_CANDIDATES)

    roles = ["red", "blue"] if args.roles == "both" else [args.roles]

    if args.dry_run:
        for name, (tpl, fn) in candidates.items():
            print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
            for role in roles:
                for n_sim, n_occ in ALL_COMPOSITIONS:
                    rng = random.Random(args.seed)
                    _, ctx = render_prompt(name, tpl, fn, n_sim, n_occ,
                                           ROLE_KW[role], rng=rng)
                    mech = "MOVE" if mechanical_move(n_sim, n_occ) else "STAY"
                    ctx_one_line = ctx.replace("\n", " / ")
                    print(f"  [{role}] occ={n_occ} sim={n_sim} mech={mech:4s} | {ctx_one_line}")
        # Show one full prompt so the frame itself gets eyeballed too.
        name, (tpl, fn) = next(iter(candidates.items()))
        full, _ = render_prompt(name, tpl, fn, 3, 8, ROLE_KW[roles[0]])
        print(f"\n{'=' * 72}\nfull prompt example ({name}, occ=8 sim=3, {roles[0]}):\n{'=' * 72}\n{full}")
        return 0

    for req in ("llm_url", "model", "label"):
        if not getattr(args, req):
            ap.error(f"--{req.replace('_', '-')} is required unless --dry-run")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    raw_records = []

    total = len(candidates) * len(roles) * len(ALL_COMPOSITIONS) * args.samples
    print(f"model   : {args.model}")
    print(f"url     : {args.llm_url}")
    print(f"temp    : {args.temperature}   samples/cell: {args.samples}   "
          f"grammar: {'ON' if args.grammar else 'off'}   roles: {roles}")
    print(f"cells   : {len(ALL_COMPOSITIONS)} compositions x {len(candidates)} candidates x {len(roles)} role(s)")
    print(f"requests: {total}\n", flush=True)
    init_slice_state(len(candidates) * len(roles))

    for name, (tpl, fn) in candidates.items():
        for role in roles:
            role_kw = ROLE_KW[role]
            # G0: one seeded layout stream per (candidate, role), giving each
            # sample a fresh arrangement at its composition — layouts are paired
            # across arms via --seed just like the parent sweep.
            g0_rng = random.Random(args.seed) if fn is grid_context else None

            per_cell = {}
            for n_sim, n_occ in ALL_COMPOSITIONS:
                if fn is grid_context:
                    rendered = [render_prompt(name, tpl, fn, n_sim, n_occ, role_kw,
                                              rng=g0_rng) for _ in range(args.samples)]
                else:
                    rendered = [render_prompt(name, tpl, fn, n_sim, n_occ, role_kw)
                                ] * args.samples
                prompts = [p for p, _ in rendered]
                replies = sample_batch(args.llm_url, args.model, prompts,
                                       args.temperature,
                                       GRAMMAR if args.grammar else None,
                                       args.concurrency)
                for i, (r, (_, ctx)) in enumerate(zip(replies, rendered)):
                    raw_records.append({
                        "candidate": name, "agent_role": role,
                        "n_occupied": n_occ, "n_similar": n_sim,
                        "n_opposite": n_occ - n_sim, "sample": i, "context": ctx,
                        "text": r["text"], "finish_reason": r["finish_reason"],
                        "completion_tokens": r["completion_tokens"],
                        "parse": parse(r["text"]),
                    })
                d = Counter(parse(r["text"]) for r in replies)
                n_bad = d["AMBIGUOUS"] + d["UNPARSEABLE"]
                per_cell[(n_sim, n_occ)] = d
                n_valid = d["MOVE"] + d["STAY"]
                rows.append({
                    "candidate": name, "agent_role": role,
                    "n_occupied": n_occ, "n_similar": n_sim,
                    "n_opposite": n_occ - n_sim, "n_empty": 8 - n_occ,
                    "n_samples": args.samples, "n_move": d["MOVE"],
                    "n_stay": d["STAY"], "n_bad": n_bad,
                    # Two explicitly-named rates; there is deliberately NO bare
                    # "move_rate" column — that name was ambiguous:
                    #   raw       = n_move / N            (single-shot, incl. bad parses)
                    #   effective = n_move / (move+stay)  (retry-equivalent = production)
                    "move_rate_raw": round(d["MOVE"] / args.samples, 4),
                    "move_rate_effective": (round(d["MOVE"] / n_valid, 4)
                                            if n_valid else ""),
                })

            # Progress line: the n_occ=8 gradient summary (parent-sweep language).
            grad = [(lambda d: d["MOVE"] / (d["MOVE"] + d["STAY"])
                     if (d["MOVE"] + d["STAY"]) else float("nan"))(per_cell[(8 - k, 8)])
                    for k in range(9)]
            rho = spearman(list(range(9)), grad)
            bad_total = sum(r["n_bad"] for r in rows
                            if r["candidate"] == name and r["agent_role"] == role)
            print(f"  {name:22s} [{role:4s}] occ8-range={max(grad) - min(grad):.2f}  "
                  f"rho={rho:+.2f}  bad={bad_total:3d}  move@8opp={grad[8]:.2f}",
                  flush=True)   # unbuffered: logs (and log-watchers) see it live
            slice_ping(args.label, f"{name} [{role}]")

    raw_path = RESULTS_DIR / "raw" / f"ratio_comparison_{args.label}_raw.jsonl.gz"
    write_raw_gz(raw_path,
                 {"label": args.label, "model": args.model,
                  "url": args.llm_url, "temperature": args.temperature,
                  "samples": args.samples, "seed": args.seed,
                  "grammar": bool(args.grammar), "roles": roles,
                  "n_cells": len(ALL_COMPOSITIONS)},
                 raw_records)

    csv_path = RESULTS_DIR / f"ratio_comparison_{args.label}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Markdown summary: per candidate/role, the occ=8 gradient row (backward-
    # comparable with prompt_comparison_*) plus a compact occupancy table of
    # move_rate at the half-and-half diagonal.
    md = [
        f"# Ratio prompt comparison — `{args.model}`",
        "",
        f"- endpoint: `{args.llm_url}`   grammar: **{'YES' if args.grammar else 'no'}**",
        f"- **{args.samples} samples per cell at T={args.temperature}**, production payload "
        "(same `SAMPLER_PARAMS`, `max_tokens=5`, no `stop`), production MOVE/STAY parse rule.",
        f"- Full composition sweep: 45 cells (n_occupied 0..8 x n_similar 0..n_occ), roles: {roles}.",
        "- Prompts state neighbor counts only — no empties, no walls, no 'of 8' "
        "(see ratio_prompt_templates.py for the rationale).",
        f"- Resolution ~1/{args.samples}; ratio cells are a single fixed prompt each, so a 0% cell "
        "means that prompt produces no movement at that composition.",
        "",
        "## EFFECTIVE MOVE rate (n_move/(n_move+n_stay), retry-equivalent = what "
        "production does) across the n_occupied=8 gradient (0..8 opposite)",
        "",
        "| candidate | role | " + " | ".join(f"{k}opp" for k in range(9)) + " |",
        "|---|---|" + "---|" * 9,
    ]
    for name in candidates:
        for role in roles:
            grad = {r["n_opposite"]: (r["move_rate_effective"] if r["move_rate_effective"] != "" else float("nan")) for r in rows
                    if r["candidate"] == name and r["agent_role"] == role
                    and r["n_occupied"] == 8}
            md.append(f"| `{name}` | {role} | "
                      + " | ".join(f"{grad[k]:.2f}" for k in range(9)) + " |")
    md += [
        "",
        "Mechanical reference (Agent.py, threshold 0.5): MOVE iff n_opposite/n_occupied > 0.5 "
        "-> occ=8 row = 0,0,0,0,0,1,1,1,1.",
        "",
        f"Full 45-cell table in `{csv_path.name}`; every raw reply in `raw/{raw_path.name}`.",
        "",
    ]
    md_path = RESULTS_DIR / f"ratio_comparison_{args.label}.md"
    md_path.write_text("\n".join(md))

    print(f"\nwrote {raw_path}  ({len(raw_records)} raw replies)")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
