#!/usr/bin/env python3
"""Score MOVE/STAY prompt candidates on COST and QUALITY, for one served model.

Run ONE model at a time (one llama.cpp server), then point this at it:

    # Llama-3.3-70B (port 8082)
    python prompt_refinement/evaluate_prompts.py \
        --llm-url http://localhost:8082/v1/completions \
        --model Llama-3.3-70B-Instruct-Q4_K_M \
        --label llama-3.3-70b-instruct-q4_k_m

    # Gemma-4-31B (port 8083)
    python prompt_refinement/evaluate_prompts.py \
        --llm-url http://localhost:8083/v1/completions \
        --model gemma-4-31B-it-Q5_K_M \
        --label gemma-4-31b-it-q5_k_m

Writes results/prompt_comparison_<label>.{md,csv} — one set per model.

WHAT IS MEASURED
----------------
COST     recomputed_tokens : tokens after {context}. llama.cpp caches only a prompt
                             PREFIX, and the grid is the first thing that varies, so
                             every token after it is re-evaluated on EVERY call.
                             Lower = faster. Directly sets wall-clock on the ~10^5
                             calls a production run makes.

QUALITY  Each (candidate, neighbourhood) is SAMPLED --samples times using the exact
         request settings llm_runner.py sends (same SAMPLER_PARAMS, same temperature,
         same max_tokens, no stop) and parsed with the same MOVE/STAY rule. So the
         reported move_rate is what the agents literally do in the simulation.

         Swept over a clean gradient of 0..8 out-group neighbours. Each sample draws a
         FRESH random arrangement at its density (same set across candidates, --seed), so
         move_rate reflects the density itself, not one arbitrary fixed layout. Giving:
           move_range : spread of move_rate over the gradient. Higher = the decision
                        actually responds to the neighbourhood.
           monotonic  : Spearman of move_rate vs out-group count. A prompt whose MOVE
                        rate FALLS as hostility rises encodes an incoherent preference
                        and is unusable no matter how fast it is.
           bad_parses : ambiguous/unparseable replies -> retries in the real runner.

NOTE ON SAMPLING vs LOGPROBS
         At T=0.3 the sampler is near-deterministic, so a prompt whose raw P(MOVE) is
         0.15 and one whose raw P(MOVE) is 0.005 can BOTH sample 0/50 MOVE and tie at
         zero. Sampling measures what the agent does; it does not resolve differences
         below its resolution (~1/samples). A tie at 0% is itself the finding: at this
         temperature that prompt produces no movement at all.
"""
import argparse
import csv
import gzip
import json
import math
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompt_templates import CANDIDATES          # noqa: E402
from llm_runner import SAMPLER_PARAMS            # noqa: E402  (the pinned pure-temperature sampler)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Roles held fixed across candidates so only the TEMPLATE varies.
KW = dict(agent_type="red team resident", opposite_type="blue team resident")

# Permissive GBNF grammar (--grammar): admits exactly what the MOVE/STAY parser accepts --
# optional leading whitespace/newlines + any casing of one of the two words -- and nothing
# else. Wide enough to preserve the model's natural surface form (its habit of opening with
# a newline, its preferred casing) so the measured distribution is distorted as little as
# possible, while still making every reply parseable by construction and halting generation
# the moment the word completes (no run-on commentary). llama.cpp masks illegal tokens at
# each sampling step and renormalises over the survivors.
GRAMMAR = r"""
root   ::= ws answer
ws     ::= [ 	
]*
answer ::= move | stay
move   ::= [Mm] [Oo] [Vv] [Ee]
stay   ::= [Ss] [Tt] [Aa] [Yy]
"""


def grid_with(n_out: int, rng=None) -> str:
    """3x3 neighbourhood with n_out out-group ('O') and 8-n_out same-group ('S').

    No 'E' (empty house) is placed: a vacancy is itself a strong MOVE cue, and varying
    it alongside composition would confound the gradient.

    If `rng` (a random.Random) is given, the O/S cells are SHUFFLED, so repeated calls at
    the same density yield different spatial arrangements. Sampling over fresh layouts
    marginalises the move-rate over geometry, instead of measuring one fixed (arbitrary)
    layout per density and mistaking a position/adjacency effect for a density effect.
    Token count is arrangement-invariant, so cost measurement passes rng=None for a
    stable, reproducible prompt.
    """
    cells = ["O"] * n_out + ["S"] * (8 - n_out)
    if rng is not None:
        rng.shuffle(cells)
    c = cells[:4] + ["X"] + cells[4:]
    return "\n".join(" ".join(c[i * 3:i * 3 + 3]) for i in range(3))


def n_tokens(base: str, text: str) -> int:
    r = requests.post(f"{base}/tokenize", json={"content": text}, timeout=60)
    r.raise_for_status()
    return len(r.json()["tokens"])


def sample_once(url: str, model: str, prompt: str, temperature: float,
                grammar: str | None = None) -> dict:
    """One decision, using the SAME payload llm_runner.py sends (see LLMAgent).

    Endpoint is inferred from the URL: `/chat/completions` sends the prompt as a single
    user `messages` turn (so the server applies the model's chat template -- special tokens,
    role framing), while `/completions` sends the raw `prompt` string unwrapped. Everything
    else (sampler, stop, max_tokens) is identical, so the two runs isolate the effect of the
    chat interface itself. Response parsing handles both `text` and `message.content`.
    """
    # Mirror llm_runner.py's production payload exactly: NO "stop" (a leading newline is a
    # plausible first token on a raw completion; a "\n" stop would truncate it to an empty
    # string and burn a retry) and max_tokens=5 (slack absorbs leading whitespace; the
    # parser only looks for MOVE/STAY anyway).
    payload = {
        "model": model,
        "stream": False,
        "temperature": temperature,
        "max_tokens": 5,
        **SAMPLER_PARAMS,
    }
    if grammar is not None:
        payload["grammar"] = grammar
    if "/chat/completions" in url:
        payload["messages"] = [{"role": "user", "content": prompt}]
    else:
        payload["prompt"] = prompt
    r = requests.post(url, timeout=600, json=payload)
    r.raise_for_status()
    j = r.json()
    c = j["choices"][0]
    text = c.get("text") or (c.get("message") or {}).get("content", "") or ""
    return {
        "text": text,
        "finish_reason": c.get("finish_reason"),
        "completion_tokens": (j.get("usage") or {}).get("completion_tokens"),
    }


def parse(text: str) -> str:
    """The exact MOVE/STAY rule llm_runner.py uses (substring match, ambiguity = bad)."""
    u = text.strip().upper()
    has_move, has_stay = "MOVE" in u, "STAY" in u
    if has_move and has_stay:
        return "AMBIGUOUS"
    if has_move:
        return "MOVE"
    if has_stay:
        return "STAY"
    return "UNPARSEABLE"


def spearman(xs, ys) -> float:
    """Rank correlation, no scipy dependency. +1 = perfectly increasing."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm-url", required=True, help="e.g. http://localhost:8082/v1/completions")
    ap.add_argument("--model", required=True, help="model id as served (GET /v1/models)")
    ap.add_argument("--label", required=True, help="slug for the output filenames")
    ap.add_argument("--samples", type=int, default=50,
                    help="samples per (candidate, neighbourhood). Resolution is ~1/samples.")
    ap.add_argument("--temperature", type=float, default=0.3,
                    help="MUST match the simulation's temperature (config default: 0.3)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="parallel in-flight requests; keep <= the server's -np slots")
    ap.add_argument("--candidates", default=None,
                    help="comma-separated candidate names to run (default: all in CANDIDATES)")
    ap.add_argument("--grammar", action="store_true",
                    help="constrain generation with the permissive MOVE/STAY GBNF grammar "
                         "(optional leading whitespace + any casing); llama.cpp only")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the per-sample random neighbourhood layouts (paired across candidates)")
    ap.add_argument("--prompt-eval-tps", type=float, default=60.0,
                    help="measured prompt-eval tokens/sec, used to convert tokens -> seconds")
    args = ap.parse_args()

    if args.candidates:
        wanted = [c.strip() for c in args.candidates.split(",")]
        missing = [c for c in wanted if c not in CANDIDATES]
        if missing:
            ap.error(f"unknown candidate(s): {missing}; known: {list(CANDIDATES)}")
        candidates = {c: CANDIDATES[c] for c in wanted}
    else:
        candidates = CANDIDATES

    base = args.llm_url.split("/v1/")[0].rstrip("/")
    gradient = list(range(9))                       # 0..8 out-group neighbours
    # Paired design: build ONE set of `samples` random layouts per density, SHARED across
    # all candidates, so every candidate is scored on the identical neighbourhoods and the
    # comparison is not confounded by which arrangements each happened to draw. Seeded for
    # reproducibility (--seed). Cost uses a deterministic grid (token count is layout-invariant).
    rng = random.Random(args.seed)
    layouts = {n: [grid_with(n, rng) for _ in range(args.samples)] for n in gradient}
    cost_grid = grid_with(4)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    raw_records = []   # every (candidate, neighbourhood, sample) reply, dumped to results/raw/

    total = len(candidates) * len(gradient) * args.samples
    print(f"model   : {args.model}")
    print(f"url     : {args.llm_url}")
    print(f"temp    : {args.temperature}   samples/cell: {args.samples}   grammar: {'ON' if args.grammar else 'off'}")
    print(f"requests: {total}\n")

    for name, tpl in candidates.items():
        prefix = tpl.split("{context}")[0].format(**KW)
        full = tpl.format(context=cost_grid, **KW)
        cached = n_tokens(base, prefix)
        recomputed = n_tokens(base, full) - cached

        move_rates, bad_counts, examples = [], [], Counter()
        for n in gradient:
            prompts = [tpl.format(context=g, **KW) for g in layouts[n]]
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                replies = list(ex.map(          # ex.map preserves input order -> replies[i] pairs with layouts[n][i]
                    lambda p: sample_once(args.llm_url, args.model, p, args.temperature,
                                          GRAMMAR if args.grammar else None),
                    prompts))
            texts = [r["text"] for r in replies]
            # RAW LOG: every reply, verbatim, for post-experiment analysis (e.g. was a
            # bad parse a refusal or a max_tokens truncation? did truncation of an
            # echoed question create a spurious MOVE/STAY?). One JSON object per line.
            for i, (r, g) in enumerate(zip(replies, layouts[n])):
                raw_records.append({
                    "candidate": name, "n_out": n, "sample": i, "grid": g,
                    "text": r["text"], "finish_reason": r["finish_reason"],
                    "completion_tokens": r["completion_tokens"],
                    "parse": parse(r["text"]),
                })
            d = Counter(parse(t) for t in texts)
            examples.update(repr(t.strip()) for t in texts)
            bad = d["AMBIGUOUS"] + d["UNPARSEABLE"]
            move_rates.append(d["MOVE"] / args.samples)
            bad_counts.append(bad)

        rho = spearman(gradient, move_rates)
        rows.append({
            "candidate": name,
            "cached_prefix_tokens": cached,
            "recomputed_tokens": recomputed,
            "est_prompt_eval_s": round(recomputed / args.prompt_eval_tps, 2),
            "samples_per_cell": args.samples,
            "temperature": args.temperature,
            "move_rate_min": round(min(move_rates), 3),
            "move_rate_max": round(max(move_rates), 3),
            "move_range": round(max(move_rates) - min(move_rates), 3),
            "spearman_monotonic": round(rho, 3),
            "bad_parses_total": sum(bad_counts),
            "top_replies": "; ".join(f"{t}x{c}" for t, c in examples.most_common(3)),
            **{f"move_rate_{n}of8": round(p, 3) for n, p in zip(gradient, move_rates)},
        })
        print(f"  {name:26s} recomp={recomputed:4d}tok  range={max(move_rates)-min(move_rates):.2f}  "
              f"rho={rho:+.2f}  bad={sum(bad_counts):3d}  move@8/8={move_rates[-1]:.2f}")

    raw_dir = RESULTS_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"prompt_comparison_{args.label}_raw.jsonl.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8") as f:
        meta = {"_meta": True, "label": args.label, "model": args.model, "url": args.llm_url,
                "temperature": args.temperature, "samples": args.samples, "seed": args.seed,
                "grammar": bool(args.grammar)}
        f.write(json.dumps(meta) + "\n")
        for rec in raw_records:
            f.write(json.dumps(rec) + "\n")

    csv_path = RESULTS_DIR / f"prompt_comparison_{args.label}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = [
        f"# Prompt comparison — `{args.model}`",
        "",
        f"- endpoint: `{args.llm_url}` (queried live)",
        f"- grammar-constrained: **{'YES -- permissive MOVE/STAY GBNF' if args.grammar else 'no'}**",
        f"- **{args.samples} samples per cell at T={args.temperature}**, using the exact payload "
        f"`llm_runner.py` sends (same `SAMPLER_PARAMS`, `max_tokens=5`, no `stop`) and the same "
        f"MOVE/STAY parse rule. So `move_rate` is what the agents literally do in the simulation.",
        "- Gradient: 0..8 out-group neighbours, no empty cell (a vacancy is itself a MOVE cue).",
        f"- Each sample draws a FRESH random layout at its density (seed={args.seed}), the same "
        "set across all candidates (paired), so move_rate reflects density, not one fixed arrangement.",
        f"- Resolution is ~1/{args.samples}. Two prompts can both sample 0% and tie — at T={args.temperature} "
        "the sampler is near-deterministic, so a 0% row means *that prompt produces no movement at all*.",
        "",
        "## Summary",
        "",
        "| candidate | recomputed tok/call | est. prompt-eval | MOVE rate range | monotonic (rho) | bad parses |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| `{r['candidate']}` | {r['recomputed_tokens']} | {r['est_prompt_eval_s']}s | "
            f"{r['move_rate_min']:.2f} – {r['move_rate_max']:.2f} ({r['move_range']:.2f}) | "
            f"{r['spearman_monotonic']:+.2f} | {r['bad_parses_total']} |"
        )
    md += [
        "",
        "`recomputed tok/call` — tokens after `{context}`; re-evaluated on EVERY call (llama.cpp caches only a prefix). Lower is faster.",
        "`monotonic (rho)` — Spearman of MOVE rate vs out-group count. **Negative or ~0 means the agent does not want to leave a hostile neighbourhood** — incoherent, and unusable regardless of speed.",
        "`bad parses` — ambiguous/unparseable replies; each one costs a retry in the real runner.",
        "",
        f"## MOVE rate across the out-group gradient ({args.samples} samples, T={args.temperature})",
        "",
        "| candidate | " + " | ".join(f"{n}/8" for n in gradient) + " |",
        "|---|" + "---|" * len(gradient),
    ]
    for r in rows:
        md.append(f"| `{r['candidate']}` | "
                  + " | ".join(f"{r[f'move_rate_{n}of8']:.2f}" for n in gradient) + " |")
    md.append("")

    md_path = RESULTS_DIR / f"prompt_comparison_{args.label}.md"
    md_path.write_text("\n".join(md))

    print(f"\nwrote {raw_path}  ({len(raw_records)} raw replies)")
    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
