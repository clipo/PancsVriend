#!/usr/bin/env python3
"""Does the probability a llama-server returns depend on the BATCH it is in?

    python value_functions/batch_numerics/batch_numerics_probe.py --label qwen3.6-27b-chat-grammar
    python value_functions/batch_numerics/batch_numerics_probe.py --label ... --cells 8 --repeats 16 --seq-samples 300
    python value_functions/batch_numerics/batch_numerics_probe.py --label gemma-4-31b-q8-chat-grammar \
        --from-exact --match-cells gemma-4-31b-chat-grammar --seq-samples 300

--from-exact (2026-09-07): rank the candidate cells on the label's EXACT
tables (value_functions/results/llm_logprob/tables/vf_<label>-lp__*) instead
of a sampled campaign — for a label that never had one (the sampling route is
retired). The campaign columns are then NaN. --match-cells <other label> adds
every cell another probe run measured (results/probe_<other>_summary.csv), so
two quantisations of one model are probed on the same cells.

Everything about this artifact lives in this folder: the probe (this file), its
results (results/probe_<label>*.csv/json/png) and README.md. All requests go
through /v1/chat/completions — the campaign's endpoint — never /completion.

Standalone diagnostic for the batch-numerics artifact found 2026-09-05 (see
KV_CACHE_SAMPLING_ARTIFACT.md, addendum): with several requests in flight
(`-np 4`, continuous batching) a byte-identical request returns DIFFERENT
next-token probabilities — tens of points on transition cells — while one
request at a time is bit-reproducible. Needs the model's llama-server up on
--url with the campaign's flags; run_batch_numerics_study.py does that for
every model once (a one-time study, not a pipeline step).

For each probed cell (the unsaturated cells of the sampled artifact with the
largest spread, plus --extra-cells random ones) it records P(MOVE) read from
the chat endpoint's post-sampling probabilities under:

    sequential      one request in flight, --repeats times   (should be constant)
    concurrent      --concurrency identical requests in flight, --repeats times
    interleaved     the cell's request in flight with other cells' requests
    seq_sampling    --seq-samples campaign-payload draws, one in flight (optional)

and the campaign's sampled rate with its Wilson CI for reference.

Outputs (value_functions/batch_numerics/results/):
    probe_<label>.csv           one row per request (condition, repeat, p_move, prompt_n)
    probe_<label>_summary.csv   one row per cell: sequential value, its spread,
                                concurrent mean/sd/min/max, interleaved same,
                                campaign p and CI, sequential-sampling p and CI
    probe_<label>.json          label, server, flags, per-condition aggregate
    probe_<label>.png           per cell: sequential value, concurrent and
                                interleaved spreads, campaign CI, sequential-sampling CI
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

_THIS = Path(__file__).resolve().parent            # value_functions/batch_numerics
REPO_ROOT = _THIS.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from value_functions.paths import SAMPLED_DIR, LOGPROB_TABLES_DIR, add_import_paths  # noqa: E402
add_import_paths()

from sampling_common import load_value_function, role_keywords, wilson_ci  # noqa: E402
from ratio_prompt_templates import RATIO_CANDIDATES  # noqa: E402
from evaluate_ratio_prompts import render_prompt  # noqa: E402
from llm_runner import MOVE_STAY_GRAMMAR, SAMPLER_PARAMS  # noqa: E402

VF_DIR = SAMPLED_DIR
OUT_DIR = _THIS / "results"


def chat_probs(url, model, prompt, temperature, timeout=120):
    """(p_move, p_stay_partial_or_full, prompt_n): post-sampling top probs of the
    first answer token via the chat endpoint (the campaign's endpoint)."""
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
               "n_probs": 16, "post_sampling_probs": True, "temperature": temperature,
               "cache_prompt": False, "grammar": MOVE_STAY_GRAMMAR, "stream": False, **SAMPLER_PARAMS}
    r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    top = {t["token"]: t["prob"] for t in j["choices"][0]["logprobs"]["content"][0]["top_probs"]}
    # MOVE may appear as a whole token or as its first piece; report the mass on
    # tokens that begin the word MOVE (case-insensitive) after optional whitespace
    def starts(tok, word):
        s = tok.lstrip(" \t\n").lower()
        return bool(s) and word.startswith(s)
    p_move = sum(p for t, p in top.items() if starts(t, "move"))
    p_stay = sum(p for t, p in top.items() if starts(t, "stay"))
    z = p_move + p_stay
    return (p_move / z if z else float("nan")), (j.get("timings") or {}).get("prompt_n")


def sample_sequential(url, model, prompt, temperature, n, seed0):
    move = stay = 0
    for i in range(n):
        payload = {"model": model, "stream": False, "temperature": temperature, "max_tokens": 5,
                   **SAMPLER_PARAMS, "grammar": MOVE_STAY_GRAMMAR, "cache_prompt": False,
                   "seed": seed0 + i, "messages": [{"role": "user", "content": prompt}]}
        txt = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=120).json()["choices"][0]["message"]["content"].strip().upper()
        move += txt == "MOVE"; stay += txt == "STAY"
    return move, stay


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--style", default="R3_dual_count")
    ap.add_argument("--url", default="http://localhost:8085")
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--cells", type=int, default=8, help="most-uncertain unsaturated cells to probe")
    ap.add_argument("--extra-cells", type=int, default=2, help="random additional cells")
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seq-samples", type=int, default=0, help="0 = skip sequential sampling")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-exact", action="store_true",
                    help="rank candidate cells on the label's exact (-lp) tables, not a sampled campaign "
                         "(labels without one); campaign columns are NaN")
    ap.add_argument("--match-cells", default=None, metavar="LABEL",
                    help="also probe every cell of results/probe_<LABEL>_summary.csv (same cells across "
                         "two quantisations of one model)")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tpl, fn = RATIO_CANDIDATES[args.style]
    # candidate cells: unsaturated, ranked by p(1-p) (where the artifact is most sensitive).
    # Tuple: (p(1-p), scenario, role, n_similar, n_occupied, campaign n_move, campaign n, meta);
    # campaign counts are None when the ranking comes from the exact tables.
    cands, allcells = [], {}
    if args.from_exact:
        src = sorted(LOGPROB_TABLES_DIR.glob(f"vf_{args.label}-lp__*__{args.style}.json"))
    else:
        src = sorted(VF_DIR.glob(f"vf_{args.label}__*__{args.style}.json"))
    for f in src:
        vf = load_value_function(f); scenario = f.name.split("__")[1]
        for role, rows in vf["compositions"].items():
            for c in rows:
                p = c["p_move_effective"]
                if args.from_exact:
                    nm, n = None, None
                    unsat = p is not None and 0.005 < p < 0.995
                else:
                    nm, n = c["n_move"], c["n_move"] + c["n_stay"]
                    unsat = bool(n) and 0 < nm < n
                rec = (p * (1 - p) if p is not None else -1.0, scenario, role,
                       c["n_similar"], c["n_occupied"], nm, n, vf["meta"])
                allcells[(scenario, role, c["n_similar"], c["n_occupied"])] = rec
                if unsat:
                    cands.append(rec)
    if not cands:
        print("no unsaturated cells in the " + ("exact tables" if args.from_exact else "sampled artifacts")); return 2
    meta = cands[0][-1]
    model = args.model or meta.get("model") or args.label
    temperature = args.temperature if args.temperature is not None else float(meta["temperature"])
    cands.sort(key=lambda x: -x[0])
    rng = random.Random(args.seed)
    chosen = cands[:args.cells] + rng.sample(cands[args.cells:], min(args.extra_cells, max(0, len(cands) - args.cells)))
    if args.match_cells:
        ref = pd.read_csv(OUT_DIR / f"probe_{args.match_cells}_summary.csv")
        have = {(c[1], c[2], c[3], c[4]) for c in chosen}
        added = 0
        for r in ref.itertuples():
            key = (r.scenario, r.role, int(r.n_similar), int(r.n_occupied))
            if key in have or key not in allcells:
                continue
            chosen.append(allcells[key]); have.add(key); added += 1
        print(f"--match-cells {args.match_cells}: {added} cell(s) added (shared with that probe)")
    props = requests.get(f"{args.url}/props", timeout=60).json()
    print(f"label={args.label} model={model} T={temperature} server={props.get('model_path')} "
          f"slots={props.get('total_slots')} cells={len(chosen)}")

    prompts = {}
    for _, scenario, role, ns, no, *_r in chosen:
        prompts[(scenario, role, ns, no)] = render_prompt(args.style, tpl, fn, ns, no,
                                                          role_keywords(scenario)[role])[0]
    rows = []
    def rec(key, cond, rep, p, pn):
        scenario, role, ns, no = key
        rows.append({"label": args.label, "scenario": scenario, "role": role, "n_similar": ns,
                     "n_occupied": no, "condition": cond, "repeat": rep, "p_move": p, "prompt_n": pn,
                     "t": time.time()})
    keys = list(prompts)
    # sequential
    for key in keys:
        for i in range(args.repeats):
            p, pn = chat_probs(args.url, model, prompts[key], temperature); rec(key, "sequential", i, p, pn)
    # concurrent, identical prompt
    for key in keys:
        with ThreadPoolExecutor(args.concurrency) as ex:
            res = list(ex.map(lambda i: chat_probs(args.url, model, prompts[key], temperature),
                              range(args.repeats)))
        for i, (p, pn) in enumerate(res):
            rec(key, "concurrent", i, p, pn)
    # interleaved: every in-flight batch mixes cells
    jobs = [(key, i) for i in range(args.repeats) for key in keys]
    rng.shuffle(jobs)
    with ThreadPoolExecutor(args.concurrency) as ex:
        res = list(ex.map(lambda kj: (kj, chat_probs(args.url, model, prompts[kj[0]], temperature)), jobs))
    for (key, i), (p, pn) in res:
        rec(key, "interleaved", i, p, pn)
    # optional sequential sampling
    seq = {}
    if args.seq_samples > 0:
        for k_i, key in enumerate(keys):
            mv, st = sample_sequential(args.url, model, prompts[key], temperature, args.seq_samples,
                                       700000 + 1000 * k_i)
            seq[key] = (mv, mv + st)
            print(f"  seq-sampling {key}: {mv}/{mv + st}")

    df = pd.DataFrame(rows)
    df.to_csv(out / f"probe_{args.label}.csv", index=False)
    summ = []
    campaign = {(sc, role, ns, no): (nm, n) for _, sc, role, ns, no, nm, n, _m in chosen}
    for key in keys:
        d = df[(df.scenario == key[0]) & (df.role == key[1]) & (df.n_similar == key[2]) & (df.n_occupied == key[3])]
        s = d[d.condition == "sequential"].p_move; c = d[d.condition == "concurrent"].p_move; il = d[d.condition == "interleaved"].p_move
        nm, n = campaign[key]
        if n:
            lo, hi = wilson_ci(nm, n); cp = nm / n
        else:                                   # --from-exact: no campaign
            lo = hi = cp = float("nan"); n = 0
        row = {"label": args.label, "scenario": key[0], "role": key[1], "n_similar": key[2], "n_occupied": key[3],
               "sequential_p": s.iloc[0], "sequential_range": float(s.max() - s.min()),
               "concurrent_mean": float(c.mean()), "concurrent_sd": float(c.std(ddof=1)),
               "concurrent_min": float(c.min()), "concurrent_max": float(c.max()),
               "interleaved_mean": float(il.mean()), "interleaved_sd": float(il.std(ddof=1)),
               "interleaved_min": float(il.min()), "interleaved_max": float(il.max()),
               "campaign_p": cp, "campaign_n": n, "campaign_ci_low": lo, "campaign_ci_high": hi,
               "campaign_minus_sequential": cp - s.iloc[0]}
        if key in seq:
            mv, tot = seq[key]; lo2, hi2 = wilson_ci(mv, tot)
            row.update({"seq_sampling_p": mv / tot, "seq_sampling_n": tot, "seq_sampling_ci_low": lo2,
                        "seq_sampling_ci_high": hi2, "sequential_inside_seq_sampling_ci": lo2 <= s.iloc[0] <= hi2})
        summ.append(row)
        print(f"  {key[0][:10]:10s}/{key[1]} {key[2]}/{key[3]}: sequential {s.iloc[0]:.4f} (range {row['sequential_range']:.1e})  "
              f"concurrent {c.min():.3f}-{c.max():.3f} (sd {row['concurrent_sd']:.3f})  campaign {cp:.3f} [{lo:.3f},{hi:.3f}]")
    sm = pd.DataFrame(summ); sm.to_csv(out / f"probe_{args.label}_summary.csv", index=False)
    agg = {"label": args.label, "model": model, "temperature": temperature,
           "server": {k: props.get(k) for k in ("model_path", "build_info", "total_slots")},
           "cells": len(keys), "repeats": args.repeats, "concurrency": args.concurrency,
           "sequential_max_range": float(sm.sequential_range.max()),
           "concurrent_sd_median": float(sm.concurrent_sd.median()),
           "concurrent_span_median": float((sm.concurrent_max - sm.concurrent_min).median()),
           "concurrent_span_max": float((sm.concurrent_max - sm.concurrent_min).max()),
           "campaign_minus_sequential_median_abs": (float(sm.campaign_minus_sequential.abs().median())
                                                   if sm.campaign_n.gt(0).any() else None),
           "campaign_minus_sequential_max_abs": (float(sm.campaign_minus_sequential.abs().max())
                                                if sm.campaign_n.gt(0).any() else None),
           "cells_from": "exact tables" if args.from_exact else "sampled campaign",
           "matched_cells_from": args.match_cells,
           "created": datetime.now().isoformat(timespec="seconds")}
    (out / f"probe_{args.label}.json").write_text(json.dumps(agg, indent=1))
    fig_probe(df, sm, args.label, out / f"probe_{args.label}.png")
    print(json.dumps(agg, indent=1))
    return 0


def fig_probe(df, sm, label, path):
    """One row per cell: the sequential value (a single mark), every concurrent
    and interleaved value (strips), the campaign CI and the sequential-sampling
    CI. Two validated categorical hues (#4C6EF5 / #E8590C) + neutral greys."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    cells = [(r.scenario, r.role, r.n_similar, r.n_occupied) for r in sm.itertuples()]
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(cells) + 1.8))
    for y, key in enumerate(cells):
        d = df[(df.scenario == key[0]) & (df.role == key[1]) & (df.n_similar == key[2]) & (df.n_occupied == key[3])]
        row = sm.iloc[y]
        # campaign CI (grey band) and sequential-sampling CI (dark band), behind the points
        if not np.isnan(row.campaign_ci_low):
            ax.plot([row.campaign_ci_low, row.campaign_ci_high], [y + 0.22, y + 0.22], color="#ADB5BD", lw=5, solid_capstyle="butt",
                    label="campaign rate, Wilson 95% CI (concurrency 4)" if y == 0 else None, zorder=1)
        if "seq_sampling_ci_low" in row and not np.isnan(row.get("seq_sampling_ci_low", np.nan)):
            ax.plot([row.seq_sampling_ci_low, row.seq_sampling_ci_high], [y - 0.22, y - 0.22], color="#495057", lw=5,
                    solid_capstyle="butt", label="sequential sampling, Wilson 95% CI" if y == 0 else None, zorder=1)
        c = d[d.condition == "concurrent"].p_move; il = d[d.condition == "interleaved"].p_move
        jit = np.random.default_rng(y).uniform(-0.08, 0.08, len(c))
        ax.scatter(c, y + 0.08 + jit, s=16, color="#E8590C", alpha=0.7, label="concurrent (4 identical in flight)" if y == 0 else None, zorder=3)
        jit = np.random.default_rng(y + 100).uniform(-0.08, 0.08, len(il))
        ax.scatter(il, y - 0.08 + jit, s=16, color="#E8590C", alpha=0.7, marker="s",
                   label="interleaved with other cells" if y == 0 else None, zorder=3)
        s = d[d.condition == "sequential"].p_move
        ax.scatter([s.iloc[0]], [y], s=90, color="#4C6EF5", marker="D", edgecolors="white", linewidths=1.2,
                   label="sequential (one in flight; identical on every repeat)" if y == 0 else None, zorder=4)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([f"{k[0][:12]}/{k[1]} {k[2]}/{k[3]}" for k in cells], fontsize=8)
    ax.set_xlabel("P(MOVE) returned by the chat endpoint (post-sampling, T as campaign)")
    ax.set_xlim(-0.02, 1.02); ax.grid(axis="x", alpha=0.2)
    ax.set_title(f"{label}: does the probability depend on the batch? ({len(df[df.condition=='concurrent'])//max(len(cells),1)} repeats per condition)", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
