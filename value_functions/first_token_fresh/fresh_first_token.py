#!/usr/bin/env python3
"""Fresh first-token validation: re-query one model and compare with its stored records.

    python value_functions/first_token_fresh/fresh_first_token.py --model phi4
    python value_functions/first_token_fresh/fresh_first_token.py --model phi4 --concurrency 4 --subset 60 --tag c4
    python value_functions/first_token_fresh/fresh_first_token.py --model phi4 --check-only     # no server

For every stored case of one model (6 scenarios x 90 (role, cell) cases) this
asks llama-server for the FIRST token only, with the request the extraction
itself used for its first step, and compares the fresh table with the stored
one.

Like for like, by construction rather than by re-typing settings:
  * the request is `logprob_value_function.Server.next_token_probs(prompt, "", "")`,
    the extraction's own first-step call: /v1/chat/completions, the prompt as
    the single user message, max_tokens 1 (one predicted token), n_probs 64,
    post_sampling_probs, cache_prompt off, llm_runner.MOVE_STAY_GRAMMAR and
    llm_runner.SAMPLER_PARAMS;
  * temperature and n_probs are read from each record's header, and the run
    refuses to start unless the header's temperature, n_probs, sampler_params
    and grammar_sha256 equal the production values in llm_runner;
  * every prompt is rebuilt with the extraction's own renderer and must hash
    to the prompt_sha256 stored with its case;
  * the server is started through value_functions.serving.PinnedServer (the
    model's registered gguf, flags and pinned clock), and the served file must
    be the one named in the record headers;
  * requests are issued with the header's concurrency (1) unless --concurrency
    says otherwise, which is how a batching effect is tested.

Readouts (from value_functions/first_token/first_token_vs_path_sum.py):
  ft_mask     P(MOVE) from the first token, masked to the grammar and renormalised
  whole_word  P(MOVE) from the most likely whole-word MOVE token vs STAY token
Each is compared, case by case, with (a) the same readout of the STORED
first-step table and (b) the stored path sum.

Outputs (default value_functions/first_token_fresh/results/):
  fresh_<label>[_<tag>]_cases.csv        one row per case
  fresh_<label>[_<tag>]_tables.jsonl.gz  every fresh top-n table as returned (the raw record)
  fresh_<label>[_<tag>]_summary.json     settings, checks and the difference statistics
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "value_functions" / "first_token"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import first_token_vs_path_sum as ft  # noqa: E402  (classify, first_token_readouts, canon_ratio, adapters)
from llm_runner import MOVE_STAY_GRAMMAR, SAMPLER_PARAMS  # noqa: E402
from value_functions.logprob import logprob_value_function as lpv  # noqa: E402
from value_functions.paths import LOGPROB_DIR  # noqa: E402
from value_functions.serving import MODELS  # noqa: E402

RAW_DIR = LOGPROB_DIR / "raw"
OUT_DIR = Path(__file__).resolve().parent / "results"
SAT_EPS = lpv.SAT_EPS            # 0.005: unsaturated means SAT_EPS < path sum < 1 - SAT_EPS
GATE = 0.01                      # differences above this are the ones that would matter
PRODUCTION_TEMPERATURE = lpv.DEFAULT_TEMPERATURE   # 0.3, every campaign's value
PRODUCTION_N_PROBS = 64
DIFFS = ("d_mask_vs_stored_first", "d_mask_vs_path", "d_word_vs_stored_word", "d_word_vs_path")


# ------------------------------------------------------------------ stored --

def stored_files(label):
    return sorted(RAW_DIR.glob(f"vflp_{label}__*_states.jsonl.gz"))


def load_stored(label):
    """[(header, cell)] for every stored case of `label`, in file order."""
    cases = []
    for f in stored_files(label):
        with gzip.open(f, "rt") as fh:
            header = json.loads(fh.readline())
            for line in fh:
                cases.append((header, json.loads(line)))
    return cases


def check_header(header):
    """Raise unless the record's settings are the production settings."""
    want = {"temperature": PRODUCTION_TEMPERATURE, "n_probs": PRODUCTION_N_PROBS,
            "sampler_params": SAMPLER_PARAMS,
            "grammar_sha256": hashlib.sha256(MOVE_STAY_GRAMMAR.encode()).hexdigest()}
    got = {k: header.get(k) for k in want}
    if got != want:
        raise ValueError(f"record header differs from production settings: {got} != {want}")
    if not str(header.get("endpoint", "")).startswith("/v1/chat/completions"):
        raise ValueError(f"record was not extracted on the chat endpoint: {header.get('endpoint')!r}")
    return want


def build_prompt(header, cell):
    """The user prompt of one stored case, rebuilt with the extraction's renderer."""
    style = header["style"]
    tpl, fn = lpv.RATIO_CANDIDATES[style]
    scenario_file = header.get("scenario_file")
    kw = lpv.role_keywords(header["scenario"], str(REPO / scenario_file) if scenario_file else None)
    return lpv.render_prompt(style, tpl, fn, cell["n_similar"], cell["n_occupied"], kw[cell["role"]])[0]


def prompts_for(cases):
    """Prompts for all cases; raises if any does not hash to its stored prompt_sha256."""
    out, bad = [], []
    for header, cell in cases:
        prompt = build_prompt(header, cell)
        if hashlib.sha256(prompt.encode()).hexdigest() != cell["prompt_sha256"]:
            bad.append((header["scenario"], cell["role"], cell["n_similar"], cell["n_occupied"]))
        out.append(prompt)
    if bad:
        raise ValueError(f"{len(bad)} rebuilt prompts do not match the stored prompt_sha256, e.g. {bad[:3]}")
    return out


# ----------------------------------------------------------------- compare --

def readouts(pairs, temperature):
    """ft_mask, whole_word (stored T), whole_word at T = 1, whitespace-first share."""
    m_off, ft_mask, _ft_mask_t1, word, word_t1 = ft.first_token_readouts(pairs, temperature)
    return {"ft_mask": ft_mask, "whole_word": word, "whole_word_t1": word_t1, "m_off": m_off}


def table_max_abs_diff(a, b):
    """Largest |p_a - p_b| over the union of tokens (a token missing from one side counts as 0)."""
    da, db = {}, {}
    for piece, p in a:
        da[piece] = da.get(piece, 0.0) + p
    for piece, p in b:
        db[piece] = db.get(piece, 0.0) + p
    return max(abs(da.get(k, 0.0) - db.get(k, 0.0)) for k in set(da) | set(db))


def _absdiff(x, y):
    return math.nan if (math.isnan(x) or math.isnan(y)) else abs(x - y)


def compare_case(header, cell, fresh_pairs, prompt_n=None):
    """One results row: fresh vs stored first step, and fresh vs stored path sum."""
    t = float(header["temperature"])
    stored_pairs = ft.get_root_table(cell)
    fresh, stored = readouts(fresh_pairs, t), readouts(stored_pairs, t)
    path = ft.get_path_p_move(cell)
    root = next(s for s in cell["states"] if s.get("prefix") == "")
    return {
        "model": header["label"], "scenario": header["scenario"], "role": cell["role"],
        "n_similar": cell["n_similar"], "n_occupied": cell["n_occupied"],
        "unsaturated": int(SAT_EPS < path < 1 - SAT_EPS),
        "stored_path_sum": path, "stored_ft_mask": stored["ft_mask"], "stored_whole_word": stored["whole_word"],
        "fresh_ft_mask": fresh["ft_mask"], "fresh_whole_word": fresh["whole_word"],
        "fresh_whole_word_t1": fresh["whole_word_t1"], "fresh_m_off": fresh["m_off"],
        "d_mask_vs_stored_first": _absdiff(fresh["ft_mask"], stored["ft_mask"]),
        "d_mask_vs_path": _absdiff(fresh["ft_mask"], path),
        "d_word_vs_stored_word": _absdiff(fresh["whole_word"], stored["whole_word"]),
        "d_word_vs_path": _absdiff(fresh["whole_word"], path),
        "table_max_abs_diff": table_max_abs_diff(fresh_pairs, stored_pairs),
        "table_identical": int(fresh_pairs == stored_pairs),
        "fresh_prompt_n": prompt_n, "stored_prompt_n": root.get("prompt_n"),
        "prompt_sha256": cell["prompt_sha256"],
    }


def diff_stats(values, gate=GATE):
    """n, NaN count, max, median and the share above `gate` of a list of absolute differences."""
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {"n": 0, "n_nan": len(values), "max": None, "median": None, "share_above_gate": None}
    return {"n": len(vals), "n_nan": len(values) - len(vals), "max": max(vals),
            "median": statistics.median(vals), "share_above_gate": sum(v > gate for v in vals) / len(vals)}


def summarise(rows):
    """{diff: {"all": stats, "unsaturated": stats}} for the four differences."""
    out = {}
    for d in DIFFS:
        out[d] = {"all": diff_stats([r[d] for r in rows]),
                  "unsaturated": diff_stats([r[d] for r in rows if r["unsaturated"]])}
    return out


# ------------------------------------------------------------------- query --

def query_fresh(server, prompts, concurrency=1, progress_every=90):
    """[(pairs, prompt_n)] in the order of `prompts`; `concurrency` requests in flight."""
    def one(prompt):
        pairs = server.next_token_probs(prompt, "", "")       # the extraction's own first-step request
        return pairs, server.last_prompt_n
    if concurrency <= 1:
        out = []
        for i, prompt in enumerate(prompts, 1):
            out.append(one(prompt))
            if i % progress_every == 0 or i == len(prompts):
                print(f"  [fresh-progress] {i}/{len(prompts)}", flush=True)
        return out
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        # prompt_n is read from a shared attribute, so it is not reliable with several in flight
        return [(pairs, None) for pairs, _ in pool.map(one, prompts)]


def pick_subset(rows_or_cases, n, key):
    """n cases spread evenly over the ordering given by `key` (largest first)."""
    order = sorted(range(len(rows_or_cases)), key=lambda i: key(rows_or_cases[i]), reverse=True)
    if n >= len(order):
        return sorted(order)
    step = len(order) / n
    return sorted({order[int(k * step)] for k in range(n)})


# -------------------------------------------------------------------- main --

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="phi4", choices=sorted(MODELS), help="key in value_functions.serving.MODELS")
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--concurrency", type=int, default=None, help="requests in flight (default: the record header's, 1)")
    ap.add_argument("--subset", type=int, default=0, help="query only this many cases, the least saturated first")
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--check-only", action="store_true", help="run the settings and prompt checks and stop (no server)")
    ap.add_argument("--skip-weights-hash", action="store_true")
    args = ap.parse_args()

    model = MODELS[args.model]
    cases = load_stored(model.label)
    if not cases:
        print(f"no stored records for {model.label} under {RAW_DIR}")
        return 2
    headers = {json.dumps(h, sort_keys=True): h for h, _ in cases}.values()
    for h in headers:
        check_header(h)
    served = {Path(h["server"]["model_path"]).name for h in headers}
    if served != {Path(model.gguf).name}:
        raise ValueError(f"records were served from {served}, the registry serves {Path(model.gguf).name}")
    stored_concurrency = {h.get("concurrency") for h in headers}
    concurrency = args.concurrency or (stored_concurrency.pop() if len(stored_concurrency) == 1 else 1)
    prompts = prompts_for(cases)
    print(f"{model.label}: {len(cases)} stored cases in {len(stored_files(model.label))} files; settings and all "
          f"{len(prompts)} prompt hashes match; concurrency {concurrency}")
    if args.check_only:
        return 0

    idx = list(range(len(cases)))
    if args.subset:
        idx = pick_subset(cases, args.subset, key=lambda hc: min(hc[1]["p_move"], 1 - hc[1]["p_move"]))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fresh_{model.label}" + (f"_{args.tag}" if args.tag else "")

    from value_functions.serving import PinnedServer, weights_sha256
    weights = None if args.skip_weights_hash else weights_sha256(REPO / model.gguf)
    t0 = time.time()
    with PinnedServer(model, port=args.port, log_tag=f"first_token_fresh_{model.key}") as srv:
        props = lpv.requests.get(f"{srv.base_url}/props", timeout=120).json()
        name = Path(props.get("model_path") or model.label).stem          # the extraction's payload model name
        t = float(cases[0][0]["temperature"])
        server = lpv.Server(srv.base_url, name, t, int(cases[0][0]["n_probs"]))
        print(f"server up: {props.get('model_path')} build={props.get('build_info')} "
              f"slots={props.get('total_slots')} pin={srv.pin}", flush=True)
        fresh = query_fresh(server, [prompts[i] for i in idx], concurrency)
        server_info = {"model_path": props.get("model_path"), "build_info": props.get("build_info"),
                       "total_slots": props.get("total_slots"), "pin": srv.pin,
                       "server_env": lpv.server_env(srv.base_url), "cmdline": lpv.server_cmdline(srv.base_url)}

    rows = []
    with gzip.open(out_dir / f"{stem}_tables.jsonl.gz", "wt") as fh:
        for i, (pairs, prompt_n) in zip(idx, fresh):
            header, cell = cases[i]
            rows.append(compare_case(header, cell, pairs, prompt_n))
            fh.write(json.dumps({"scenario": header["scenario"], "role": cell["role"], "n_similar": cell["n_similar"],
                                 "n_occupied": cell["n_occupied"], "prompt_sha256": cell["prompt_sha256"],
                                 "prompt_n": prompt_n, "top": [{"token": p, "prob": q} for p, q in pairs]}) + "\n")
    with open(out_dir / f"{stem}_cases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    stored_build = {h["server"]["build_info"] for h in headers}
    stored_weights = {(h.get("weights") or {}).get("sha256_combined") for h in headers} - {None}
    summary = {
        "label": model.label, "created": datetime.now().isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1), "cases": len(rows), "unsaturated": sum(r["unsaturated"] for r in rows),
        "concurrency": concurrency, "subset": args.subset or None,
        "settings": {"temperature": t, "n_probs": server.n_probs, "max_tokens": 1,
                     "sampler_params": SAMPLER_PARAMS, "endpoint": "/v1/chat/completions",
                     "grammar_sha256": hashlib.sha256(MOVE_STAY_GRAMMAR.encode()).hexdigest()},
        "checks": {"prompt_sha256_match": len(prompts), "stored_build": sorted(stored_build),
                   "fresh_build": server_info["build_info"], "same_build": stored_build == {server_info["build_info"]},
                   "weights_sha256": weights, "stored_weights_sha256": sorted(stored_weights),
                   "same_weights": (weights in stored_weights) if (weights and stored_weights) else None,
                   "prompt_n_equal": sum(r["fresh_prompt_n"] == r["stored_prompt_n"] for r in rows)
                   if concurrency <= 1 else None},
        "server": server_info,
        "tables_identical": sum(r["table_identical"] for r in rows),
        "table_max_abs_diff": max(r["table_max_abs_diff"] for r in rows),
        "max_fresh_m_off": max(r["fresh_m_off"] for r in rows),
        "gate": GATE, "sat_eps": SAT_EPS, "differences": summarise(rows),
    }
    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(rows)} cases ({summary['unsaturated']} unsaturated) in {summary['seconds']} s; "
          f"identical tables: {summary['tables_identical']}; largest token-probability difference: "
          f"{summary['table_max_abs_diff']:.3g}")
    for d, s in summary["differences"].items():
        for scope in ("all", "unsaturated"):
            v = s[scope]
            if v["n"]:
                print(f"  {d:24} {scope:12} n={v['n']:4} (NaN {v['n_nan']}) max={v['max']:.3g} "
                      f"median={v['median']:.3g} share>{GATE}={v['share_above_gate']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
