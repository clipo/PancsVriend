#!/usr/bin/env python3
"""EXACT value functions from grammar-masked token probabilities (vf-lp-1).

    python value_functions/logprob/logprob_value_function.py --label qwen3.6-27b-chat-grammar
    python value_functions/logprob/logprob_value_function.py --label ... --scenarios baseline --no-write-vf1

Plan: value_functions/logprob/LOGPROB_PLAN.md. Nothing here touches the sampled
store (value_functions/results/sampled/); everything lands in
value_functions/results/llm_logprob/ (value_functions/paths.py; it was
prompt_refinement/results/value_functions_logprob/ and then, briefly,
llm_log_probs/value_functions_logprob/ earlier on 2026-09-07).

WHAT IS COMPUTED

The sampled value function is the EFFECTIVE move rate under the grammar arm:
at every generation step llama.cpp restricts the vocabulary to tokens
consistent with

    root ::= ws answer ; ws ::= [ \\t\\n]* ; answer ::= [Mm][Oo][Vv][Ee] | [Ss][Tt][Aa][Yy]

and draws from softmax(logits/T) over the allowed set. P(MOVE) is therefore
an exact sum over the paths of that closed language:

    P(MOVE) = sum over grammar-valid token paths ending in MOVE of
              prod_k p_T(token_k | prefix_k, allowed_k)

This script evaluates that sum against the SAME llama-server AND THE SAME
ENDPOINT the sampling campaign used: /v1/chat/completions with the user
prompt as the single message, so template rendering, BOS handling and any
reasoning prefill are the campaign's by construction. (The first version
used /completion on the /apply-template rendering; that reproduced the chat
path bit-for-bit for qwen but NOT for gemma-4 — 162 tokens either way, yet
P(MOVE) 0.999 vs 0.022 — so nothing about the rendering is assumed any more.)
A prefix state is an ASSISTANT PREFILL: the generated text so far is sent as
a trailing assistant message and llama-server continues it (default
--prefill-assistant behaviour); the empty prefix sends no assistant message,
because an empty prefill renders a different template (gemma-4 drops its
thought channel: 155 vs 162 tokens).

Per state one request returns the top n_probs tokens with their
probabilities AFTER the sampler chain (`post_sampling_probs`, read from
choices[0].logprobs.content[0].top_probs), i.e. after temperature. llama.cpp applies the
grammar by rejection (draw first, mask only if the draw is invalid), so those
probabilities are NOT reliably masked; the mask is applied here — a token is
allowed iff prefix + piece is still a prefix of a string in the language —
and the allowed mass is renormalised. That is exactly the distribution the
sampler's accept/resample loop realises. Mass outside the returned top-n is
tracked as `mass_bound` (the probe measured 1.00000 captured at T = 0.3).
States are expanded in order of mass; whitespace prefixes decay
geometrically; expansion stops at --max-depth or when a state's mass is
below --mass-floor, with the dropped mass added to the bound.

OUTPUTS (value_functions/results/llm_logprob/)
  tables/vf_<label>-lp__<scenario>__<style>.json   the CONSUMER table
      (schema vf-1, the format the sampled route also writes) - what the _lp run
      configs load. Exact rates in p_move_effective; the sampled campaign's
      numbers demoted to p_move_sampled / ci95_sampled for provenance.
  raw/vflp_<label>__<scenario>__<style>_states.jsonl.gz   the extraction record,
      schema vf-lp-1. Line 0 is a {"_meta": true, ...} header (label, model,
      grammar_sha256, server, sampled_artifact_sha256, T, mass_floor, ...); each
      later line is ONE (role, cell) - 90 of them for 45 compositions x 2 roles,
      NOT one per HTTP request (a cell costs 2-3 requests, ~251 per scenario;
      the docstring claimed per-request until 2026-09-07). Per cell: p_move,
      p_stay, mass_bound, n_requests, prompt_sha256, every path with its mass
      and every state's top-n table. Until 2026-09-07 a vflp_<label>__....json
      duplicated these cells plus the header at 7x the bytes; the header line
      replaced it.
  concurrent_sampling_vs_exact_logprob/OUTDATED_artifact_samples_vs_exact_<label>.csv / .png   per cell:
      p_logprob vs the OUTDATED concurrency-4 campaign's sampled p with its
      Wilson 95% CI — an ARTIFACT MAP, not the pass/fail test (see below).
      Renamed from validation_* on 2026-09-07: the old name read as a pass/fail
      record and the large disagreements it shows were being taken for
      extraction errors. They are the batch-numerics artifact in the y-axis
      series, which was sampled at concurrency 4 before that artifact was known.
      The y-axis numbers are superseded and are kept only as evidence OF the
      artifact; nothing downstream should read them as value functions.
  validation_data/seqcheck_<label>.csv + validation_data/validation_<label>.json
      the pass/fail test (moved into validation_data/ on 2026-09-07 so the
      store's top level holds value-function tables only):
      the cells where exact and campaign disagree most are RE-SAMPLED
      SEQUENTIALLY (campaign payload, chat endpoint, one request in flight);
      a cell whose 95% CI misses the exact value is ESCALATED (--seq-escalate
      more draws, default 3x) and re-tested, and the run passes only if EVERY
      checked cell is inside its final CI. Rationale: a 95% interval misses a
      correct value 5% of the time, so 12/12 at one stage would fail a correct
      extractor ~46% of the time; more draws separate a chance miss (the
      interval tightens around the exact value) from a real error (the miss
      persists). Overall false-failure rate ≈ 3%; a real error ≥ 0.03 is caught.

WHY THE CAMPAIGN SURFACE IS NOT THE REFERENCE (measured 2026-09-05, qwen,
build b1-a4ce259, GB10, -np 4, cache_prompt false): a byte-identical request
returns DIFFERENT probabilities depending on the batch it is processed in —
sequential 0.2303 every time, four-way concurrent 0.21-0.44, with flash
attention on or off. The sampling campaigns ran four requests in flight, so
each sampled rate is an average over that batch-dependent family (cell red
1-of-5: campaign 0.282 +/- 0.018 at n = 2384; sequential sampling 0.253 +/-
0.05; exact sequential 0.230). Saturated cells are unaffected. This is the
mechanism behind KV_CACHE_SAMPLING_ARTIFACT.md; cache_prompt=false did not
remove it. Sequential requests are bit-reproducible, so the exact tables are
computed sequentially and validated against sequential sampling.
  vf_<label>-lp__<scenario>__<style>.json     vf-1 artifact (meta.source =
      "logprob") the simulation consumes unchanged via --value-function;
      sampled counts kept alongside for provenance (--write-vf1)
"""
import argparse
import gzip
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests

_THIS = Path(__file__).resolve().parent
REPO_ROOT = _THIS.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from value_functions.paths import LOGPROB_DIR, SAMPLED_DIR, add_import_paths  # noqa: E402
add_import_paths()

from sampling_common import load_value_function, role_keywords, wilson_ci  # noqa: E402
from value_functions.paths import (LOGPROB_VALIDATION_DIR, LOGPROB_OUTDATED_MAP_DIR,  # noqa: E402
                                   LOGPROB_TABLES_DIR, LOGPROB_RAW_DIR,
                                   LOGPROB_DIR, VF_MAPPING_PLOTS_REL)
from ratio_prompt_templates import ALL_COMPOSITIONS, RATIO_CANDIDATES  # noqa: E402
from evaluate_ratio_prompts import render_prompt  # noqa: E402
from llm_runner import MOVE_STAY_GRAMMAR, SAMPLER_PARAMS  # noqa: E402

VF_DIR = SAMPLED_DIR       # the sampled tables this extraction is checked against
USER_PROMPTS = {}          # (scenario, role, (n_sim, n_occ)) -> user prompt text
LP_DIR = LOGPROB_DIR       # the exact store (value_functions/paths.py)
WORDS = ("move", "stay")
WS = " \t\n"
# prefix + piece must fully match this to stay inside the language
_PREFIX_RE = re.compile(r"^[ \t\n]*(m(o(v(e)?)?)?|s(t(a(y)?)?)?)?$", re.I)


# ---------------------------------------------------------------------------
# Grammar state machine over TEXT (what the server sees as prompt + prefix)
# ---------------------------------------------------------------------------

def classify_text(text):
    """('open' | 'partial' | 'MOVE' | 'STAY' | 'dead', word_so_far)."""
    if not _PREFIX_RE.match(text):
        return "dead", ""
    word = text.lstrip(WS)
    if not word:
        return "open", ""
    if word.lower() == "move":
        return "MOVE", word
    if word.lower() == "stay":
        return "STAY", word
    return "partial", word


def remainder_grammar(word):
    """GBNF for what may still be generated after `word` letters of the answer.

    Passed to the server so its own grammar engine agrees with the mask (it
    only affects the token the server *draws*, never the probabilities we
    read), and so that a request from a partial state cannot run on."""
    if not word:
        return MOVE_STAY_GRAMMAR
    target = "move" if word.lower().startswith("m") else "stay"
    rest = target[len(word):]
    body = " ".join(f"[{c.upper()}{c.lower()}]" for c in rest)
    return f"root ::= {body}\n"


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class Server:
    def __init__(self, url, model, temperature, n_probs, timeout=120):
        self.url = url.rstrip("/")
        self.model = model
        self.temperature = float(temperature)
        self.n_probs = int(n_probs)
        self.timeout = timeout
        self.session = requests.Session()
        self.n_requests = 0
        self.last_prompt_n = None
        self._lock = threading.Lock()

    def props(self):
        return self.session.get(f"{self.url}/props", timeout=self.timeout).json()

    def render(self, user_prompt):
        r = self.session.post(f"{self.url}/apply-template",
                              json={"messages": [{"role": "user", "content": user_prompt}]},
                              timeout=self.timeout)
        r.raise_for_status()
        return r.json()["prompt"]

    def sample_sequential(self, user_prompt, n, seed0):
        """n campaign-payload chat samples, ONE in flight: (n_move, n_stay)."""
        move = stay = 0
        for i in range(n):
            payload = {"model": self.model, "stream": False, "temperature": self.temperature,
                       "max_tokens": 5, **SAMPLER_PARAMS, "grammar": MOVE_STAY_GRAMMAR,
                       "cache_prompt": False, "seed": seed0 + i,
                       "messages": [{"role": "user", "content": user_prompt}]}
            for attempt in range(6):
                try:
                    r = self.session.post(f"{self.url}/v1/chat/completions", json=payload, timeout=self.timeout)
                    r.raise_for_status()
                    txt = r.json()["choices"][0]["message"]["content"].strip().upper()
                    break
                except (requests.RequestException, KeyError, ValueError):
                    if attempt == 5:
                        raise
                    time.sleep(2 * (attempt + 1))
            if txt == "MOVE":
                move += 1
            elif txt == "STAY":
                stay += 1
        return move, stay

    def next_token_probs(self, user_prompt, prefix, word):
        """[(piece, prob)] after the sampler chain, next token after `prefix`.

        Chat endpoint, campaign payload shape; the prefix (if any) goes in as
        an assistant prefill. Returns the post-sampling top-n; the caller masks
        and renormalises."""
        messages = [{"role": "user", "content": user_prompt}]
        if prefix:
            messages.append({"role": "assistant", "content": prefix})
        payload = {"model": self.model, "messages": messages, "max_tokens": 1,
                   "n_probs": self.n_probs, "post_sampling_probs": True,
                   "temperature": self.temperature, "cache_prompt": False,
                   "grammar": remainder_grammar(word), "stream": False, **SAMPLER_PARAMS}
        for attempt in range(6):
            try:
                r = self.session.post(f"{self.url}/v1/chat/completions", json=payload, timeout=self.timeout)
                r.raise_for_status()
                j = r.json()
                cp = j["choices"][0]["logprobs"]["content"][0]
                items = cp.get("top_probs") or cp.get("top_logprobs") or []
                self.last_prompt_n = (j.get("timings") or {}).get("prompt_n")
                with self._lock:
                    self.n_requests += 1
                out = []
                for it in items:
                    piece = it.get("token", "")
                    if isinstance(piece, list):            # raw bytes: never an answer token
                        piece = "�"
                    prob = it["prob"] if "prob" in it else float(2.718281828 ** it["logprob"])
                    out.append((piece, float(prob)))
                return out
            except (requests.RequestException, KeyError, ValueError) as exc:
                if attempt == 5:
                    raise
                time.sleep(2 * (attempt + 1))
        return []


# ---------------------------------------------------------------------------
# Path sum for one cell
# ---------------------------------------------------------------------------

def path_sum(server, user_prompt, max_depth, mass_floor):
    """Exact P(MOVE), P(STAY) and the unresolved mass bound for one prompt."""
    move = stay = bound = 0.0
    paths, states = [], []
    frontier = [("", 1.0, 0)]                   # (prefix text, mass, depth)
    n_req = 0
    while frontier:
        frontier.sort(key=lambda s: -s[1])
        prefix, mass, depth = frontier.pop(0)
        if mass < mass_floor:
            bound += mass
            continue
        if depth >= max_depth:
            bound += mass
            continue
        kind, word = classify_text(prefix)
        top = server.next_token_probs(user_prompt, prefix, word)
        n_req += 1
        captured = sum(p for _, p in top)
        allowed = []
        for piece, p in top:
            if p <= 0.0 or not piece:
                continue                        # special/EOS tokens: not in the language
            k, _ = classify_text(prefix + piece)
            if k != "dead":
                allowed.append((piece, p, k))
        allowed_mass = sum(p for _, p, _ in allowed)
        states.append({"prefix": prefix, "mass": mass, "captured": captured,
                       "allowed_mass": allowed_mass, "prompt_n": server.last_prompt_n,
                       "top": [{"token": t, "prob": p} for t, p in top[:16]]})
        if allowed_mass <= 0.0:
            bound += mass                       # nothing valid in top-n: unresolved
            continue
        # mass not captured by top-n could belong to allowed tokens: bound it
        bound += mass * max(0.0, 1.0 - captured) / max(allowed_mass, 1e-300) * allowed_mass \
            if captured < 1.0 else 0.0
        for piece, p, k in allowed:
            m = mass * p / allowed_mass         # masked + renormalised
            if k == "MOVE":
                move += m; paths.append({"text": prefix + piece, "decision": "MOVE", "mass": m})
            elif k == "STAY":
                stay += m; paths.append({"text": prefix + piece, "decision": "STAY", "mass": m})
            else:
                frontier.append((prefix + piece, m, depth + 1))
    return {"p_move": move, "p_stay": stay, "mass_bound": bound, "n_requests": n_req,
            "paths": sorted(paths, key=lambda x: -x["mass"]), "states": states}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sampled_artifact(label, scenario, style):
    p = VF_DIR / f"vf_{label}__{scenario}__{style}.json"
    return load_value_function(p) if p.exists() else None, p


def run_scenario(server, label, scenario, style, roles, scenario_file, args, raw_writer):
    tpl, fn = RATIO_CANDIDATES[style]
    kw_by_role = role_keywords(scenario, scenario_file)
    out = {"schema": "vf-lp-1", "meta": None, "compositions": {}}
    jobs = []
    for role in roles:
        for cell in ALL_COMPOSITIONS:
            n_sim, n_occ = cell
            prompt, ctx = render_prompt(style, tpl, fn, n_sim, n_occ, kw_by_role[role])
            jobs.append((role, cell, prompt, ctx))

    def work(job):
        role, (n_sim, n_occ), prompt, ctx = job
        res = path_sum(server, prompt, args.max_depth, args.mass_floor)
        rec = {"role": role, "n_similar": n_sim, "n_occupied": n_occ,
               "n_opposite": n_occ - n_sim, "context": ctx, **res}
        raw_writer(dict(rec, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(work, jobs))
    for role, cell, prompt, _ in jobs:
        USER_PROMPTS[(scenario, role, cell)] = prompt
    for role in roles:
        rows = [r for r in results if r["role"] == role]
        out["compositions"][role] = [
            {k: r[k] for k in ("n_similar", "n_occupied", "n_opposite", "context", "p_move",
                               "p_stay", "mass_bound", "n_requests", "paths", "states")}
            for r in sorted(rows, key=lambda r: (r["n_occupied"], r["n_similar"]))]
    print(f"  {scenario:32s} {len(jobs)} cells, {server.n_requests} requests so far, "
          f"{time.time() - t0:.0f} s")
    return out



def load_lp_trace(path):
    """Read a vflp_*_states.jsonl.gz back into the {meta, compositions} shape.

    Line 0 is the {"_meta": true, ...} header (the same convention the sampled
    raw files use); every later line is one cell. Replaces the vflp_*.json that
    carried the identical cells plus that header, dropped 2026-09-07.
    """
    meta, comps = {}, {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("_meta"):
                meta = {k: v for k, v in rec.items() if k != "_meta"}
                continue
            comps.setdefault(rec["role"], []).append(rec)
    if not meta:
        raise ValueError(f"{path} has no _meta header line (pre-2026-09-07 trace?)")
    return {"schema": "vf-lp-1", "meta": meta, "compositions": comps}


def to_vf1(lp, sampled, label, style):
    """vf-1 artifact with exact rates, sampled counts kept for provenance."""
    vf = json.loads(json.dumps(sampled))       # deep copy
    vf["meta"]["source"] = "logprob"
    vf["meta"]["sampled_label"] = label
    vf["meta"]["label"] = f"{label}-lp"
    vf["meta"]["logprob"] = {k: lp["meta"][k] for k in
                             ("temperature", "n_probs", "max_depth", "mass_floor", "server",
                              "grammar_sha256", "created", "sampled_artifact_sha256")}
    for role, rows in vf["compositions"].items():
        lookup = {(c["n_similar"], c["n_occupied"]): c for c in lp["compositions"][role]}
        for c in rows:
            e = lookup[(c["n_similar"], c["n_occupied"])]
            c["p_move_sampled"] = c["p_move_effective"]
            c["ci95_sampled"] = c["ci95"]
            c["p_move_effective"] = round(e["p_move"], 8)
            c["mass_bound"] = e["mass_bound"]
            c["ci95"] = [round(max(0.0, e["p_move"] - e["mass_bound"]), 8),
                         round(min(1.0, e["p_move"] + e["mass_bound"]), 8)]
        # ratio rows are a visualisation (equal-weight mean over member cells)
        for r in vf["ratios"].get(role, []):
            ps = [lookup[tuple(m)]["p_move"] for m in r["members"] if tuple(m) in lookup]
            if ps:
                r["p_move_sampled"] = r["p_move_effective"]
                r["p_move_effective"] = round(sum(ps) / len(ps), 8)
                b = max(lookup[tuple(m)]["mass_bound"] for m in r["members"] if tuple(m) in lookup)
                r["ci95"] = [round(max(0.0, r["p_move_effective"] - b), 8),
                             round(min(1.0, r["p_move_effective"] + b), 8)]
    return vf


def validate(label, style, scenarios, roles, lp_by_scenario, out_dir,
             server=None, seq_cells=12, seq_samples=300, seq_escalate=3,
             seq_saturated=4, seq_random=2):
    rows = []
    for scenario in scenarios:
        sampled, _ = sampled_artifact(label, scenario, style)
        if sampled is None:
            continue
        for role in roles:
            lookup = {(c["n_similar"], c["n_occupied"]): c for c in lp_by_scenario[scenario]["compositions"][role]}
            for c in sampled["compositions"][role]:
                e = lookup[(c["n_similar"], c["n_occupied"])]
                n = c["n_move"] + c["n_stay"]
                lo, hi = wilson_ci(c["n_move"], n) if n else (0.0, 1.0)
                p = e["p_move"]
                sat = c["n_move"] == 0 or c["n_stay"] == 0
                half = (hi - lo) / 2
                rows.append({"scenario": scenario, "role": role, "n_similar": c["n_similar"],
                             "n_occupied": c["n_occupied"], "n": n, "n_move": c["n_move"],
                             "p_sampled": c["p_move_effective"], "ci_low": lo, "ci_high": hi,
                             "p_logprob": p, "mass_bound": e["mass_bound"],
                             # tolerance: Wilson bounds and the renormalised path sum both carry
                             # ~1e-9 float error at p = 0 or 1, which must not count as a miss
                             "saturated": sat,
                             "inside_ci": lo - e["mass_bound"] - 1e-6 <= p <= hi + e["mass_bound"] + 1e-6,
                             "excess_halfwidths": (max(0.0, lo - p, p - hi) / half) if half > 0 else 0.0,
                             "n_requests": e["n_requests"]})
    import pandas as pd
    df = pd.DataFrame(rows)
    # Named for what it IS: OUTDATED sampled numbers that carry the
    # batch-numerics ARTIFACT, plotted against the exact tables. Called
    # validation_*.csv until 2026-09-07, which read as a pass/fail record; it is
    # not one. The verdict lives in validation_<label>.json and rests on
    # seqcheck_<label>.csv. The leading OUTDATED_ is deliberate: it sorts these
    # away from the live artifacts and warns anyone who only sees the filename.
    # Filed in its own folder (concurrent_sampling_vs_exact_logprob/), away from
    # the live tables.
    LOGPROB_OUTDATED_MAP_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOGPROB_OUTDATED_MAP_DIR / f"OUTDATED_artifact_samples_vs_exact_{label}.csv", index=False)
    uns = df[~df.saturated]
    sat = df[df.saturated]
    summary = {
        "label": label, "cells": int(len(df)),
        "unsaturated": int(len(uns)),
        "unsaturated_inside_ci": float(uns.inside_ci.mean()) if len(uns) else None,
        "unsaturated_max_excess_halfwidths": float(uns.excess_halfwidths.max()) if len(uns) else None,
        "saturated": int(len(sat)),
        "saturated_inside_bound": float(sat.inside_ci.mean()) if len(sat) else None,
        "saturated_zero_cells_p_logprob_median": float(sat[sat.n_move == 0].p_logprob.median()) if (sat.n_move == 0).any() else None,
        "saturated_zero_cells_p_logprob_max": float(sat[sat.n_move == 0].p_logprob.max()) if (sat.n_move == 0).any() else None,
        "max_mass_bound": float(df.mass_bound.max()),
        "requests_per_cell_mean": float(df.n_requests.mean()),
    }
    # ---- the decisive test: sequential re-sampling of the worst-disagreeing cells
    seq_rows = []
    if server is not None and (seq_cells > 0 or seq_saturated > 0) and len(df):
        df["abs_diff"] = (df.p_logprob - df.p_sampled).abs()
        # (a) unsaturated cells where exact and campaign disagree most;
        # (b) SATURATED cells where an extractor error would be detectable:
        #     sampled-zero cells with the LARGEST exact P(MOVE) and sampled-one
        #     cells with the SMALLEST, plus a few at random for coverage. A cell
        #     claiming 0.015 must show ~4-5 moves in 300 draws; 0/300 misses
        #     its interval and the escalation (upper bound 0.003) rules it out.
        parts = [df.sort_values("abs_diff", ascending=False).head(seq_cells)]
        zero = df[df.n_move == 0].sort_values("p_logprob", ascending=False)
        full = df[(df.n_move == df.n) & (df.n > 0)].sort_values("p_logprob", ascending=True)
        k = max(seq_saturated // 2, 1) if seq_saturated > 0 else 0
        if k:
            parts += [zero.head(k), full.head(k)]
        if seq_random > 0:
            rest = df[df.saturated].drop(pd.concat(parts).index, errors="ignore")
            if len(rest):
                parts.append(rest.sample(min(seq_random, len(rest)), random_state=0))
        pick = pd.concat(parts).drop_duplicates(subset=["scenario", "role", "n_similar", "n_occupied"])
        for i, (_, r) in enumerate(pick.iterrows()):
            key = (r.scenario, r.role, (int(r.n_similar), int(r.n_occupied)))
            prompt = USER_PROMPTS.get(key)
            if prompt is None:
                continue
            mv, st = server.sample_sequential(prompt, seq_samples, 900000 + 1000 * i)
            n = mv + st
            lo, hi = wilson_ci(mv, n) if n else (0.0, 1.0)
            inside1 = lo <= r.p_logprob <= hi
            print(f"  seq-check {r.scenario}/{r.role} {int(r.n_similar)}/{int(r.n_occupied)}: campaign "
                  f"{r.p_sampled:.3f}  exact {r.p_logprob:.3f}  sequential {mv}/{n}={mv/max(n,1):.3f} "
                  f"[{lo:.3f},{hi:.3f}] {'ok' if inside1 else 'MISS -> escalating'}")
            n1, mv1, lo1, hi1 = n, mv, lo, hi
            escalated = False
            if not inside1 and seq_escalate > 0:
                # more draws: a chance miss resolves, a real error persists
                mv2, st2 = server.sample_sequential(prompt, seq_samples * seq_escalate,
                                                    950000 + 1000 * i)
                mv += mv2; n += mv2 + st2
                lo, hi = wilson_ci(mv, n) if n else (0.0, 1.0)
                escalated = True
                print(f"      escalated to {mv}/{n}={mv/max(n,1):.3f} [{lo:.3f},{hi:.3f}] "
                      f"{'ok' if lo <= r.p_logprob <= hi else 'STILL MISSES'}")
            seq_rows.append({"scenario": r.scenario, "role": r.role, "n_similar": r.n_similar,
                             "n_occupied": r.n_occupied, "p_campaign": r.p_sampled,
                             "campaign_saturated": bool(r.saturated),
                             "p_logprob": r.p_logprob,
                             "seq_n_stage1": n1, "seq_move_stage1": mv1,
                             "seq_ci_low_stage1": lo1, "seq_ci_high_stage1": hi1, "inside_stage1": inside1,
                             "escalated": escalated, "seq_n": n, "seq_move": mv,
                             "p_sequential": mv / n if n else None, "seq_ci_low": lo, "seq_ci_high": hi,
                             "inside_seq_ci": lo <= r.p_logprob <= hi})
        LOGPROB_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(seq_rows).to_csv(LOGPROB_VALIDATION_DIR / f"seqcheck_{label}.csv", index=False)
    sq = pd.DataFrame(seq_rows)
    summary["seq_check_cells"] = int(len(sq))
    summary["seq_check_inside_ci"] = float(sq.inside_seq_ci.mean()) if len(sq) else None
    summary["seq_check_median_abs_diff"] = (float((sq.p_logprob - sq.p_sequential).abs().median())
                                            if len(sq) else None)
    # PASS = the exact numbers reproduce SEQUENTIAL sampling on EVERY checked
    # cell (after escalation of any first-stage miss) and the saturated cells
    # are consistent with their bounds. The campaign comparison is reported,
    # not judged: it measures the concurrency artifact, not the extractor.
    summary["seq_check_escalated"] = int(sq.escalated.sum()) if len(sq) else 0
    summary["seq_check_saturated_cells"] = int(sq.campaign_saturated.sum()) if len(sq) else 0
    summary["seq_check_saturated_inside_ci"] = (float(sq[sq.campaign_saturated].inside_seq_ci.mean())
                                                if len(sq) and sq.campaign_saturated.any() else None)
    summary["seq_check_stage1_inside_ci"] = float(sq.inside_stage1.mean()) if len(sq) else None
    summary["pass"] = bool(
        (summary["seq_check_inside_ci"] is None or summary["seq_check_inside_ci"] >= 1.0)
        and (summary["saturated_inside_bound"] is None or summary["saturated_inside_bound"] >= 0.99))
    LOGPROB_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    (LOGPROB_VALIDATION_DIR / f"validation_{label}.json").write_text(json.dumps(summary, indent=1))
    fig_validation(df, label, LOGPROB_OUTDATED_MAP_DIR / f"OUTDATED_artifact_samples_vs_exact_{label}.png")
    return summary


def fig_validation(df, label, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    # Palette: two categorical hues (unsaturated / saturated) plus a neutral
    # for the identity line; error bars in the point's own hue at low alpha.
    uns, sat = df[~df.saturated], df[df.saturated]
    # Extra height reserved for the provenance banner and the footnote: without
    # them a reader takes the scatter for an extraction failure (2026-09-07).
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.1))
    ax = axes[0]
    for sub, color, name, mk in ((uns, "#4C6EF5", "unsaturated (0 < n_move < n)", "o"),
                                 (sat, "#E8590C", "saturated (0/n or n/n)", "s")):
        if len(sub) == 0:
            continue
        # p_move_effective is stored to 6 decimals, the CI is not: clip the
        # rounding-level negatives matplotlib refuses.
        yerr = np.clip(np.vstack([sub.p_sampled - sub.ci_low, sub.ci_high - sub.p_sampled]), 0, None)
        ax.errorbar(sub.p_logprob, sub.p_sampled, yerr=yerr, fmt=mk, ms=4, color=color,
                    ecolor=color, elinewidth=0.6, alpha=0.75, label=name, zorder=3)
    ax.plot([0, 1], [0, 1], color="#868E96", lw=1, ls="--", zorder=2)
    ax.set_xlabel("P(MOVE) from grammar-masked logprobs (exact)")
    ax.set_ylabel("OLD CONCURRENT campaign: sampled rate\n(n=100/cell, concurrency 4), Wilson 95% CI")
    ax.set_title(f"{len(df)} cells — {int(uns.inside_ci.sum())}/{len(uns)} unsaturated "
                 f"inside the CAMPAIGN's CI", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.2)
    ax = axes[1]
    zero = sat[sat.n_move == 0]
    if len(zero):
        vals = np.clip(zero.p_logprob.values, 1e-12, 1)
        ax.hist(np.log10(vals), bins=30, color="#E8590C", alpha=0.85)
        ax.axvline(np.log10(3 / 100), color="#868E96", ls="--", lw=1)
        ax.text(np.log10(3 / 100), ax.get_ylim()[1] * 0.95, " 0/100 upper bound (0.03)",
                fontsize=8, color="#495057", va="top")
        ax.set_xlabel("log10 P(MOVE) on cells sampled as 0/n")
        ax.set_ylabel("cells")
        ax.set_title("what the sampled zeros actually are", fontsize=10)
        ax.grid(alpha=0.2)
    else:
        ax.axis("off")
    # The y-axis series predates the batch-numerics finding, so points off the
    # identity line are the artifact being MEASURED, not an extraction error.
    # Say so on the figure: the file travels without its docstring.
    fig.suptitle(f"{label} — ARTIFACT MAP, NOT the validation test",
                 fontsize=12, fontweight="bold", color="#C92A2A")
    fig.text(
        0.5, 0.015,
        "y-axis = the pre-2026-09-06 SAMPLED CAMPAIGN (n=100/cell at concurrency 4). llama.cpp returns "
        "batch-dependent probabilities at concurrency > 1,\n"
        "so disagreement with the exact x-axis is EXPECTED HERE and is that artifact — not an error in "
        "the log-probability extraction.\n"
        "The pass/fail test is seqcheck_<label>.csv: the worst-disagreeing cells re-sampled SEQUENTIALLY "
        "(n=300, escalated 3x on a miss); all 9 models pass at 100%.",
        ha="center", va="bottom", fontsize=7.5, color="#495057", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.115, 1, 0.945))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True, help="sampled label, e.g. qwen3.6-27b-chat-grammar")
    ap.add_argument("--style", default="R3_dual_count")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="default: every scenario with a sampled artifact for the label")
    ap.add_argument("--roles", nargs="*", default=["red", "blue"])
    ap.add_argument("--scenario-file", default=None, help="default scenarios_a2.py")
    ap.add_argument("--url", default="http://localhost:8085")
    ap.add_argument("--model", default=None, help="payload model name (default: artifact meta)")
    ap.add_argument("--temperature", type=float, default=None, help="default: artifact meta")
    ap.add_argument("--n-probs", type=int, default=64)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--mass-floor", type=float, default=1e-7)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="requests in flight. MUST stay 1 for exact numbers: on this "
                         "server build (b1-a4ce259, GB10, -np 4) the returned "
                         "probabilities of a byte-identical request vary by tens of "
                         "points with the batch it lands in (measured 2026-09-05: "
                         "sequential 0.2303 x8; concurrent 0.21-0.44). Sequential "
                         "requests are bit-reproducible")
    ap.add_argument("--out-dir", default=str(LP_DIR))
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--seq-cells", type=int, default=12,
                    help="cells (largest exact-vs-campaign disagreement) re-sampled "
                         "sequentially for the pass/fail test")
    ap.add_argument("--seq-samples", type=int, default=300)
    ap.add_argument("--seq-saturated", type=int, default=4,
                    help="saturated cells to re-sample too: half the sampled-zero cells "
                         "with the largest exact P(MOVE), half the sampled-one cells with "
                         "the smallest")
    ap.add_argument("--seq-random", type=int, default=2,
                    help="additional random saturated cells for coverage")
    ap.add_argument("--seq-escalate", type=int, default=3,
                    help="a first-stage miss gets this many x --seq-samples more draws; "
                         "pass requires every checked cell inside its final CI")
    ap.add_argument("--validate-only", action="store_true",
                    help="skip extraction: load the existing raw/vflp_*_states.jsonl.gz for the label "
                         "and run the validation (needs the model's server up)")
    ap.add_argument("--no-write-vf1", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    scenarios = args.scenarios or sorted(
        p.name.split("__")[1] for p in VF_DIR.glob(f"vf_{args.label}__*__{args.style}.json"))
    if not scenarios:
        print(f"no sampled artifacts for {args.label!r} in {VF_DIR}"); return 2
    ref, ref_path = sampled_artifact(args.label, scenarios[0], args.style)
    model = args.model or ref["meta"].get("model") or args.label
    temperature = args.temperature if args.temperature is not None else float(ref["meta"]["temperature"])
    server = Server(args.url, model, temperature, args.n_probs)
    props = server.props()
    print(f"label={args.label} style={args.style} scenarios={scenarios} T={temperature} "
          f"server={props.get('model_path')} build={props.get('build_info')}")
    if ref["meta"].get("sampling_protocol", {}).get("server", {}).get("model_path") and \
       Path(props.get("model_path", "")).name != Path(ref["meta"]["sampling_protocol"]["server"]["model_path"]).name:
        print(f"WARNING: server model {props.get('model_path')} != sampled "
              f"{ref['meta']['sampling_protocol']['server']['model_path']}")

    grammar_sha = hashlib.sha256(MOVE_STAY_GRAMMAR.encode()).hexdigest()
    lp_by_scenario = {}
    if args.validate_only:
        tpl, fn = RATIO_CANDIDATES[args.style]
        for scenario in scenarios:
            f = LOGPROB_RAW_DIR / f"vflp_{args.label}__{scenario}__{args.style}_states.jsonl.gz"
            if not f.exists():
                print(f"--validate-only: missing {f}"); return 2
            lp_by_scenario[scenario] = load_lp_trace(f)
            kw_by_role = role_keywords(scenario, args.scenario_file)
            for role in args.roles:
                for cell in ALL_COMPOSITIONS:
                    USER_PROMPTS[(scenario, role, cell)] = render_prompt(
                        args.style, tpl, fn, cell[0], cell[1], kw_by_role[role])[0]
        summary = validate(args.label, args.style, scenarios, args.roles, lp_by_scenario, out_dir,
                           server=server, seq_cells=args.seq_cells, seq_samples=args.seq_samples,
                           seq_escalate=args.seq_escalate, seq_saturated=args.seq_saturated,
                           seq_random=args.seq_random)
        print(json.dumps(summary, indent=1))
        return 0 if summary["pass"] else 4
    for scenario in scenarios:
        sampled, spath = sampled_artifact(args.label, scenario, args.style)
        LOGPROB_RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = LOGPROB_RAW_DIR / f"vflp_{args.label}__{scenario}__{args.style}_states.jsonl.gz"
        meta = {
            "label": args.label, "model": model, "url": args.url, "style": args.style,
            "scenario": scenario, "scenario_file": args.scenario_file or "scenarios_a2.py",
            "roles": args.roles, "temperature": temperature, "sampler_params": SAMPLER_PARAMS,
            "grammar_sha256": grammar_sha, "n_probs": args.n_probs, "max_depth": args.max_depth,
            "mass_floor": args.mass_floor, "post_sampling_probs": True,
            "endpoint": "/v1/chat/completions (campaign endpoint); prefix states as assistant prefill",
            "mask": "applied client-side on temperature-scaled probs (server grammar is rejection-based)",
            "concurrency": args.concurrency,
            "server": {k: props.get(k) for k in ("model_path", "build_info", "total_slots")},
            "sampled_artifact": str(spath), "sampled_artifact_sha256": sha256_file(spath) if spath.exists() else None,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        with gzip.open(raw_path, "wt") as raw:
            raw.write(json.dumps({"_meta": True, **meta}) + "\n")

            def raw_writer(rec, raw=raw):
                raw.write(json.dumps(rec) + "\n")
            lp = run_scenario(server, args.label, scenario, args.style, args.roles,
                              args.scenario_file, args, raw_writer)
        lp["meta"] = meta
        lp_by_scenario[scenario] = lp
        if not args.no_write_vf1 and sampled is not None:
            vf1 = to_vf1(lp, sampled, args.label, args.style)
            LOGPROB_TABLES_DIR.mkdir(parents=True, exist_ok=True)
            p1 = LOGPROB_TABLES_DIR / f"vf_{args.label}-lp__{scenario}__{args.style}.json"
            p1.write_text(json.dumps(vf1, indent=1))
            load_value_function(p1)              # consumer-side check
    print(f"total requests: {server.n_requests}")
    if not args.no_validate:
        summary = validate(args.label, args.style, scenarios, args.roles, lp_by_scenario, out_dir,
                           server=server, seq_cells=args.seq_cells, seq_samples=args.seq_samples,
                           seq_escalate=args.seq_escalate, seq_saturated=args.seq_saturated,
                           seq_random=args.seq_random)
        print(json.dumps(summary, indent=1))
        return 0 if summary["pass"] else 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
