#!/usr/bin/env python3
"""EXACT value functions from grammar-masked token probabilities (vf-lp-1).

    python value_functions/logprob/logprob_value_function.py --label qwen3.6-27b-chat-grammar
    python value_functions/logprob/logprob_value_function.py --label ... --scenarios baseline --no-write-vf1
    python value_functions/logprob/logprob_value_function.py --label gemma-4-31b-q8-chat-grammar   # new model

NOTHING IS READ FROM THE SAMPLED TABLES (user decision 2026-09-07: they are
not canonical — they carry the batch-numerics artifact — and must not be used
to derive anything). Every label, old or new, is extracted the same way: the
scenarios come from the scenario file (scenarios_a2.py: the six the campaigns
ran; --scenarios narrows), the temperature from --temperature (default
DEFAULT_TEMPERATURE = 0.3, every campaign's value), the payload model name
from the served gguf. The vf-1 table is assembled from a blank count store
(build_value_function._assemble_artifact) so it carries no sampled counts,
and the sequential check picks its cells from the exact surface itself.
Before this date the extractor took its skeleton, scenarios and check cells
from the label's sampled table; the nine tables extracted on 2026-09-05/06
therefore still carry p_move_sampled / ci95_sampled slots (informational
only) and an OUTDATED_* artifact map each. Those maps are no longer written.

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
      (schema vf-1, the format the sampled route also wrote) - what the _lp run
      configs load. Exact rates in p_move_effective, +/- mass_bound as ci95;
      every count slot is 0 (meta.source = "logprob" tells the plots so).
  raw/vflp_<label>__<scenario>__<style>_states.jsonl.gz   the extraction record,
      schema vf-lp-1. Line 0 is a {"_meta": true, ...} header (label, model,
      grammar_sha256, server, T, mass_floor, ...); each
      later line is ONE (role, cell) - 90 of them for 45 compositions x 2 roles,
      NOT one per HTTP request (a cell costs 2-3 requests, ~251 per scenario;
      the docstring claimed per-request until 2026-09-07). Per cell: p_move,
      p_stay, mass_bound, n_requests, prompt_sha256, every path with its mass
      and every state's top-n table. Until 2026-09-07 a vflp_<label>__....json
      duplicated these cells plus the header at 7x the bytes; the header line
      replaced it.
  validation_data/seqcheck_<label>.csv + validation_data/validation_<label>.json
  seqcheck_plots/seqcheck_<label>.png   drawn automatically when the check is written
      the pass/fail test (in validation_data/ since 2026-09-07 so the
      store's top level holds value-function tables only): cells chosen from
      the exact surface — --seq-cells unsaturated cells at evenly spaced ranks
      of exact P(MOVE), the saturated cells nearest the detectable boundary
      (largest exact P(MOVE) among the ~0 cells, smallest among the ~1 cells)
      and a couple at random — are RE-SAMPLED SEQUENTIALLY (campaign payload,
      chat endpoint, one request in flight);
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
import os
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
from value_functions.paths import LOGPROB_DIR, add_import_paths  # noqa: E402
from value_functions.paths import REPO_ROOT as REPO_ROOT_FOR_WEIGHTS  # noqa: E402
add_import_paths()

from sampling_common import load_value_function, role_keywords, wilson_ci  # noqa: E402
from value_functions.paths import (LOGPROB_VALIDATION_DIR,  # noqa: E402
                                   LOGPROB_TABLES_DIR, LOGPROB_RAW_DIR,
                                   LOGPROB_DIR, VF_MAPPING_PLOTS_REL)
from ratio_prompt_templates import ALL_COMPOSITIONS, RATIO_CANDIDATES  # noqa: E402
from evaluate_ratio_prompts import render_prompt  # noqa: E402
from llm_runner import MOVE_STAY_GRAMMAR, SAMPLER_PARAMS  # noqa: E402

DEFAULT_TEMPERATURE = 0.3  # every campaign and every exact table so far
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

def scenario_names(scenario_file=None):
    """Every scenario in the scenario registry (default scenarios_a2.py), in
    file order — the campaigns' six."""
    import importlib.util
    path = Path(scenario_file) if scenario_file else REPO_ROOT / "scenarios_a2.py"
    spec = importlib.util.spec_from_file_location("_vf_scenarios", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.CONTEXT_SCENARIOS)


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


def to_vf1(lp, label, style, scenario, scenario_file, roles, props):
    """The consumer table: a vf-1 skeleton (blank count store, the same
    builder the sampled route used) carrying the EXACT rates. Nothing sampled
    is read or attached — the sampled tables are not canonical (user decision
    2026-09-07); the count slots stay at zero and the plots draw no N panels
    (plot_value_functions.is_exact keys on meta.source)."""
    from build_value_function import _assemble_artifact, _blank_counts
    m = lp["meta"]
    kw_by_role = role_keywords(scenario, scenario_file)
    meta = {
        "label": f"{label}-lp", "model": m["model"],
        "url": m["url"].rstrip("/") + "/v1/chat/completions",
        "arm": "chat+grammar", "style": style, "scenario": scenario,
        "scenario_file": scenario_file or "scenarios_a2.py",
        "role_to_type": {"red": "type_a", "blue": "type_b"},
        "role_labels": {r: kw_by_role[r]["agent_type"] for r in roles},
        "temperature": m["temperature"], "sampler_params": SAMPLER_PARAMS,
        "grammar": True, "grammar_sha256": m["grammar_sha256"],
        "samples_mode": None, "samples_per": 0, "seed": None,
        "sampling_protocol": {"cache_prompt": False, "seeded": None,
                              "server": {k: props.get(k) for k in ("model_path", "build_info", "total_slots")}},
        "created": m["created"], "sources": [], "raw_replies": None,
        "source": "logprob",
        "logprob": {k: m[k] for k in ("temperature", "n_probs", "max_depth", "mass_floor", "server",
                                      "grammar_sha256", "created")},
    }
    vf = _assemble_artifact(meta, _blank_counts(roles), roles)
    for role, rows in vf["compositions"].items():
        lookup = {(c["n_similar"], c["n_occupied"]): c for c in lp["compositions"][role]}
        for c in rows:
            e = lookup[(c["n_similar"], c["n_occupied"])]
            c["p_move_effective"] = round(e["p_move"], 8)
            c["mass_bound"] = e["mass_bound"]
            c["ci95"] = [round(max(0.0, e["p_move"] - e["mass_bound"]), 8),
                         round(min(1.0, e["p_move"] + e["mass_bound"]), 8)]
        # ratio rows are a visualisation (equal-weight mean over member cells)
        for r in vf["ratios"].get(role, []):
            ps = [lookup[tuple(m_)]["p_move"] for m_ in r["members"] if tuple(m_) in lookup]
            if ps:
                r["p_move_effective"] = round(sum(ps) / len(ps), 8)
                b = max(lookup[tuple(m_)]["mass_bound"] for m_ in r["members"] if tuple(m_) in lookup)
                r["ci95"] = [round(max(0.0, r["p_move_effective"] - b), 8),
                             round(min(1.0, r["p_move_effective"] + b), 8)]
    return vf


# exact P(MOVE) at or beyond this is treated as saturated for the sequential
# check (a 300-draw check cannot resolve anything finer anyway)
SAT_EPS = 0.005


# Wilson limits are analytically 0 and 1 at k=0 and k=n but land ~1e-16 inside
# in floating point, so an exact value of exactly 0.0 or 1.0 read as OUTSIDE
# a 1200/1200 interval (llama, 2026-09-11: a phantom MISS failed a run whose
# 17 other cells were inside with median |diff| 0.004). Same fix as
# comparison/sanity_vs_exact.py CI_EPS.
CI_EPS = 1e-9


def _inside(p, lo, hi):
    return (lo - CI_EPS) <= p <= (hi + CI_EPS)


def _seq_summary(summary, sq):
    """The seq-check fields of validation_<label>.json from the seqcheck rows."""
    summary["seq_check_cells"] = int(len(sq))
    summary["seq_check_inside_ci"] = float(sq.inside_seq_ci.mean()) if len(sq) else None
    summary["seq_check_median_abs_diff"] = (float((sq.p_logprob - sq.p_sequential).abs().median())
                                            if len(sq) else None)
    summary["seq_check_escalated"] = int(sq.escalated.sum()) if len(sq) else 0
    summary["seq_check_saturated_cells"] = int(sq.exact_saturated.sum()) if len(sq) else 0
    summary["seq_check_saturated_inside_ci"] = (float(sq[sq.exact_saturated].inside_seq_ci.mean())
                                                if len(sq) and sq.exact_saturated.any() else None)
    summary["seq_check_stage1_inside_ci"] = float(sq.inside_stage1.mean()) if len(sq) else None
    # PASS = the exact numbers reproduce SEQUENTIAL sampling on EVERY checked
    # cell (after escalation of any first-stage miss).
    summary["pass"] = bool(summary["seq_check_inside_ci"] is None or summary["seq_check_inside_ci"] >= 1.0)
    return summary


def reverdict(label, plot=True):
    """Recompute validation_<label>.json from the seqcheck_<label>.csv already on
    disk — no server, no new draws. For when the containment rule changes (the
    CI_EPS fix) and the sampled counts are still valid evidence."""
    import pandas as pd
    csv_p = LOGPROB_VALIDATION_DIR / f"seqcheck_{label}.csv"
    js_p = LOGPROB_VALIDATION_DIR / f"validation_{label}.json"
    sq = pd.read_csv(csv_p)
    sq["inside_stage1"] = [_inside(p, lo, hi) for p, lo, hi in
                           zip(sq.p_logprob, sq.seq_ci_low_stage1, sq.seq_ci_high_stage1)]
    sq["inside_seq_ci"] = [_inside(p, lo, hi) for p, lo, hi in
                           zip(sq.p_logprob, sq.seq_ci_low, sq.seq_ci_high)]
    sq.to_csv(csv_p, index=False)
    summary = json.loads(js_p.read_text())
    summary = _seq_summary(summary, sq)
    summary["reverdict"] = f"recomputed {datetime.now().isoformat(timespec='seconds')} from seqcheck counts with CI_EPS={CI_EPS}"
    js_p.write_text(json.dumps(summary, indent=1))
    if plot and len(sq):
        try:
            from plot_seqcheck import plot_one
            plot_one(label)
        except Exception as e:                                  # the verdict stands without the figure
            print(f"[warn] seqcheck plot not redrawn: {e}")
    return summary


def validate(label, style, scenarios, roles, lp_by_scenario, out_dir,
             server=None, seq_cells=12, seq_samples=300, seq_escalate=3,
             seq_saturated=4, seq_random=2, plot=True):
    """The pass/fail test: sequential re-sampling of cells chosen from the
    EXACT surface itself (no sampled campaign is consulted — 2026-09-07):
      * seq_cells unsaturated cells at evenly spaced ranks of exact P(MOVE),
        so the check spans the whole transition region;
      * the saturated cells where an extraction error would be detectable:
        the largest exact P(MOVE) among the ~0 cells and the smallest among
        the ~1 cells (a cell claiming 0.015 must show ~4-5 moves in 300 draws);
      * seq_random more saturated cells at random.
    A first-stage CI miss is escalated (seq_escalate x more draws) and the
    run passes only if EVERY checked cell ends inside its final CI."""
    import pandas as pd
    rows = []
    for scenario in scenarios:
        for role in roles:
            for e in lp_by_scenario[scenario]["compositions"][role]:
                p = e["p_move"]
                rows.append({"scenario": scenario, "role": role, "n_similar": e["n_similar"],
                             "n_occupied": e["n_occupied"], "p_logprob": p,
                             "mass_bound": e["mass_bound"],
                             "saturated": bool(p <= SAT_EPS or p >= 1 - SAT_EPS),
                             "n_requests": e["n_requests"]})
    df = pd.DataFrame(rows)
    uns = df[~df.saturated]
    sat = df[df.saturated]
    zero = sat[sat.p_logprob <= SAT_EPS]
    summary = {
        "label": label, "cells": int(len(df)),
        "unsaturated": int(len(uns)),
        "saturated": int(len(sat)),
        "saturated_zero_cells_p_logprob_median": float(zero.p_logprob.median()) if len(zero) else None,
        "saturated_zero_cells_p_logprob_max": float(zero.p_logprob.max()) if len(zero) else None,
        "max_mass_bound": float(df.mass_bound.max()),
        "requests_per_cell_mean": float(df.n_requests.mean()),
    }
    seq_rows = []
    if server is not None and (seq_cells > 0 or seq_saturated > 0) and len(df):
        import numpy as np
        u = uns.sort_values("p_logprob")
        if len(u) and seq_cells > 0:
            idx = sorted(set(np.linspace(0, len(u) - 1, min(seq_cells, len(u))).round().astype(int)))
            parts = [u.iloc[idx]]
        else:
            parts = [u.head(0)]
        k = max(seq_saturated // 2, 1) if seq_saturated > 0 else 0
        if k:
            parts += [zero.sort_values("p_logprob", ascending=False).head(k),
                      sat[sat.p_logprob >= 1 - SAT_EPS].sort_values("p_logprob").head(k)]
        if seq_random > 0:
            rest = sat.drop(pd.concat(parts).index, errors="ignore")
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
            inside1 = _inside(r.p_logprob, lo, hi)
            print(f"  seq-check {r.scenario}/{r.role} {int(r.n_similar)}/{int(r.n_occupied)}: "
                  f"exact {r.p_logprob:.3f}  sequential {mv}/{n}={mv/max(n,1):.3f} "
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
                      f"{'ok' if _inside(r.p_logprob, lo, hi) else 'STILL MISSES'}")
            seq_rows.append({"scenario": r.scenario, "role": r.role, "n_similar": r.n_similar,
                             "n_occupied": r.n_occupied,
                             "exact_saturated": bool(r.saturated),
                             "p_logprob": r.p_logprob,
                             "seq_n_stage1": n1, "seq_move_stage1": mv1,
                             "seq_ci_low_stage1": lo1, "seq_ci_high_stage1": hi1, "inside_stage1": inside1,
                             "escalated": escalated, "seq_n": n, "seq_move": mv,
                             "p_sequential": mv / n if n else None, "seq_ci_low": lo, "seq_ci_high": hi,
                             "inside_seq_ci": _inside(r.p_logprob, lo, hi)})
        LOGPROB_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(seq_rows).to_csv(LOGPROB_VALIDATION_DIR / f"seqcheck_{label}.csv", index=False)
    sq = pd.DataFrame(seq_rows)
    summary = _seq_summary(summary, sq)
    LOGPROB_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    (LOGPROB_VALIDATION_DIR / f"validation_{label}.json").write_text(json.dumps(summary, indent=1))
    # The figure that belongs to this verdict is the SEQUENTIAL check
    # (seqcheck_plots/seqcheck_<label>.png), drawn as soon as its data is on disk.
    if plot and len(sq):
        from plot_seqcheck import plot_one
        plot_one(label)
    return summary





def server_env(url, keys=("FAKETIME", "LD_PRELOAD", "TZ")):
    """The clock-pinning environment of the server answering `url` (from
    /proc/<pid>/environ). llama.cpp injects today's date into every chat
    template (common_chat_extra_context: date_string, datetime) and minja's
    strftime_now reads the real clock, so a Llama-3 or Mistral prompt changes
    every calendar day unless the server's clock is pinned (libfaketime).
    The pin belongs in the trace next to the numbers it produced."""
    port = re.search(r":(\d+)(?:/|$)", url)
    port = port.group(1) if port else None
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
                if not (argv and argv[0].decode(errors="replace").endswith("llama-server")
                        and (port is None or port.encode() in argv)):
                    continue
                env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
            except OSError:
                continue
            d = dict(e.decode(errors="replace").split("=", 1) for e in env if b"=" in e)
            return {k: d.get(k) for k in keys}
    except OSError:
        pass
    return None


def server_cmdline(url):
    """Best effort: the argv of the llama-server answering `url`, from /proc.
    /props does not report attention or offload flags, and `-fa off` alone
    moved llama's transition cells by up to 0.42 (2026-09-11), so the launch
    line is the record that pins the serving configuration. None if no
    matching process is visible (remote server, other user)."""
    port = re.search(r":(\d+)(?:/|$)", url)
    port = port.group(1) if port else None
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            except OSError:
                continue
            argv = [a.decode(errors="replace") for a in argv if a]
            if argv and argv[0].endswith("llama-server") and (port is None or port in argv):
                return argv
    except OSError:
        pass
    return None



def _rendered_probe(server):
    try:
        return server.render("probe")
    except Exception as e:                                       # never block extraction on it
        return f"<apply-template failed: {e}>"


def weights_digest(model_path, chunk=1 << 24):
    """sha256 of the served weights as the server reads them (through the page
    cache — the same bytes mmap hands to the model), every shard of a split
    gguf included. Recorded in each trace's meta since 2026-09-11: llama's
    2026-09-06 extraction differed from three bit-identical 2026-09-11 launches
    on every cell (~0.18 nats in logit space) with nothing in the code, flags,
    build, driver or the file's mtime having changed; the file hashes to the
    upstream LFS oid today, so the earlier regime is unverifiable after the fact.
    A checksum in the trace makes that comparison possible next time. ~3 min for
    42 GB from cache; --no-weights-hash skips it."""
    path = Path(model_path)
    files = [path]
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})(\.gguf)$", path.name)
    if m:
        stem, _, n, ext = m.groups()
        files = [path.with_name(f"{stem}-{i:05d}-of-{n}{ext}") for i in range(1, int(n) + 1)]
    out, combined = [], hashlib.sha256()
    for f in files:
        if not f.exists():
            out.append({"path": str(f), "missing": True}); continue
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(chunk), b""):
                h.update(block)
        d = h.hexdigest(); combined.update(d.encode())
        out.append({"path": str(f), "size": f.stat().st_size, "sha256": d})
    return {"files": out, "sha256_combined": combined.hexdigest() if len(out) > 1 else out[0].get("sha256"),
            "hashed_via": "page cache read (what mmap serves)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True, help="table label, e.g. qwen3.6-27b-chat-grammar "
                                                   "(the tables are written as <label>-lp)")
    ap.add_argument("--style", default="R3_dual_count")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="default: every scenario in the scenario file (the campaigns' six)")
    ap.add_argument("--roles", nargs="*", default=["red", "blue"])
    ap.add_argument("--scenario-file", default=None, help="default scenarios_a2.py")
    ap.add_argument("--url", default="http://localhost:8085")
    ap.add_argument("--model", default=None, help="payload model name (default: the served gguf's stem)")
    ap.add_argument("--no-weights-hash", action="store_true",
                    help="skip the sha256 of the served gguf that goes into each trace's meta")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help=f"sampling temperature the table is extracted at (default {DEFAULT_TEMPERATURE}, "
                         "every campaign's value)")
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
    ap.add_argument("--no-plot", action="store_true",
                    help="skip the figures (per-scenario + combined value-function plots, and the sequential-check plot)")
    ap.add_argument("--seq-cells", type=int, default=12,
                    help="unsaturated cells (evenly spaced over the exact P(MOVE) range) re-sampled "
                         "sequentially for the pass/fail test")
    ap.add_argument("--seq-samples", type=int, default=300)
    ap.add_argument("--seq-saturated", type=int, default=4,
                    help="saturated cells to re-sample too: half the ~0 cells with the largest "
                         "exact P(MOVE), half the ~1 cells with the smallest")
    ap.add_argument("--seq-random", type=int, default=2,
                    help="additional random saturated cells for coverage")
    ap.add_argument("--seq-escalate", type=int, default=3,
                    help="a first-stage miss gets this many x --seq-samples more draws; "
                         "pass requires every checked cell inside its final CI")
    ap.add_argument("--validate-only", action="store_true",
                    help="skip extraction: load the existing raw/vflp_*_states.jsonl.gz for the label "
                         "and run the validation (needs the model's server up)")
    ap.add_argument("--no-write-vf1", action="store_true")
    ap.add_argument("--reverdict-only", action="store_true",
                    help="no server, no draws: recompute validation_<label>.json from the "
                         "seqcheck_<label>.csv on disk under the current containment rule")
    args = ap.parse_args()
    if args.reverdict_only:
        print(json.dumps(reverdict(args.label, plot=not args.no_plot), indent=1))
        return 0 if reverdict(args.label, plot=False)["pass"] else 4

    out_dir = Path(args.out_dir); (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    # Nothing is read from a sampled table (user decision 2026-09-07: the
    # sampled tables are not canonical): scenarios come from the scenario
    # file, the temperature from the CLI, the model name from the server.
    scenarios = list(args.scenarios) if args.scenarios else scenario_names(args.scenario_file)
    props = requests.get(f"{args.url.rstrip('/')}/props", timeout=120).json()
    model = args.model or Path(props.get("model_path") or args.label).stem
    temperature = float(args.temperature)
    server = Server(args.url, model, temperature, args.n_probs)
    print(f"label={args.label} style={args.style} scenarios={scenarios} T={temperature} "
          f"server={props.get('model_path')} build={props.get('build_info')}")

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
                           seq_random=args.seq_random, plot=not args.no_plot)
        print(json.dumps(summary, indent=1))
        return 0 if summary["pass"] else 4
    weights = None
    if not args.no_weights_hash:
        mp = props.get("model_path") or ""
        t0 = time.time()
        weights = weights_digest(mp if Path(mp).exists() else REPO_ROOT_FOR_WEIGHTS / mp)
        print(f"weights sha256 {weights['sha256_combined']} ({len(weights['files'])} file(s), "
              f"{time.time() - t0:.0f} s)", flush=True)
    for scenario in scenarios:
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
            # the whole /props reply: the table is exact only for the serving
            # configuration — `-fa off` moved llama's transition cells by up to
            # 0.42 (launch probe, 2026-09-11) — so every setting the server
            # reports is kept with the numbers it produced
            "server_props": props,
            "server_cmdline": server_cmdline(args.url),
            "server_env": server_env(args.url),
            # the SERVER-rendered prompt for a probe message: exposes the date
            # the template baked in (prompt_sha256 per cell hashes only the
            # client's user turn, which is why a daily date went unnoticed)
            "rendered_probe_prompt": _rendered_probe(server),
            "weights": weights,
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
        if not args.no_write_vf1:
            vf1 = to_vf1(lp, args.label, args.style, scenario, args.scenario_file, args.roles, props)
            LOGPROB_TABLES_DIR.mkdir(parents=True, exist_ok=True)
            p1 = LOGPROB_TABLES_DIR / f"vf_{args.label}-lp__{scenario}__{args.style}.json"
            p1.write_text(json.dumps(vf1, indent=1))
            load_value_function(p1)              # consumer-side check
            if not args.no_plot:
                from build_value_function import plot_vf, vf_plot_path
                plot_vf(vf1, vf_plot_path(LOGPROB_DIR, f"{args.label}-lp",
                                          scenario, args.style, "png"))
    if not args.no_plot and not args.no_write_vf1:
        from plot_value_functions import render_label
        render_label(LOGPROB_TABLES_DIR, LOGPROB_DIR / VF_MAPPING_PLOTS_REL,
                     f"{args.label}-lp", args.style)
    print(f"total requests: {server.n_requests}")
    if not args.no_validate:
        summary = validate(args.label, args.style, scenarios, args.roles, lp_by_scenario, out_dir,
                           server=server, seq_cells=args.seq_cells, seq_samples=args.seq_samples,
                           seq_escalate=args.seq_escalate, seq_saturated=args.seq_saturated,
                           seq_random=args.seq_random, plot=not args.no_plot)
        print(json.dumps(summary, indent=1))
        return 0 if summary["pass"] else 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
