"""Shared sampling / parsing / aggregation helpers for the prompt_refinement
harnesses (evaluate_prompts.py, evaluate_ratio_prompts.py,
build_value_function.py).

Single home for everything that used to be duplicated between the two evaluate
scripts (2026-08-21 dedup): the slice-ping ntfy helper, the role->keyword
mapping, the per-cell ThreadPool sampling block, the gzip raw-reply writer, and
the production-payload sampler itself (moved here verbatim from
evaluate_prompts.py; that module re-exports the old names so existing imports
keep working).

Also the home of the RATIO math for the 23-point value function:
every distinct opposite-neighbour fraction reachable in a Moore neighbourhood
of <= 8 occupied neighbours (the Farey sequence F_8, |F_8| = 23), plus the
separate no-neighbours point. See ratio_of / ratio_groups / NO_NEIGHBORS_KEY.
"""
import gzip
import json
import math
import subprocess
import sys
import time
import zlib
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import requests

_PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = _PKG_DIR.parent
RESULTS_DIR = _PKG_DIR / "results"
for _p in (_PKG_DIR, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from llm_runner import SAMPLER_PARAMS, MOVE_STAY_GRAMMAR  # noqa: E402

# The production grammar bytes (llm_runner.MOVE_STAY_GRAMMAR) are the canonical
# +grammar arm payload: evaluate_prompts.py's historical GRAMMAR constant encodes
# the same language with literal whitespace chars — same accepted strings,
# different bytes. New code should use GRAMMAR from here.
GRAMMAR = MOVE_STAY_GRAMMAR

# ---------------------------------------------------------------------------
# Role -> prompt keywords
# ---------------------------------------------------------------------------

# Baseline scenario keywords, held fixed across prompt candidates so only the
# template varies. KW stays the red mapping for legacy importers.
KW = dict(agent_type="red team resident", opposite_type="blue team resident")
KW_BY_ROLE = {
    "red": KW,
    "blue": dict(agent_type="blue team resident", opposite_type="red team resident"),
}
ROLE_KW = KW_BY_ROLE   # evaluate_ratio_prompts historical alias


def role_keywords(scenario="baseline", scenario_file=None):
    """{'red': {agent_type, opposite_type}, 'blue': {...}} for a named scenario.

    red == the scenario's type_a, blue == type_b (llm_runner's _agent_role_key
    convention). Labels come from the scenario registry (default
    scenarios_a2.py, the ratio-frame's ancestor); for 'baseline' this returns
    exactly KW_BY_ROLE, asserted so the two sources can never drift.
    """
    if scenario == "baseline" and scenario_file is None:
        return KW_BY_ROLE
    import importlib.util
    path = Path(scenario_file) if scenario_file else REPO_ROOT / "scenarios_a2.py"
    spec = importlib.util.spec_from_file_location("_vf_scenarios", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sc = mod.CONTEXT_SCENARIOS[scenario]
    kw = {
        "red": dict(agent_type=sc["type_a"], opposite_type=sc["type_b"]),
        "blue": dict(agent_type=sc["type_b"], opposite_type=sc["type_a"]),
    }
    if scenario == "baseline":
        assert kw == KW_BY_ROLE, "scenario file baseline labels drifted from KW_BY_ROLE"
    return kw


# ---------------------------------------------------------------------------
# Sampling (moved verbatim from evaluate_prompts.py)
# ---------------------------------------------------------------------------

def request_seed(*parts) -> int:
    """Deterministic per-request sampling seed: crc32 over the identifying
    tuple (stage, scenario, style, role, n_sim, n_occ, sample_index, ...).

    Pure function of its arguments — no global counter to keep in sync across
    top-up stages or resumed runs, and any request's seed can be recomputed
    from its raw-log identity for bitwise replication (see
    LLAMA_CPP_SERVING_NOTES.md reproducibility table)."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def server_fingerprint(url: str) -> dict | None:
    """Build/model identity from llama-server's /props endpoint, for artifact
    provenance and the replication kit. None if unavailable."""
    try:
        base = url.split("/v1/")[0].rstrip("/")
        j = requests.get(f"{base}/props", timeout=10).json()
        return {"build_info": j.get("build_info"),
                "model_path": j.get("model_path"),
                "total_slots": j.get("total_slots")}
    except Exception:
        return None


def sample_once(url: str, model: str, prompt: str, temperature: float,
                grammar: str | None = None, cache_prompt: bool | None = None,
                seed: int | None = None) -> dict:
    """One decision, using the SAME payload llm_runner.py sends (see LLMAgent).

    cache_prompt/seed default None => field OMITTED => payload byte-identical
    to the historical harness (legacy sweeps stay self-comparable). The clean
    measurement protocol (KV_CACHE_SAMPLING_ARTIFACT.md) passes
    cache_prompt=False plus a distinct request_seed() per sample.

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
    if cache_prompt is not None:
        payload["cache_prompt"] = cache_prompt
    if seed is not None:
        payload["seed"] = seed
    if "/chat/completions" in url:
        payload["messages"] = [{"role": "user", "content": prompt}]
    else:
        payload["prompt"] = prompt
    # Bounded retry, then degrade to an unparseable sample — NEVER raise.
    # Rationale (2026-08-04): a single HTTP 500 used to abort a whole arm and
    # discard hours of completed sampling (gemma blue completions died twice at
    # ~7k requests). Two distinct 500 sources:
    #   * transient server-side allocation hiccups -> a retry succeeds;
    #   * llama.cpp's content parser rejecting the model's own output, e.g.
    #     "The model produced output that does not match the expected
    #     Content-only format" after a partial-UTF-8 token ("// �") from
    #     max_tokens=5 truncation. Retrying re-samples; if it keeps failing the
    #     reply IS unparseable model output, so record it as such (text="",
    #     parse -> UNPARSEABLE) instead of crashing. Production does the same
    #     class of thing: llm_runner retries 5xx (llm_runner.py:368).
    last_err = None
    for attempt in range(3):
        try:
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
        except (requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                ValueError, KeyError, IndexError) as e:
            last_err = e
            if attempt < 2:
                time.sleep(1.0 + attempt)
    print(f"    [sample_once] giving up after 3 attempts ({type(last_err).__name__}: "
          f"{str(last_err)[:120]}) -> recording as UNPARSEABLE", flush=True)
    return {"text": "", "finish_reason": "server_error", "completion_tokens": None}


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


def sample_batch(url, model, prompts, temperature, grammar, concurrency,
                 cache_prompt=None, seed_fn=None):
    """The per-cell ThreadPool block both evaluate scripts carried inline.

    `prompts` is the full list of rendered prompt strings for one batch (one
    per requested sample); returns the reply dicts in order. `seed_fn(i)`
    supplies request i's sampling seed (clean protocol); None omits the field.
    """
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(
            lambda ip: sample_once(url, model, ip[1], temperature, grammar,
                                   cache_prompt=cache_prompt,
                                   seed=None if seed_fn is None else seed_fn(ip[0])),
            enumerate(prompts)))


def aggregate_counts(replies):
    """(n_move, n_stay, n_bad) from a batch of reply dicts (bad = AMBIG + UNPARSE)."""
    c = Counter(parse(r["text"]) for r in replies)
    return c["MOVE"], c["STAY"], c["AMBIGUOUS"] + c["UNPARSEABLE"]


def wilson_ci(count: int, total: int, z: float = 1.96) -> tuple:
    """Wilson score interval, clamped to [0, 1]; (0, 1) when total <= 0.

    Canonical home going forward; original in
    llm_utility_approximation/branching_vs_sampling_comparison.py:136 (not
    imported from there because that module pulls numpy + local modules).
    """
    if total <= 0:
        return (0.0, 1.0)
    phat = count / total
    denom = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Raw-reply persistence
# ---------------------------------------------------------------------------

class RawWriter:
    """Incremental gzip-jsonl raw-reply writer with the standard `_meta` header.

    The sweep scripts accumulated every raw record in memory and wrote once at
    the end — a killed run lost all raw data. This writes through per record.
    Use as a context manager.
    """

    def __init__(self, path, meta):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._f = gzip.open(path, "wt", encoding="utf-8")
        self._f.write(json.dumps({"_meta": True, **meta}) + "\n")
        self.n = 0

    def write(self, record):
        self._f.write(json.dumps(record) + "\n")
        self.n += 1

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_raw_gz(path, meta, records):
    """One-shot variant for the evaluate scripts' end-of-run write."""
    with RawWriter(path, meta) as w:
        for rec in records:
            w.write(rec)


# ---------------------------------------------------------------------------
# Slice pings (flag-file gated ntfy progress, shared state)
# ---------------------------------------------------------------------------

SLICE_PING_FLAG = REPO_ROOT / "logs" / "ntfy_slice_pings.flag"
_SLICE_STATE = {"t0": None, "done": 0, "total": None}


def init_slice_state(total):
    _SLICE_STATE.update(t0=time.time(), done=0, total=total)


def slice_ping(label, what):
    """Per-slice ntfy ping with arm ETA; no-ops unless the flag file exists
    and init_slice_state() was called. `what` is the finished item's name
    (callers format candidate/role themselves)."""
    st = _SLICE_STATE
    st["done"] += 1
    if not SLICE_PING_FLAG.exists() or st["total"] is None:
        return
    elapsed = time.time() - st["t0"]
    remaining = st["total"] - st["done"]
    eta_s = int(elapsed / st["done"] * remaining) if remaining else 0
    body = (f"slice {st['done']}/{st['total']}; elapsed {int(elapsed // 60)}m; "
            f"arm ETA ~{eta_s // 60}m" if remaining else
            f"slice {st['done']}/{st['total']} — arm complete")
    subprocess.run([str(REPO_ROOT / "ntfy.sh"), f"{label}: {what} done", body,
                    "hourglass", "low"], cwd=str(REPO_ROOT), check=False,
                   capture_output=True)


# ---------------------------------------------------------------------------
# Ratio math for the 23-point value function
# ---------------------------------------------------------------------------

# Key for the zero-neighbour composition (0, 0): it has NO defined ratio and is
# its own datapoint (mechanical reference: satisfied -> STAY).
NO_NEIGHBORS_KEY = "none"


def ratio_of(n_similar, n_occupied):
    """Opposite-neighbour fraction as an exact Fraction, or None for no neighbours.

    Fraction() reduces automatically, so (1,2), (2,4), (3,6), (4,8) all key to
    Fraction(1, 2) — the aliasing that makes the 23-point axis coarser than the
    45 compositions.
    """
    if n_occupied == 0:
        return None
    return Fraction(n_occupied - n_similar, n_occupied)


def ratio_groups(max_n=8):
    """OrderedDict[Fraction -> list[(n_similar, n_occupied)]], ascending ratio.

    Every occupied composition of a Moore neighbourhood with <= max_n
    neighbours, grouped by its reduced opposite fraction. For max_n=8 there are
    exactly 23 groups (the Farey sequence F_8); the (0, 0) no-neighbour cell is
    NOT included — it has no ratio and is handled as its own datapoint.
    """
    groups = {}
    for n_occ in range(1, max_n + 1):
        for n_sim in range(n_occ + 1):
            groups.setdefault(ratio_of(n_sim, n_occ), []).append((n_sim, n_occ))
    out = OrderedDict((f, groups[f]) for f in sorted(groups))
    assert len(out) == 23, f"expected 23 distinct ratios for max_n=8, got {len(out)}"
    return out


def ratio_key(frac):
    """Canonical string key for a ratio: 'p/q' reduced ('0/1', '1/2', '1/1'),
    or NO_NEIGHBORS_KEY for the zero-neighbour datapoint (frac is None)."""
    if frac is None:
        return NO_NEIGHBORS_KEY
    return f"{frac.numerator}/{frac.denominator}"


def load_value_function(path):
    """Load a vf_*.json artifact; returns the parsed dict after schema checks.

    Fails loudly AT LOAD TIME if any ratio datapoint lacks a defined
    p_move_effective (all-bad-parse cells) — the predecessor log-prob policy
    raised KeyError mid-simulation instead, which discarded whole runs.
    """
    with open(path) as f:
        vf = json.load(f)
    if vf.get("schema") != "vf-1":
        raise ValueError(f"{path}: unknown schema {vf.get('schema')!r} (want 'vf-1')")
    expected = {ratio_key(f) for f in ratio_groups()} | {NO_NEIGHBORS_KEY}
    for role, rows in vf["ratios"].items():
        keys = {r["ratio"] for r in rows}
        missing = expected - keys
        undefined = sorted(r["ratio"] for r in rows if r["p_move_effective"] is None)
        if missing:
            raise ValueError(f"{path}: role {role} missing ratio datapoints: {sorted(missing)}")
        if undefined:
            raise ValueError(
                f"{path}: role {role} has ratio datapoints with no defined "
                f"P(MOVE) (all samples unparseable): {undefined} — resample these")
    return vf
