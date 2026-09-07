import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass  # Older Pythons or non-tty streams

import math
import numpy as np
import json
import os
import random
from datetime import datetime
import config as cfg
from Agent import Agent
import requests
from tqdm import tqdm
import time
from context_scenarios import CONTEXT_SCENARIOS
import argparse
import run_files
from base_simulation import Simulation, convergence_from_step_moves
from multiprocessing import Pool, cpu_count
import pandas as pd
import ast
import glob
import re
import gzip
from pathlib import Path


# Sampler parameters are pinned EXPLICITLY rather than left to the server's defaults.
# Two reasons:
#   1. Reproducibility — "temperature only" silently inherits each backend's own
#      defaults (llama.cpp native server, llama_cpp.server, Ollama, vLLM all differ
#      in top_k/top_p/min_p/penalties), so results would depend on the backend.
#   2. The probability estimator computes a pure-temperature softmax. Pinning the
#      sampler to pure temperature (all truncation + penalties disabled) makes the
#      live sampler match that assumption, so the estimator has a well-defined target.
# Pure temperature: top_k<=0 disables top-k (full vocab), top_p=1.0 / min_p=0.0 disable
# nucleus / min-p truncation, penalties=1.0/0.0 disable repetition shaping.
# NOTE: these are pinned here (not yet plumbed through the YAML run configs); to change
# the sampling regime, edit this dict. Temperature is still supplied per-request.
SAMPLER_PARAMS = {
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

# Request styles selectable via --llm-style (CLI) / contexts_args.llm_style (YAML).
LLM_STYLES = ("completions", "completions+grammar", "chat", "chat+grammar")

# Permissive MOVE/STAY GBNF grammar: optional leading whitespace + any casing of
# exactly one of the two words, nothing else. Byte-identical to the version
# validated in prompt_refinement/ (2026-07: 0 bad parses on Llama/Gemma/Qwen;
# move-rate curves unchanged vs unconstrained sampling). Generation halts at the
# word boundary, so max_tokens=5 becomes a never-binding safety ceiling.
MOVE_STAY_GRAMMAR = r"""
root   ::= ws answer
ws     ::= [ \t\n]*
answer ::= move | stay
move   ::= [Mm] [Oo] [Vv] [Ee]
stay   ::= [Ss] [Tt] [Aa] [Yy]
"""


def resolve_llm_style(llm_style):
    """Normalise/validate an llm_style value. None/'' -> None (legacy behaviour)."""
    if llm_style is None or str(llm_style).strip() == "":
        return None
    style = str(llm_style).strip().lower().replace(" ", "+").replace("_", "+")
    while "++" in style:
        style = style.replace("++", "+")
    if style not in LLM_STYLES:
        raise ValueError(f"Unknown llm_style '{llm_style}'. Valid: {', '.join(LLM_STYLES)}")
    return style


def resolve_llm_request_url(llm_url, llm_style):
    """Rewrite llm_url's path to the style's endpoint. style None: URL unchanged."""
    style = resolve_llm_style(llm_style)
    if style is None:
        return llm_url
    base = llm_url.split("/v1/")[0].rstrip("/")
    endpoint = "chat/completions" if style.startswith("chat") else "completions"
    return f"{base}/v1/{endpoint}"


def build_llm_request(llm_url, llm_style, llm_model, prompt, temperature, max_tokens=5):
    """Return (request_url, payload) for the given style.

    completions[+grammar]  raw /v1/completions "prompt" payload (no chat template)
    chat[+grammar]         /v1/chat/completions single-user-turn "messages" payload
    +grammar variants add the MOVE/STAY GBNF grammar (llama.cpp).
    style=None keeps legacy behaviour: raw payload against the unmodified llm_url,
    switching to a chat payload only if that URL already targets /chat/completions.
    """
    style = resolve_llm_style(llm_style)
    url = resolve_llm_request_url(llm_url, style)
    payload = {
        "model": llm_model,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 10000,    # 10 second timeout in milliseconds
        **SAMPLER_PARAMS,    # pinned pure-temperature sampler (see module top)
    }
    is_chat = (style or "").startswith("chat") or "/chat/completions" in url
    if is_chat:
        payload["messages"] = [{"role": "user", "content": prompt}]
    else:
        payload["prompt"] = prompt
    if style is not None and style.endswith("+grammar"):
        payload["grammar"] = MOVE_STAY_GRAMMAR
    return url, payload


def grid_config_from_args(grid_size=None, num_type_a=None, num_type_b=None):
    """Build the apply_grid_config payload from CLI values, or None if all unset.

    Returning None when nothing was given keeps config.py the default: only a
    run that explicitly asks for a geometry gets one.
    """
    provided = {'GRID_SIZE': grid_size, 'NUM_TYPE_A': num_type_a, 'NUM_TYPE_B': num_type_b}
    provided = {k: v for k, v in provided.items() if v is not None}
    return provided or None


def apply_grid_config(grid_config):
    """Set GRID_SIZE / NUM_TYPE_A / NUM_TYPE_B on the config module, in place.

    The grid used to be reachable only by editing config.py, which is ambient
    global state: it changed every runner, test and GUI launch at once, and a
    run's geometry was whatever the file happened to say that day rather than
    something the run recorded. Passing it down means the run YAML — which the
    orchestrator freezes into the run directory — is the record of what was
    simulated, and config.json picks it up because the parent applies it before
    the config snapshot is written.

    Workers re-apply this (see run_single_simulation) so it is fork/spawn safe:
    under spawn a worker re-imports config fresh and would otherwise run the
    default 10x10 while the parent reported 20x20.
    """
    if not grid_config:
        return
    size = int(grid_config.get('GRID_SIZE', cfg.GRID_SIZE))
    n_a = int(grid_config.get('NUM_TYPE_A', cfg.NUM_TYPE_A))
    n_b = int(grid_config.get('NUM_TYPE_B', cfg.NUM_TYPE_B))
    if size < 1:
        raise ValueError(f"grid_size must be >= 1, got {size}")
    if n_a < 0 or n_b < 0:
        raise ValueError(f"agent counts must be non-negative, got {n_a} and {n_b}")
    if n_a + n_b > size * size:
        raise ValueError(
            f"{n_a} + {n_b} agents do not fit on a {size}x{size} grid "
            f"({size * size} cells)")
    cfg.GRID_SIZE, cfg.NUM_TYPE_A, cfg.NUM_TYPE_B = size, n_a, n_b


def apply_scenario_file(scenario_file):
    """Load CONTEXT_SCENARIOS from a python module path and swap them in, IN PLACE.

    In-place mutation is the point: every module that did `from context_scenarios
    import CONTEXT_SCENARIOS` shares this dict object, so clear()+update() re-points
    them all (validation in run_all_contexts included) at the file's scenarios.
    Workers re-apply this (see run_single_simulation), so it is fork/spawn safe.
    """
    import importlib.util
    path = str(scenario_file)
    # A bare/relative path must survive callers that chdir (the YAML pipeline runs
    # run_all_contexts with cwd=<run_dir>): fall back to resolving it against the
    # repo root (this file's directory) before giving up.
    if not os.path.isabs(path) and not os.path.exists(path):
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if os.path.exists(candidate):
            path = candidate
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"scenario file not found: {path}")
    spec = importlib.util.spec_from_file_location("_scenario_file_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = getattr(module, "CONTEXT_SCENARIOS", None)
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError(f"{path} must define a non-empty CONTEXT_SCENARIOS dict")
    for name, entry in loaded.items():
        for field in ("type_a", "type_b", "prompt_template"):
            if field not in entry:
                raise ValueError(f"scenario '{name}' in {path} is missing '{field}'")
    CONTEXT_SCENARIOS.clear()
    CONTEXT_SCENARIOS.update(loaded)
    return path


PROGRESS_EVERY = 100     # [run-progress] cadence for value-function runs


def format_run_progress(scenario, done, total, elapsed_s, n_processes,
                        progress_offset=0, progress_total=None):
    """One structured, machine-readable log line per completed run.

    This is part of the experimental record (per-run wall timing) and doubles as
    the contract consumed by watch_progress.sh, which tails the run log and turns
    these lines into ntfy pings with avg runtime + ETA. Notification transport
    deliberately lives OUTSIDE the experiment code so a replication run has zero
    notification side-effects — see watch_progress.sh / ntfy.sh.
    """
    line = (f"[run-progress] scenario={scenario} done={done} total={total} "
            f"elapsed_s={elapsed_s:.1f} procs={n_processes}")
    if progress_total:
        line += f" overall_done={progress_offset + done} overall_total={progress_total}"
    return line


def _sanitize_model_for_path_component(name: str) -> str:
    """Return filesystem-safe model slug used by llm_token_probabilities outputs."""
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '-', str(name).strip())
    sanitized = sanitized.rstrip(' .')
    sanitized = re.sub(r'-{2,}', '-', sanitized)
    return sanitized or "unknown-model"


def _temp_slug(temperature: float) -> str:
    """Format a temperature into a filesystem-safe slug matching branching_probability_estimator."""
    return f"T{float(temperature):.3f}".replace(".", "p")


def _resolve_value_function_file(vf_path, scenario):
    """Accept a vf-1 JSON file, a '{scenario}' template, OR a directory.

    Template ('.../vf_<label>__{scenario}__<style>.json') pins label and style
    exactly while substituting the current scenario — the right form for
    multi-scenario batches (run_all_contexts / the YAML orchestrator) once
    several labels (e.g. the -half sufficiency artifacts) share a directory.
    Directory resolution: scenario is in
    the filename) but with NO silent fallback: zero matches is an error naming
    the expected pattern, several matches is an error listing them.
    """
    def _anchor(path):
        """Resolve relative to cwd, else to the repo root — the orchestrator
        runs the contexts stage with cwd inside the run directory, while
        configs state artifact paths repo-relative."""
        p = Path(path)
        if p.exists():
            return p
        repo_p = Path(__file__).resolve().parent / path
        return repo_p if repo_p.exists() else p

    if "{scenario}" in str(vf_path):
        if not scenario:
            raise ValueError(f"value-function template {vf_path!r} needs a scenario")
        resolved = _anchor(str(vf_path).format(scenario=scenario))
        if not resolved.exists():
            raise FileNotFoundError(f"value-function template resolved to missing file: {resolved}")
        return str(resolved)
    p = _anchor(vf_path)
    if not p.is_dir():
        return str(p)
    matches = sorted(p.glob(f"vf_*__{scenario}__*.json"))
    if not matches:
        raise FileNotFoundError(
            f"no value function for scenario {scenario!r} under {p} "
            f"(expected vf_<label>__{scenario}__<style>.json)")
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous value function for scenario {scenario!r} under {p}: "
            f"{[m.name for m in matches]} — pass the file explicitly")
    return str(matches[0])


def load_value_function_policy(vf_path, scenario=None):
    """Load a vf-1 value-function artifact (prompt_refinement/build_value_function.py)
    into {(type_key, "comp:n_similar,n_occupied"): move_probability}.

    COMPOSITION-level lookup only (user decision 2026-08-25, ratio mode
    removed): the agent's exact (n_similar, n_occupied) cell drives the
    decision — no ratio aliasing. The measured surfaces show members of one
    reduced ratio can genuinely differ (e.g. gemma baseline red at
    all-opposite: 1-of-1 p=0.00, 2-of-2 p=0.46, 8-of-8 p=1.00), so the
    aggregated 23-ratio curve in the artifact is a visualization, not a
    policy. All 45 cells per role — including the zero-neighbour (0,0) cell —
    must carry a defined rate; coverage is validated here at load time, so
    a mid-run KeyError on an unseen composition cannot occur.
    `vf_path` may be a directory when `scenario` is given (per-scenario
    filename resolution; see _resolve_value_function_file).
    """
    pr_dir = str(Path(__file__).resolve().parent / "prompt_refinement")
    if pr_dir not in sys.path:
        sys.path.insert(0, pr_dir)
    from sampling_common import load_value_function  # deferred: avoids import cycle

    vf_path = _resolve_value_function_file(vf_path, scenario)
    vf = load_value_function(vf_path)
    # Artifact roles are red/blue (scenario type_a/type_b); the simulation keys
    # agents as type_a/type_b (LLMAgent._agent_role_key).
    role_map = vf["meta"].get("role_to_type", {"red": "type_a", "blue": "type_b"})
    policy = {}
    for role, rows in vf["compositions"].items():
        type_key = role_map[role]
        for c in rows:
            if c["p_move_effective"] is None:
                raise ValueError(
                    f"{vf_path}: composition ({c['n_similar']},{c['n_occupied']}) "
                    f"role {role} has no defined rate — cannot build the policy")
            policy[(type_key, f"comp:{c['n_similar']},{c['n_occupied']}")] = \
                float(c["p_move_effective"])
    return policy, str(vf_path), vf["meta"]


def query_server_slots(llm_url, timeout=5):
    """Ask a llama.cpp server how many continuous-batching slots it actually has.

    Returns the server's `total_slots` (from GET /props) or None if it can't be
    determined (non-llama.cpp backend, server down, old build). Used to cap the
    client Pool so we never launch more concurrent runs than there are slots —
    the server's `-np` (set from `processes` by the run_*.sh launcher) is the
    authoritative slot count, and this keeps the client in sync with it.
    """
    if not llm_url:
        return None
    # /props lives at the server root, not under the /v1/... OpenAI path.
    base = re.split(r"/v1(?:/|$)", llm_url, maxsplit=1)[0].rstrip("/")
    try:
        resp = requests.get(f"{base}/props", timeout=timeout)
        resp.raise_for_status()
        slots = resp.json().get("total_slots")
        return int(slots) if slots else None
    except Exception:
        return None


def check_llm_connection(llm_model=None, llm_url=None, llm_api_key=None, timeout=10,
                         max_retries=5, llm_style=None):
    """
    Check if LLM connection is active and working

    Parameters:
    - llm_model: Model to use (overrides config.py)
    - llm_url: API URL (overrides config.py)
    - llm_api_key: API key (overrides config.py)
    - timeout: Connection timeout in seconds
    - max_retries: Max retry attempts on transient failures (timeouts/connection errors)
    - llm_style: request style the run will actually use (completions[+grammar] /
      chat[+grammar]). The check MUST mirror it: a chat endpoint rejects a raw
      "prompt" payload with HTTP 400 ('messages' is required), which would abort
      the whole run before a single simulation step.

    Returns:
    - True if connection successful
    - False if connection failed
    """
    model = llm_model or cfg.OLLAMA_MODEL
    url = llm_url or cfg.OLLAMA_URL
    api_key = llm_api_key or cfg.OLLAMA_API_KEY

    # Same builder the simulation uses, so the probe hits the same endpoint with
    # the same payload shape. The grammar is dropped: it constrains output to
    # MOVE/STAY, and here we just want the server to echo something back.
    url, test_payload = build_llm_request(
        url, llm_style, model,
        "Respond with only the word 'OK' and nothing else.",
        temperature=0, max_tokens=10,
    )
    test_payload.pop("grammar", None)

    print("\nChecking LLM connection...")
    print(f"URL: {url}")
    print(f"Model: {model}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for attempt in range(1, max_retries + 1):
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=test_payload, timeout=timeout)
            elapsed = time.time() - start_time

            if response.status_code != 200:
                print(f"❌ LLM connection failed - HTTP {response.status_code}")
                print(f"Response: {response.text}")
                # Retry only for server errors (5xx) if attempts remain
                if attempt < max_retries and 500 <= response.status_code < 600:
                    print(f"↻ Retrying... ({attempt}/{max_retries})")
                    time.sleep(1)
                    continue
                return False

            data = response.json()

            # Check for proper response structure
            if "choices" not in data or not data["choices"]:
                print("❌ LLM connection failed - Invalid response structure")
                print(f"Response: {data}")
                return False

            choice0 = data["choices"][0]
            content = (choice0.get("text")
                       or (choice0.get("message") or {}).get("content", "")).strip()
            print(f"✅ LLM connection successful (response time: {elapsed:.2f}s)")
            print(f"Test response: '{content}'")
            return True

        except requests.exceptions.Timeout:
            print(f"❌ Timeout after {timeout}s (attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(5)
                continue
            print("The LLM server is not responding. Please check:")
            print("1. Is the Ollama server running?")
            print("2. Is the URL correct?")
            print("3. Is the model loaded?")
            return False

        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection error (attempt {attempt}/{max_retries})")
            print(f"Error: {e}")
            if attempt < max_retries:
                time.sleep(1)
                continue
            print("\nPlease check:")
            print("1. Is the Ollama server running?")
            print("2. Is the URL correct?")
            print("3. Is your network connection working?")
            return False

        except Exception as e:
            print("❌ LLM connection failed - Unexpected error")
            print(f"Error: {type(e).__name__}: {e}")
            return False

    return False

# ---------------------------------------------------------------------------
# Value-function decision RNG: one uniform per (run, step, agent, purpose)
# ---------------------------------------------------------------------------
# Until 2026-09-04 value-function decisions drew from the shared `random`
# stream: a STAY consumed one draw, a MOVE two (accept + destination). Two runs
# with the same seed but tables differing in ONE cell therefore desynchronised
# at the first differing decision and re-rolled every later draw — so the
# "paired" arms of vf_multisplit_check (same run seeds, half vs full table)
# and the cross-scenario pairing of vf_rank_stability were paired only up to
# the initial grid. Keying each uniform to its identity instead makes the
# streams true common random numbers: two tables now produce runs that differ
# only at decisions whose uniform falls between the two probabilities.
#
# VF_RNG_SCHEME=shared reproduces pre-2026-09-04 runs from their run_id; the
# scheme in force is recorded as `rng_scheme` in each experiment's config.json.
VF_RNG_SCHEME = os.environ.get("VF_RNG_SCHEME", "keyed")
if VF_RNG_SCHEME not in ("keyed", "shared"):
    raise ValueError(f"VF_RNG_SCHEME must be 'keyed' or 'shared', got {VF_RNG_SCHEME!r}")

_MASK64 = (1 << 64) - 1
_U53 = 1.0 / (1 << 53)


def _splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def keyed_uniform(run_id, step, agent_id, salt):
    """Uniform in [0, 1) that is a pure function of (run_id, step, agent_id, salt).

    salt 0 = the accept/reject draw, 1 = the destination draw. splitmix64 is
    chained over the four keys, so neighbouring keys give independent-looking
    outputs (the same mixer numpy's SeedSequence and Java's SplittableRandom
    rely on); 53 high bits become the double, matching random.random().
    """
    h = 0
    for k in (run_id, step, agent_id, salt):
        h = _splitmix64(h ^ (int(k) & _MASK64))
    return (h >> 11) * _U53


def _empty_cells(grid):
    """Empty cells in row-major order — the order the destination draw indexes.

    One vectorised None-mask over the object grid instead of a Python scan of
    every cell for every mover (2026-09-05). Same list, so int(u * len) and
    random.choice pick the same cell as before.
    """
    if not isinstance(grid, np.ndarray):
        grid = np.asarray(grid, dtype=object)
    return [(int(row), int(col)) for row, col in np.argwhere(grid == None)]  # noqa: E711


class LLMAgent(Agent):
    def __init__(self, type_id, scenario='baseline', llm_model=None, llm_url=None, llm_api_key=None,
                 run_id=None, step=None, temperature=None, llm_style=None,
                 value_function_policy=None, value_function_path=None):
        super().__init__(type_id)
        self.scenario = scenario
        self.context_info = CONTEXT_SCENARIOS[scenario]
        self.agent_type = self.context_info['type_a'] if type_id == 0 else self.context_info['type_b']
        self.opposite_type = self.context_info['type_b'] if type_id == 0 else self.context_info['type_a']
        self.llm_model = llm_model or cfg.OLLAMA_MODEL
        self.llm_url = llm_url or cfg.OLLAMA_URL
        self.llm_api_key = llm_api_key or cfg.OLLAMA_API_KEY
        self.llm_style = resolve_llm_style(llm_style)
        self.llm_request_url = resolve_llm_request_url(self.llm_url, self.llm_style)
        if temperature is None:
            raise ValueError("temperature must be provided to LLMAgent")
        self.temperature = float(temperature)
        self.run_id = run_id
        # Initialize LLM tracking metrics
        self.llm_call_count = 0
        self.llm_call_time = 0.0
        self.step = step
        # Composition-keyed value-function policy:
        # {(type_key, "comp:n_sim,n_occ"): p_move}, loaded by load_value_function_policy.
        self.value_function_policy = value_function_policy or {}
        self.value_function_path = value_function_path
        self.store_llm_responses = (
            getattr(cfg, 'STORE_LLM_RESPONSES', False) or
            os.environ.get('STORE_LLM_RESPONSES', '').lower() in ('true', '1', 'yes')
        )
        self.last_llm_response_raw = None
        self.last_llm_parsed_decision = None
        self.last_llm_parse_status = None

    _warned_unkeyed = False

    def _agent_role_key(self):
        return "type_a" if self.type_id == 0 else "type_b"

    def _context_arrangement_code(self, r, c, grid):
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < cfg.GRID_SIZE and 0 <= nc < cfg.GRID_SIZE:
                    neighbor = grid[nr][nc]
                    if neighbor is None:
                        neighbors.append("E")
                    elif neighbor.type_id == self.type_id:
                        neighbors.append("S")
                    else:
                        neighbors.append("O")
                else:
                    neighbors.append("#")
        return "".join(neighbors)

    def _get_value_function_decision(self, r, c, grid):
        """Sample MOVE/STAY from the composition-level value function.

        The key is the agent's exact (n_similar, n_occupied) count over the 8
        surrounding cells (walls and empties excluded — Agent._unlike_ratio
        semantics). Every cell, including the zero-neighbour (0,0) one, is
        guaranteed present by the load-time coverage check.

        Randomness: under VF_RNG_SCHEME=keyed (default since 2026-09-04) the
        accept draw and the destination draw are keyed_uniform(run_id, step,
        agent_id, salt) — common random numbers across tables and scenarios,
        see the note above keyed_uniform. `shared` is the legacy shared-stream
        behaviour; it is also used, with a one-time warning, if a caller never
        gave the agent a run_id/step/agent_id (base_simulation always does).
        """
        # Count neighbours directly rather than rendering the 8-character
        # arrangement code and counting letters in it: this ran for every
        # agent every step and was the single largest line in the profile.
        n_sim = n_occ = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < cfg.GRID_SIZE and 0 <= nc < cfg.GRID_SIZE:
                    neighbor = grid[nr][nc]
                    if neighbor is not None:
                        n_occ += 1
                        if neighbor.type_id == self.type_id:
                            n_sim += 1

        type_key = self._agent_role_key()
        move_probability = self.value_function_policy[
            (type_key, f"comp:{n_sim},{n_occ}")]

        # Keep accounting aligned with LLM runs for downstream summaries.
        self.llm_call_count += 1

        agent_id = getattr(self, "agent_id", None)
        keyed = (VF_RNG_SCHEME == "keyed" and self.run_id is not None
                 and self.step is not None and agent_id is not None)
        if VF_RNG_SCHEME == "keyed" and not keyed and not LLMAgent._warned_unkeyed:
            LLMAgent._warned_unkeyed = True
            print("[LLMAgent] value-function decision without run_id/step/agent_id: "
                  "falling back to the shared random stream for this process")

        u_accept = (keyed_uniform(self.run_id, self.step, agent_id, 0) if keyed
                    else random.random())
        choose_move = u_accept < move_probability
        if self.store_llm_responses:
            self.last_llm_response_raw = (
                f"VALUE_FUNCTION(comp={n_sim}/{n_occ}, move={move_probability:.6f})"
            )
            self.last_llm_parsed_decision = "MOVE" if choose_move else "STAY"
            self.last_llm_parse_status = "OK"

        if not choose_move:
            return None
        empty_spaces = _empty_cells(grid)
        if not empty_spaces:
            return None
        if keyed:
            u_dest = keyed_uniform(self.run_id, self.step, agent_id, 1)
            return empty_spaces[int(u_dest * len(empty_spaces))]
        return random.choice(empty_spaces)

    def get_context_grid(self, r, c, grid):
        """
        Create a 3x3 neighborhood context string for the LLM prompt.
        
        Parameters:
        -----------
        r : int
            Row position of the agent
        c : int  
            Column position of the agent
        grid : list
            2D grid representing the simulation state
            
        Returns:
        --------
        str
            Formatted context string showing the 3x3 neighborhood with:
            - X: Current agent position (center)
            - S: Same type agent
            - O: Opposite type agent  
            - E: Empty space
            - #: Out of bounds
        """
        # Construct 3x3 neighborhood context
        context = []
        for dr in [-1, 0, 1]:
            row = []
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < cfg.GRID_SIZE and 0 <= nc < cfg.GRID_SIZE:
                    neighbor = grid[nr][nc]
                    if neighbor is None:
                        row.append("E")  # Empty
                    elif neighbor.type_id == self.type_id:
                        row.append("S")  # Same
                    else:
                        row.append("O")  # Opposite
                else:
                    row.append("#")  # Out of bounds
            context.append(row)
        
        # Format context for prompt - mark current position
        context_with_position = []
        for i, row in enumerate(context):
            new_row = []
            for j, cell in enumerate(row):
                if i == 1 and j == 1:  # Center position (current location)
                    new_row.append("X")
                else:
                    new_row.append(cell)
            context_with_position.append(new_row)
        
        return "\n".join([" ".join(row) for row in context_with_position])
    
    def get_llm_decision(self, r, c, grid, max_retries=300):
        """Get movement decision from LLM with retry logic (max_retries attempts)"""
        if self.value_function_policy:
            return self._get_value_function_decision(r, c, grid)

        # Debug flag - set via environment variable
        debug = os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes')
        if self.store_llm_responses:
            self.last_llm_response_raw = None
            self.last_llm_parsed_decision = None
            self.last_llm_parse_status = None
        parse_failures = 0
        
        if debug:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] [DEBUG] LLM Decision Request for agent at ({r},{c})")
            print(f"[DEBUG] Agent type: {self.agent_type} | Scenario: {self.scenario}")
        
        # Get context string using the new method
        context_str = self.get_context_grid(r, c, grid)
        
        # Create prompt using context scenario template
        prompt = self.context_info['prompt_template'].format(
            agent_type=self.agent_type,
            opposite_type=self.opposite_type,
            context=context_str
        )
        
        for attempt in range(max_retries + 1):
            try:
                # Endpoint + payload follow llm_style (build_llm_request):
                #   completions[+grammar]  raw /v1/completions "prompt" payload -- no chat
                #       template, nothing emits EOS, so max_tokens=5 caps run-on (do NOT
                #       add a "\n" stop: a leading newline is a plausible first token and
                #       would truncate to an empty string and burn a retry).
                #   chat[+grammar]         /v1/chat/completions messages payload -- the
                #       server applies the model's chat template.
                # +grammar constrains output to (whitespace + MOVE/STAY in any casing):
                # zero unparseable replies and generation halts at the word boundary.
                # style=None preserves the legacy raw behaviour against llm_url as given.
                request_url, payload = build_llm_request(
                    self.llm_request_url, self.llm_style, self.llm_model,
                    prompt, self.temperature,
                )
                
                headers = {
                    "Authorization": f"Bearer {self.llm_api_key}",
                    "Content-Type": "application/json"
                }
                
                if debug:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] [DEBUG] Sending LLM request to: {request_url} (style={self.llm_style or 'legacy'})")
                    print(f"[DEBUG] Model: {self.llm_model}")
                    print(f"[DEBUG] Context grid:\n{context_str}")
                
                # Track timing and calls
                start_time = time.time()
                response = requests.post(request_url, headers=headers, json=payload, timeout=20)
                response_time = time.time() - start_time
                
                # Update agent's LLM metrics
                self.llm_call_count += 1
                self.llm_call_time += response_time
                
                if debug:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] [DEBUG] LLM Response received in {response_time:.2f}s")
                    print(f"[DEBUG] Status code: {response.status_code}")

                
                response.raise_for_status()
                data = response.json()
                choice0 = data["choices"][0]
                text = choice0.get("text")
                if text is None:
                    text = (choice0.get("message") or {}).get("content", "")
                if self.store_llm_responses:
                    self.last_llm_response_raw = text
                
                if debug:
                    print(f"[DEBUG] LLM Response text: '{text}'")
                    print("[DEBUG] Attempting to parse decision...")
                
                # Parse MOVE/STAY response
                text_upper = text.strip().upper()
                has_move = "MOVE" in text_upper
                has_stay = "STAY" in text_upper

                if has_move and has_stay:
                    if self.store_llm_responses:
                        self.last_llm_parsed_decision = None
                        self.last_llm_parse_status = None
                    parse_failures += 1
                    if debug:
                        print(f"[DEBUG] Ambiguous MOVE/STAY in response: '{text}'")
                        print(f"[DEBUG] Retrying due to parse failure ({parse_failures}/{max_retries})")
                    if attempt < max_retries:
                        continue
                    if self.store_llm_responses:
                        self.last_llm_parse_status = "FAILURE"
                    raise Exception(
                        f"Unable to parse MOVE/STAY after {max_retries + 1} attempts: ambiguous response '{text}'"
                    )

                if has_move:
                    if self.store_llm_responses:
                        self.last_llm_parsed_decision = "MOVE"
                        self.last_llm_parse_status = "OK"
                    if debug:
                        print("[DEBUG] Decision: MOVE - finding random empty space")
                    
                    # Find all empty spaces on the grid
                    empty_spaces = []
                    for row in range(cfg.GRID_SIZE):
                        for col in range(cfg.GRID_SIZE):
                            if grid[row][col] is None:  # Empty space
                                empty_spaces.append((row, col))
                    
                    # Return random empty space if available
                    if empty_spaces:
                        chosen_pos = random.choice(empty_spaces)
                        if debug:
                            print(f"[DEBUG] Moving to random empty position: {chosen_pos}")
                        return chosen_pos
                    else:
                        if debug:
                            print("[DEBUG] No empty spaces available, staying put")
                        return None
                
                elif has_stay:
                    if self.store_llm_responses:
                        self.last_llm_parsed_decision = "STAY"
                        self.last_llm_parse_status = "OK"
                    if debug:
                        print("[DEBUG] Decision: STAY")
                    return None
                
                else:
                    if self.store_llm_responses:
                        self.last_llm_parsed_decision = None
                        self.last_llm_parse_status = None
                    parse_failures += 1
                    if debug:
                        print(f"[DEBUG] Could not parse MOVE/STAY from: '{text}'")
                        print(f"[DEBUG] Retrying due to parse failure ({parse_failures}/{max_retries})")
                    if attempt < max_retries:
                        continue
                    if self.store_llm_responses:
                        self.last_llm_parse_status = "FAILURE"
                    raise Exception(
                        f"Unable to parse MOVE/STAY after {max_retries + 1} attempts: response '{text}'"
                    )
            except requests.exceptions.Timeout:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if attempt < max_retries:
                    print(f"[{timestamp}] [LLM Timeout] Retry {attempt + 1}/{max_retries} for agent at ({r},{c}) [run {self.run_id}, step {self.step}]")
                    time.sleep(5)  # Wait 5 seconds before retry
                    continue
                else:
                    print(f"[{timestamp}] [LLM Error] Max retries exceeded ({max_retries}) for agent at ({r},{c}) [run {self.run_id}, step {self.step}]")
                    raise Exception(f"LLM timeout after {max_retries} retries")
            except Exception as e:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if attempt < max_retries:
                    print(f"[{timestamp}] [LLM Error] Exception - Retry {attempt + 1}/{max_retries}: {e} [run {self.run_id}, step {self.step}]")
                    time.sleep(5)  # Wait 5 seconds before retry
                    continue
                else:
                    print(f"[{timestamp}] [LLM Error] Exception: Unhandled error after {max_retries} retries: {e} [run {self.run_id}, step {self.step}]")
                    raise Exception(f"LLM error after {max_retries} retries: {e}")
                    raise Exception(f"LLM error after {max_retries} retries: {e}")

def llm_decision_function(agent, r, c, grid):
    """Decision function for LLM agents with retry logic (no fallback to mechanical decision)"""
    return agent.get_llm_decision(r, c, grid)

class LLMSimulation(Simulation):
    def __init__(self, run_id, scenario='baseline', llm_model=None, llm_url=None, llm_api_key=None, random_seed=None,
                 initial_int_grid=None, initial_step=None, initial_no_move_steps=None,
                 temperature=None, llm_style=None,
                 value_function_policy=None, value_function_path=None):
        # Store LLM parameters for agent creation
        self.scenario = scenario
        self.llm_model = llm_model or cfg.OLLAMA_MODEL
        self.llm_url = llm_url or cfg.OLLAMA_URL
        self.llm_api_key = llm_api_key or cfg.OLLAMA_API_KEY
        self.llm_style = resolve_llm_style(llm_style)
        if temperature is None:
            raise ValueError("temperature must be provided to LLMSimulation")
        self.temperature = float(temperature)
        self.value_function_policy = value_function_policy or {}
        self.value_function_path = value_function_path

        super().__init__(
            run_id=run_id, 
            agent_factory=self._create_llm_agent, 
            decision_func=llm_decision_function, 
            scenario=scenario,
            random_seed=random_seed,
            initial_int_grid=initial_int_grid,
            initial_step=initial_step,
            initial_no_move_steps=initial_no_move_steps,
            # Live-LLM runs keep one record per decision because that is
            # where the raw LLM reply is stored; value-function runs make no
            # LLM calls and get the per-step format (see run_files).
            full_move_log=not self.value_function_policy,
        )
        
        # Track LLM metrics across all agents
        self.total_llm_calls = 0
        self.total_llm_time = 0.0
    
    def _create_llm_agent(self, type_id):
        """Create LLM agent with simulation parameters"""
        return LLMAgent(
            type_id,
            self.scenario,
            self.llm_model,
            self.llm_url,
            self.llm_api_key,
            self.run_id,
            self.step,
            temperature=self.temperature,
            llm_style=self.llm_style,
            value_function_policy=self.value_function_policy,
            value_function_path=self.value_function_path,
        )

    def _collect_llm_counters(self):
        """Fold every agent's call count / time into the run totals and reset
        them. Once per run, after the loop (2026-09-05): this used to be a
        400-cell scan at the top of every step, and the step/run_id sync it
        also did is redundant — update_agents sets agent.step before each
        decision and run_id is fixed at construction."""
        for agent in self.grid[self.grid != None]:  # noqa: E711
            if hasattr(agent, 'llm_call_count'):
                self.total_llm_calls += agent.llm_call_count
                self.total_llm_time += agent.llm_call_time
                agent.llm_call_count = 0
                agent.llm_call_time = 0.0

    def run_step(self, verbose_move_log: bool = False):
        """Override run_step to track step timing"""
        step_start_time = datetime.now()

        # Call parent run_step
        result = super().run_step(verbose_move_log=verbose_move_log)

        # Add timestamp for step completion
        step_end_time = datetime.now()
        step_duration = (step_end_time - step_start_time).total_seconds()

        # Only print timestamp for longer steps or periodically
        if step_duration > 10: # or (hasattr(self, 'step') and self.step % 10 == 0)
            print(f"[{step_end_time.strftime('%Y-%m-%d %H:%M:%S')}] Step {getattr(self, 'step', '?')} completed in {step_duration:.1f}s [run {self.run_id}]")

        return result

    def run_single_simulation(self, output_dir=None, max_steps=1000, show_progress=False, save_every_steps=None):
        """Override to show progress bar for LLM simulations and add timestamps"""
        start_time = datetime.now()
        live = not self.value_function_policy      # per-run lines only for live-LLM runs
        if live:
            print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting LLM simulation run {self.run_id}")
        
        result = super().run_single_simulation(
            output_dir=output_dir,
            max_steps=max_steps,
            show_progress=show_progress,
            save_every_steps=save_every_steps,
        )
        self._collect_llm_counters()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        if live:
            print(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] Completed LLM simulation run {self.run_id} in {duration:.1f}s")
        
        return result

def run_single_simulation(args):
    """Run a single LLM simulation - compatible with baseline_runner structure.
    Supports optional resume seeds and max_steps.
        Args tuples supported:
            (run_id, scenario, llm_model, llm_url, llm_api_key, output_dir)
            (run_id, scenario, llm_model, llm_url, llm_api_key, output_dir, max_steps)
            (run_id, scenario, llm_model, llm_url, llm_api_key, output_dir, max_steps, initial_int_grid, initial_step)
                (..., initial_no_move_steps, save_every_steps, temperature, llm_style,
             scenario_file, value_function_policy, value_function_path)
    """
    if len(args) < 6:
        raise ValueError(
            f"incomplete worker args: expected at least 6 required values, got {len(args)}"
        )

    run_id, scenario, llm_model, llm_url, llm_api_key, output_dir = args[:6]
    max_steps = args[6] if len(args) >= 7 and args[6] is not None else 1000
    initial_int_grid = args[7] if len(args) >= 8 else None
    initial_step = args[8] if len(args) >= 9 else None
    initial_no_move_steps = args[9] if len(args) >= 10 else None
    save_every_steps = args[10] if len(args) >= 11 else None
    if len(args) < 12:
        raise ValueError(
            f"incomplete worker args: expected 12 values including temperature, got {len(args)}"
        )
    temperature = args[11]
    if temperature is None:
        raise ValueError("temperature argument missing in run_single_simulation worker args")
    llm_style = args[12] if len(args) >= 13 else None
    scenario_file = args[13] if len(args) >= 14 else None
    value_function_policy = args[14] if len(args) >= 15 else None
    value_function_path = args[15] if len(args) >= 16 else None
    grid_config = args[16] if len(args) >= 17 else None
    # Re-apply in the worker for the same reason scenario_file is: required
    # under spawn, harmless (idempotent) under fork.
    apply_grid_config(grid_config)
    if scenario_file:
        # Re-apply in the worker: idempotent, and required under spawn start methods
        # (under fork the parent's apply already covers it).
        apply_scenario_file(scenario_file)

    sim = LLMSimulation(run_id, scenario, llm_model, llm_url, llm_api_key,
                        # Seed by run_id: run k of ANY two batches shares its
                        # initial grid and RNG streams, making full-vs-half
                        # value-function checks and LLM-vs-mechanical
                        # comparisons paired (base_simulation seeds numpy +
                        # stdlib random when a seed is given, 2026-08-21).
                        random_seed=run_id,
                        initial_int_grid=initial_int_grid, initial_step=initial_step,
                        initial_no_move_steps=initial_no_move_steps,
                        temperature=temperature, llm_style=llm_style,
                        value_function_policy=value_function_policy,
                        value_function_path=value_function_path)
    if initial_int_grid is not None:
        sim.preload_record(output_dir)
    result = sim.run_single_simulation(output_dir=output_dir, max_steps=max_steps, save_every_steps=save_every_steps)
    
    # Add LLM-specific metrics to the result
    result.update({
        'scenario': scenario,
        'llm_call_count': sim.total_llm_calls,
        'avg_llm_call_time': sim.total_llm_time / max(sim.total_llm_calls, 1),
        'decision_source': ('value_function' if value_function_policy else 'llm_api'),
        'value_function_path': value_function_path,
    })
    return result

def _analyze_run_status(output_dir, run_id, max_steps):
    """Classify a run by inspecting logs and extract resume seeds if needed.
    Returns dict with keys: status in {'converged','reached_max','aborted','missing'},
    last_step (int or None), next_step (int or None), seed_grid (2D list or None).
    """
    empty = {'status': 'missing', 'last_step': None, 'next_step': None,
             'seed_grid': None, 'convergence_step': None, 'no_move_streak': 0}
    step_log = run_files.load_step_log(output_dir, run_id)
    if step_log is None or step_log.empty:
        # No usable move log; a states file alone means an aborted run whose
        # step count is unknown — seed from its last grid.
        frames = run_files.load_frames(output_dir, run_id)
        if frames is None or len(frames) == 0:
            return empty
        return dict(empty, status='aborted', seed_grid=frames[-1].tolist())

    step_moves = run_files.step_moves(step_log)
    last_step = max(step_moves)
    threshold = getattr(cfg, 'NO_MOVE_THRESHOLD', 5)
    converged, convergence_step, _ = convergence_from_step_moves(step_moves, threshold)
    # Count trailing zero-move steps for resuming aborted runs
    no_move_streak = 0
    for step in sorted(step_moves, reverse=True):
        if step_moves[step] != 0:
            break
        no_move_streak += 1
    reached_max = (last_step + 1) >= max_steps
    status = 'converged' if converged else ('reached_max' if reached_max else 'aborted')

    seed_grid = None
    next_step = None
    if status == 'aborted':
        final = run_files.load_final_grid(output_dir, run_id)
        seed_grid = None if final is None else np.asarray(final).tolist()
        next_step = last_step + 1

    return {
        'status': status,
        'last_step': last_step,
        'next_step': next_step,
        'seed_grid': seed_grid,
        'convergence_step': convergence_step if converged else None,
        'no_move_streak': int(no_move_streak)
    }

def list_available_experiments():
    """List all available experiments that can be resumed"""
    exp_dir = "experiments"
    if not os.path.exists(exp_dir):
        print("No experiments directory found.")
        return []
    
    experiments = []
    for exp_name in os.listdir(exp_dir):
        exp_path = os.path.join(exp_dir, exp_name)
        if os.path.isdir(exp_path):
            # Use the updated check_existing_experiment function
            exists, completed_runs, _, existing_run_ids = check_existing_experiment(exp_name)
            
            # Load config to get total runs
            config_file = os.path.join(exp_path, "config.json")
            total_runs = "unknown"
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        total_runs = config.get('n_runs', 'unknown')
                except Exception:
                    pass
            
            experiments.append({
                'name': exp_name,
                'completed': completed_runs,
                'total': total_runs,
                'path': exp_path
            })
    
    return experiments

def check_existing_experiment(experiment_name):
    """
    Check if an experiment already exists and find completed run IDs
    
    Parameters:
    -----------
    experiment_name : str
        Name of the experiment directory to check
        
    Returns:
    --------
    tuple
        (exists, completed_runs, output_dir, existing_run_ids) where:
        - exists: bool indicating if experiment directory exists
        - completed_runs: int number of completed simulation runs found
        - output_dir: str path to the experiment directory
        - existing_run_ids: set of run IDs that already exist
    """
    output_dir = f"experiments/{experiment_name}"
    
    if not os.path.exists(output_dir):
        return False, 0, output_dir, set()
    
    # Every run with a move log, per-run or packed (run_files knows both
    # layouts); plus the original run_<id>.json.gz result files if any.
    import glob
    import re
    existing_run_ids = set(run_files.list_run_ids(output_dir))
    for pattern in (os.path.join(output_dir, "run_*.json.gz"),
                    os.path.join(output_dir, "states", "states_run_*.npz"),
                    os.path.join(output_dir, "states_run_*.npz")):
        for file_path in glob.glob(pattern):
            match = re.search(r'run_(\d+)', os.path.basename(file_path))
            if match:
                existing_run_ids.add(int(match.group(1)))
    
    completed_runs = len(existing_run_ids)
    return True, completed_runs, output_dir, existing_run_ids

def run_llm_experiment(scenario=None, n_runs=None, max_steps=None, llm_model=None, llm_url=None, llm_api_key=None,
                       parallel=True, n_processes=None, resume_experiment=None,
                       save_every_steps=None, temperature=None,
                       llm_style=None, scenario_file=None, value_function=None,
                       grid_config=None, full_move_log=False, progress_offset=0, progress_total=None):
    """
    Run LLM experiments with specified scenario - compatible with baseline_runner structure
    
    Parameters:
    -----------
    scenario : str or None
        Scenario context to use. When resuming and omitted, falls back to the stored experiment config.
    n_runs : int or None
        Number of simulation runs to perform. When resuming and omitted, uses the stored experiment config.
    max_steps : int or None
        Maximum steps per simulation. When resuming and omitted, uses the stored experiment config.
    llm_model : str or None
        LLM model to use (overrides config.py if provided; otherwise uses config/defaults).
    llm_url : str or None
        LLM API URL (overrides config.py if provided; otherwise uses config/defaults).
    llm_api_key : str or None
        LLM API key to use (overrides config.py if provided; otherwise uses config/defaults).
    parallel : bool
        Whether to use parallel processing
    n_processes : int, optional
        Number of CPU processes to use for parallel execution.
        If None, uses min(cpu_count(), n_runs). If 1, forces sequential execution.
    resume_experiment : str, optional
        Name of an existing experiment to resume (skip runs that are already completed)
    save_every_steps : int or None
        Persist states and move logs every N simulation steps (default: 1 / every step).
        Keeps all details; only changes disk write frequency.
    temperature : float
        Sampling temperature for live LLM API requests.
        
    Returns:
    --------
    tuple
        (output_dir, results) where results contains simulation outcomes
    """
    
    llm_style = resolve_llm_style(llm_style)   # fail fast on typos
    # Applied in the parent as well as in each worker: the config.json snapshot
    # below reads cfg.GRID_SIZE, so without this the run would execute at 20x20
    # while its own provenance recorded the default.
    apply_grid_config(grid_config)
    if full_move_log:
        # Environment rather than an args-tuple slot: inherited by spawned
        # workers, and read by Simulation.__init__ in every runner.
        os.environ['FULL_MOVE_LOG'] = '1'
    resolved_scenario_file = None
    if scenario_file:
        # keep the RESOLVED path: callers may pass a bare name and run from a
        # different cwd, so abspath() here would record a file that isn't there.
        resolved_scenario_file = apply_scenario_file(scenario_file)

    # Handle experiment resumption
    if resume_experiment:
        print(f"Checking for existing experiment: {resume_experiment}")
        exists, completed_runs, output_dir, existing_run_ids = check_existing_experiment(resume_experiment)
        
        if not exists:
            print(f"❌ Experiment '{resume_experiment}' not found in experiments/ directory")
            print("Available experiments:")
            exp_dir = "experiments"
            if os.path.exists(exp_dir):
                for exp in os.listdir(exp_dir):
                    if os.path.isdir(os.path.join(exp_dir, exp)):
                        print(f"  - {exp}")
            return None, []
        
        print(f"✅ Found existing experiment with {completed_runs} completed runs")
        if existing_run_ids:
            print(f"   Existing run IDs: {sorted(existing_run_ids)}")
        
        # Load existing config to match original parameters when CLI does not override them
        config_file = os.path.join(output_dir, "config.json")
        config_defaults_used = []
        config_values_from_file = {}
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                existing_config = json.load(f)
                if scenario is None:
                    scenario_from_config = existing_config.get('scenario')
                    if scenario_from_config:
                        scenario = scenario_from_config
                        config_defaults_used.append('scenario')
                        config_values_from_file['scenario'] = scenario_from_config
                if max_steps is None:
                    max_steps_from_config = existing_config.get('max_steps')
                    if max_steps_from_config is not None:
                        max_steps = max_steps_from_config
                        config_defaults_used.append('max_steps')
                        config_values_from_file['max_steps'] = max_steps_from_config
                if llm_model is None:
                    llm_model_from_config = existing_config.get('llm_model')
                    if llm_model_from_config:
                        llm_model = llm_model_from_config
                        config_defaults_used.append('llm_model')
                        config_values_from_file['llm_model'] = llm_model_from_config
                if llm_url is None:
                    llm_url_from_config = existing_config.get('llm_url')
                    if llm_url_from_config:
                        llm_url = llm_url_from_config
                        config_defaults_used.append('llm_url')
                        config_values_from_file['llm_url'] = llm_url_from_config
                if llm_api_key is None and existing_config.get('llm_api_key_last4'):
                    # API key cannot be reconstructed from last4; fall back to cfg if not provided
                    pass
                if temperature is None:
                    temperature_from_config = existing_config.get('temperature')
                    if temperature_from_config is not None:
                        try:
                            temperature = float(temperature_from_config)
                            config_defaults_used.append('temperature')
                            config_values_from_file['temperature'] = temperature
                        except (TypeError, ValueError):
                            pass
                if n_runs is None:
                    n_runs_from_config = existing_config.get('n_runs')
                    if n_runs_from_config is not None:
                        n_runs = n_runs_from_config
                        config_defaults_used.append('n_runs')
                        config_values_from_file['n_runs'] = n_runs_from_config
                if n_processes is None:
                    n_proc_from_config = existing_config.get('n_processes')
                    if isinstance(n_proc_from_config, int) and n_proc_from_config > 0:
                        n_processes = n_proc_from_config
                        config_defaults_used.append('n_processes')
                        config_values_from_file['n_processes'] = n_proc_from_config
                if value_function is None:
                    vf_from_config = existing_config.get('value_function_file')
                    if isinstance(vf_from_config, str) and vf_from_config.strip():
                        value_function = vf_from_config
                        config_defaults_used.append('value_function')
                        config_values_from_file['value_function'] = vf_from_config
                if save_every_steps is None:
                    save_every_steps_from_config = existing_config.get('save_every_steps')
                    if isinstance(save_every_steps_from_config, int) and save_every_steps_from_config > 0:
                        save_every_steps = save_every_steps_from_config
                        config_defaults_used.append('save_every_steps')
                        config_values_from_file['save_every_steps'] = save_every_steps_from_config

                threshold_from_config = existing_config.get('no_move_threshold')
                if threshold_from_config is not None:
                    try:
                        parsed_threshold = int(threshold_from_config)
                    except (TypeError, ValueError):
                        parsed_threshold = None
                    if parsed_threshold and parsed_threshold > 0:
                        cfg.NO_MOVE_THRESHOLD = parsed_threshold

                # Persist new n_runs value if CLI provided different target count
                orig_runs = existing_config.get('n_runs')
                if orig_runs is not None and n_runs is not None and orig_runs != n_runs:
                    print(f"📝 Updating config n_runs from {orig_runs} to {n_runs} to match CLI")
                    existing_config['n_runs'] = n_runs
                    try:
                        with open(config_file, 'w') as fw:
                            json.dump(existing_config, fw, indent=2)
                    except Exception as e:
                        print(f"Warning: failed to update config.json: {e}")
        else:
            existing_config = {}
            print("⚠️ config.json not found for this experiment; falling back to default settings where necessary.")

        # Apply fallback defaults if config did not supply missing values
        scenario = scenario or 'baseline'
        max_steps = max_steps if (isinstance(max_steps, int) and max_steps > 0) else 1000
        n_runs = n_runs if (isinstance(n_runs, int) and n_runs > 0) else (completed_runs if completed_runs > 0 else 10)
        llm_model = llm_model or cfg.OLLAMA_MODEL
        llm_url = llm_url or cfg.OLLAMA_URL
        llm_api_key = llm_api_key or cfg.OLLAMA_API_KEY
        if temperature is None:
            raise ValueError("temperature must be provided via caller arguments")
        try:
            temperature = float(temperature)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid temperature value: {temperature}") from exc
        if not isinstance(n_processes, int) or n_processes < 1:
            n_processes = None
        # None means NO intermediate saves (base_simulation._normalize_save_every_steps,
        # 2026-08-27) — do NOT coerce it to 1 here. Coercing was silently
        # reinstating the per-step save, whose full-file rewrite makes total I/O
        # O(steps^2): a 1000-step deepseek run took ~30 min instead of 7.5 s.
        if save_every_steps is not None and (
                not isinstance(save_every_steps, int) or save_every_steps < 1):
            save_every_steps = None

        value_map = {
            'scenario': scenario,
            'max_steps': max_steps,
            'llm_model': llm_model,
            'llm_url': llm_url,
            'temperature': temperature,
            'n_runs': n_runs,
            'n_processes': n_processes,
            'save_every_steps': save_every_steps,
        }
        config_defaults_used = [key for key in config_defaults_used if value_map.get(key) == config_values_from_file.get(key)]

        print("📋 Resuming with experiment parameters:")
        print(f"   Scenario: {scenario}{' (from config)' if 'scenario' in config_defaults_used else ''}")
        print(f"   Max steps: {max_steps}{' (from config)' if 'max_steps' in config_defaults_used else ''}")
        if 'n_runs' in config_defaults_used:
            print(f"   Total planned runs: {n_runs} (from config)")
        else:
            print(f"   Total planned runs: {n_runs}")
        if 'llm_model' in config_defaults_used:
            print(f"   LLM model: {llm_model} (from config)")
        if 'llm_url' in config_defaults_used:
            print(f"   LLM URL: {llm_url} (from config)")
        if 'temperature' in config_defaults_used:
            print(f"   Temperature: {temperature} (from config)")
        else:
            print(f"   Temperature: {temperature}")
        if n_processes is not None:
            suffix = " (from config)" if 'n_processes' in config_defaults_used else ""
            print(f"   Parallel processes: {n_processes}{suffix}")
        if 'save_every_steps' in config_defaults_used:
            print(f"   Save every N steps: {save_every_steps} (from config)")
        else:
            print(f"   Save every N steps: {save_every_steps}")
        
        # Plan and classify all target run IDs
        planned_run_ids = list(range(n_runs))
        statuses = {}
        for rid in planned_run_ids:
            statuses[rid] = _analyze_run_status(output_dir, rid, max_steps)

        if value_function:
            # A resumed run reseeds from run_id at the resume point, so it is
            # not the run an uninterrupted process would have produced. Runs
            # driven by a value function cost a fraction of a second: redo
            # them from scratch and keep every run reproducible from run_id.
            for s in statuses.values():
                if s['status'] == 'aborted':
                    s['status'] = 'missing'

        completed_ids = [rid for rid, s in statuses.items() if s['status'] in ('converged', 'reached_max')]
        aborted_items = [(rid, s) for rid, s in statuses.items() if s['status'] == 'aborted']
        missing_run_ids = [rid for rid, s in statuses.items() if s['status'] == 'missing']

        print(f"   Classified runs → completed: {len(completed_ids)}, aborted: {len(aborted_items)}, missing: {len(missing_run_ids)}")
        print(f"     Aborted IDs: {[rid for rid,_ in aborted_items]}, Missing IDs: {missing_run_ids}")

        if len(completed_ids) >= n_runs and not aborted_items and not missing_run_ids:
            print(f"⚠️  Experiment already complete! ({len(completed_ids)}/{n_runs} runs)")
            print("Loading existing results...")
            # Load analysis from stored outputs for accurate results
            out_dir, results, convergence_data = Simulation.load_and_analyze_results(output_dir, force_recompute=False)
            return out_dir, results

        # Prepare args: aborted runs (resume) first, then missing runs (new)
        pending_run_specs = []
        for rid, s in aborted_items:
            seed_grid = s.get('seed_grid')
            next_step = s.get('next_step')
            no_move_streak = s.get('no_move_streak')
            pending_run_specs.append((rid, seed_grid, next_step, no_move_streak))
        if aborted_items:
            print(f"   Will resume aborted run IDs first: {[rid for rid,_ in aborted_items]}")
            for rid, s in aborted_items:
                if s.get('no_move_streak'):
                    print(f"     Run {rid}: trailing no-move steps = {s['no_move_streak']}")
        if missing_run_ids:
            print(f"   Will execute missing run IDs: {missing_run_ids}")
        for rid in missing_run_ids:
            pending_run_specs.append((rid, None, None, None))

        runs_to_execute = len(pending_run_specs)
        remaining_runs = runs_to_execute
    else:
        completed_runs = 0
        existing_run_ids = set()
        scenario = scenario or 'baseline'
        max_steps = max_steps or 1000
        n_runs = n_runs or 10
        llm_model = llm_model or cfg.OLLAMA_MODEL
        llm_url = llm_url or cfg.OLLAMA_URL
        llm_api_key = llm_api_key or cfg.OLLAMA_API_KEY
        if temperature is None:
            raise ValueError("temperature must be provided via caller arguments")
        try:
            temperature = float(temperature)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid temperature value: {temperature}") from exc
        # None means NO intermediate saves (base_simulation._normalize_save_every_steps,
        # 2026-08-27) — do NOT coerce it to 1 here. Coercing was silently
        # reinstating the per-step save, whose full-file rewrite makes total I/O
        # O(steps^2): a 1000-step deepseek run took ~30 min instead of 7.5 s.
        if save_every_steps is not None and (
                not isinstance(save_every_steps, int) or save_every_steps < 1):
            save_every_steps = None
        # Create output directory for new experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"llm_{scenario}_{timestamp}"
        # Value-function batches finish in under a second, so a seconds-resolution
        # timestamp can COLLIDE with the previous batch and silently mix two
        # experiments into one directory. Suffix until unique.
        k = 2
        while os.path.exists(f"experiments/{experiment_name}"):
            experiment_name = f"llm_{scenario}_{timestamp}-{k}"
            k += 1
        output_dir = f"experiments/{experiment_name}"
        os.makedirs(output_dir, exist_ok=True)
        remaining_runs = n_runs
        pending_run_specs = [(i, None, None, None) for i in range(n_runs)]
    
    value_function_policy = None
    value_function_path = None
    vf_meta = None

    if value_function:
        # Composition-keyed value-function policy:
        # no live LLM server needed, full cell coverage validated at load time.
        value_function_policy, value_function_path, vf_meta = \
            load_value_function_policy(value_function, scenario=scenario)
        print(f"✅ Loaded value-function policy from: {value_function_path}")
        print(f"   model={vf_meta.get('model')}  style={vf_meta.get('style')}  "
              f"scenario={vf_meta.get('scenario')}  arm={vf_meta.get('arm')}")
        print(f"   Policy entries: {len(value_function_policy)} "
              f"(45 composition cells x 2 roles)")
        # A value function is measured PER SCENARIO (the identity labels are
        # part of the prompt): simulating scenario X with scenario Y's tables
        # would silently attribute Y's behaviour to X's social context.
        vf_scenario = vf_meta.get("scenario")
        if vf_scenario and scenario and vf_scenario != scenario:
            raise ValueError(
                f"value function was measured on scenario {vf_scenario!r} but this "
                f"experiment simulates {scenario!r} — use the matching "
                f"vf_*__{scenario}__*.json artifact")
    else:
        # Check LLM connection first with potentially custom parameters
        if not check_llm_connection(llm_model, llm_url, llm_api_key, llm_style=llm_style):
            print("\n⚠️  Cannot proceed with LLM experiments - connection check failed!")
            print("Please ensure the LLM server is running and accessible.")
            return None, []

    # Build finalized worker args with optional policy payload
    args_list = [
        (
            rid,
            scenario,
            llm_model,
            llm_url,
            llm_api_key,
            output_dir,
            max_steps,
            seed_grid,
            next_step,
            no_move_streak,
            save_every_steps,
            temperature,
            llm_style,
            scenario_file,
            value_function_policy,
            value_function_path,
            grid_config,
        )
        for rid, seed_grid, next_step, no_move_streak in pending_run_specs
    ]
    
    # Create or update config (for new experiments only)
    if not resume_experiment:
        config_dict = {
            'llm_style': llm_style,
            'scenario_file': resolved_scenario_file,
            'n_runs': n_runs,
            'max_steps': max_steps,
            'grid_size': cfg.GRID_SIZE,
            'num_type_a': cfg.NUM_TYPE_A,
            'num_type_b': cfg.NUM_TYPE_B,
            'scenario': scenario,
            'llm_model': llm_model,
            # In value-function mode no LLM endpoint is ever contacted; the
            # sampled model/URL live in value_function_meta instead. Recording
            # the (irrelevant) config.py defaults here misled downstream
            # tooling into reading VF runs as live-LLM runs.
            'llm_url': None if value_function_policy else llm_url,
            'temperature': temperature,
            'llm_api_key_last4': (None if value_function_policy
                                  else ((llm_api_key)[-4:] if llm_api_key else None)),
            'no_move_threshold': cfg.NO_MOVE_THRESHOLD,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'context_info': CONTEXT_SCENARIOS[scenario],
            'parallel_execution': parallel,
            'n_processes': n_processes if parallel else 1,
            'cpu_count': cpu_count(),
            'value_function_file': value_function_path,
            # Decision-RNG scheme for value-function runs (keyed_uniform doc):
            # 'keyed' since 2026-09-04, 'shared' before. Live-LLM runs draw
            # no decision uniforms, so the field is None for them.
            'rng_scheme': (VF_RNG_SCHEME if value_function_policy else None),
            'value_function_meta': ({k: vf_meta.get(k) for k in
                                     ('label', 'model', 'url', 'style', 'arm',
                                      'scenario', 'temperature', 'grammar_sha256',
                                      'created')}
                                    if vf_meta else None),
            'save_every_steps': save_every_steps,
        }
        
        with open(f"{output_dir}/config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    # Determine number of processes to use
    runs_to_execute = remaining_runs if resume_experiment else n_runs
    if n_processes is None:
        n_processes = min(cpu_count(), runs_to_execute)
    elif n_processes == 1:
        parallel = False  # Force sequential execution if only 1 process requested
    elif n_processes > cpu_count():
        print(f"⚠️  Warning: Requested {n_processes} processes but only {cpu_count()} CPU cores available.")
        print(f"   Using {cpu_count()} processes instead.")
        n_processes = cpu_count()
    elif n_processes > runs_to_execute:
        print(f"⚠️  Warning: Requested {n_processes} processes but only {runs_to_execute} runs to execute.")
        print(f"   Using {runs_to_execute} processes instead.")
        n_processes = runs_to_execute
    elif n_processes < 1:
        print(f"⚠️  Warning: Invalid number of processes ({n_processes}). Using 1 process (sequential).")
        n_processes = 1
        parallel = False

    # Cap client concurrency to the server's actual slot count. The server's `-np`
    # is set from `processes` by the run_*.sh launcher; if for any reason it is
    # below `processes`, sending more concurrent runs than there are slots just
    # queues them. Querying /props keeps the two in sync (fail-open: skip if unavailable).
    server_slots = query_server_slots(llm_url)
    if server_slots and n_processes and n_processes > server_slots:
        print(f"⚠️  Server exposes {server_slots} slot(s) (/props) but {n_processes} processes "
              f"requested — capping to {server_slots} to match.")
        n_processes = server_slots
        if n_processes <= 1:
            parallel = False

    _run_started_at = time.time()
    # A [run-progress] line per completed run for live-LLM runs (minutes
    # each); every PROGRESS_EVERY for value-function runs (tens of ms each,
    # 10k per scenario), and always the last, which is what the watchers
    # read (they take the latest line, watch_progress.sh / watch_campaign_eta.sh).
    progress_every = 1 if not value_function_policy else PROGRESS_EVERY

    def _progress(done, procs):
        if done % progress_every == 0 or done == runs_to_execute:
            print(format_run_progress(scenario, done, runs_to_execute,
                                      time.time() - _run_started_at, procs,
                                      progress_offset, progress_total), flush=True)

    if parallel and n_processes > 1:
        print(f"Running {runs_to_execute} simulations using {n_processes} parallel processes...")
        results = []
        with Pool(n_processes) as pool:
            for _res in tqdm(pool.imap(run_single_simulation, args_list),
                             total=runs_to_execute, desc="Running LLM simulations", ncols=80):
                results.append(_res)
                _progress(len(results), n_processes)
    else:
        print(f"Running {runs_to_execute} simulations sequentially...")
        results = []
        for args in tqdm(args_list, desc="Running LLM simulations", ncols=80):
            results.append(run_single_simulation(args))
            _progress(len(results), 1)

    # Load existing results if resuming
    if resume_experiment and completed_runs > 0:
        print(f"Loading {completed_runs} existing results...")
        # Load existing results and filter out run_ids that were just re-executed
        newly_executed_run_ids = {r['run_id'] for r in results}
        
        existing_results = []
        seen = set()
        for result_file in sorted(glob.glob(os.path.join(output_dir, "run_*.json.gz"))):
            with gzip.open(result_file, 'rt') as f:
                result = json.load(f)
                # Skip if this run was just re-executed (avoids duplicates)
                if result.get('run_id') not in newly_executed_run_ids:
                    existing_results.append(result)
                    seen.add(result.get('run_id'))
        # Every other stored run (per-run files or packed, via run_files) gets
        # a placeholder; analyze_results keeps its real rows from disk.
        for run_id in run_files.list_run_ids(output_dir):
            if run_id in newly_executed_run_ids or run_id in seen:
                continue
            existing_results.append({
                'run_id': run_id,
                'converged': True,  # Assume converged if the record exists
                'convergence_step': None,  # Unknown from the record alone
                'final_step': 'unknown',
                'metrics_history': [],  # analyze_results keeps the stored rows
            })
        
        # Combine existing and new results (no duplicates now)
        all_results = existing_results + results
        total_runs = len(all_results)
        print(f"Combined {len(existing_results)} existing + {len(results)} new = {total_runs} total results")
    else:
        all_results = results
        total_runs = n_runs

    # Analyze results using Simulation's analyze_results method
    output_dir, final_results, convergence_data = Simulation.analyze_results(all_results, output_dir, total_runs)

    # Value-function runs write the per-step record; fold the completed
    # experiment's per-run files into the packed containers (run_files).
    if value_function_policy and run_files.pack_enabled():
        run_files.pack_run_record(output_dir)
    
    print(f"\nExperiment completed. Results saved to: {output_dir}")
    if resume_experiment:
        print(f"Resumed experiment: {completed_runs} existing + {len(results)} new = {total_runs} total runs")
    else:
        print(f"Total runs: {total_runs}")
    print(f"Converged runs: {sum(1 for r in convergence_data if r['converged'])}")
    converged_steps = [r['convergence_step'] for r in convergence_data if r['convergence_step'] is not None]
    if converged_steps:
        print(f"Average convergence step: {np.mean(converged_steps):.2f}")
    
    # Calculate LLM-specific statistics
    llm_calls = [r.get('llm_call_count', 0) for r in final_results if 'llm_call_count' in r]
    llm_times = [r.get('avg_llm_call_time', 0) for r in final_results if 'avg_llm_call_time' in r]
    if llm_calls:
        # "agent calls", not "LLM calls": the counter is incremented once per
        # agent decision, and in value-function mode no endpoint is contacted
        # at all — a 20x20 VF run reports 640 of these having made zero
        # requests. The response time below really is LLM-only (0.000s in VF
        # mode), so the two lines count different things.
        print(f"Average agent calls per run: {np.mean(llm_calls):.1f}")
        print(f"Average LLM response time: {np.mean(llm_times):.3f}s")
    
    return output_dir, final_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-based Schelling segregation simulations")
    parser.add_argument('--runs', type=int, default=None, help='Number of simulation runs')
    parser.add_argument('--max-steps', type=int, default=None, help='Maximum steps per simulation')
    parser.add_argument('--grid-size', type=int, default=None,
                        help='Cells per side (overrides config.GRID_SIZE for this run only)')
    parser.add_argument('--num-type-a', type=int, default=None,
                        help='Type A agent count (overrides config.NUM_TYPE_A)')
    parser.add_argument('--num-type-b', type=int, default=None,
                        help='Type B agent count (overrides config.NUM_TYPE_B)')
    parser.add_argument('--full-move-log', action='store_true',
                        help='Record every agent decision and a grid frame per decision '
                             '(the pre-2026-09-02 format) instead of per-step counts and '
                             'frames. Runs are deterministic per run_id, so this regenerates '
                             'per-move detail for any run exactly.')
    parser.add_argument('--scenario', type=str, default=None, choices=list(CONTEXT_SCENARIOS.keys()), help='Scenario to simulate')
    parser.add_argument('--llm-model', type=str, help='LLM model to use (overrides config.py)')
    parser.add_argument('--llm-url', type=str, help='LLM API URL (overrides config.py)')
    parser.add_argument('--llm-api-key', type=str, help='LLM API key (overrides config.py)')
    parser.add_argument('--temperature', type=float, default=0.3, help='Sampling temperature for live LLM API requests')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--processes', type=int, default=None, 
                       help=f'Number of CPU processes to use (default: min(cpu_count={cpu_count()}, n_runs)). Use 1 for sequential execution.')
    parser.add_argument('--resume', type=str, help='Resume existing experiment by name (e.g., "llm_baseline_20250706_143022")')
    parser.add_argument('--list-experiments', action='store_true', help='List all available experiments that can be resumed')
    parser.add_argument(
        '--value-function',
        type=str,
        default=None,
        help='Path to a vf-1 value-function JSON (prompt_refinement/build_value_function.py), '
             'or a DIRECTORY of them (the per-scenario file is resolved by filename, '
             'vf_<label>__<scenario>__<style>.json; a '
             'missing/ambiguous match is a hard error). Decides MOVE/STAY from the exact '
             '(n_similar, n_occupied) composition-cell P(MOVE) rates instead of live LLM '
             'calls (composition-level lookup).',
    )
    parser.add_argument(
        '--save-every-steps',
        type=int,
        default=None,
        help='Persist states/move logs every N steps (default: 1). Keeps all detail; only write frequency changes.',
    )
    parser.add_argument(
        '--llm-style',
        type=str,
        default=None,
        choices=list(LLM_STYLES),
        help='Request style: completions / completions+grammar / chat / chat+grammar '
             '(default: legacy raw behaviour against --llm-url as given)',
    )
    parser.add_argument(
        '--scenario-file',
        type=str,
        default=None,
        help='Python module defining CONTEXT_SCENARIOS; replaces context_scenarios.py definitions',
    )
    args = parser.parse_args()

    # Handle listing experiments
    if args.list_experiments:
        experiments = list_available_experiments()
        if not experiments:
            print("No experiments found.")
        else:
            print("\nAvailable experiments:")
            print("-" * 80)
            for exp in experiments:
                status = f"{exp['completed']}/{exp['total']}"
                if exp['completed'] == exp['total'] and exp['total'] != 'unknown':
                    status += " (complete)"
                elif exp['total'] != 'unknown' and exp['completed'] < exp['total']:
                    status += " (incomplete - can resume)"
                print(f"{exp['name']:<50} {status}")
            print("-" * 80)
            print("\nTo resume an experiment, use: --resume <experiment_name>")
        exit(0)

    run_llm_experiment(
        scenario=args.scenario,
        n_runs=args.runs,
        max_steps=args.max_steps,
        llm_model=args.llm_model,
        llm_url=args.llm_url,
        llm_api_key=args.llm_api_key,
        temperature=args.temperature,
        parallel=not args.no_parallel,
        n_processes=args.processes,
        resume_experiment=args.resume,
        value_function=args.value_function,
        llm_style=args.llm_style,
        scenario_file=args.scenario_file,
        grid_config=grid_config_from_args(args.grid_size, args.num_type_a, args.num_type_b),
        full_move_log=args.full_move_log,
        save_every_steps=args.save_every_steps,
    )