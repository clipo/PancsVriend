"""Llama.cpp branchwise MOVE/STAY probability extraction.

Scenario-only pipeline that:
- iterates valid Schelling neighborhood contexts for selected scenario(s)
- computes MOVE/STAY probabilities using branchwise following-token traces
- stores all traces in one SQLite database per run/output directory
- writes downstream-compatible CSV artifacts consumed by llm_runner.load_log_prob_policy

Output files per scenario:
- <model_slug>_<scenario_slug>_stay_move_probability_split.csv
- <model_slug>_<scenario_slug>_stay_move_probability_split_summary.csv

Optional debug artifact (disabled by default):
- <model_slug>_<scenario_slug>_token_probabilities.csv

Progress manifest:
- <model_slug>_per_scenario_progress.json

Trace database:
- trace_store.sqlite
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from llama_cpp import Llama

try:
    from context_scenarios import CONTEXT_SCENARIOS
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from context_scenarios import CONTEXT_SCENARIOS


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHELLING_GRID_SIZE = 10
NON_WALL_CONTEXT_ELEMENTS = ["S", "O", "E"]
NEIGHBOR_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.3
    max_tokens: int = 16
    n_ctx: int = 256
    top_k: int = 0
    top_p: float = 0.9999
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass(frozen=True)
class BranchingEstimatorConfig:
    beam_width: int = 16
    candidate_top_n: int = 1
    candidate_top_n_cap: int = 2048
    min_step_retained_mass: float = 0.999
    early_stop_move_stay_mass: float = 0.999


@dataclass
class TokenRecord:
    sample_index: int
    token_index: int
    token: str
    logprob: float
    probability: float
    top_rank: int
    top_token: str
    top_logprob: float
    top_probability: float


@dataclass(frozen=True)
class TraceStoreConfig:
    db_filename: str = "trace_store.sqlite"
    commit_every: int = 100
    full_logits_dtype: str = "float16"
    full_logits_compression_level: int = 6


GENERATION_CONFIG = GenerationConfig()
BRANCHING_ESTIMATOR_CONFIG = BranchingEstimatorConfig()
TRACE_STORE_CONFIG = TraceStoreConfig()

llm: Llama | None = None
model_file: Path | None = None


@dataclass
class RuntimeProfile:
    enabled: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def add_time(self, key: str, seconds: float) -> None:
        if not self.enabled:
            return
        self.timings[key] = float(self.timings.get(key, 0.0) + max(0.0, float(seconds)))

    def add_count(self, key: str, value: int = 1) -> None:
        if not self.enabled:
            return
        self.counters[key] = int(self.counters.get(key, 0) + int(value))

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "timings_seconds": {k: float(v) for k, v in sorted(self.timings.items())},
            "counters": {k: int(v) for k, v in sorted(self.counters.items())},
        }


@contextlib.contextmanager
def _profile_time(profile: RuntimeProfile | None, key: str):
    if profile is None or not profile.enabled:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        profile.add_time(key, time.perf_counter() - start)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Llama.cpp branchwise token probability extractor")
    parser.add_argument(
        "--llm-model",
        type=str,
        required=True,
        help="Model label used for output naming (kept for workflow compatibility)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to GGUF model file for llama.cpp",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        default="all",
        help="Scenario key from context_scenarios.py, or 'all' to run every scenario",
    )
    parser.add_argument(
        "--agent-role",
        type=str,
        choices=["type_a", "type_b", "both"],
        default="both",
        help="Which scenario agent role(s) to evaluate",
    )
    parser.add_argument(
        "--repeats-per-context",
        type=int,
        default=1,
        help="Repeated evaluations per neighborhood arrangement",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=GENERATION_CONFIG.temperature,
        help="Temperature for branchwise probability computation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=GENERATION_CONFIG.max_tokens,
        help="Maximum generated-token horizon for branchwise expansion",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=GENERATION_CONFIG.n_ctx,
        help="llama.cpp context window",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=BRANCHING_ESTIMATOR_CONFIG.beam_width,
        help="Beam width for branchwise state pruning",
    )
    parser.add_argument(
        "--candidate-top-n",
        type=int,
        default=BRANCHING_ESTIMATOR_CONFIG.candidate_top_n,
        help="Minimum number of top tokens expanded per state",
    )
    parser.add_argument(
        "--candidate-top-n-cap",
        type=int,
        default=BRANCHING_ESTIMATOR_CONFIG.candidate_top_n_cap,
        help="Maximum number of top tokens expanded per state",
    )
    parser.add_argument(
        "--min-step-retained-mass",
        type=float,
        default=BRANCHING_ESTIMATOR_CONFIG.min_step_retained_mass,
        help="Minimum retained candidate probability mass per state",
    )
    parser.add_argument(
        "--early-stop-move-stay-mass",
        type=float,
        default=BRANCHING_ESTIMATOR_CONFIG.early_stop_move_stay_mass,
        help="Early stop threshold for resolved MOVE+STAY mass",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root directory override; writes to <output-dir>/<model_slug> unless already namespaced",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume scenario runs by skipping completed scenarios from manifest",
    )
    parser.add_argument(
        "--write-token-probabilities-csv",
        action="store_true",
        help="Write optional token-probabilities CSV debug artifact",
    )
    parser.add_argument(
        "--store-full-logits",
        action="store_true",
        help="Store compressed full-vocabulary logits in trace payloads for exact multi-temperature replay",
    )
    parser.add_argument(
        "--full-logits-dtype",
        type=str,
        choices=["float16", "float32"],
        default=TRACE_STORE_CONFIG.full_logits_dtype,
        help="Data type used to store full logits before compression",
    )
    parser.add_argument(
        "--full-logits-compression-level",
        type=int,
        default=TRACE_STORE_CONFIG.full_logits_compression_level,
        help="zlib compression level for full-logits payloads (0-9)",
    )
    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        help="Enable lightweight stage timing and profiling summary output",
    )
    parser.add_argument(
        "--profiling-context-limit",
        type=int,
        default=10,
        help="When profiling is enabled, cap neighborhood permutations to this many contexts",
    )
    parser.add_argument(
        "--db-commit-every",
        type=int,
        default=TRACE_STORE_CONFIG.commit_every,
        help="Commit SQLite writes every N processed requests",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=None,
        help="Optional llama.cpp CPU thread count override",
    )
    parser.add_argument(
        "--n-batch",
        type=int,
        default=None,
        help="Optional llama.cpp batch size override",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Optional llama.cpp GPU offload layer count override",
    )
    parser.add_argument(
        "--enable-kv-state-reuse",
        action="store_true",
        help=(
            "Enable prompt-prefill KV state reuse for branch eval: prefill prompt once, "
            "restore state per branch, and eval only generated suffix tokens"
        ),
    )
    parser.add_argument(
        "--step-heartbeat-every",
        type=int,
        default=0,
        help=(
            "Emit heartbeat logs every N branch steps inside a request (0 disables). "
            "Useful when request-level progress appears stalled"
        ),
    )
    parser.add_argument(
        "--step-heartbeat-min-seconds",
        type=float,
        default=10.0,
        help="Minimum elapsed seconds between heartbeat logs when enabled",
    )
    args, ignored_args = parser.parse_known_args()
    setattr(args, "ignored_cli_args", ignored_args)
    return args


def _warn_ignored_args(ignored_args: list[str]) -> None:
    if len(ignored_args) == 0:
        return
    print(
        "[warning] Ignoring unsupported arguments for llama.cpp branchwise mode: "
        + " ".join(ignored_args),
        flush=True,
    )


def _warn_kv_reuse_runtime_risk(args: argparse.Namespace) -> None:
    if not bool(args.enable_kv_state_reuse):
        return

    risky_beam = int(args.beam_width) >= 16
    risky_horizon = int(args.max_tokens) >= 6
    risky_mass = float(args.min_step_retained_mass) >= 0.995

    if not (risky_beam or risky_horizon or risky_mass):
        return

    print(
        (
            "[warning] KV state reuse is enabled with branch-heavy settings. "
            "On this workload, repeated state load/copy can be slower than full re-eval. "
            "If throughput regresses, disable --enable-kv-state-reuse or reduce "
            "--beam-width / --max-tokens / --min-step-retained-mass."
        ),
        flush=True,
    )


def _sanitize_model_for_path_component(name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '-', str(name).strip())
    sanitized = sanitized.rstrip(' .')
    sanitized = re.sub(r'-{2,}', '-', sanitized)
    return sanitized or "unknown-model"


def _resolve_model_output_dir(model: str) -> str:
    model_slug = _sanitize_model_for_path_component(model)
    return str(REPO_ROOT / "llm_log_probs" / model_slug)


def _resolve_output_dir_for_model(model: str, output_dir_arg: str | None) -> str:
    model_slug = _sanitize_model_for_path_component(model)
    if isinstance(output_dir_arg, str) and output_dir_arg.strip():
        provided_root = Path(output_dir_arg.strip())
        if provided_root.name == model_slug:
            return str(provided_root)
        return str(provided_root / model_slug)
    return _resolve_model_output_dir(model)


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_per_scenario_progress_manifest(
    output_dir: str,
    model: str,
    per_scenario_outputs: dict[str, dict[str, Any]],
    run_parameters: dict[str, Any] | None = None,
) -> str:
    ensure_directory(output_dir)
    model_slug = _sanitize_model_for_path_component(model)
    manifest_path = os.path.join(output_dir, f"{model_slug}_per_scenario_progress.json")
    payload = {
        "model": model,
        "completed_scenarios": sorted(per_scenario_outputs.keys()),
        "outputs": per_scenario_outputs,
        "run_parameters": run_parameters or {},
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return manifest_path


def _read_per_scenario_progress_manifest(
    output_dir: str,
    model: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    model_slug = _sanitize_model_for_path_component(model)
    manifest_path = os.path.join(output_dir, f"{model_slug}_per_scenario_progress.json")
    if not os.path.exists(manifest_path):
        return {}, {}

    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}, {}

    run_parameters = payload.get("run_parameters", {})
    validated_params: dict[str, Any] = run_parameters if isinstance(run_parameters, dict) else {}

    outputs = payload.get("outputs", {})
    if isinstance(outputs, dict):
        validated: dict[str, dict[str, Any]] = {}
        for scenario_key, value in outputs.items():
            if isinstance(scenario_key, str) and isinstance(value, dict):
                validated[scenario_key] = {
                    k: v for k, v in value.items() if isinstance(k, str) and (isinstance(v, str) or v is None)
                }
        return validated, validated_params

    return {}, validated_params


def generate_neighbor_context(neighbors: list[str]) -> str:
    if len(neighbors) != 8:
        raise ValueError(f"Expected 8 neighbors, got {len(neighbors)}")

    grid_rows: list[list[str]] = []
    idx = 0
    for row_idx in range(3):
        row: list[str] = []
        for col_idx in range(3):
            if row_idx == 1 and col_idx == 1:
                row.append("X")
            else:
                row.append(neighbors[idx])
                idx += 1
        grid_rows.append(row)

    return "\n".join(" ".join(row) for row in grid_rows)


def _wall_mask_for_position(row: int, col: int, grid_size: int) -> tuple[bool, ...]:
    mask: list[bool] = []
    for dr, dc in NEIGHBOR_OFFSETS:
        nr = row + dr
        nc = col + dc
        mask.append(not (0 <= nr < grid_size and 0 <= nc < grid_size))
    return tuple(mask)


def _all_valid_wall_masks(grid_size: int) -> list[tuple[bool, ...]]:
    unique_masks: set[tuple[bool, ...]] = set()
    for row in range(grid_size):
        for col in range(grid_size):
            boundary_walls = int(row == 0) + int(row == grid_size - 1) + int(col == 0) + int(col == grid_size - 1)
            if boundary_walls > 2:
                continue
            unique_masks.add(_wall_mask_for_position(row, col, grid_size))
    return sorted(unique_masks)


def generate_all_valid_schelling_neighbors(grid_size: int = SCHELLING_GRID_SIZE) -> list[list[str]]:
    arrangements: list[list[str]] = []
    for wall_mask in _all_valid_wall_masks(grid_size):
        open_indices = [idx for idx, is_wall in enumerate(wall_mask) if not is_wall]
        for open_values in product(NON_WALL_CONTEXT_ELEMENTS, repeat=len(open_indices)):
            neighbors = ["#" if wall_mask[idx] else "" for idx in range(8)]
            for slot_idx, symbol in zip(open_indices, open_values):
                neighbors[slot_idx] = symbol
            arrangements.append(neighbors)
    return arrangements


def build_scenario_prompt(scenario_key: str, context: str, agent_label: str, opposite_label: str) -> str:
    scenario = CONTEXT_SCENARIOS[scenario_key]
    return str(scenario["prompt_template"]).format(
        agent_type=agent_label,
        opposite_type=opposite_label,
        context=context,
    )


def parse_decision(raw_response: str) -> str:
    text_upper = raw_response.strip().upper()
    has_move = "MOVE" in text_upper
    has_stay = "STAY" in text_upper
    if has_move and has_stay:
        return "UNKNOWN"
    if has_move:
        return "MOVE"
    if has_stay:
        return "STAY"
    return "UNKNOWN"


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _stable_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits_scaled = logits / max(float(temperature), 1e-6)
    exp_logits = np.exp(logits_scaled - np.max(logits_scaled))
    return exp_logits / np.sum(exp_logits)


def _adaptive_top_indices(
    probs: np.ndarray,
    min_retained_mass: float,
    min_top_n: int,
    top_n_cap: int,
) -> np.ndarray:
    # Use iterative partial selection to avoid full-vocabulary O(V log V) sorts.
    vocab_size = int(len(probs))
    max_n = int(max(1, min(top_n_cap, vocab_size)))
    start_n = int(max(1, min(min_top_n, max_n)))
    target_mass = float(min_retained_mass)

    k = start_n
    best_sorted: np.ndarray | None = None
    while True:
        candidate = np.argpartition(probs, -k)[-k:]
        # Only sort selected head for deterministic descending order.
        sorted_candidate = candidate[np.argsort(probs[candidate])[::-1]]
        retained_mass = float(np.sum(probs[sorted_candidate]))
        best_sorted = sorted_candidate

        if retained_mass >= target_mass or k >= max_n:
            break
        k = int(min(max_n, max(k + 1, k * 2)))

    if best_sorted is None:
        return np.array([], dtype=int)
    return best_sorted


def _sort_indices_by_probability_desc(probs: np.ndarray, indices: set[int]) -> list[int]:
    return sorted((int(idx) for idx in indices), key=lambda idx: float(probs[idx]), reverse=True)


def _normalize_compression_level(level: int) -> int:
    return int(max(0, min(9, int(level))))


def _compress_full_logits(raw_logits: np.ndarray, dtype: str, compression_level: int) -> dict[str, Any]:
    dtype_name = "float16" if str(dtype).lower() != "float32" else "float32"
    target_dtype = np.float16 if dtype_name == "float16" else np.float32
    logits = np.asarray(raw_logits, dtype=target_dtype)
    logits_bytes = logits.tobytes(order="C")
    level = _normalize_compression_level(compression_level)
    compressed_bytes = zlib.compress(logits_bytes, level=level)
    return {
        "encoding": "zlib+base64",
        "dtype": dtype_name,
        "vocab_size": int(logits.shape[0]),
        "compression_level": int(level),
        "compressed_b64": base64.b64encode(compressed_bytes).decode("ascii"),
        "compressed_bytes": int(len(compressed_bytes)),
        "uncompressed_bytes": int(len(logits_bytes)),
    }


def _detokenize_single_token_cached(model: Llama, token_id: int, cache: dict[int, str]) -> str:
    token_int = int(token_id)
    cached = cache.get(token_int)
    if cached is not None:
        return cached
    token_text = model.detokenize([token_int]).decode("utf-8", errors="replace")
    cache[token_int] = token_text
    return token_text


def _init_llama(
    model_path: str,
    temperature: float,
    n_ctx: int,
    n_threads: int | None,
    n_batch: int | None,
    n_gpu_layers: int | None,
) -> None:
    global llm
    global model_file
    model_file = Path(model_path).expanduser().resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    llama_kwargs: dict[str, Any] = {
        "model_path": str(model_file),
        "logits_all": True,
        "verbose": False,
        "temperature": float(temperature),
        "n_ctx": int(n_ctx),
    }
    if n_threads is not None and int(n_threads) > 0:
        llama_kwargs["n_threads"] = int(n_threads)
    if n_batch is not None and int(n_batch) > 0:
        llama_kwargs["n_batch"] = int(n_batch)
    if n_gpu_layers is not None:
        llama_kwargs["n_gpu_layers"] = int(n_gpu_layers)

    llm = Llama(**llama_kwargs)


def _require_llm() -> Llama:
    if llm is None:
        raise RuntimeError("llama.cpp model is not initialized")
    return llm


def _safe_log(probability: float) -> float:
    if probability <= 0.0:
        return float("-inf")
    return float(math.log(probability))


def _model_supports_state_io(model: Llama) -> bool:
    save_state_fn = getattr(model, "save_state", None)
    load_state_fn = getattr(model, "load_state", None)
    return callable(save_state_fn) and callable(load_state_fn)


def capture_following_token_logit_trace(
    prompt: str,
    prompt_tokens: list[int] | None,
    temperature: float,
    max_tokens: int,
    config: BranchingEstimatorConfig,
    store_full_logits: bool,
    full_logits_dtype: str,
    full_logits_compression_level: int,
    enable_kv_state_reuse: bool,
    step_heartbeat_every: int,
    step_heartbeat_min_seconds: float,
    heartbeat_context_label: str,
    profile: RuntimeProfile | None,
) -> dict[str, Any]:
    model = _require_llm()
    with _profile_time(profile, "tokenize_prompt_seconds"):
        effective_prompt_tokens = (
            list(int(t) for t in prompt_tokens)
            if prompt_tokens is not None
            else list(model.tokenize(prompt.encode("utf-8")))
        )
    quality_flags: list[str] = []
    steps: list[dict[str, Any]] = []
    single_token_text_cache: dict[int, str] = {}

    kv_reuse_enabled = bool(enable_kv_state_reuse)
    kv_reuse_active = False
    prompt_state_blob: Any = None
    prompt_prefill_logits: np.ndarray | None = None

    if kv_reuse_enabled:
        if not _model_supports_state_io(model):
            kv_reuse_enabled = False
            quality_flags.append("kv_reuse_state_io_unsupported")
        else:
            save_state_fn = getattr(model, "save_state")
            try:
                with _profile_time(profile, "kv_prompt_prefill_seconds"):
                    model.reset()
                    model.eval(effective_prompt_tokens)
                if model.scores is not None and len(model.scores) > 0:
                    prompt_prefill_logits = np.asarray(model.scores[len(effective_prompt_tokens) - 1], dtype=float)
                with _profile_time(profile, "kv_save_prompt_state_seconds"):
                    prompt_state_blob = save_state_fn()
                kv_reuse_active = prompt_state_blob is not None
                if profile is not None and profile.enabled and kv_reuse_active:
                    profile.add_count("kv_prompt_state_saved", 1)
            except Exception:
                kv_reuse_active = False
                prompt_state_blob = None
                prompt_prefill_logits = None
                quality_flags.append("kv_reuse_prompt_prefill_failed")

    if profile is not None and profile.enabled:
        profile.add_count("trace_calls", 1)

    active_states: list[tuple[tuple[int, ...], float]] = [(tuple(), 1.0)]
    move_mass = 0.0
    stay_mass = 0.0
    unknown_mass = 0.0
    last_heartbeat_time = 0.0

    for step_idx in range(1, int(max_tokens) + 1):
        if len(active_states) == 0:
            break

        next_state_masses: dict[tuple[int, ...], float] = {}
        step_states: list[dict[str, Any]] = []

        for generated_ids, state_mass in active_states:
            if state_mass <= 0.0:
                continue

            with _profile_time(profile, "detokenize_prefix_seconds"):
                generated_prefix_text = model.detokenize(list(generated_ids)).decode("utf-8", errors="replace")
            prefix_decision = parse_decision(generated_prefix_text)

            state_record: dict[str, Any] = {
                "generated_token_ids": [int(t) for t in generated_ids],
                "generated_prefix_text": generated_prefix_text,
                "state_mass": float(state_mass),
                "prefix_decision": prefix_decision,
                "candidates": [],
            }

            if prefix_decision == "MOVE":
                move_mass += state_mass
                state_record["resolved_mass"] = float(state_mass)
                step_states.append(state_record)
                continue
            if prefix_decision == "STAY":
                stay_mass += state_mass
                state_record["resolved_mass"] = float(state_mass)
                step_states.append(state_record)
                continue

            raw_logits: np.ndarray | None = None
            if kv_reuse_active and prompt_state_blob is not None:
                try:
                    if len(generated_ids) == 0 and prompt_prefill_logits is not None:
                        raw_logits = prompt_prefill_logits
                    else:
                        load_state_fn = getattr(model, "load_state")
                        with _profile_time(profile, "kv_load_prompt_state_seconds"):
                            model.reset()
                            load_state_fn(prompt_state_blob)
                        with _profile_time(profile, "llama_eval_suffix_only_seconds"):
                            model.eval(list(generated_ids))
                        if model.scores is not None and len(model.scores) > 0:
                            raw_logits = np.asarray(model.scores[len(generated_ids) - 1], dtype=float)
                except Exception:
                    quality_flags.append(f"step_{step_idx}_kv_reuse_fallback")
                    raw_logits = None

            if raw_logits is None:
                eval_tokens = effective_prompt_tokens + list(generated_ids)
                with _profile_time(profile, "llama_eval_seconds"):
                    model.reset()
                    model.eval(eval_tokens)
                if model.scores is None or len(model.scores) == 0:
                    unknown_mass += state_mass
                    state_record["failure"] = "missing_scores"
                    step_states.append(state_record)
                    quality_flags.append(f"step_{step_idx}_missing_scores")
                    continue
                raw_logits = np.asarray(model.scores[len(eval_tokens) - 1], dtype=float)

            with _profile_time(profile, "softmax_seconds"):
                probs = _stable_softmax(raw_logits, temperature=temperature)

            if bool(store_full_logits):
                with _profile_time(profile, "compress_full_logits_seconds"):
                    state_record["full_logits"] = _compress_full_logits(
                        raw_logits=raw_logits,
                        dtype=full_logits_dtype,
                        compression_level=full_logits_compression_level,
                    )

            candidate_ids = _adaptive_top_indices(
                probs=probs,
                min_retained_mass=float(config.min_step_retained_mass),
                min_top_n=int(config.candidate_top_n),
                top_n_cap=int(config.candidate_top_n_cap),
            )
            retained_mass = float(np.sum(probs[candidate_ids]))
            tail_mass = max(0.0, 1.0 - retained_mass)
            if tail_mass > 0.0:
                unknown_mass += state_mass * tail_mass
                state_record["unresolved_tail_mass"] = float(tail_mass)
                if tail_mass > (1.0 - float(config.min_step_retained_mass) + 1e-12):
                    quality_flags.append(f"step_{step_idx}_low_retained_mass")

            candidate_rows: list[dict[str, Any]] = []
            with _profile_time(profile, "sort_candidates_seconds"):
                sorted_candidate_ids = [int(i) for i in candidate_ids]

            for token_id in sorted_candidate_ids:
                token_prob = float(probs[token_id])
                branch_mass = state_mass * token_prob
                with _profile_time(profile, "detokenize_candidate_seconds"):
                    token_text = _detokenize_single_token_cached(model, int(token_id), single_token_text_cache)
                decision = parse_decision(generated_prefix_text + token_text)

                candidate_rows.append(
                    {
                        "token_id": int(token_id),
                        "token_text": token_text,
                        "raw_logit": float(raw_logits[token_id]),
                        "token_probability": token_prob,
                        "next_decision": decision,
                        "branch_mass": float(branch_mass),
                    }
                )

                if decision == "MOVE":
                    move_mass += branch_mass
                elif decision == "STAY":
                    stay_mass += branch_mass
                else:
                    next_key = tuple(list(generated_ids) + [int(token_id)])
                    next_state_masses[next_key] = float(next_state_masses.get(next_key, 0.0) + branch_mass)

            state_record["candidates"] = candidate_rows
            step_states.append(state_record)

        sorted_states = sorted(next_state_masses.items(), key=lambda item: item[1], reverse=True)
        beam_cutoff = int(max(1, config.beam_width))
        pruned_mass = float(sum(mass for _, mass in sorted_states[beam_cutoff:]))
        if pruned_mass > 0.0:
            unknown_mass += pruned_mass
            quality_flags.append(f"step_{step_idx}_beam_pruned_mass")

        active_states = sorted_states[:beam_cutoff]

        steps.append(
            {
                "step_index": int(step_idx),
                "num_active_states_in": int(len(step_states)),
                "num_next_states_before_prune": int(len(sorted_states)),
                "num_next_states_after_prune": int(len(active_states)),
                "pruned_beam_mass": float(pruned_mass),
                "states": step_states,
            }
        )

        resolved_mass = float(move_mass + stay_mass)
        if resolved_mass >= float(config.early_stop_move_stay_mass):
            quality_flags.append(f"step_{step_idx}_early_stop")
            break

        heartbeat_every = int(max(0, step_heartbeat_every))
        if heartbeat_every > 0 and (step_idx % heartbeat_every == 0):
            now = time.time()
            min_seconds = max(0.0, float(step_heartbeat_min_seconds))
            if (now - last_heartbeat_time) >= min_seconds:
                print(
                    (
                        f"[heartbeat][{heartbeat_context_label}] "
                        f"step={step_idx}/{int(max_tokens)} "
                        f"active={len(active_states)} "
                        f"move_mass={move_mass:.6f} "
                        f"stay_mass={stay_mass:.6f} "
                        f"unknown_mass={unknown_mass:.6f}"
                    ),
                    flush=True,
                )
                last_heartbeat_time = now

    if len(active_states) > 0:
        unknown_mass += float(sum(mass for _, mass in active_states))

    total_mass = float(move_mass + stay_mass + unknown_mass)
    if total_mass <= 0.0:
        move_probability = 0.0
        stay_probability = 0.0
        unknown_probability = 1.0
    else:
        move_probability = float(move_mass / total_mass)
        stay_probability = float(stay_mass / total_mass)
        unknown_probability = float(unknown_mass / total_mass)

    parseable_mass = move_probability + stay_probability
    move_probability_parseable = float(move_probability / parseable_mass) if parseable_mass > 0 else 0.0
    stay_probability_parseable = float(stay_probability / parseable_mass) if parseable_mass > 0 else 0.0

    parsed_decision = "UNKNOWN"
    if move_probability > stay_probability and move_probability > unknown_probability:
        parsed_decision = "MOVE"
    elif stay_probability > move_probability and stay_probability > unknown_probability:
        parsed_decision = "STAY"

    # Emit a final heartbeat snapshot so long requests always show resolved end-state mass.
    if int(max(0, step_heartbeat_every)) > 0:
        print(
            (
                f"[heartbeat-final][{heartbeat_context_label}] "
                f"steps={len(steps)}/{int(max_tokens)} "
                f"move_mass={move_mass:.6f} "
                f"stay_mass={stay_mass:.6f} "
                f"unknown_mass={unknown_mass:.6f} "
                f"move_prob={move_probability:.6f} "
                f"stay_prob={stay_probability:.6f} "
                f"unknown_prob={unknown_probability:.6f} "
                f"parsed={parsed_decision}"
            ),
            flush=True,
        )

    return {
        "schema_version": "3.0" if store_full_logits else "2.1",
        "payload_mode": "full_logits_compressed" if store_full_logits else "candidate_only",
        "capture_mode": "branchwise_following_token",
        "prompt_hash": _prompt_hash(prompt),
        "prompt": prompt,
        "prompt_token_ids": [int(t) for t in effective_prompt_tokens],
        "model_path": str(model_file) if model_file is not None else "",
        "generation_config": {
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "n_ctx": int(GENERATION_CONFIG.n_ctx),
            "top_k": int(GENERATION_CONFIG.top_k),
            "top_p": float(GENERATION_CONFIG.top_p),
            "min_p": float(GENERATION_CONFIG.min_p),
            "repeat_penalty": float(GENERATION_CONFIG.repeat_penalty),
            "frequency_penalty": float(GENERATION_CONFIG.frequency_penalty),
            "presence_penalty": float(GENERATION_CONFIG.presence_penalty),
        },
        "branching_config": {
            "beam_width": int(config.beam_width),
            "candidate_top_n": int(config.candidate_top_n),
            "candidate_top_n_cap": int(config.candidate_top_n_cap),
            "min_step_retained_mass": float(config.min_step_retained_mass),
            "early_stop_move_stay_mass": float(config.early_stop_move_stay_mass),
        },
        "seed_policy": "deterministic_branching_expansion",
        "num_steps": int(len(steps)),
        "steps": steps,
        "final_mass": {
            "move_mass": float(move_mass),
            "stay_mass": float(stay_mass),
            "unknown_mass": float(unknown_mass),
            "total_mass": float(total_mass),
            "move_probability": float(move_probability),
            "stay_probability": float(stay_probability),
            "unknown_probability": float(unknown_probability),
            "move_probability_parseable": float(move_probability_parseable),
            "stay_probability_parseable": float(stay_probability_parseable),
        },
        "parsed_decision": parsed_decision,
        "trace_quality_flags": quality_flags,
    }


def _trace_candidates_first_step(trace_payload: dict[str, Any]) -> list[dict[str, Any]]:
    steps = trace_payload.get("steps", [])
    if not isinstance(steps, list) or len(steps) == 0:
        return []
    first_step = steps[0]
    states = first_step.get("states", [])
    if not isinstance(states, list) or len(states) == 0:
        return []
    first_state = states[0]
    candidates = first_state.get("candidates", [])
    if not isinstance(candidates, list):
        return []
    return candidates


def _token_records_from_trace(trace_payload: dict[str, Any], sample_index: int) -> list[TokenRecord]:
    candidates = _trace_candidates_first_step(trace_payload)
    sorted_candidates = sorted(
        candidates,
        key=lambda row: float(row.get("token_probability", 0.0)),
        reverse=True,
    )

    rows: list[TokenRecord] = []
    for rank, row in enumerate(sorted_candidates):
        prob = float(row.get("token_probability", 0.0))
        token_text = str(row.get("token_text", ""))
        rows.append(
            TokenRecord(
                sample_index=int(sample_index),
                token_index=0,
                token=token_text,
                logprob=_safe_log(prob),
                probability=float(prob),
                top_rank=int(rank),
                top_token=token_text,
                top_logprob=_safe_log(prob),
                top_probability=float(prob),
            )
        )
    return rows


def _trace_db_path(output_dir: str) -> Path:
    return Path(output_dir) / TRACE_STORE_CONFIG.db_filename


def _init_trace_db(output_dir: str) -> tuple[sqlite3.Connection, Path]:
    db_path = _trace_db_path(output_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trace_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_label TEXT NOT NULL,
            scenario TEXT NOT NULL,
            agent_role TEXT NOT NULL,
            arrangement_code TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            repeat_index INTEGER NOT NULL,
            parsed_decision TEXT NOT NULL,
            trace_quality_flags TEXT,
            move_probability REAL NOT NULL,
            stay_probability REAL NOT NULL,
            unknown_probability REAL NOT NULL,
            move_probability_parseable REAL NOT NULL,
            stay_probability_parseable REAL NOT NULL,
            payload_sha256 TEXT NOT NULL,
            payload_uncompressed_bytes INTEGER NOT NULL,
            payload_compressed_bytes INTEGER NOT NULL,
            payload_json_zlib BLOB NOT NULL,
            trace_schema_version TEXT,
            trace_payload_mode TEXT,
            capture_temperature REAL,
            created_at_utc TEXT NOT NULL,
            UNIQUE(scenario, agent_role, arrangement_code, sample_index)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_trace_records_lookup
        ON trace_records (scenario, agent_role, arrangement_code, sample_index)
        """
    )
    _ensure_trace_db_columns(conn)
    conn.commit()
    return conn, db_path


def _ensure_trace_db_columns(conn: sqlite3.Connection) -> None:
    existing_cols = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(trace_records)")
    }
    required_cols: dict[str, str] = {
        "trace_schema_version": "TEXT",
        "trace_payload_mode": "TEXT",
        "capture_temperature": "REAL",
    }
    for col_name, col_type in required_cols.items():
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE trace_records ADD COLUMN {col_name} {col_type}")


def _store_trace_record(
    conn: sqlite3.Connection,
    *,
    model_label: str,
    scenario: str,
    agent_role: str,
    arrangement_code: str,
    sample_index: int,
    repeat_index: int,
    trace_payload: dict[str, Any],
) -> tuple[int, str]:
    final_mass = trace_payload.get("final_mass", {})
    quality_flags = trace_payload.get("trace_quality_flags", [])
    quality_flags_text = "|".join(str(x) for x in quality_flags)

    payload_text = json.dumps(trace_payload, ensure_ascii=False, separators=(",", ":"))
    payload_bytes = payload_text.encode("utf-8")
    payload_compressed = zlib.compress(payload_bytes, level=6)
    payload_sha = hashlib.sha256(payload_bytes).hexdigest()

    cursor = conn.execute(
        """
        INSERT INTO trace_records (
            model_label,
            scenario,
            agent_role,
            arrangement_code,
            sample_index,
            repeat_index,
            parsed_decision,
            trace_quality_flags,
            move_probability,
            stay_probability,
            unknown_probability,
            move_probability_parseable,
            stay_probability_parseable,
            payload_sha256,
            payload_uncompressed_bytes,
            payload_compressed_bytes,
            payload_json_zlib,
            trace_schema_version,
            trace_payload_mode,
            capture_temperature,
            created_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(scenario, agent_role, arrangement_code, sample_index) DO UPDATE SET
            model_label=excluded.model_label,
            repeat_index=excluded.repeat_index,
            parsed_decision=excluded.parsed_decision,
            trace_quality_flags=excluded.trace_quality_flags,
            move_probability=excluded.move_probability,
            stay_probability=excluded.stay_probability,
            unknown_probability=excluded.unknown_probability,
            move_probability_parseable=excluded.move_probability_parseable,
            stay_probability_parseable=excluded.stay_probability_parseable,
            payload_sha256=excluded.payload_sha256,
            payload_uncompressed_bytes=excluded.payload_uncompressed_bytes,
            payload_compressed_bytes=excluded.payload_compressed_bytes,
            payload_json_zlib=excluded.payload_json_zlib,
            trace_schema_version=excluded.trace_schema_version,
            trace_payload_mode=excluded.trace_payload_mode,
            capture_temperature=excluded.capture_temperature,
            created_at_utc=excluded.created_at_utc
        """,
        (
            str(model_label),
            str(scenario),
            str(agent_role),
            str(arrangement_code),
            int(sample_index),
            int(repeat_index),
            str(trace_payload.get("parsed_decision", "UNKNOWN")),
            quality_flags_text,
            float(final_mass.get("move_probability", 0.0)),
            float(final_mass.get("stay_probability", 0.0)),
            float(final_mass.get("unknown_probability", 1.0)),
            float(final_mass.get("move_probability_parseable", 0.0)),
            float(final_mass.get("stay_probability_parseable", 0.0)),
            payload_sha,
            int(len(payload_bytes)),
            int(len(payload_compressed)),
            payload_compressed,
            str(trace_payload.get("schema_version", "")),
            str(trace_payload.get("payload_mode", "")),
            float(trace_payload.get("generation_config", {}).get("temperature", 0.0)),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    row_id_raw = cursor.lastrowid
    row_id = int(row_id_raw) if row_id_raw is not None else -1
    return row_id, payload_sha


def _write_profile_report(output_dir: str, model: str, profile: RuntimeProfile) -> str:
    ensure_directory(output_dir)
    model_slug = _sanitize_model_for_path_component(model)
    profile_path = os.path.join(output_dir, f"{model_slug}_profiling_summary.json")
    with open(profile_path, "w", encoding="utf-8") as handle:
        json.dump(profile.as_dict(), handle, indent=2)
    return profile_path


def summarize_label_probability_split(label_split_df: pd.DataFrame) -> pd.DataFrame:
    if len(label_split_df) == 0:
        return pd.DataFrame(
            columns=[
                "scenario",
                "agent_role",
                "arrangement_code",
                "num_trials",
                "mean_stay_probability",
                "mean_move_probability",
                "mean_total_labeled_probability",
                "mean_stay_share",
                "mean_move_share",
            ]
        )

    group_cols = [
        "scenario",
        "agent_role",
        "arrangement_code",
    ]

    summary_df = (
        label_split_df.groupby(group_cols, as_index=False)
        .agg(
            num_trials=("sample_index", "count"),
            mean_stay_probability=("stay_probability", "mean"),
            mean_move_probability=("move_probability", "mean"),
            mean_total_labeled_probability=("total_labeled_probability", "mean"),
            mean_stay_share=("stay_share", "mean"),
            mean_move_share=("move_share", "mean"),
        )
        .sort_values(group_cols)
    )

    return summary_df


def save_outputs(
    output_dir: str,
    model: str,
    scenario: str,
    records: list[TokenRecord],
    label_split_df: pd.DataFrame,
    label_split_summary_df: pd.DataFrame,
    write_token_probabilities_csv: bool,
) -> tuple[str | None, str, str]:
    ensure_directory(output_dir)
    model_slug = _sanitize_model_for_path_component(model)
    scenario_slug = _sanitize_model_for_path_component(scenario)

    tokens_csv: str | None = None
    if write_token_probabilities_csv:
        tokens_df = pd.DataFrame([r.__dict__ for r in records])
        tokens_csv = os.path.join(output_dir, f"{model_slug}_{scenario_slug}_token_probabilities.csv")
        tokens_df.to_csv(tokens_csv, index=False)

    label_split_csv = os.path.join(output_dir, f"{model_slug}_{scenario_slug}_stay_move_probability_split.csv")
    label_split_df.to_csv(label_split_csv, index=False)

    label_split_summary_csv = os.path.join(
        output_dir,
        f"{model_slug}_{scenario_slug}_stay_move_probability_split_summary.csv",
    )
    label_split_summary_df.to_csv(label_split_summary_csv, index=False)

    return tokens_csv, label_split_csv, label_split_summary_csv


def _validate_repeats(args: argparse.Namespace) -> int:
    repeats_int = int(args.repeats_per_context)
    if repeats_int < 1:
        raise ValueError("--repeats-per-context must be >= 1")
    return repeats_int


def _resolve_scenarios(raw_scenario: str) -> list[str]:
    if raw_scenario.lower() == "all":
        return sorted(CONTEXT_SCENARIOS.keys())
    if raw_scenario not in CONTEXT_SCENARIOS:
        available = ", ".join(sorted(CONTEXT_SCENARIOS.keys()))
        raise ValueError(f"Unknown --scenario '{raw_scenario}'. Available: {available}, or 'all'")
    return [raw_scenario]


def _role_configs(scenario_key: str, agent_role: str) -> list[tuple[str, str, str]]:
    scenario_info = CONTEXT_SCENARIOS[scenario_key]
    configs: list[tuple[str, str, str]] = []
    if agent_role in {"type_a", "both"}:
        configs.append(("type_a", str(scenario_info["type_a"]), str(scenario_info["type_b"])))
    if agent_role in {"type_b", "both"}:
        configs.append(("type_b", str(scenario_info["type_b"]), str(scenario_info["type_a"])))
    return configs


def evaluate_scenario_permutations(
    scenario_key: str,
    agent_role: str,
    repeats_per_context: int,
    model_label: str,
    trace_db_conn: sqlite3.Connection,
    generation_temperature: float,
    max_tokens: int,
    branching_config: BranchingEstimatorConfig,
    store_full_logits: bool,
    full_logits_dtype: str,
    full_logits_compression_level: int,
    enable_kv_state_reuse: bool,
    step_heartbeat_every: int,
    step_heartbeat_min_seconds: float,
    profile: RuntimeProfile | None,
    profiling_context_limit: int,
    db_commit_every: int,
) -> tuple[list[TokenRecord], pd.DataFrame]:
    if repeats_per_context < 1:
        raise ValueError("--repeats-per-context must be >= 1")

    role_configs = _role_configs(scenario_key, agent_role)
    with _profile_time(profile, "generate_neighbor_arrangements_seconds"):
        all_neighbor_arrangements = generate_all_valid_schelling_neighbors(SCHELLING_GRID_SIZE)

    if profile is not None and profile.enabled:
        cap = int(max(1, profiling_context_limit))
        if len(all_neighbor_arrangements) > cap:
            print(
                (
                    f"[profiling][scenario={scenario_key}] limiting contexts "
                    f"from {len(all_neighbor_arrangements)} to {cap}."
                ),
                flush=True,
            )
            all_neighbor_arrangements = all_neighbor_arrangements[:cap]

    all_records: list[TokenRecord] = []
    context_rows: list[dict[str, Any]] = []

    total_requests = len(role_configs) * len(all_neighbor_arrangements) * repeats_per_context
    processed_requests = 0
    global_sample_idx = 0
    progress_start_time = time.time()

    print(
        (
            f"[progress] Starting scenario '{scenario_key}' with {len(role_configs)} role(s), "
            f"{len(all_neighbor_arrangements)} contexts, {repeats_per_context} repeat(s) each "
            f"({total_requests} total requests)."
        ),
        flush=True,
    )

    for role_name, role_label, opposite_label in role_configs:
        for arrangement_idx, neighbors in enumerate(all_neighbor_arrangements):
            with _profile_time(profile, "build_prompt_seconds"):
                context_grid = generate_neighbor_context(neighbors)
                prompt = build_scenario_prompt(scenario_key, context_grid, role_label, opposite_label)
            with _profile_time(profile, "tokenize_prompt_seconds"):
                prompt_token_ids = list(_require_llm().tokenize(prompt.encode("utf-8")))
            arrangement_code = "".join(neighbors)

            for repeat_index in range(repeats_per_context):
                sample_index = int(global_sample_idx)
                global_sample_idx += 1

                with _profile_time(profile, "trace_capture_seconds"):
                    trace_payload = capture_following_token_logit_trace(
                        prompt=prompt,
                        prompt_tokens=prompt_token_ids,
                        temperature=float(generation_temperature),
                        max_tokens=int(max_tokens),
                        config=branching_config,
                        store_full_logits=bool(store_full_logits),
                        full_logits_dtype=str(full_logits_dtype),
                        full_logits_compression_level=int(full_logits_compression_level),
                        enable_kv_state_reuse=bool(enable_kv_state_reuse),
                        step_heartbeat_every=int(step_heartbeat_every),
                        step_heartbeat_min_seconds=float(step_heartbeat_min_seconds),
                        heartbeat_context_label=(
                            f"scenario={scenario_key};role={role_name};arr={arrangement_idx + 1};"
                            f"sample={sample_index}"
                        ),
                        profile=profile,
                    )

                with _profile_time(profile, "store_trace_seconds"):
                    trace_row_id, trace_payload_sha = _store_trace_record(
                        trace_db_conn,
                        model_label=model_label,
                        scenario=scenario_key,
                        agent_role=role_name,
                        arrangement_code=arrangement_code,
                        sample_index=sample_index,
                        repeat_index=repeat_index,
                        trace_payload=trace_payload,
                    )

                final_mass = trace_payload.get("final_mass", {})
                stay_probability = float(final_mass.get("stay_probability", 0.0))
                move_probability = float(final_mass.get("move_probability", 0.0))
                stay_share = float(final_mass.get("stay_probability_parseable", 0.0))
                move_share = float(final_mass.get("move_probability_parseable", 0.0))
                total_labeled_probability = float(stay_probability + move_probability)
                quality_flags = trace_payload.get("trace_quality_flags", [])
                quality_flags_text = "|".join(str(x) for x in quality_flags)

                context_rows.append(
                    {
                        "scenario": scenario_key,
                        "agent_role": role_name,
                        "arrangement_code": arrangement_code,
                        "context": context_grid,
                        "repeat_index": repeat_index,
                        "sample_index": sample_index,
                        "stay_probability": stay_probability,
                        "move_probability": move_probability,
                        "total_labeled_probability": total_labeled_probability,
                        "stay_share": stay_share,
                        "move_share": move_share,
                        "trace_quality_flags": quality_flags_text,
                        "trace_row_id": int(trace_row_id),
                        "trace_payload_sha256": trace_payload_sha,
                    }
                )

                with _profile_time(profile, "token_records_from_trace_seconds"):
                    all_records.extend(_token_records_from_trace(trace_payload, sample_index=sample_index))

                processed_requests += 1
                if processed_requests % int(max(1, db_commit_every)) == 0:
                    with _profile_time(profile, "db_commit_seconds"):
                        trace_db_conn.commit()
                if processed_requests % 1 == 0 or processed_requests == total_requests:
                    now = time.time()
                    elapsed = now - progress_start_time
                    ratio = processed_requests / total_requests if total_requests > 0 else 1.0
                    rate = processed_requests / elapsed if elapsed > 0 else 0.0
                    eta = (total_requests - processed_requests) / rate if rate > 0 else float("inf")
                    eta_display = f"{eta:.1f}s" if math.isfinite(eta) else "unknown"
                    print(
                        (
                            f"[progress][scenario={scenario_key}] {processed_requests}/{total_requests} "
                            f"({ratio * 100:.2f}%) | elapsed={elapsed:.1f}s | eta={eta_display}"
                        ),
                        flush=True,
                    )

    context_rows.sort(key=lambda row: int(row.get("sample_index", -1)))
    all_records.sort(key=lambda rec: (rec.sample_index, rec.token_index, rec.top_rank))
    with _profile_time(profile, "db_commit_seconds"):
        trace_db_conn.commit()
    return all_records, pd.DataFrame(context_rows)


def main() -> None:
    args = parse_args()
    _warn_ignored_args(getattr(args, "ignored_cli_args", []))

    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be >= 1")

    repeats_per_context = _validate_repeats(args)
    scenario_keys = _resolve_scenarios(args.scenario)
    if int(args.db_commit_every) < 1:
        raise ValueError("--db-commit-every must be >= 1")
    if int(args.profiling_context_limit) < 1:
        raise ValueError("--profiling-context-limit must be >= 1")
    if int(args.step_heartbeat_every) < 0:
        raise ValueError("--step-heartbeat-every must be >= 0")
    if float(args.step_heartbeat_min_seconds) < 0.0:
        raise ValueError("--step-heartbeat-min-seconds must be >= 0")

    _warn_kv_reuse_runtime_risk(args)

    runtime_profile = RuntimeProfile(enabled=bool(args.enable_profiling))

    _init_llama(
        model_path=args.model_path,
        temperature=float(args.temperature),
        n_ctx=int(args.n_ctx),
        n_threads=args.n_threads,
        n_batch=args.n_batch,
        n_gpu_layers=args.n_gpu_layers,
    )

    branching_config = BranchingEstimatorConfig(
        beam_width=int(args.beam_width),
        candidate_top_n=int(args.candidate_top_n),
        candidate_top_n_cap=int(args.candidate_top_n_cap),
        min_step_retained_mass=float(args.min_step_retained_mass),
        early_stop_move_stay_mass=float(args.early_stop_move_stay_mass),
    )

    model = str(args.llm_model)
    output_dir = _resolve_output_dir_for_model(model, args.output_dir)
    ensure_directory(output_dir)
    trace_db_conn, trace_db_path = _init_trace_db(output_dir)

    current_run_parameters: dict[str, Any] = {
        "llm_model": model,
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "scenario_request": args.scenario,
        "effective_agent_role": args.agent_role,
        "repeats_per_context": repeats_per_context,
        "temperature": float(args.temperature),
        "max_tokens": int(args.max_tokens),
        "n_ctx": int(args.n_ctx),
        "beam_width": int(args.beam_width),
        "candidate_top_n": int(args.candidate_top_n),
        "candidate_top_n_cap": int(args.candidate_top_n_cap),
        "min_step_retained_mass": float(args.min_step_retained_mass),
        "early_stop_move_stay_mass": float(args.early_stop_move_stay_mass),
        "output_dir": output_dir,
        "trace_db_path": str(trace_db_path),
        "grid_size": SCHELLING_GRID_SIZE,
        "non_wall_context_elements": NON_WALL_CONTEXT_ELEMENTS,
        "mode": "llama_cpp_branchwise",
        "store_full_logits": bool(args.store_full_logits),
        "full_logits_dtype": str(args.full_logits_dtype),
        "full_logits_compression_level": int(_normalize_compression_level(args.full_logits_compression_level)),
        "profiling_enabled": bool(args.enable_profiling),
        "profiling_context_limit": int(args.profiling_context_limit),
        "db_commit_every": int(args.db_commit_every),
        "n_threads": args.n_threads,
        "n_batch": args.n_batch,
        "n_gpu_layers": args.n_gpu_layers,
        "enable_kv_state_reuse": bool(args.enable_kv_state_reuse),
        "step_heartbeat_every": int(args.step_heartbeat_every),
        "step_heartbeat_min_seconds": float(args.step_heartbeat_min_seconds),
    }

    resume_outputs: dict[str, dict[str, str]] = {}
    if args.resume:
        resume_outputs, _ = _read_per_scenario_progress_manifest(output_dir, model)
        if len(resume_outputs) > 0:
            print(
                f"[progress] Resume enabled: found {len(resume_outputs)} completed scenario(s) in manifest.",
                flush=True,
            )
        scenario_keys = [scenario_key for scenario_key in scenario_keys if scenario_key not in resume_outputs]
        if len(scenario_keys) == 0:
            result = {
                "success": True,
                "model": model,
                "scenario": args.scenario,
                "agent_role": args.agent_role,
                "grid_size": SCHELLING_GRID_SIZE,
                "non_wall_context_elements": NON_WALL_CONTEXT_ELEMENTS,
                "repeats_per_context": repeats_per_context,
                "outputs": {"per_scenario": resume_outputs},
                "resume": True,
                "message": "All requested scenarios are already completed.",
            }
            print(json.dumps(result, indent=2))
            return

    per_scenario_outputs: dict[str, dict[str, Any]] = dict(resume_outputs)
    total_records: list[TokenRecord] = []

    try:
        for scenario_key in scenario_keys:
            scenario_records, scenario_split_df = evaluate_scenario_permutations(
                scenario_key=scenario_key,
                agent_role=args.agent_role if args.scenario.lower() != "all" else "both",
                repeats_per_context=repeats_per_context,
                model_label=model,
                trace_db_conn=trace_db_conn,
                generation_temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                branching_config=branching_config,
                store_full_logits=bool(args.store_full_logits),
                full_logits_dtype=str(args.full_logits_dtype),
                full_logits_compression_level=int(_normalize_compression_level(args.full_logits_compression_level)),
                enable_kv_state_reuse=bool(args.enable_kv_state_reuse),
                step_heartbeat_every=int(args.step_heartbeat_every),
                step_heartbeat_min_seconds=float(args.step_heartbeat_min_seconds),
                profile=runtime_profile,
                profiling_context_limit=int(args.profiling_context_limit),
                db_commit_every=int(args.db_commit_every),
            )

            scenario_summary_df = summarize_label_probability_split(scenario_split_df)
            tokens_csv, label_split_csv, label_split_summary_csv = save_outputs(
                output_dir=output_dir,
                model=model,
                scenario=scenario_key,
                records=scenario_records,
                label_split_df=scenario_split_df,
                label_split_summary_df=scenario_summary_df,
                write_token_probabilities_csv=bool(args.write_token_probabilities_csv),
            )

            per_scenario_outputs[scenario_key] = {
                "token_csv": tokens_csv,
                "stay_move_split_csv": label_split_csv,
                "stay_move_split_summary_csv": label_split_summary_csv,
                "trace_db": str(trace_db_path),
            }
            total_records.extend(scenario_records)

            manifest_path = _write_per_scenario_progress_manifest(
                output_dir=output_dir,
                model=model,
                per_scenario_outputs=per_scenario_outputs,
                run_parameters=current_run_parameters,
            )
            print(
                f"[progress][scenario={scenario_key}] Saved outputs and updated manifest: {manifest_path}",
                flush=True,
            )
    finally:
        trace_db_conn.commit()
        trace_db_conn.close()

    if len(total_records) == 0:
        raise RuntimeError("No token probability records were extracted")

    outputs_payload: dict[str, Any] = {"per_scenario": per_scenario_outputs}
    if len(scenario_keys) == 1:
        only_key = scenario_keys[0]
        outputs_payload.update(per_scenario_outputs[only_key])

    profile_report_path: str | None = None
    if runtime_profile.enabled:
        profile_report_path = _write_profile_report(output_dir=output_dir, model=model, profile=runtime_profile)
        top_timings = sorted(runtime_profile.timings.items(), key=lambda item: item[1], reverse=True)
        print("[profiling] Top timing buckets (seconds):", flush=True)
        for key, value in top_timings[:12]:
            print(f"[profiling] {key}: {value:.6f}", flush=True)
        if profile_report_path:
            print(f"[profiling] Saved profiling summary: {profile_report_path}", flush=True)

    result = {
        "success": True,
        "model": model,
        "url": "llama.cpp",
        "trace_db": str(trace_db_path),
        "scenario": args.scenario,
        "agent_role": args.agent_role if args.scenario.lower() != "all" else "both",
        "grid_size": SCHELLING_GRID_SIZE,
        "non_wall_context_elements": NON_WALL_CONTEXT_ELEMENTS,
        "repeats_per_context": repeats_per_context,
        "prompt": "scenario_mode_only",
        "outputs": outputs_payload,
        "mode": "llama_cpp_branchwise",
        "profiling": runtime_profile.as_dict() if runtime_profile.enabled else {"enabled": False},
        "profiling_report": profile_report_path,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
