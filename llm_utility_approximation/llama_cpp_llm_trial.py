import json
import csv
import time
import os
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any
import numpy as np
from llama_cpp import Llama
from dataclasses import dataclass, asdict


PROMPT_TEMPLATE = """You are a red team resident living in a neighborhood, considering whether to move to a different house.

{context}

Where:
- X = Your current position (center)
- S = neighbors who are also red team residents like you
- O = neighbors from the blue team resident community
- E = empty houses you could move to
- # = area outside the neighborhood

As a real person, you have your own preferences about neighborhood composition, but you also consider practical factors like proximity to friends, community ties, and comfort level.

Based on this neighborhood, do you want to move to a different position or stay where you are?

IMPORTANT: Respond with ONLY one word: MOVE or STAY. Do not provide any explanation, reasoning, or additional text.

Your response:
"""

PROMPT_TEXT = PROMPT_TEMPLATE.format(
    context="# O E\n# X S\n# O S"
)

TOTAL_NEIGHBORS = 8


@dataclass(frozen=True)
class GenerationConfig:
    """Shared generation settings used by both sampling and replay paths."""

    temperature: float = 0.3
    max_tokens: int = 8
    n_ctx: int = 256
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass(frozen=True)
class LogitTraceConfig:
    """Storage and quality settings for per-step following-token traces."""

    storage_mode: str = "top_logits"  # full_logits or top_logits
    top_n: int = 200
    top_n_cap: int = 4000
    min_step_retained_mass: float = 0.995
    fallback_to_full_logits_on_insufficient_mass: bool = True


@dataclass(frozen=True)
class BranchingEstimatorConfig:
    """Controls for branch-aware following-token probability estimation."""

    beam_width: int = 48
    candidate_top_n: int = 512
    candidate_top_n_cap: int = 4096
    min_step_retained_mass: float = 0.02


GENERATION_CONFIG = GenerationConfig()
LOGIT_TRACE_CONFIG = LogitTraceConfig()
BRANCHING_ESTIMATOR_CONFIG = BranchingEstimatorConfig()

_DECISION_HINT_TOKEN_IDS: set[int] | None = None
_TOKEN_DECISION_CATEGORY_CACHE: dict[int, str] | None = None

# DEFAULT_MODEL_PATH = Path(r"C:\Users\Sriki\.ollama\models")
# MODEL: "gemma3:4b"

# Load the model from the model path
# model_file = DEFAULT_MODEL_PATH / "gemma3:4b" / "model.gguf"
model_file = Path       (r"C:\Users\Sriki\PancsVriend\llms\gemma-3-4b-it-q4_0.gguf")
llm = Llama(
    model_path=str(model_file),
    logits_all=True,
    verbose=False,
    temperature=GENERATION_CONFIG.temperature,
    n_ctx=GENERATION_CONFIG.n_ctx,
)
ORGANIZED_RESPONSE_PATH = Path(__file__).with_name("llama_cpp_response_organized.txt")
ALL_LOGITS_PATH = Path(__file__).with_name("llama_cpp_all_logits_first_token.jsonl")
SAMPLES_TABLE_PATH = Path(__file__).with_name("llama_cpp_prompt_samples_table.csv")
PROBABILITY_SUMMARY_PATH = Path(__file__).with_name("llama_cpp_move_stay_probability.txt")
LOGITS_PROBABILITY_SUMMARY_PATH = Path(__file__).with_name(
    "llama_cpp_logits_move_stay_probability.txt"
)
BATCH_OUTPUT_DIR = Path(__file__).resolve().parent / "llama_cpp_neighborhood_trials"
TRACE_OUTPUT_DIR = BATCH_OUTPUT_DIR / "logit_traces"


_WORKER_LLM: Llama | None = None


def _init_sampling_worker(worker_model_path: str) -> None:
    """Initialize a per-process llama-cpp model instance for parallel sampling."""
    global _WORKER_LLM
    _WORKER_LLM = Llama(
        model_path=worker_model_path,
        logits_all=False,
        verbose=False,
        temperature=GENERATION_CONFIG.temperature,
        n_ctx=GENERATION_CONFIG.n_ctx,
    )


def _parallel_sample_once(
    sample_index: int,
    prompt: str,
    temperature: float,
) -> tuple[int, str, str]:
    """Worker task: run one prompt sample and return parsed decision."""
    global _WORKER_LLM
    if _WORKER_LLM is None:
        raise RuntimeError("Worker model is not initialized")

    response = _WORKER_LLM(
        prompt,
        max_tokens=GENERATION_CONFIG.max_tokens,
        temperature=temperature,
        top_k=GENERATION_CONFIG.top_k,
        top_p=GENERATION_CONFIG.top_p,
        min_p=GENERATION_CONFIG.min_p,
        repeat_penalty=GENERATION_CONFIG.repeat_penalty,
        frequency_penalty=GENERATION_CONFIG.frequency_penalty,
        presence_penalty=GENERATION_CONFIG.presence_penalty,
    )
    raw_text = extract_response_text(response)
    decision = parse_decision(raw_text)
    return sample_index, raw_text, decision


def _stable_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled softmax with numerical stability."""
    logits_scaled = logits / max(float(temperature), 1e-6)
    exp_logits = np.exp(logits_scaled - np.max(logits_scaled))
    return exp_logits / np.sum(exp_logits)


def _top_indices_by_probability(probs: np.ndarray, top_n: int) -> np.ndarray:
    """Return token indices for the top-N probabilities in descending order."""
    n = int(max(1, min(top_n, len(probs))))
    candidate = np.argpartition(probs, -n)[-n:]
    return candidate[np.argsort(probs[candidate])[::-1]]


def _adaptive_top_indices(
    probs: np.ndarray,
    min_retained_mass: float,
    min_top_n: int,
    top_n_cap: int,
) -> np.ndarray:
    """Pick enough top indices to meet retained mass threshold up to a cap."""
    max_n = int(max(1, min(top_n_cap, len(probs))))
    start_n = int(max(1, min(min_top_n, max_n)))
    ordered = np.argsort(probs)[::-1]

    cumulative = np.cumsum(probs[ordered])
    threshold_idx = int(np.searchsorted(cumulative, min_retained_mass, side="left")) + 1
    required_n = int(max(start_n, min(threshold_idx, max_n)))
    return ordered[:required_n]


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def capture_following_token_logit_trace(
    prompt: str,
    temperature: float = GENERATION_CONFIG.temperature,
    max_tokens: int = GENERATION_CONFIG.max_tokens,
    trace_config: LogitTraceConfig = LOGIT_TRACE_CONFIG,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Capture per-step following-token logits conditioned on realized prefixes.

    This uses one shared parser (`parse_decision`) and the same max token horizon
    used by empirical sampling so replay and sampling are directly comparable.
    """
    if trace_config.storage_mode not in {"full_logits", "top_logits"}:
        raise ValueError("trace_config.storage_mode must be 'full_logits' or 'top_logits'")

    local_rng = rng or np.random.default_rng()
    prompt_tokens = llm.tokenize(prompt.encode("utf-8"))
    llm.reset()
    llm.eval(prompt_tokens)

    generated_token_ids: list[int] = []
    steps: list[dict[str, Any]] = []
    trace_quality_flags: list[str] = []

    for step_index in range(1, int(max_tokens) + 1):
        if llm.scores is None or len(llm.scores) == 0:
            raise RuntimeError("Could not read logits from llama-cpp during trace capture")

        score_index = len(prompt_tokens) + len(generated_token_ids) - 1
        if score_index < 0 or score_index >= len(llm.scores):
            raise RuntimeError("Could not locate logits row for current trace step")
        raw_logits = llm.scores[score_index]
        probs = _stable_softmax(raw_logits, temperature=temperature)
        top_indices = _adaptive_top_indices(
            probs,
            min_retained_mass=trace_config.min_step_retained_mass,
            min_top_n=trace_config.top_n,
            top_n_cap=trace_config.top_n_cap,
        )
        retained_mass = float(np.sum(probs[top_indices]))
        unresolved_tail_mass = float(max(0.0, 1.0 - retained_mass))
        effective_storage_mode = trace_config.storage_mode

        if retained_mass < trace_config.min_step_retained_mass:
            trace_quality_flags.append(
                f"step_{step_index}_retained_mass_below_threshold:{retained_mass:.6f}"
            )
            if (
                trace_config.storage_mode == "top_logits"
                and trace_config.fallback_to_full_logits_on_insufficient_mass
            ):
                effective_storage_mode = "full_logits"
                unresolved_tail_mass = 0.0
                trace_quality_flags.append(
                    f"step_{step_index}_fallback_to_full_logits"
                )

        sampled_token_id = int(local_rng.choice(len(probs), p=probs))
        sampled_token_text = llm.detokenize([sampled_token_id]).decode("utf-8", errors="replace")

        step_record: dict[str, Any] = {
            "step_index": step_index,
            "prefix_token_ids": [int(t) for t in (prompt_tokens + generated_token_ids)],
            "generated_prefix_token_ids": [int(t) for t in generated_token_ids],
            "generated_prefix_text": llm.detokenize(generated_token_ids).decode(
                "utf-8", errors="replace"
            ),
            "sampled_token_id": sampled_token_id,
            "sampled_token_text": sampled_token_text,
            "retained_mass": retained_mass,
            "unresolved_tail_mass": unresolved_tail_mass,
            "trace_storage_mode": effective_storage_mode,
        }

        if effective_storage_mode == "full_logits":
            step_record["full_logits"] = [float(x) for x in raw_logits]
        else:
            top_entries: list[dict[str, Any]] = []
            for token_id in top_indices:
                token_int = int(token_id)
                token_text = llm.detokenize([token_int]).decode("utf-8", errors="replace")
                top_entries.append(
                    {
                        "token_id": token_int,
                        "token_text": token_text,
                        "logit": float(raw_logits[token_int]),
                        "probability": float(probs[token_int]),
                    }
                )
            step_record["top_logits"] = top_entries

        steps.append(step_record)

        generated_token_ids.append(sampled_token_id)
        llm.eval([sampled_token_id])

        generated_text = llm.detokenize(generated_token_ids).decode("utf-8", errors="replace")
        decision = parse_decision(generated_text)
        if decision in {"MOVE", "STAY"}:
            break

    final_text = llm.detokenize(generated_token_ids).decode("utf-8", errors="replace")
    final_decision = parse_decision(final_text)

    return {
        "schema_version": "1.0",
        "prompt_hash": _prompt_hash(prompt),
        "prompt": prompt,
        "prompt_token_ids": [int(t) for t in prompt_tokens],
        "model_path": str(model_file),
        "generation_config": asdict(GENERATION_CONFIG),
        "trace_config": asdict(trace_config),
        "replay_temperature": float(temperature),
        "seed_policy": "random_seed_empirical_sampling",
        "num_steps": len(steps),
        "steps": steps,
        "generated_token_ids": generated_token_ids,
        "generated_text": final_text,
        "parsed_decision": final_decision,
        "trace_quality_flags": trace_quality_flags,
    }


def write_logit_trace(trace_payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trace_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _trace_step_generated_prefix_text(step_record: dict[str, Any], prompt_token_len: int) -> str:
    """Decode generated-response prefix text used for parser-consistent replay.

    Replay must parse generated output text only (not prompt text), otherwise
    prompt instructions containing both MOVE and STAY force UNKNOWN outcomes.
    """
    generated_prefix_text = step_record.get("generated_prefix_text")
    if isinstance(generated_prefix_text, str):
        return generated_prefix_text

    generated_ids = step_record.get("generated_prefix_token_ids", [])
    if isinstance(generated_ids, list):
        try:
            token_ids = [int(x) for x in generated_ids]
            return llm.detokenize(token_ids).decode("utf-8", errors="replace")
        except Exception:
            return ""

    # Backward compatibility for traces written before generated-prefix fields existed.
    prefix_ids = step_record.get("prefix_token_ids", [])
    if not isinstance(prefix_ids, list):
        return ""
    try:
        token_ids = [int(x) for x in prefix_ids]
    except Exception:
        return ""

    if prompt_token_len < 0:
        prompt_token_len = 0
    generated_only_ids = token_ids[prompt_token_len:]
    return llm.detokenize(generated_only_ids).decode("utf-8", errors="replace")


def _step_candidates_from_trace(step_record: dict[str, Any]) -> tuple[list[tuple[str, float]], float, str]:
    """Return (token_text, raw_logit) candidates and unresolved tail mass estimate."""
    storage_mode = str(step_record.get("trace_storage_mode", ""))
    unresolved_tail = float(step_record.get("unresolved_tail_mass", 0.0) or 0.0)

    if storage_mode == "full_logits":
        full_logits = step_record.get("full_logits", [])
        if not isinstance(full_logits, list) or len(full_logits) == 0:
            return [], 1.0, storage_mode

        raw_logits = np.asarray(full_logits, dtype=float)
        candidates: list[tuple[str, float]] = []
        for token_id, raw_logit in enumerate(raw_logits):
            token_text = llm.detokenize([int(token_id)]).decode("utf-8", errors="replace")
            candidates.append((token_text, float(raw_logit)))
        return candidates, 0.0, storage_mode

    if storage_mode == "top_logits":
        top_entries = step_record.get("top_logits", [])
        if not isinstance(top_entries, list) or len(top_entries) == 0:
            return [], 1.0, storage_mode

        candidates = []
        for entry in top_entries:
            if not isinstance(entry, dict):
                continue
            token_text = str(entry.get("token_text", ""))
            raw_logit = float(entry.get("logit", 0.0))
            candidates.append((token_text, raw_logit))
        return candidates, max(0.0, min(1.0, unresolved_tail)), storage_mode

    return [], 1.0, storage_mode


def _get_decision_hint_token_ids() -> set[int]:
    """Return single-token ids likely relevant to MOVE/STAY decoding paths."""
    global _DECISION_HINT_TOKEN_IDS
    if _DECISION_HINT_TOKEN_IDS is not None:
        return _DECISION_HINT_TOKEN_IDS

    fragments = [
        "M", "MO", "MOV", "MOVE",
        "S", "ST", "STA", "STAY",
        " M", " S", "\nM", "\nS", "\n",
        "`", "```,", "```", " ",
    ]

    token_ids: set[int] = set()
    for fragment in fragments:
        tokenized = llm.tokenize(fragment.encode("utf-8"))
        if len(tokenized) == 1:
            token_ids.add(int(tokenized[0]))

    _DECISION_HINT_TOKEN_IDS = token_ids
    return token_ids


def _get_token_decision_category(token_id: int) -> str:
    """Return MOVE/STAY/UNKNOWN for a token id with lazy caching."""
    global _TOKEN_DECISION_CATEGORY_CACHE
    if _TOKEN_DECISION_CATEGORY_CACHE is None:
        _TOKEN_DECISION_CATEGORY_CACHE = {}

    cached = _TOKEN_DECISION_CATEGORY_CACHE.get(int(token_id))
    if cached is not None:
        return cached

    token_text = llm.detokenize([int(token_id)]).decode("utf-8", errors="replace")
    category = parse_decision(token_text)
    _TOKEN_DECISION_CATEGORY_CACHE[int(token_id)] = category
    return category


def clear_runtime_caches(reset_decision_hints: bool = False) -> None:
    """Clear runtime caches so runs start from a clean cache state.

    Args:
        reset_decision_hints: Also clear cached decision-hint token ids.
            Default keeps hint ids because they are model-static and cheap to reuse.
    """
    global _TOKEN_DECISION_CATEGORY_CACHE, _DECISION_HINT_TOKEN_IDS
    _TOKEN_DECISION_CATEGORY_CACHE = None
    if reset_decision_hints:
        _DECISION_HINT_TOKEN_IDS = None


def estimate_move_stay_probability_branching(
    prompt: str,
    temperature: float,
    max_tokens: int = GENERATION_CONFIG.max_tokens,
    config: BranchingEstimatorConfig = BRANCHING_ESTIMATOR_CONFIG,
) -> dict[str, float | str | int]:
    """Estimate MOVE/STAY probabilities by expanding following-token branches.

    Uses the same parser and max-token horizon as sampling, but tracks multi-token
    paths so high-probability first tokens like markdown/newlines can still route
    mass toward MOVE/STAY at later positions.
    """
    if int(max_tokens) <= 0:
        raise ValueError("max_tokens must be positive")

    prompt_tokens = llm.tokenize(prompt.encode("utf-8"))
    quality_flags: list[str] = []

    # Each state is (generated_token_ids, probability_mass).
    active_states: list[tuple[tuple[int, ...], float]] = [(tuple(), 1.0)]
    move_mass = 0.0
    stay_mass = 0.0
    unknown_mass = 0.0

    for step_idx in range(1, int(max_tokens) + 1):
        if len(active_states) == 0:
            break

        next_state_masses: dict[tuple[int, ...], float] = {}

        for generated_ids, state_mass in active_states:
            if state_mass <= 0.0:
                continue

            llm.reset()
            llm.eval(prompt_tokens + list(generated_ids))
            if llm.scores is None or len(llm.scores) == 0:
                quality_flags.append(f"step_{step_idx}_missing_scores")
                unknown_mass += state_mass
                continue

            score_index = len(prompt_tokens) + len(generated_ids) - 1
            if score_index < 0 or score_index >= len(llm.scores):
                quality_flags.append(f"step_{step_idx}_invalid_score_index")
                unknown_mass += state_mass
                continue
            raw_logits = llm.scores[score_index]
            probs = _stable_softmax(raw_logits, temperature=temperature)
            # Build a candidate set, then classify only those tokens.
            top_indices = _adaptive_top_indices(
                probs,
                min_retained_mass=config.min_step_retained_mass,
                min_top_n=config.candidate_top_n,
                top_n_cap=config.candidate_top_n_cap,
            )
            hint_ids = _get_decision_hint_token_ids()
            top_index_set = {int(x) for x in top_indices.tolist()}
            for hint_id in hint_ids:
                if 0 <= hint_id < len(probs):
                    top_index_set.add(int(hint_id))
            candidate_indices = sorted(top_index_set)

            immediate_move = 0.0
            immediate_stay = 0.0
            unresolved_indices: list[int] = []
            for token_id in candidate_indices:
                token_prob = float(probs[int(token_id)])
                category = _get_token_decision_category(int(token_id))
                if category == "MOVE":
                    immediate_move += token_prob
                elif category == "STAY":
                    immediate_stay += token_prob
                else:
                    unresolved_indices.append(int(token_id))

            # If prefix already contains decision tokens, parser semantics can differ.
            # In normal unresolved branching prefixes this will be UNKNOWN.
            generated_prefix_text = llm.detokenize(list(generated_ids)).decode(
                "utf-8", errors="replace"
            )
            prefix_decision = parse_decision(generated_prefix_text)
            if prefix_decision == "MOVE":
                move_mass += state_mass
                continue
            if prefix_decision == "STAY":
                stay_mass += state_mass
                continue

            unresolved_mass_in_candidates = float(
                np.sum(probs[np.asarray(unresolved_indices, dtype=int)])
            ) if unresolved_indices else 0.0
            retained_mass = float(immediate_move + immediate_stay + unresolved_mass_in_candidates)
            tail_mass = float(max(0.0, 1.0 - retained_mass))

            move_mass += state_mass * immediate_move
            stay_mass += state_mass * immediate_stay

            if unresolved_mass_in_candidates <= 1e-15:
                if tail_mass > 0.0:
                    unknown_mass += state_mass * tail_mass
                continue

            if retained_mass < config.min_step_retained_mass:
                quality_flags.append(
                    f"step_{step_idx}_branch_retained_mass_below_threshold:{retained_mass:.6f}"
                )

            local_known_mass = 0.0
            for token_int in unresolved_indices:
                token_int = int(token_int)
                token_prob = float(probs[token_int])
                token_text = llm.detokenize([token_int]).decode("utf-8", errors="replace")
                decision = parse_decision(generated_prefix_text + token_text)

                branch_mass = state_mass * token_prob
                local_known_mass += token_prob

                if decision == "UNKNOWN":
                    next_ids = tuple(list(generated_ids) + [token_int])
                    next_state_masses[next_ids] = next_state_masses.get(next_ids, 0.0) + branch_mass
                elif decision == "MOVE":
                    move_mass += branch_mass
                elif decision == "STAY":
                    stay_mass += branch_mass

            # Treat truncated tail as unresolved unknown because token identity is missing.
            if tail_mass > 0.0:
                unknown_mass += state_mass * tail_mass

            # Defensive guard for numeric drift.
            if local_known_mass > 1.000001:
                quality_flags.append(f"step_{step_idx}_local_known_mass_gt_one:{local_known_mass:.6f}")

        # Beam pruning by probability mass.
        sorted_states = sorted(next_state_masses.items(), key=lambda item: item[1], reverse=True)
        active_states = sorted_states[: int(max(1, config.beam_width))]

    # Any unresolved states remaining at horizon contribute to UNKNOWN.
    if len(active_states) > 0:
        unknown_mass += float(sum(mass for _, mass in active_states))

    total_mass = float(move_mass + stay_mass + unknown_mass)
    if total_mass <= 0.0:
        return {
            "move_probability": 0.0,
            "stay_probability": 0.0,
            "unknown_probability": 1.0,
            "move_probability_parseable": 0.0,
            "stay_probability_parseable": 0.0,
            "quality_flags": "|".join(quality_flags + ["zero_total_mass"]),
            "num_final_active_states": 0,
        }

    move_probability = float(move_mass / total_mass)
    stay_probability = float(stay_mass / total_mass)
    unknown_probability = float(unknown_mass / total_mass)
    parseable_mass = move_probability + stay_probability
    move_probability_parseable = float(move_probability / parseable_mass) if parseable_mass > 0 else 0.0
    stay_probability_parseable = float(stay_probability / parseable_mass) if parseable_mass > 0 else 0.0

    return {
        "move_probability": move_probability,
        "stay_probability": stay_probability,
        "unknown_probability": unknown_probability,
        "move_probability_parseable": move_probability_parseable,
        "stay_probability_parseable": stay_probability_parseable,
        "quality_flags": "|".join(quality_flags),
        "num_final_active_states": int(len(active_states)),
    }


def replay_move_stay_probability_from_trace(
    trace_payload: dict[str, Any],
    replay_temperature: float,
    max_tokens: int = GENERATION_CONFIG.max_tokens,
) -> dict[str, float | str]:
    """Replay MOVE/STAY probabilities from stored following-token logits.

    Notes:
    - This replay uses the same parse function as empirical sampling.
    - For top-logit traces, tail mass is tracked as unresolved/unknown mass.
    - Replay consumes the same max-token horizon as sampling.
    """
    generation_cfg = trace_payload.get("generation_config", {})
    trace_steps = trace_payload.get("steps", [])
    prompt_token_ids = trace_payload.get("prompt_token_ids", [])
    if not isinstance(trace_steps, list):
        raise ValueError("Trace payload has invalid steps")
    prompt_token_len = len(prompt_token_ids) if isinstance(prompt_token_ids, list) else 0

    trace_max_tokens = int(generation_cfg.get("max_tokens", max_tokens))
    if int(max_tokens) != trace_max_tokens:
        raise ValueError(
            f"Shared max-token mismatch: replay max_tokens={max_tokens} trace max_tokens={trace_max_tokens}"
        )

    active_mass = 1.0
    move_mass = 0.0
    stay_mass = 0.0
    unknown_mass = 0.0
    replay_quality_flags: list[str] = []

    for step_record in trace_steps[:trace_max_tokens]:
        candidates, unresolved_tail_mass, storage_mode = _step_candidates_from_trace(step_record)
        if len(candidates) == 0:
            unknown_mass += active_mass
            replay_quality_flags.append("step_missing_candidates")
            active_mass = 0.0
            break

        raw_logits = np.asarray([raw_logit for _, raw_logit in candidates], dtype=float)
        candidate_probs = _stable_softmax(raw_logits, temperature=replay_temperature)

        prefix_text = _trace_step_generated_prefix_text(step_record, prompt_token_len)
        local_move = 0.0
        local_stay = 0.0
        local_unknown = 0.0

        for (token_text, _), token_prob in zip(candidates, candidate_probs):
            decision = parse_decision(prefix_text + token_text)
            if decision == "MOVE":
                local_move += float(token_prob)
            elif decision == "STAY":
                local_stay += float(token_prob)
            else:
                local_unknown += float(token_prob)

        if storage_mode == "top_logits":
            local_unknown += unresolved_tail_mass

        step_total = local_move + local_stay + local_unknown
        if step_total <= 0:
            unknown_mass += active_mass
            replay_quality_flags.append("step_probability_zero")
            active_mass = 0.0
            break

        local_move /= step_total
        local_stay /= step_total
        local_unknown /= step_total

        move_mass += active_mass * local_move
        stay_mass += active_mass * local_stay
        unknown_mass += active_mass * local_unknown
        active_mass = active_mass * local_unknown

        if active_mass <= 1e-12:
            break

    unknown_mass += active_mass
    move_mass = float(max(0.0, move_mass))
    stay_mass = float(max(0.0, stay_mass))
    unknown_mass = float(max(0.0, unknown_mass))

    total_mass = move_mass + stay_mass + unknown_mass
    if total_mass > 0:
        move_mass /= total_mass
        stay_mass /= total_mass
        unknown_mass /= total_mass

    parseable_mass = move_mass + stay_mass
    move_parseable = move_mass / parseable_mass if parseable_mass > 0 else 0.0
    stay_parseable = stay_mass / parseable_mass if parseable_mass > 0 else 0.0

    return {
        "replay_temperature": float(replay_temperature),
        "move_probability": float(move_mass),
        "stay_probability": float(stay_mass),
        "unknown_probability": float(unknown_mass),
        "move_probability_parseable": float(move_parseable),
        "stay_probability_parseable": float(stay_parseable),
        "quality_flags": "|".join(replay_quality_flags),
    }


def generate_neighbor_arrangements(num_similar: int) -> list[list[str]]:
    """Reuse arrangement generation logic from llm_utility_approximation.py."""
    if not 0 <= num_similar <= TOTAL_NEIGHBORS:
        raise ValueError("num_similar must be between 0 and 8 inclusive")

    arrangements: list[list[str]] = []
    for similar_indices in combinations(range(TOTAL_NEIGHBORS), num_similar):
        neighbors = ["O"] * TOTAL_NEIGHBORS
        for idx in similar_indices:
            neighbors[idx] = "S"
        arrangements.append(neighbors)
    return arrangements


def generate_neighbor_context(neighbors: list[str]) -> str:
    """Reuse context formatting logic from llm_utility_approximation.py."""
    if len(neighbors) != TOTAL_NEIGHBORS:
        raise ValueError(f"Expected {TOTAL_NEIGHBORS} neighbors, got {len(neighbors)}")

    grid_rows: list[list[str]] = []
    idx = 0
    for r in range(3):
        row: list[str] = []
        for c in range(3):
            if r == 1 and c == 1:
                row.append("X")
            else:
                row.append(neighbors[idx])
                idx += 1
        grid_rows.append(row)

    return "\n".join(" ".join(row) for row in grid_rows)


def build_prompt_for_context(context: str) -> str:
    return PROMPT_TEMPLATE.format(context=context)


def neighborhood_label(context: str) -> str:
    """Generate a filename-safe label for a neighborhood context string."""
    compact = context.replace(" ", "").replace("\n", "_")
    sanitized = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in compact)
    return sanitized


def get_ten_neighborhood_contexts() -> list[str]:
    """Build 10 deterministic neighborhood configurations."""
    contexts: list[str] = []

    # One configuration for each similarity level 0..8 gives 9 contexts.
    for num_similar in range(TOTAL_NEIGHBORS + 1):
        arrangements = generate_neighbor_arrangements(num_similar)
        contexts.append(generate_neighbor_context(arrangements[0]))

    # Add a tenth configuration from the middle of the 4-similar arrangement set.
    four_similar = generate_neighbor_arrangements(4)
    contexts.append(generate_neighbor_context(four_similar[len(four_similar) // 2]))

    # Deduplicate while preserving order, then keep the first 10.
    unique_contexts = list(dict.fromkeys(contexts))
    return unique_contexts[:3]


def parse_decision(raw_response: str) -> str:
    """Parse MOVE/STAY using the same logic as get_llm_decision in llm_runner.py."""
    text_upper = raw_response.strip().upper()
    has_move = "MOVE" in text_upper
    has_stay = "STAY" in text_upper

    # Ambiguous responses are treated as unparseable.
    if has_move and has_stay:
        return "UNKNOWN"

    if has_move:
        return "MOVE"
    if has_stay:
        return "STAY"

    return "UNKNOWN"


def extract_response_text(payload: Any) -> str:
    """Extract response text from either completion or chat-style payloads."""
    if not isinstance(payload, dict):
        return ""

    choices = payload.get("choices", [])
    if not choices:
        return ""

    first_choice = choices[0]
    if isinstance(first_choice, dict):
        text = first_choice.get("text")
        if isinstance(text, str):
            return text.strip()

        message = first_choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()

    return ""


def sample_prompt_move_stay_probabilities(
    prompt: str,
    n_samples: int = 1000,
    temperature: float = GENERATION_CONFIG.temperature,
    table_path: Path = SAMPLES_TABLE_PATH,
    probability_txt_path: Path = PROBABILITY_SUMMARY_PATH,
    num_workers: int = 1,
) -> dict[str, float]:
    """Sample the same prompt repeatedly and estimate MOVE/STAY probabilities."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rows: list[dict[str, Any]] = []
    move_count = 0
    stay_count = 0
    unknown_count = 0
    start_time = time.time()
    progress_every = max(1, n_samples // 20)  # ~5% updates

    max_workers = max(1, int(num_workers))
    if max_workers > 1:
        print(
            f"Starting sampling: {n_samples} total prompt evaluations with "
            f"{max_workers} parallel workers"
        )
    else:
        print(f"Starting sampling: {n_samples} total prompt evaluations")
    print(f"Status update frequency: every {progress_every} samples")

    if max_workers > 1:
        completed = 0
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_sampling_worker,
            initargs=(str(model_file),),
        ) as executor:
            futures = [
                executor.submit(_parallel_sample_once, sample_index, prompt, float(temperature))
                for sample_index in range(1, n_samples + 1)
            ]

            for future in as_completed(futures):
                sample_index, raw_text, decision = future.result()
                completed += 1

                if decision == "MOVE":
                    move_count += 1
                elif decision == "STAY":
                    stay_count += 1
                else:
                    unknown_count += 1

                rows.append(
                    {
                        "sample_index": sample_index,
                        "raw_response": raw_text,
                        "parsed_decision": decision,
                    }
                )

                if completed % progress_every == 0 or completed == n_samples:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    remaining = (n_samples - completed) / rate if rate > 0 else 0.0
                    print(
                        f"[Progress] {completed}/{n_samples} "
                        f"({(completed / n_samples) * 100:.1f}%) | "
                        f"MOVE={move_count} STAY={stay_count} UNKNOWN={unknown_count} | "
                        f"elapsed={elapsed:.1f}s eta={remaining:.1f}s"
                    )
    else:
        for sample_index in range(1, n_samples + 1):
            response = llm(
                prompt,
                max_tokens=GENERATION_CONFIG.max_tokens,
                temperature=temperature,
                top_k=GENERATION_CONFIG.top_k,
                top_p=GENERATION_CONFIG.top_p,
                min_p=GENERATION_CONFIG.min_p,
                repeat_penalty=GENERATION_CONFIG.repeat_penalty,
                frequency_penalty=GENERATION_CONFIG.frequency_penalty,
                presence_penalty=GENERATION_CONFIG.presence_penalty,
            )
            raw_text = extract_response_text(response)
            decision = parse_decision(raw_text)

            if decision == "MOVE":
                move_count += 1
            elif decision == "STAY":
                stay_count += 1
            else:
                unknown_count += 1

            rows.append(
                {
                    "sample_index": sample_index,
                    "raw_response": raw_text,
                    "parsed_decision": decision,
                }
            )

            if sample_index % progress_every == 0 or sample_index == n_samples:
                elapsed = time.time() - start_time
                rate = sample_index / elapsed if elapsed > 0 else 0.0
                remaining = (n_samples - sample_index) / rate if rate > 0 else 0.0
                print(
                    f"[Progress] {sample_index}/{n_samples} "
                    f"({(sample_index / n_samples) * 100:.1f}%) | "
                    f"MOVE={move_count} STAY={stay_count} UNKNOWN={unknown_count} | "
                    f"elapsed={elapsed:.1f}s eta={remaining:.1f}s"
                )

    rows.sort(key=lambda row: int(row["sample_index"]))

    with table_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["sample_index", "raw_response", "parsed_decision"],
        )
        writer.writeheader()
        writer.writerows(rows)

    total = float(n_samples)
    parseable_total = float(move_count + stay_count)
    move_probability = move_count / total
    stay_probability = stay_count / total
    unknown_probability = unknown_count / total
    move_probability_parseable = (move_count / parseable_total) if parseable_total else 0.0
    stay_probability_parseable = (stay_count / parseable_total) if parseable_total else 0.0

    summary_lines = [
        "MOVE/STAY Probability Summary",
        "=" * 80,
        f"Samples: {n_samples}",
        f"MOVE count: {move_count}",
        f"STAY count: {stay_count}",
        f"UNKNOWN count: {unknown_count}",
        "",
        "Probabilities over all samples:",
        f"P(MOVE) = {move_probability:.6f}",
        f"P(STAY) = {stay_probability:.6f}",
        f"P(UNKNOWN) = {unknown_probability:.6f}",
        "",
        "Probabilities over parseable samples only (MOVE+STAY):",
        f"P(MOVE | parseable) = {move_probability_parseable:.6f}",
        f"P(STAY | parseable) = {stay_probability_parseable:.6f}",
        "",
        f"Table file: {table_path}",
    ]
    probability_txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "move_probability": move_probability,
        "stay_probability": stay_probability,
        "unknown_probability": unknown_probability,
        "move_probability_parseable": move_probability_parseable,
        "stay_probability_parseable": stay_probability_parseable,
    }


def save_response_organized(payload: dict) -> None:
    lines = ["=" * 80, "LLM RESPONSE (ORGANIZED)", "=" * 80]

    top_level_order = ["id", "object", "created", "model", "usage", "choices"]
    lines.append("\n[Top-Level Fields]")
    for key in top_level_order:
        if key in payload and key != "choices":
            lines.append(f"- {key}: {payload[key]}")

    choices = payload.get("choices", [])
    lines.append(f"\n[Choices] count={len(choices)}")
    for idx, choice in enumerate(choices):
        lines.append(f"\n  Choice #{idx}")
        lines.append(f"  - index: {choice.get('index')}")
        lines.append(f"  - finish_reason: {choice.get('finish_reason')}")
        lines.append(f"  - text: {repr(choice.get('text', ''))}")

        logprobs = choice.get("logprobs")
        if isinstance(logprobs, dict):
            tokens = logprobs.get("tokens", [])
            offsets = logprobs.get("text_offset", [])
            token_logprobs = logprobs.get("token_logprobs", [])
            top_logprobs = logprobs.get("top_logprobs", [])

            lines.append("  - logprobs:")
            for tok_i in range(len(tokens)):
                token = tokens[tok_i] if tok_i < len(tokens) else None
                offset = offsets[tok_i] if tok_i < len(offsets) else None
                tok_lp = token_logprobs[tok_i] if tok_i < len(token_logprobs) else None
                lines.append(
                    f"      token[{tok_i}] token={repr(token)} offset={offset} logprob={tok_lp}"
                )

                if tok_i < len(top_logprobs) and isinstance(top_logprobs[tok_i], dict):
                    lines.append("        top_logprobs:")
                    for candidate, candidate_lp in top_logprobs[tok_i].items():
                        lines.append(f"          - {repr(candidate)}: {candidate_lp}")

    lines.append("\n[Full Raw Payload]")
    # default=str handles non-JSON-native values like numpy.float32.
    lines.append(json.dumps(payload, indent=2, default=str))

    organized_text = "\n".join(lines)
    # Overwrite output file on each run.
    ORGANIZED_RESPONSE_PATH.write_text(organized_text + "\n", encoding="utf-8")


def dump_all_logits_first_completion_token(payload: dict, model: Llama) -> None:
    usage = payload.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens")

    if not isinstance(prompt_tokens, int):
        return

    if model.scores is None or len(model.scores) == 0:
        return

    score_index = prompt_tokens - 1
    if score_index < 0 or score_index >= len(model.scores):
        return

    logits_row = model.scores[score_index]

    with ALL_LOGITS_PATH.open("w", encoding="utf-8") as handle:
        sorted_candidates = sorted(
            enumerate(logits_row), key=lambda item: item[1], reverse=True
        )
        for token_id, logit in sorted_candidates:
            token_text = model.detokenize([token_id]).decode("utf-8", errors="replace")
            row = {
                "token_id": token_id,
                "token_text": token_text,
                "logit": float(logit),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def get_probabilities(prompt, temperature=1.0):
    # 2. Tokenize and evaluate the prompt
    tokens = llm.tokenize(prompt.encode("utf-8"))
    llm.reset()
    llm.eval(tokens)
    
    # 3. Extract the raw logits for the "first" predicted token
    # llm.scores stores logits for every token evaluated. 
    # The last index represents the prediction for the next (first generated) token.
    raw_logits = llm.scores[len(tokens) - 1]
    
    # 4. Apply Temperature and Softmax
    # We subtract the max logit for numerical stability (prevents overflow)
    logits_scaled = raw_logits / max(temperature, 1e-6) # Avoid division by zero
    exp_logits = np.exp(logits_scaled - np.max(logits_scaled))
    probabilities = exp_logits / np.sum(exp_logits)
    
    return probabilities


def logits_to_probabilities(raw_logits: np.ndarray, temperature: float = 0.3) -> np.ndarray:
    """Convert logits to probabilities using temperature-scaled softmax."""
    return _stable_softmax(raw_logits, temperature=temperature)


def map_token_to_move_stay(token_text: str) -> str | None:
    """Map a token to MOVE/STAY by prefix, e.g., MO->MOVE and ST->STAY."""
    normalized = token_text.strip().upper()
    if normalized.startswith("MO"):
        return "MOVE"
    if normalized.startswith("ST"):
        return "STAY"
    return None


def get_move_stay_probability_from_logits(
    prompt: str,
    temperature: float = 0.3,
) -> dict[str, float]:
    """Compute final MOVE/STAY probabilities for a prompt from first-token logits."""
    tokens = llm.tokenize(prompt.encode("utf-8"))
    llm.reset()
    llm.eval(tokens)

    if llm.scores is None or len(llm.scores) == 0:
        raise RuntimeError("Could not read logits from llama-cpp (scores missing)")

    raw_logits = llm.scores[len(tokens) - 1]
    probabilities = logits_to_probabilities(raw_logits, temperature=temperature)

    move_probability = 0.0
    stay_probability = 0.0

    for token_id, token_probability in enumerate(probabilities):
        token_text = llm.detokenize([token_id]).decode("utf-8", errors="replace")
        decision = map_token_to_move_stay(token_text)
        if decision == "MOVE":
            move_probability += float(token_probability)
        elif decision == "STAY":
            stay_probability += float(token_probability)

    unresolved_probability = max(0.0, 1.0 - (move_probability + stay_probability))
    normalized_total = move_probability + stay_probability
    move_probability_normalized = (
        move_probability / normalized_total if normalized_total > 0 else 0.0
    )
    stay_probability_normalized = (
        stay_probability / normalized_total if normalized_total > 0 else 0.0
    )

    return {
        "temperature": temperature,
        "move_probability": move_probability,
        "stay_probability": stay_probability,
        "unresolved_probability": unresolved_probability,
        "move_probability_normalized": move_probability_normalized,
        "stay_probability_normalized": stay_probability_normalized,
    }


def get_top_n_logits_for_prompt(
    prompt: str,
    top_n: int = 10,
    temperature: float = 0.3,
) -> list[dict[str, Any]]:
    """Return top-N first-token logits with probabilities for manual inspection."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    tokens = llm.tokenize(prompt.encode("utf-8"))
    llm.reset()
    llm.eval(tokens)

    if llm.scores is None or len(llm.scores) == 0:
        raise RuntimeError("Could not read logits from llama-cpp (scores missing)")

    raw_logits = llm.scores[len(tokens) - 1]
    probabilities = logits_to_probabilities(raw_logits, temperature=temperature)

    ranked = sorted(
        enumerate(raw_logits),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:top_n]

    rows: list[dict[str, Any]] = []
    for rank, (token_id, logit) in enumerate(ranked, start=1):
        token_text = llm.detokenize([token_id]).decode("utf-8", errors="replace")
        rows.append(
            {
                "rank": rank,
                "token_id": int(token_id),
                "token_text": token_text,
                "logit": float(logit),
                "probability_temp_0p3": float(probabilities[token_id]),
            }
        )

    return rows


def write_top_logits_files(
    top_logits_rows: list[dict[str, Any]],
    csv_path: Path,
    json_path: Path,
) -> None:
    """Write top-logits rows to CSV and JSON files."""
    sorted_rows = sorted(top_logits_rows, key=lambda row: float(row["logit"]), reverse=True)
    for rank, row in enumerate(sorted_rows, start=1):
        row["rank"] = rank

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["rank", "token_id", "token_text", "logit", "probability_temp_0p3"],
        )
        writer.writeheader()
        writer.writerows(sorted_rows)

    json_path.write_text(json.dumps(sorted_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_logits_probability_summary_txt(
    probabilities: dict[str, float],
    output_path: Path = LOGITS_PROBABILITY_SUMMARY_PATH,
) -> None:
    """Write logits-derived MOVE/STAY probabilities to a txt file."""
    lines = [
        "Logits-Based MOVE/STAY Probability Summary",
        "=" * 80,
        f"Temperature: {probabilities['temperature']}",
        f"P(MOVE) = {probabilities['move_probability']:.6f}",
        f"P(STAY) = {probabilities['stay_probability']:.6f}",
        f"P(UNRESOLVED) = {probabilities['unresolved_probability']:.6f}",
        "",
        "Normalized over MOVE/STAY mass only:",
        f"P(MOVE | MOVE/STAY) = {probabilities['move_probability_normalized']:.6f}",
        f"P(STAY | MOVE/STAY) = {probabilities['stay_probability_normalized']:.6f}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_trials_for_ten_neighborhoods(
    n_samples: int = 1000,
    temperature: float = 0.3,
    output_dir: Path = BATCH_OUTPUT_DIR,
    num_workers: int = 1,
    max_contexts: int | None = None,
) -> None:
    """Run logits + sampling trials for 10 contexts and store labeled outputs."""
    # Ensure reproducible per-run cache behavior for branching estimators.
    clear_runtime_caches()

    output_dir.mkdir(parents=True, exist_ok=True)
    contexts = get_ten_neighborhood_contexts()
    if max_contexts is not None:
        if max_contexts <= 0:
            raise ValueError("max_contexts must be positive when provided")
        contexts = contexts[: int(max_contexts)]
    summary_rows: list[dict[str, Any]] = []

    print(f"Batch output directory: {output_dir}")
    print(f"Running trials for {len(contexts)} neighborhood configurations")

    for idx, context in enumerate(contexts, start=1):
        label = neighborhood_label(context)
        prompt = build_prompt_for_context(context)

        prompt_path = output_dir / f"{label}_prompt.txt"
        logits_txt_path = output_dir / f"{label}_logits_probability.txt"
        top_logits_csv_path = output_dir / f"{label}_top10_logits.csv"
        top_logits_json_path = output_dir / f"{label}_top10_logits.json"
        sample_csv_path = output_dir / f"{label}_samples.csv"
        sample_txt_path = output_dir / f"{label}_sample_probability.txt"
        trace_json_path = TRACE_OUTPUT_DIR / f"{label}_trace.json"

        print("-" * 80)
        print(f"[Neighborhood {idx}/{len(contexts)}] {label}")
        print(context)

        prompt_path.write_text(prompt + "\n", encoding="utf-8")

        logit_probs = get_move_stay_probability_from_logits(prompt, temperature=temperature)
        write_logits_probability_summary_txt(logit_probs, output_path=logits_txt_path)

        top_logits_rows = get_top_n_logits_for_prompt(prompt, top_n=10, temperature=temperature)
        write_top_logits_files(top_logits_rows, top_logits_csv_path, top_logits_json_path)

        sample_probs = sample_prompt_move_stay_probabilities(
            prompt,
            n_samples=n_samples,
            temperature=temperature,
            table_path=sample_csv_path,
            probability_txt_path=sample_txt_path,
            num_workers=num_workers,
        )

        trace_payload = capture_following_token_logit_trace(
            prompt=prompt,
            temperature=temperature,
            max_tokens=GENERATION_CONFIG.max_tokens,
            trace_config=LOGIT_TRACE_CONFIG,
        )
        write_logit_trace(trace_payload, trace_json_path)

        replay_probs = replay_move_stay_probability_from_trace(
            trace_payload=trace_payload,
            replay_temperature=temperature,
            max_tokens=GENERATION_CONFIG.max_tokens,
        )
        branch_probs = estimate_move_stay_probability_branching(
            prompt=prompt,
            temperature=temperature,
            max_tokens=GENERATION_CONFIG.max_tokens,
            config=BRANCHING_ESTIMATOR_CONFIG,
        )

        branch_move_probability = float(branch_probs["move_probability"])
        branch_move_probability_parseable = float(branch_probs["move_probability_parseable"])

        summary_rows.append(
            {
                "neighborhood_label": label,
                "context": context.replace("\n", " | "),
                "first_token_logits_move_probability": logit_probs["move_probability"],
                "logits_stay_probability": logit_probs["stay_probability"],
                "sample_move_probability": sample_probs["move_probability"],
                "sample_stay_probability": sample_probs["stay_probability"],
                "sample_unknown_probability": sample_probs["unknown_probability"],
                "derived_estimator": "branch_following_token",
                "derived_move_probability": float(branch_probs["move_probability"]),
                "derived_stay_probability": float(branch_probs["stay_probability"]),
                "derived_unknown_probability": float(branch_probs["unknown_probability"]),
                "derived_move_probability_parseable": float(branch_probs["move_probability_parseable"]),
                "derived_stay_probability_parseable": float(branch_probs["stay_probability_parseable"]),
                "logits_txt": str(logits_txt_path),
                "top_logits_csv": str(top_logits_csv_path),
                "top_logits_json": str(top_logits_json_path),
                "sample_txt": str(sample_txt_path),
                "sample_csv": str(sample_csv_path),
                "logit_trace_json": str(trace_json_path),
                "trace_decision": trace_payload.get("parsed_decision", "UNKNOWN"),
                "trace_steps": int(trace_payload.get("num_steps", 0)),
                "trace_quality_flags": "|".join(trace_payload.get("trace_quality_flags", [])),
                "replay_move_probability": replay_probs["move_probability"],
                "replay_stay_probability": replay_probs["stay_probability"],
                "replay_unknown_probability": replay_probs["unknown_probability"],
                "replay_move_probability_parseable": replay_probs["move_probability_parseable"],
                "replay_stay_probability_parseable": replay_probs["stay_probability_parseable"],
                "replay_quality_flags": replay_probs["quality_flags"],
                "branch_move_probability": branch_probs["move_probability"],
                "branch_stay_probability": branch_probs["stay_probability"],
                "branch_unknown_probability": branch_probs["unknown_probability"],
                "branch_move_probability_parseable": branch_probs["move_probability_parseable"],
                "branch_stay_probability_parseable": branch_probs["stay_probability_parseable"],
                "branch_quality_flags": branch_probs["quality_flags"],
                "branch_num_final_active_states": branch_probs["num_final_active_states"],
                "branch_delta_move_probability": float(
                    branch_move_probability - sample_probs["move_probability"]
                ),
                "branch_abs_delta_move_probability": float(
                    abs(branch_move_probability - sample_probs["move_probability"])
                ),
                "branch_delta_move_probability_parseable": float(
                    branch_move_probability_parseable
                    - sample_probs["move_probability_parseable"]
                ),
                "branch_abs_delta_move_probability_parseable": float(
                    abs(
                        branch_move_probability_parseable
                        - sample_probs["move_probability_parseable"]
                    )
                ),
            }
        )

    summary_csv_path = output_dir / "all_neighborhoods_summary.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("=" * 80)
    print(f"Completed all neighborhood trials. Summary: {summary_csv_path}")

if __name__ == "__main__":
    run_start = time.time()
    default_workers = min(1, max(1, os.cpu_count() or 1))
    run_trials_for_ten_neighborhoods(
        n_samples=1000,
        temperature=0.3,
        num_workers=default_workers,
        max_contexts=10,
    )
    print(f"Total runtime: {time.time() - run_start:.1f}s")

    # Optional: single-response inspection helpers
    # response = llm(PROMPT_TEXT, logprobs=5, max_tokens=5)
    # save_response_organized(response)
    # dump_all_logits_first_completion_token(response, llm)





