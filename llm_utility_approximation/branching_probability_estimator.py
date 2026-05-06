"""Branching-token probability estimator for Schelling LLM policies.

Port of the branching / trace / replay machinery from
``llama_cpp_llm_trial.py`` scaled up to every valid 3x3 Schelling
neighborhood × every scenario × every agent role, with outputs labelled by
model name and temperature and compatible with
``llm_runner.load_log_prob_policy``.

Usage (debug slice)::

    python branching_probability_estimator.py \\
        --model-path C:/Users/Sriki/PancsVriend/llms/gemma-3-4b-it-q4_0.gguf \\
        --model-name gemma-3-4b-it-q4_0 \\
        --temperature 0.3 \\
        --scenarios baseline \\
        --agent-roles type_a \\
        --debug-subset 5 \\
        --capture-trace

Usage (production sweep with resume)::

    python branching_probability_estimator.py \\
        --model-path C:/Users/Sriki/PancsVriend/llms/gemma-3-4b-it-q4_0.gguf \\
        --model-name gemma-3-4b-it-q4_0 \\
        --temperature 0.3 \\
        --scenarios baseline race_white_black \\
        --agent-roles type_a type_b \\
        --resume

The summary CSV lands at both
``<output-root>/<model_slug>/T<temp_slug>/<model_slug>_<scenario>_T<temp_slug>_stay_move_probability_split_summary.csv``
and (for backwards-compatible loading) at
``<output-root>/<model_slug>/<model_slug>_<scenario>_stay_move_probability_split_summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Locate sibling modules.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for candidate in (_THIS_DIR, _REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from context_scenarios import CONTEXT_SCENARIOS  # noqa: E402
from llm_token_probabilities import (  # noqa: E402
    SCHELLING_GRID_SIZE,
    _sanitize_model_for_path_component,
    build_scenario_prompt,
    generate_all_valid_schelling_neighbors,
    generate_neighbor_context,
)

try:
    from llama_cpp import Llama  # type: ignore
except ImportError as exc:  # pragma: no cover - environment error
    raise SystemExit(
        "llama-cpp-python is required; install with `pip install llama-cpp-python`.\n"
        f"Underlying error: {exc}"
    )


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.3
    max_tokens: int = 16
    n_ctx: int = 512
    top_k: int = 0
    top_p: float = 0.9999
    min_p: float = 0.0
    repeat_penalty: float = 1.0


@dataclass(frozen=True)
class BranchingConfig:
    beam_width: int = 16
    candidate_top_n: int = 1
    candidate_top_n_cap: int = 2048
    min_step_retained_mass: float = 0.999
    early_stop_move_stay_mass: float = 0.999


@dataclass
class ArrangementTask:
    scenario: str
    agent_role: str
    agent_label: str
    opposite_label: str
    arrangement_code: str
    context_string: str
    prompt: str
    num_similar: int
    num_opposite: int
    num_empty: int
    num_wall: int
    ratio_similar: float


@dataclass
class ArrangementResult:
    scenario: str
    agent_role: str
    agent_label: str
    opposite_label: str
    arrangement_code: str
    context: str
    num_similar: int
    num_opposite: int
    num_empty: int
    num_wall: int
    ratio_similar: float
    move_probability: float
    stay_probability: float
    unknown_probability: float
    move_probability_parseable: float
    stay_probability_parseable: float
    num_final_active_states: int
    num_steps: int
    quality_flags: str
    seconds_elapsed: float
    trace_path: str = ""

    # Columns used by llm_runner.load_log_prob_policy (kept verbatim):
    mean_stay_share: float = 0.0
    mean_move_share: float = 0.0
    mean_stay_probability: float = 0.0
    mean_move_probability: float = 0.0
    mean_total_labeled_probability: float = 0.0
    mean_stay_logprob: float = 0.0
    mean_move_logprob: float = 0.0
    num_trials: int = 1
    num_meaningful_errors: int = 0
    mean_meaningful_reasks_used: float = 0.0


# ----------------------------------------------------------------------------
# Branching engine (single-process, single Llama instance)
# ----------------------------------------------------------------------------


class BranchingEngine:
    """Wraps a single llama.cpp instance for branch-token probability estimation."""

    _DECISION_HINT_FRAGMENTS: tuple[str, ...] = (
        "M", "MO", "MOV", "MOVE",
        "S", "ST", "STA", "STAY",
        " M", " S", "\nM", "\nS", "\n",
        "`", "```", " ",
    )

    def __init__(
        self,
        model_path: str,
        gen_config: GenerationConfig,
        branch_config: BranchingConfig,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
    ) -> None:
        self.model_path = model_path
        self.gen_config = gen_config
        self.branch_config = branch_config

        kwargs: dict[str, Any] = dict(
            model_path=model_path,
            logits_all=True,
            verbose=False,
            temperature=gen_config.temperature,
            n_ctx=gen_config.n_ctx,
            n_gpu_layers=int(n_gpu_layers),
        )
        if n_threads is not None and int(n_threads) > 0:
            kwargs["n_threads"] = int(n_threads)
        self.llm = Llama(**kwargs)
        self._hint_token_ids: set[int] | None = None

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _stable_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
        scaled = logits / max(float(temperature), 1e-6)
        exp_logits = np.exp(scaled - np.max(scaled))
        return exp_logits / np.sum(exp_logits)

    @staticmethod
    def _adaptive_top_indices(
        probs: np.ndarray,
        min_retained_mass: float,
        min_top_n: int,
        top_n_cap: int,
    ) -> np.ndarray:
        max_n = int(max(1, min(top_n_cap, len(probs))))
        start_n = int(max(1, min(min_top_n, max_n)))
        ordered = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[ordered])
        threshold_idx = int(np.searchsorted(cumulative, min_retained_mass, side="left")) + 1
        required_n = int(max(start_n, min(threshold_idx, max_n)))
        return ordered[:required_n]

    @staticmethod
    def _sort_indices_desc(probs: np.ndarray, indices: set[int]) -> list[int]:
        return sorted((int(i) for i in indices), key=lambda i: float(probs[i]), reverse=True)

    def _get_decision_hint_token_ids(self) -> set[int]:
        if self._hint_token_ids is not None:
            return self._hint_token_ids
        hints: set[int] = set()
        for fragment in self._DECISION_HINT_FRAGMENTS:
            tokenized = self.llm.tokenize(fragment.encode("utf-8"))
            if len(tokenized) == 1:
                hints.add(int(tokenized[0]))
        self._hint_token_ids = hints
        return hints

    @staticmethod
    def parse_decision(raw_response: str) -> str:
        """Mirror llm_runner.py MOVE/STAY parsing — ambiguous → UNKNOWN."""
        text_upper = str(raw_response).strip().upper()
        has_move = "MOVE" in text_upper
        has_stay = "STAY" in text_upper
        if has_move and has_stay:
            return "UNKNOWN"
        if has_move:
            return "MOVE"
        if has_stay:
            return "STAY"
        return "UNKNOWN"

    # ---- branching estimator ---------------------------------------------

    def estimate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        capture_trace: bool = False,
    ) -> dict[str, Any]:
        if int(max_tokens) <= 0:
            raise ValueError("max_tokens must be positive")

        llm = self.llm
        branch_config = self.branch_config
        quality_flags: list[str] = []
        step_traces: list[dict[str, Any]] = []

        prompt_tokens = llm.tokenize(prompt.encode("utf-8"))

        active_states: list[tuple[tuple[int, ...], float]] = [(tuple(), 1.0)]
        move_mass = 0.0
        stay_mass = 0.0
        unknown_mass = 0.0
        steps_executed = 0

        for step_idx in range(1, int(max_tokens) + 1):
            if len(active_states) == 0:
                break
            steps_executed = step_idx

            next_state_masses: dict[tuple[int, ...], float] = {}
            step_state_records: list[dict[str, Any]] = [] if capture_trace else []

            for generated_ids, state_mass in active_states:
                if state_mass <= 0.0:
                    continue

                generated_prefix_text = llm.detokenize(list(generated_ids)).decode(
                    "utf-8", errors="replace"
                )

                prefix_decision = self.parse_decision(generated_prefix_text)
                if prefix_decision == "MOVE":
                    move_mass += state_mass
                    if capture_trace:
                        step_state_records.append({
                            "generated_token_ids": [int(t) for t in generated_ids],
                            "generated_prefix_text": generated_prefix_text,
                            "state_mass": float(state_mass),
                            "prefix_decision": "MOVE",
                            "resolved_mass": float(state_mass),
                            "candidates": [],
                        })
                    continue
                if prefix_decision == "STAY":
                    stay_mass += state_mass
                    if capture_trace:
                        step_state_records.append({
                            "generated_token_ids": [int(t) for t in generated_ids],
                            "generated_prefix_text": generated_prefix_text,
                            "state_mass": float(state_mass),
                            "prefix_decision": "STAY",
                            "resolved_mass": float(state_mass),
                            "candidates": [],
                        })
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

                raw_logits = np.asarray(llm.scores[score_index])
                probs = self._stable_softmax(raw_logits, temperature=temperature)
                top_indices = self._adaptive_top_indices(
                    probs,
                    min_retained_mass=branch_config.min_step_retained_mass,
                    min_top_n=branch_config.candidate_top_n,
                    top_n_cap=branch_config.candidate_top_n_cap,
                )
                top_index_set = {int(x) for x in top_indices.tolist()}
                for hint_id in self._get_decision_hint_token_ids():
                    if 0 <= hint_id < len(probs):
                        top_index_set.add(int(hint_id))
                candidate_indices = self._sort_indices_desc(probs, top_index_set)

                retained_candidate_mass = 0.0
                candidate_records: list[dict[str, Any]] = []
                for token_id in candidate_indices:
                    token_int = int(token_id)
                    token_prob = float(probs[token_int])
                    retained_candidate_mass += token_prob
                    token_text = llm.detokenize([token_int]).decode("utf-8", errors="replace")
                    decision = self.parse_decision(generated_prefix_text + token_text)
                    branch_mass = state_mass * token_prob

                    if decision == "MOVE":
                        move_mass += branch_mass
                    elif decision == "STAY":
                        stay_mass += branch_mass
                    else:
                        next_ids = tuple(list(generated_ids) + [token_int])
                        next_state_masses[next_ids] = next_state_masses.get(next_ids, 0.0) + branch_mass

                    if capture_trace:
                        candidate_records.append({
                            "token_id": token_int,
                            "token_text": token_text,
                            "logit": float(raw_logits[token_int]),
                            "probability": token_prob,
                            "decision": decision,
                            "branch_mass": float(branch_mass),
                        })

                tail_mass = float(max(0.0, 1.0 - retained_candidate_mass))
                if tail_mass > 0.0:
                    unknown_mass += state_mass * tail_mass
                if retained_candidate_mass < branch_config.min_step_retained_mass:
                    quality_flags.append(
                        f"step_{step_idx}_branch_retained_mass_below_threshold:{retained_candidate_mass:.6f}"
                    )

                if capture_trace:
                    step_state_records.append({
                        "generated_token_ids": [int(t) for t in generated_ids],
                        "generated_prefix_text": generated_prefix_text,
                        "state_mass": float(state_mass),
                        "prefix_decision": "UNKNOWN",
                        "retained_candidate_mass": float(retained_candidate_mass),
                        "unresolved_tail_mass": float(tail_mass),
                        "candidates": candidate_records,
                    })

            sorted_states = sorted(next_state_masses.items(), key=lambda kv: kv[1], reverse=True)
            beam_cutoff = int(max(1, branch_config.beam_width))
            pruned_mass = float(sum(mass for _, mass in sorted_states[beam_cutoff:]))
            if pruned_mass > 0.0:
                unknown_mass += pruned_mass
                quality_flags.append(f"step_{step_idx}_pruned_beam_mass:{pruned_mass:.6f}")
            active_states = sorted_states[:beam_cutoff]

            if capture_trace:
                step_traces.append({
                    "step_index": step_idx,
                    "num_active_states_in": len(step_state_records),
                    "num_next_states_before_prune": len(sorted_states),
                    "num_next_states_after_prune": len(active_states),
                    "pruned_beam_mass": float(pruned_mass),
                    "states": step_state_records,
                })

            resolved_mass = float(move_mass + stay_mass)
            if resolved_mass >= float(branch_config.early_stop_move_stay_mass):
                quality_flags.append(f"early_stop_move_stay_mass_reached:{resolved_mass:.6f}")
                break

        if len(active_states) > 0:
            unknown_mass += float(sum(mass for _, mass in active_states))

        total_mass = float(move_mass + stay_mass + unknown_mass)
        if total_mass <= 0.0:
            move_probability = 0.0
            stay_probability = 0.0
            unknown_probability = 1.0
            quality_flags.append("zero_total_mass")
        else:
            move_probability = float(move_mass / total_mass)
            stay_probability = float(stay_mass / total_mass)
            unknown_probability = float(unknown_mass / total_mass)

        parseable_mass = move_probability + stay_probability
        if parseable_mass > 0:
            move_probability_parseable = float(move_probability / parseable_mass)
            stay_probability_parseable = float(stay_probability / parseable_mass)
        else:
            move_probability_parseable = 0.0
            stay_probability_parseable = 0.0

        result: dict[str, Any] = {
            "move_probability": move_probability,
            "stay_probability": stay_probability,
            "unknown_probability": unknown_probability,
            "move_probability_parseable": move_probability_parseable,
            "stay_probability_parseable": stay_probability_parseable,
            "num_final_active_states": int(len(active_states)),
            "num_steps": int(steps_executed),
            "quality_flags": "|".join(quality_flags),
        }

        if capture_trace:
            result["trace"] = {
                "schema_version": "2.0",
                "capture_mode": "branchwise_following_token",
                "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt": prompt,
                "prompt_token_ids": [int(t) for t in prompt_tokens],
                "model_path": self.model_path,
                "generation_config": asdict(self.gen_config),
                "branching_config": asdict(self.branch_config),
                "replay_temperature": float(temperature),
                "num_steps": len(step_traces),
                "steps": step_traces,
                "final_mass": {
                    "move_mass": float(move_mass),
                    "stay_mass": float(stay_mass),
                    "unknown_mass": float(unknown_mass),
                    "total_mass": float(total_mass),
                    "move_probability": move_probability,
                    "stay_probability": stay_probability,
                    "unknown_probability": unknown_probability,
                },
                "trace_quality_flags": list(quality_flags),
            }
        return result


# ----------------------------------------------------------------------------
# Output layout + IO helpers
# ----------------------------------------------------------------------------


SUMMARY_COLUMNS: tuple[str, ...] = (
    "scenario",
    "agent_role",
    "agent_label",
    "opposite_label",
    "arrangement_code",
    "context",
    "num_similar",
    "num_opposite",
    "num_empty",
    "num_wall",
    "ratio_similar",
    "num_trials",
    "mean_stay_probability",
    "mean_move_probability",
    "mean_total_labeled_probability",
    "mean_stay_logprob",
    "mean_move_logprob",
    "mean_stay_share",
    "mean_move_share",
    "num_meaningful_errors",
    "mean_meaningful_reasks_used",
    # Branching-specific diagnostic columns (ignored by load_log_prob_policy):
    "move_probability",
    "stay_probability",
    "unknown_probability",
    "move_probability_parseable",
    "stay_probability_parseable",
    "num_final_active_states",
    "num_steps",
    "quality_flags",
    "seconds_elapsed",
    "temperature",
    "model",
    "trace_path",
)


def temp_slug(temperature: float) -> str:
    return f"T{temperature:.3f}".replace(".", "p")


def scenario_slug(scenario: str) -> str:
    return _sanitize_model_for_path_component(scenario)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def summary_csv_paths(
    output_root: Path,
    model_name: str,
    scenario: str,
    temperature: float,
    debug: bool = False,
) -> dict[str, Path]:
    model_slug_ = _sanitize_model_for_path_component(model_name)
    scen_slug = scenario_slug(scenario)
    ts = temp_slug(temperature)
    suffix = "_DEBUG" if debug else ""

    model_dir = output_root / model_slug_
    temp_dir = model_dir / ts

    nested_summary = temp_dir / f"{model_slug_}_{scen_slug}_{ts}_stay_move_probability_split_summary{suffix}.csv"
    nested_full = temp_dir / f"{model_slug_}_{scen_slug}_{ts}_stay_move_probability_split{suffix}.csv"
    flat_summary = model_dir / f"{model_slug_}_{scen_slug}_stay_move_probability_split_summary{suffix}.csv"

    return {
        "nested_summary": nested_summary,
        "nested_full": nested_full,
        "flat_summary": flat_summary,
        "model_dir": model_dir,
        "temp_dir": temp_dir,
        "manifest": temp_dir / f"manifest_{scen_slug}{suffix}.json",
        "trace_dir": temp_dir / "traces" / scen_slug,
    }


def row_from_result(
    task: ArrangementTask,
    estimate: dict[str, Any],
    seconds_elapsed: float,
    temperature: float,
    model_name: str,
    trace_path: str,
) -> dict[str, Any]:
    stay_probability = float(estimate["stay_probability"])
    move_probability = float(estimate["move_probability"])
    stay_share = float(estimate.get("stay_probability_parseable", 0.0))
    move_share = float(estimate.get("move_probability_parseable", 0.0))
    total_labeled = stay_probability + move_probability

    return {
        "scenario": task.scenario,
        "agent_role": task.agent_role,
        "agent_label": task.agent_label,
        "opposite_label": task.opposite_label,
        "arrangement_code": task.arrangement_code,
        "context": task.context_string,
        "num_similar": task.num_similar,
        "num_opposite": task.num_opposite,
        "num_empty": task.num_empty,
        "num_wall": task.num_wall,
        "ratio_similar": task.ratio_similar,
        "num_trials": 1,
        "mean_stay_probability": stay_probability,
        "mean_move_probability": move_probability,
        "mean_total_labeled_probability": total_labeled,
        "mean_stay_logprob": math.log(stay_probability) if stay_probability > 0 else float("-inf"),
        "mean_move_logprob": math.log(move_probability) if move_probability > 0 else float("-inf"),
        "mean_stay_share": stay_share,
        "mean_move_share": move_share,
        "num_meaningful_errors": 0,
        "mean_meaningful_reasks_used": 0.0,
        "move_probability": move_probability,
        "stay_probability": stay_probability,
        "unknown_probability": float(estimate["unknown_probability"]),
        "move_probability_parseable": move_share,
        "stay_probability_parseable": stay_share,
        "num_final_active_states": int(estimate["num_final_active_states"]),
        "num_steps": int(estimate["num_steps"]),
        "quality_flags": str(estimate["quality_flags"]),
        "seconds_elapsed": float(seconds_elapsed),
        "temperature": float(temperature),
        "model": model_name,
        "trace_path": trace_path,
    }


def read_completed_keys(summary_csv: Path) -> set[tuple[str, str]]:
    if not summary_csv.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with summary_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            role = str(row.get("agent_role", "")).strip()
            code = str(row.get("arrangement_code", "")).strip()
            if role and code:
                done.add((role, code))
    return done


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)


def write_trace_gz(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def append_row(csv_path: Path, row: dict[str, Any], header_written: bool) -> bool:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if header_written or csv_path.exists() else "w"
    with csv_path.open(mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SUMMARY_COLUMNS), extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        writer.writerow(row)
    return True


# ----------------------------------------------------------------------------
# Enumeration
# ----------------------------------------------------------------------------


def enumerate_arrangement_tasks(
    scenario: str,
    agent_roles: Iterable[str],
    grid_size: int = SCHELLING_GRID_SIZE,
) -> list[ArrangementTask]:
    scenario_info = CONTEXT_SCENARIOS[scenario]
    type_a = str(scenario_info["type_a"])
    type_b = str(scenario_info["type_b"])

    role_configs: list[tuple[str, str, str]] = []
    for role in agent_roles:
        if role == "type_a":
            role_configs.append(("type_a", type_a, type_b))
        elif role == "type_b":
            role_configs.append(("type_b", type_b, type_a))
        else:
            raise ValueError(f"Unknown agent role: {role}")

    all_arrangements = generate_all_valid_schelling_neighbors(grid_size)

    tasks: list[ArrangementTask] = []
    seen: set[tuple[str, str]] = set()  # (role, arrangement_code)
    for role_key, role_label, opposite_label in role_configs:
        for neighbors in all_arrangements:
            arrangement_code = "".join(neighbors)
            if (role_key, arrangement_code) in seen:
                continue
            seen.add((role_key, arrangement_code))

            context_string = generate_neighbor_context(neighbors)
            prompt = build_scenario_prompt(
                scenario_key=scenario,
                context=context_string,
                agent_label=role_label,
                opposite_label=opposite_label,
            )

            num_same = int(sum(1 for item in neighbors if item == "S"))
            num_opposite = int(sum(1 for item in neighbors if item == "O"))
            num_empty = int(sum(1 for item in neighbors if item == "E"))
            num_wall = int(sum(1 for item in neighbors if item == "#"))
            num_non_wall = 8 - num_wall
            ratio_similar = (num_same / num_non_wall) if num_non_wall > 0 else 0.0

            tasks.append(ArrangementTask(
                scenario=scenario,
                agent_role=role_key,
                agent_label=role_label,
                opposite_label=opposite_label,
                arrangement_code=arrangement_code,
                context_string=context_string,
                prompt=prompt,
                num_similar=num_same,
                num_opposite=num_opposite,
                num_empty=num_empty,
                num_wall=num_wall,
                ratio_similar=ratio_similar,
            ))
    return tasks


# ----------------------------------------------------------------------------
# Sweep runner
# ----------------------------------------------------------------------------


def run_scenario_sweep(
    engine: BranchingEngine,
    scenario: str,
    agent_roles: list[str],
    output_root: Path,
    model_name: str,
    temperature: float,
    max_tokens: int,
    debug_subset: int | None,
    sample_seed: int | None,
    resume: bool,
    capture_trace: bool,
    progress_every: int = 25,
) -> dict[str, Any]:
    paths = summary_csv_paths(output_root, model_name, scenario, temperature, debug=bool(debug_subset))
    ensure_dir(paths["temp_dir"])
    ensure_dir(paths["model_dir"])

    tasks = enumerate_arrangement_tasks(scenario, agent_roles)

    if debug_subset is not None and debug_subset > 0:
        sample_count = min(int(debug_subset), len(tasks))
        if sample_seed is None:
            tasks = random.sample(tasks, sample_count)
        else:
            tasks = random.Random(int(sample_seed)).sample(tasks, sample_count)

    completed_keys: set[tuple[str, str]] = set()
    if resume:
        completed_keys = read_completed_keys(paths["nested_summary"]) | read_completed_keys(paths["flat_summary"])

    remaining = [t for t in tasks if (t.agent_role, t.arrangement_code) not in completed_keys]

    total = len(tasks)
    skipped = total - len(remaining)
    print(
        f"[sweep] scenario={scenario} roles={agent_roles} total_tasks={total} "
        f"already_done={skipped} to_run={len(remaining)} "
        f"output_nested={paths['nested_summary']}",
        flush=True,
    )

    start_time = time.time()
    nested_header_written = paths["nested_summary"].exists()
    flat_header_written = paths["flat_summary"].exists()
    full_header_written = paths["nested_full"].exists()

    for idx, task in enumerate(remaining, start=1):
        t0 = time.time()
        estimate = engine.estimate(
            prompt=task.prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            capture_trace=capture_trace,
        )
        elapsed = time.time() - t0

        trace_path_str = ""
        if capture_trace and "trace" in estimate:
            trace_path = paths["trace_dir"] / task.agent_role / f"{task.arrangement_code}.trace.json.gz"
            write_trace_gz(estimate["trace"], trace_path)
            trace_path_str = str(trace_path)
            # Free trace payload before next iteration to keep memory small
            del estimate["trace"]

        row = row_from_result(
            task=task,
            estimate=estimate,
            seconds_elapsed=elapsed,
            temperature=temperature,
            model_name=model_name,
            trace_path=trace_path_str,
        )

        nested_header_written = append_row(paths["nested_summary"], row, nested_header_written)
        flat_header_written = append_row(paths["flat_summary"], row, flat_header_written)
        full_header_written = append_row(paths["nested_full"], row, full_header_written)

        if idx % progress_every == 0 or idx == len(remaining):
            total_elapsed = time.time() - start_time
            rate = idx / total_elapsed if total_elapsed > 0 else 0.0
            eta = (len(remaining) - idx) / rate if rate > 0 else float("inf")
            eta_display = f"{eta:.1f}s" if math.isfinite(eta) else "unknown"
            print(
                f"[sweep][{scenario}] {idx}/{len(remaining)} "
                f"({(idx / max(1, len(remaining))) * 100:.1f}%) | "
                f"elapsed={total_elapsed:.1f}s eta={eta_display} | "
                f"last: role={task.agent_role} arr={task.arrangement_code} "
                f"p_move={estimate['move_probability']:.3f} p_unk={estimate['unknown_probability']:.3f}",
                flush=True,
            )

    write_manifest(paths["manifest"], {
        "scenario": scenario,
        "agent_roles": agent_roles,
        "model_name": model_name,
        "model_path": engine.model_path,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "debug_subset": debug_subset,
        "sample_seed": sample_seed,
        "resume": resume,
        "capture_trace": capture_trace,
        "total_tasks": total,
        "already_done_before_run": skipped,
        "newly_processed": len(remaining),
        "summary_csv_nested": str(paths["nested_summary"]),
        "summary_csv_flat": str(paths["flat_summary"]),
        "full_csv_nested": str(paths["nested_full"]),
        "trace_dir": str(paths["trace_dir"]),
        "generation_config": asdict(engine.gen_config),
        "branching_config": asdict(engine.branch_config),
        "wall_time_seconds": time.time() - start_time,
    })

    return {
        "total_tasks": total,
        "skipped": skipped,
        "newly_processed": len(remaining),
        "paths": {k: str(v) for k, v in paths.items()},
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Branching-token MOVE/STAY probability estimator for the Schelling simulator.",
    )
    parser.add_argument("--model-path", required=True, help="Path to the GGUF model file")
    parser.add_argument("--model-name", required=True, help="Slug used in output file names")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--n-ctx", type=int, default=512)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--n-gpu-layers", type=int, default=0)

    parser.add_argument("--scenarios", nargs="+", default=["baseline"],
                        help="Scenario keys from context_scenarios.CONTEXT_SCENARIOS (or 'all')")
    parser.add_argument("--agent-roles", nargs="+", default=["type_a", "type_b"],
                        choices=["type_a", "type_b"])

    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--candidate-top-n", type=int, default=1)
    parser.add_argument("--candidate-top-n-cap", type=int, default=2048)
    parser.add_argument("--min-step-retained-mass", type=float, default=0.999)
    parser.add_argument("--early-stop-move-stay-mass", type=float, default=0.999)

    parser.add_argument("--capture-trace", action="store_true",
                        help="Write gzipped per-arrangement branching trace JSON for replay")
    parser.add_argument("--output-root", default=str(_THIS_DIR.parent / "llm_log_probs"),
                        help="Root directory for outputs (default: <repo>/llm_log_probs)")
    parser.add_argument("--debug-subset", type=int, default=None,
                        help="Process only N arrangements per scenario, sampled randomly if --sample-seed is set")
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="Seed used when randomly sampling the debug subset (default: 0)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (agent_role, arrangement_code) pairs already present in summary CSV")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    scenarios: list[str]
    if len(args.scenarios) == 1 and args.scenarios[0].lower() == "all":
        scenarios = sorted(CONTEXT_SCENARIOS.keys())
    else:
        scenarios = list(args.scenarios)
        for scen in scenarios:
            if scen not in CONTEXT_SCENARIOS:
                raise SystemExit(
                    f"Unknown scenario '{scen}'. Available: "
                    f"{sorted(CONTEXT_SCENARIOS.keys())} or 'all'"
                )

    debug_subset = args.debug_subset
    capture_trace = args.capture_trace or (debug_subset is not None)

    gen_config = GenerationConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n_ctx=args.n_ctx,
    )
    branch_config = BranchingConfig(
        beam_width=args.beam_width,
        candidate_top_n=args.candidate_top_n,
        candidate_top_n_cap=args.candidate_top_n_cap,
        min_step_retained_mass=args.min_step_retained_mass,
        early_stop_move_stay_mass=args.early_stop_move_stay_mass,
    )

    engine = BranchingEngine(
        model_path=args.model_path,
        gen_config=gen_config,
        branch_config=branch_config,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
    )

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[config] model={args.model_name} temp={args.temperature} scenarios={scenarios} "
            f"roles={args.agent_roles} debug_subset={debug_subset} sample_seed={args.sample_seed} "
            f"capture_trace={capture_trace}",
          flush=True)

    for scenario in scenarios:
        try:
            run_scenario_sweep(
                engine=engine,
                scenario=scenario,
                agent_roles=list(args.agent_roles),
                output_root=output_root,
                model_name=args.model_name,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                debug_subset=debug_subset,
                sample_seed=args.sample_seed,
                resume=args.resume,
                capture_trace=capture_trace,
            )
        except KeyboardInterrupt:
            print(f"[sweep] interrupted during scenario={scenario}; partial results kept.", flush=True)
            raise
        except Exception as exc:
            print(f"[sweep] ERROR in scenario={scenario}: {type(exc).__name__}: {exc}", flush=True)
            raise

    print("[sweep] done.", flush=True)


if __name__ == "__main__":
    main()
