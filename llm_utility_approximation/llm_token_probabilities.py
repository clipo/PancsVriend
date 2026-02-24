"""Token probability extraction for LLM responses.

This script supports two modes:

1) Single-prompt mode
   Sends one prompt (optionally repeated) to a configured chat-completions
   endpoint and extracts token-level probabilities from returned logprobs.

2) Scenario mode (--scenario)
   Enumerates all valid local 3x3 Schelling neighborhoods for a 10x10 grid,
   where walls (#) are only those possible from real boundary positions,
   and queries the model for each context.

	Use --scenario all to iterate every scenario in context_scenarios.py.

Outputs are saved with model-prefixed filenames:
  - <model>_token_probabilities.csv
  - <model>_stay_move_probability_split.csv
  - <model>_stay_move_probability_split_summary.csv

If the endpoint/model does not return token logprobs, the script fails with a
clear error message.

Example usage:

	# Single-prompt mode
	python llm_utility_approximation/llm_token_probabilities.py \
		--prompt "Respond with exactly: MOVE" --num-samples 3 --llm-model "phi4:latest"

	# Scenario mode (all valid neighborhood contexts)
	python llm_utility_approximation/llm_token_probabilities.py \
		--scenario baseline --agent-role both --repeats-per-context 1

	# All scenarios (forces both agent roles)
	python llm_utility_approximation/llm_token_probabilities.py \
		--scenario all --llm-model "mixtral:8x22b-instruct"

Online/local endpoint routing is controlled by Use_ONLINE_API in this file.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

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

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_PROMPT = """You are a red team resident living in a neighborhood, considering whether to move to a different house.

# O E
# X S
# O O

Where:
- X = Your current position (center)
- S = neighbors who are also red team residents like you
- O = neighbors from the blue team resident community
- E = empty houses you could move to
- # = area outside the neighborhood

As a real person, you have your own preferences about neighborhood composition, but you also consider practical factors like proximity to friends, community ties, and comfort level.

Based on this neighborhood, do you want to move to a different position or stay where you are?

IMPORTANT: Respond with ONLY one word: MOVE or STAY. Do not provide any explanation, reasoning, or additional text.

Your response:"""

LLM_MODEL = "mixtral:8x22b-instruct"  #"phi4:latest"
LOCAL_OLLAMA_URLS = [
	"http://localhost:11434/api/chat",
	"http://127.0.0.1:11434/api/chat",
	"http://localhost:11434/v1/chat/completions",
	"http://127.0.0.1:11434/v1/chat/completions",
]
ONLINE_API_URL = "https://chat.binghamton.edu/ollama/api/chat"
ONLINE_API_KEY = "sk-571df6eec7f5495faef553ab5cb2c67a"
Use_ONLINE_API = True

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

@dataclass
class TokenRecord:
	"""One token probability record from one sampled completion."""

	sample_index: int
	token_index: int
	token: str
	logprob: float
	probability: float
	top_rank: int
	top_token: str
	top_logprob: float
	top_probability: float


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Extract token probabilities from LLM responses")
	parser.add_argument(
		"--prompt",
		type=str,
		default=None,
		help="Prompt text (if omitted, uses DEFAULT_PROMPT in this file)",
	)
	parser.add_argument(
		"--num-samples",
		type=int,
		default=1,
		help="Number of repeated generations for the same prompt",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=0.0,
		help="Sampling temperature",
	)
	parser.add_argument(
		"--max-tokens",
		type=int,
		default=24,
		help="Maximum generated tokens",
	)
	parser.add_argument(
		"--top-logprobs",
		type=int,
		default=20,
		help="Top-k alternatives to request per generated token",
	)
	parser.add_argument(
		"--timeout",
		type=int,
		default=30,
		help="Request timeout in seconds",
	)
	parser.add_argument(
		"--llm-model",
		type=str,
		default=None,
		help="Local Ollama model name (default: llama3.1)",
	)
	parser.add_argument(
		"--llm-url",
		type=str,
		default=None,
		help="Optional localhost Ollama OpenAI-compatible endpoint URL",
	)
	parser.add_argument(
		"--scenario",
		type=str,
		default=None,
		help="Scenario key from context_scenarios.py, or 'all' to run every scenario",
	)
	parser.add_argument(
		"--agent-role",
		type=str,
		choices=["type_a", "type_b", "both"],
		default="both",
		help="Which scenario agent role(s) to evaluate in scenario mode",
	)
	parser.add_argument(
		"--repeats-per-context",
		type=int,
		default=None,
		help="How many repeated generations per neighborhood arrangement in scenario mode (defaults to --num-samples)",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=str(SCRIPT_DIR),
		help="Directory where CSVs and plot are written (default: script folder)",
	)
	parser.add_argument(
		"--resume",
		action="store_true",
		help="Resume scenario runs by skipping scenarios already listed in the per-scenario progress manifest",
	)
	return parser.parse_args()


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


def _is_local_url(url: str) -> bool:
	parsed = urlparse(url)
	if parsed.scheme not in {"http", "https"}:
		return False
	hostname = (parsed.hostname or "").lower()
	return hostname in {"localhost", "127.0.0.1", "::1"}


def _exp_logprob(value: float) -> float:
	if value < -745:
		return 0.0
	return float(math.exp(value))


def _extract_message_content(choice: dict[str, Any]) -> str:
	message = choice.get("message", {})
	if isinstance(message, dict):
		content = message.get("content", "")
		if isinstance(content, str):
			return content
	return ""


def _extract_native_ollama_logprobs(response_json: dict[str, Any], sample_index: int) -> tuple[str, list[TokenRecord]]:
	message = response_json.get("message", {})
	response_text = ""
	if isinstance(message, dict):
		content = message.get("content", "")
		if isinstance(content, str):
			response_text = content.strip()

	logprobs = response_json.get("logprobs")
	if not isinstance(logprobs, list):
		return response_text, []

	records: list[TokenRecord] = []
	for token_index, token_item in enumerate(logprobs):
		if not isinstance(token_item, dict):
			continue

		token_text = str(token_item.get("token", ""))
		logprob_val = token_item.get("logprob")
		if not isinstance(logprob_val, (int, float)):
			continue

		top_alternatives = token_item.get("top_logprobs", [])
		if not isinstance(top_alternatives, list) or len(top_alternatives) == 0:
			top_alternatives = [{"token": token_text, "logprob": float(logprob_val)}]

		for rank, alt in enumerate(top_alternatives):
			if not isinstance(alt, dict):
				continue
			alt_token = str(alt.get("token", ""))
			alt_logprob = alt.get("logprob")
			if not isinstance(alt_logprob, (int, float)):
				continue

			records.append(
				TokenRecord(
					sample_index=sample_index,
					token_index=token_index,
					token=token_text,
					logprob=float(logprob_val),
					probability=_exp_logprob(float(logprob_val)),
					top_rank=rank,
					top_token=alt_token,
					top_logprob=float(alt_logprob),
					top_probability=_exp_logprob(float(alt_logprob)),
				)
			)

	return response_text, records


def _parse_chat_content_logprobs(choice: dict[str, Any], sample_index: int) -> list[TokenRecord]:
	logprobs = choice.get("logprobs")
	if not isinstance(logprobs, dict):
		return []

	content_items = logprobs.get("content")
	if not isinstance(content_items, list):
		return []

	records: list[TokenRecord] = []
	for token_index, token_item in enumerate(content_items):
		if not isinstance(token_item, dict):
			continue

		token_text = str(token_item.get("token", ""))
		logprob_val = token_item.get("logprob")
		if not isinstance(logprob_val, (int, float)):
			continue

		top_alternatives = token_item.get("top_logprobs", [])
		if not isinstance(top_alternatives, list) or len(top_alternatives) == 0:
			top_alternatives = [{"token": token_text, "logprob": float(logprob_val)}]

		for rank, alt in enumerate(top_alternatives):
			if not isinstance(alt, dict):
				continue
			alt_token = str(alt.get("token", ""))
			alt_logprob = alt.get("logprob")
			if not isinstance(alt_logprob, (int, float)):
				continue

			records.append(
				TokenRecord(
					sample_index=sample_index,
					token_index=token_index,
					token=token_text,
					logprob=float(logprob_val),
					probability=_exp_logprob(float(logprob_val)),
					top_rank=rank,
					top_token=alt_token,
					top_logprob=float(alt_logprob),
					top_probability=_exp_logprob(float(alt_logprob)),
				)
			)

	return records


def _parse_legacy_logprobs(choice: dict[str, Any], sample_index: int) -> list[TokenRecord]:
	logprobs = choice.get("logprobs")
	if not isinstance(logprobs, dict):
		return []

	tokens = logprobs.get("tokens")
	values = logprobs.get("token_logprobs") or logprobs.get("logprobs")
	top_logprobs = logprobs.get("top_logprobs")

	if not isinstance(tokens, list) or not isinstance(values, list) or len(tokens) != len(values):
		return []

	records: list[TokenRecord] = []
	for token_index, (token_text, token_lp) in enumerate(zip(tokens, values)):
		if not isinstance(token_lp, (int, float)):
			continue

		alts_for_token: list[dict[str, Any]] = []
		if isinstance(top_logprobs, list) and token_index < len(top_logprobs):
			entry = top_logprobs[token_index]
			if isinstance(entry, dict):
				for alt_token, alt_lp in entry.items():
					if isinstance(alt_lp, (int, float)):
						alts_for_token.append({"token": str(alt_token), "logprob": float(alt_lp)})

		if len(alts_for_token) == 0:
			alts_for_token = [{"token": str(token_text), "logprob": float(token_lp)}]

		for rank, alt in enumerate(alts_for_token):
			records.append(
				TokenRecord(
					sample_index=sample_index,
					token_index=token_index,
					token=str(token_text),
					logprob=float(token_lp),
					probability=_exp_logprob(float(token_lp)),
					top_rank=rank,
					top_token=str(alt["token"]),
					top_logprob=float(alt["logprob"]),
					top_probability=_exp_logprob(float(alt["logprob"])),
				)
			)

	return records


def extract_token_records(response_json: dict[str, Any], sample_index: int) -> tuple[str, list[TokenRecord]]:
	if isinstance(response_json.get("logprobs"), list):
		text, native_records = _extract_native_ollama_logprobs(response_json, sample_index)
		if len(native_records) > 0:
			return text, native_records

	choices = response_json.get("choices")
	if not isinstance(choices, list) or len(choices) == 0 or not isinstance(choices[0], dict):
		raise RuntimeError("Response does not contain a valid 'choices[0]' object")

	choice = choices[0]
	text = _extract_message_content(choice).strip()

	records = _parse_chat_content_logprobs(choice, sample_index)
	if len(records) > 0:
		return text, records

	records = _parse_legacy_logprobs(choice, sample_index)
	if len(records) > 0:
		return text, records

	raise RuntimeError(
		"Endpoint/model did not return token logprobs. "
		"Requested logprobs but no supported logprobs fields were found in the response."
	)


def request_with_logprobs(
	urls: list[str],
	model: str,
	prompt: str,
	temperature: float,
	max_tokens: int,
	top_logprobs: int,
	timeout: int,
	session: requests.Session,
	api_key: str | None = None,
) -> tuple[dict[str, Any], float, str]:
	headers = {"Content-Type": "application/json"}
	if api_key:
		headers["Authorization"] = f"Bearer {api_key}"

	last_error: Exception | None = None
	for url in urls:
		start = time.time()
		try:
			if url.rstrip("/").endswith("/api/chat"):
				payload: dict[str, Any] = {
					"model": model,
					"messages": [{"role": "user", "content": prompt}],
					"stream": False,
					"logprobs": True,
					"top_logprobs": top_logprobs,
					"options": {
						"temperature": temperature,
						"num_predict": max_tokens,
					},
				}
			else:
				payload = {
					"model": model,
					"messages": [{"role": "user", "content": prompt}],
					"stream": False,
					"temperature": temperature,
					"max_tokens": max_tokens,
					"logprobs": True,
					"top_logprobs": top_logprobs,
				}

			response = session.post(url, headers=headers, json=payload, timeout=timeout)
			elapsed = time.time() - start
			response.raise_for_status()
			return response.json(), elapsed, url
		except Exception as exc:
			last_error = exc

	raise RuntimeError(
		"Failed to query local Ollama endpoint. Ensure Ollama is running and OpenAI-compatible API is enabled on localhost:11434."
	) from last_error


def ensure_directory(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def _write_per_scenario_progress_manifest(
	output_dir: str,
	model: str,
	per_scenario_outputs: dict[str, dict[str, str]],
) -> str:
	ensure_directory(output_dir)
	model_slug = _sanitize_model_for_path_component(model)
	manifest_path = os.path.join(output_dir, f"{model_slug}_per_scenario_progress.json")
	payload = {
		"model": model,
		"completed_scenarios": sorted(per_scenario_outputs.keys()),
		"outputs": per_scenario_outputs,
	}
	with open(manifest_path, "w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2)
	return manifest_path


def _read_per_scenario_progress_manifest(
	output_dir: str,
	model: str,
) -> dict[str, dict[str, str]]:
	model_slug = _sanitize_model_for_path_component(model)
	manifest_path = os.path.join(output_dir, f"{model_slug}_per_scenario_progress.json")
	if not os.path.exists(manifest_path):
		return {}

	try:
		with open(manifest_path, "r", encoding="utf-8") as handle:
			payload = json.load(handle)
	except Exception:
		return {}

	outputs = payload.get("outputs", {})
	if isinstance(outputs, dict):
		validated: dict[str, dict[str, str]] = {}
		for scenario_key, value in outputs.items():
			if isinstance(scenario_key, str) and isinstance(value, dict):
				validated[scenario_key] = {
					k: v for k, v in value.items() if isinstance(k, str) and isinstance(v, str)
				}
		return validated

	return {}


def _sanitize_model_for_path_component(name: str) -> str:
	# Windows-disallowed chars for a path segment: < > : " / \ | ? * and controls.
	sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '-', name.strip())
	sanitized = sanitized.rstrip(' .')
	sanitized = re.sub(r'-{2,}', '-', sanitized)
	return sanitized or "unknown-model"


def detect_local_model_name(session: requests.Session, timeout: int) -> str:
	tags_url = "http://localhost:11434/api/tags"
	response = session.get(tags_url, timeout=timeout)
	response.raise_for_status()
	data = response.json()
	models = data.get("models", [])
	if isinstance(models, list) and len(models) > 0 and isinstance(models[0], dict):
		name = models[0].get("name")
		if isinstance(name, str) and name.strip():
			return name.strip()
	return LLM_MODEL


def _normalize_token_fragment(token: str) -> str:
	clean = token.strip().upper()
	clean = re.sub(r"[^A-Z]", "", clean)
	return clean


def _safe_log(probability: float) -> float:
	if probability <= 0:
		return float("-inf")
	return float(math.log(probability))


def _token_candidates_from_sample_records(sample_records: list[TokenRecord]) -> list[list[tuple[str, float]]]:
	if len(sample_records) == 0:
		return []

	token_df = pd.DataFrame([record.__dict__ for record in sample_records])
	candidates_per_token: list[list[tuple[str, float]]] = []
	for token_index in sorted(token_df["token_index"].unique()):
		token_rows = token_df[token_df["token_index"] == token_index]
		candidates = [(str(row["top_token"]), float(row["top_probability"])) for _, row in token_rows.iterrows()]
		candidates_per_token.append(candidates)

	return candidates_per_token


def _compute_label_metrics(token_candidates: list[list[tuple[str, float]]], labels: tuple[str, str] = ("STAY", "MOVE")) -> dict[str, float]:
	label_a, label_b = labels
	p_a = _probability_of_label_from_candidates(token_candidates, label_a)
	p_b = _probability_of_label_from_candidates(token_candidates, label_b)
	total = p_a + p_b
	share_a = (p_a / total) if total > 0 else 0.0
	share_b = (p_b / total) if total > 0 else 0.0

	return {
		f"{label_a.lower()}_probability": p_a,
		f"{label_b.lower()}_probability": p_b,
		"total_labeled_probability": total,
		f"{label_a.lower()}_logprob": _safe_log(p_a),
		f"{label_b.lower()}_logprob": _safe_log(p_b),
		f"{label_a.lower()}_share": share_a,
		f"{label_b.lower()}_share": share_b,
	}


def evaluate_scenario_permutations(
	scenario_key: str,
	agent_role: str,
	repeats_per_context: int,
	model: str,
	urls: list[str],
	temperature: float,
	max_tokens: int,
	top_logprobs: int,
	timeout: int,
	session: requests.Session,
	api_key: str | None = None,
) -> tuple[list[TokenRecord], pd.DataFrame, str]:
	if scenario_key not in CONTEXT_SCENARIOS:
		available = ", ".join(sorted(CONTEXT_SCENARIOS.keys()))
		raise ValueError(f"Unknown --scenario '{scenario_key}'. Available: {available}")
	if repeats_per_context < 1:
		raise ValueError("--repeats-per-context must be >= 1")

	scenario_info = CONTEXT_SCENARIOS[scenario_key]
	role_configs: list[tuple[str, str, str]] = []
	if agent_role in {"type_a", "both"}:
		role_configs.append(("type_a", str(scenario_info["type_a"]), str(scenario_info["type_b"])))
	if agent_role in {"type_b", "both"}:
		role_configs.append(("type_b", str(scenario_info["type_b"]), str(scenario_info["type_a"])))

	all_records: list[TokenRecord] = []
	context_rows: list[dict[str, Any]] = []
	last_url_used = ""
	global_sample_idx = 0
	all_neighbor_arrangements = generate_all_valid_schelling_neighbors(SCHELLING_GRID_SIZE)
	total_requests = len(role_configs) * len(all_neighbor_arrangements) * repeats_per_context
	processed_requests = 0
	progress_start_time = time.time()
	last_progress_log_time = progress_start_time

	print(
		f"[progress] Starting scenario '{scenario_key}' with {len(role_configs)} role(s), "
		f"{len(all_neighbor_arrangements)} contexts, {repeats_per_context} repeat(s) each "
		f"({total_requests} total requests).",
		flush=True,
	)

	for role_name, role_label, opposite_label in role_configs:
		for arrangement_idx, neighbors in enumerate(all_neighbor_arrangements):
			context_grid = generate_neighbor_context(neighbors)
			prompt = build_scenario_prompt(scenario_key, context_grid, role_label, opposite_label)
			arrangement_code = "".join(neighbors)
			num_same = int(sum(1 for item in neighbors if item == "S"))
			num_opposite = int(sum(1 for item in neighbors if item == "O"))
			num_empty = int(sum(1 for item in neighbors if item == "E"))
			num_wall = int(sum(1 for item in neighbors if item == "#"))
			ratio_similar = num_same / 8

			for repeat_index in range(repeats_per_context):
				response_json, elapsed, used_url = request_with_logprobs(
					urls=urls,
					model=model,
					prompt=prompt,
					temperature=temperature,
					max_tokens=max_tokens,
					top_logprobs=top_logprobs,
					timeout=timeout,
					session=session,
					api_key=api_key,
				)
				last_url_used = used_url

				response_text, sample_records = extract_token_records(response_json, global_sample_idx)
				all_records.extend(sample_records)

				token_candidates = _token_candidates_from_sample_records(sample_records)
				label_metrics = _compute_label_metrics(token_candidates, labels=("STAY", "MOVE"))

				context_rows.append(
					{
						"scenario": scenario_key,
						"agent_role": role_name,
						"agent_label": role_label,
						"opposite_label": opposite_label,
						"arrangement_index": arrangement_idx,
						"arrangement_code": arrangement_code,
						"context": context_grid,
						"repeat_index": repeat_index,
						"sample_index": global_sample_idx,
						"num_similar": num_same,
						"num_opposite": num_opposite,
						"num_empty": num_empty,
						"num_wall": num_wall,
						"ratio_similar": ratio_similar,
						"response_text": response_text,
						"request_url": used_url,
						"elapsed_seconds": elapsed,
						**label_metrics,
					}
				)
				global_sample_idx += 1
				processed_requests += 1

				now = time.time()
				if now - last_progress_log_time >= 50 or processed_requests == total_requests:
					elapsed_seconds = now - progress_start_time
					completion_ratio = (processed_requests / total_requests) if total_requests > 0 else 1.0
					percent_complete = completion_ratio * 100
					requests_per_second = (processed_requests / elapsed_seconds) if elapsed_seconds > 0 else 0.0
					remaining_requests = total_requests - processed_requests
					eta_seconds = (remaining_requests / requests_per_second) if requests_per_second > 0 else float("inf")
					eta_display = f"{eta_seconds:.1f}s" if math.isfinite(eta_seconds) else "unknown"

					print(
						f"[progress][scenario={scenario_key}] {processed_requests}/{total_requests} "
						f"({percent_complete:.2f}%) | elapsed={elapsed_seconds:.1f}s | "
						f"eta={eta_display} | current_role={role_name}",
						flush=True,
					)
					last_progress_log_time = now

	print("[progress] Scenario evaluation complete.", flush=True)

	context_df = pd.DataFrame(context_rows)
	return all_records, context_df, last_url_used


def _probability_of_label_from_candidates(token_candidates: list[list[tuple[str, float]]], label: str) -> float:
	label = label.upper()
	label_len = len(label)
	state: dict[int, float] = {0: 1.0}

	for position_candidates in token_candidates:
		next_state: dict[int, float] = collections.defaultdict(float)
		for prefix_len, prefix_prob in state.items():
			if prefix_prob <= 0:
				continue

			if prefix_len >= label_len:
				next_state[label_len] += prefix_prob
				continue

			for token_text, token_prob in position_candidates:
				if token_prob <= 0:
					continue
				fragment = _normalize_token_fragment(token_text)
				if not fragment:
					continue
				remaining = label[prefix_len:]
				if remaining.startswith(fragment):
					next_prefix_len = min(label_len, prefix_len + len(fragment))
					next_state[next_prefix_len] += prefix_prob * token_prob

		state = dict(next_state)
		if len(state) == 0:
			break

	return float(state.get(label_len, 0.0))


def build_label_probability_split(records: list[TokenRecord], labels: tuple[str, str] = ("STAY", "MOVE")) -> tuple[pd.DataFrame, pd.DataFrame]:
	if len(records) == 0:
		empty_per_sample = pd.DataFrame(
			columns=[
				"sample_index",
				f"{labels[0].lower()}_probability",
				f"{labels[1].lower()}_probability",
				"total_labeled_probability",
				f"{labels[0].lower()}_logprob",
				f"{labels[1].lower()}_logprob",
				f"{labels[0].lower()}_share",
				f"{labels[1].lower()}_share",
			]
		)
		empty_summary = pd.DataFrame(
			columns=[
				"metric",
				f"{labels[0].lower()}_value",
				f"{labels[1].lower()}_value",
				"total_labeled_value",
			]
		)
		return empty_per_sample, empty_summary

	rows = [r.__dict__ for r in records]
	records_df = pd.DataFrame(rows)

	per_sample_rows: list[dict[str, Any]] = []
	for sample_index in sorted(records_df["sample_index"].unique()):
		sample_df = records_df[records_df["sample_index"] == sample_index].copy()
		sample_records = [TokenRecord(**row) for row in sample_df.to_dict(orient="records")]
		token_candidates = _token_candidates_from_sample_records(sample_records)
		label_metrics = _compute_label_metrics(token_candidates, labels=labels)
		label_a, label_b = labels

		per_sample_rows.append(
			{
				"sample_index": int(sample_index),
				**label_metrics,
			}
		)

	per_sample_df = pd.DataFrame(per_sample_rows).sort_values("sample_index")

	if len(per_sample_df) > 0:
		label_a, label_b = labels
		summary_rows = [
			{
				"metric": "mean_probability",
				f"{label_a.lower()}_value": float(per_sample_df[f"{label_a.lower()}_probability"].mean()),
				f"{label_b.lower()}_value": float(per_sample_df[f"{label_b.lower()}_probability"].mean()),
				"total_labeled_value": float(per_sample_df["total_labeled_probability"].mean()),
			},
			{
				"metric": "mean_logprob",
				f"{label_a.lower()}_value": float(per_sample_df[f"{label_a.lower()}_logprob"].replace([float("-inf")], math.nan).mean()),
				f"{label_b.lower()}_value": float(per_sample_df[f"{label_b.lower()}_logprob"].replace([float("-inf")], math.nan).mean()),
				"total_labeled_value": math.nan,
			},
			{
				"metric": "mean_share",
				f"{label_a.lower()}_value": float(per_sample_df[f"{label_a.lower()}_share"].mean()),
				f"{label_b.lower()}_value": float(per_sample_df[f"{label_b.lower()}_share"].mean()),
				"total_labeled_value": 1.0,
			},
		]
		summary_df = pd.DataFrame(summary_rows)
	else:
		summary_df = pd.DataFrame(
			columns=[
				"metric",
				f"{labels[0].lower()}_value",
				f"{labels[1].lower()}_value",
				"total_labeled_value",
			]
		)

	return per_sample_df, summary_df


def summarize_label_probability_split(label_split_df: pd.DataFrame) -> pd.DataFrame:
	if len(label_split_df) == 0:
		return pd.DataFrame(
			columns=[
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
			]
		)

	group_cols = [
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
	]

	if not set(group_cols).issubset(set(label_split_df.columns)):
		summary_df = pd.DataFrame(
			[
				{
					"scenario": "single_prompt",
					"agent_role": "single_prompt",
					"agent_label": "single_prompt",
					"opposite_label": "single_prompt",
					"num_similar": math.nan,
					"ratio_similar": math.nan,
					"num_trials": int(len(label_split_df)),
					"mean_stay_probability": float(label_split_df["stay_probability"].mean()),
					"mean_move_probability": float(label_split_df["move_probability"].mean()),
					"mean_total_labeled_probability": float(label_split_df["total_labeled_probability"].mean()),
					"mean_stay_logprob": float(label_split_df["stay_logprob"].replace([float("-inf")], math.nan).mean()),
					"mean_move_logprob": float(label_split_df["move_logprob"].replace([float("-inf")], math.nan).mean()),
					"mean_stay_share": float(label_split_df["stay_share"].mean()),
					"mean_move_share": float(label_split_df["move_share"].mean()),
				}
			]
		)
		return summary_df

	summary_df = (
		label_split_df.groupby(group_cols, as_index=False)
		.agg(
			num_trials=("sample_index", "count"),
			mean_stay_probability=("stay_probability", "mean"),
			mean_move_probability=("move_probability", "mean"),
			mean_total_labeled_probability=("total_labeled_probability", "mean"),
			mean_stay_logprob=("stay_logprob", lambda s: s.replace([float("-inf")], math.nan).mean()),
			mean_move_logprob=("move_logprob", lambda s: s.replace([float("-inf")], math.nan).mean()),
			mean_stay_share=("stay_share", "mean"),
			mean_move_share=("move_share", "mean"),
		)
		.sort_values(group_cols)
	)

	return summary_df


def save_outputs(
	output_dir: str,
	model: str,
	scenario: str | None,
	records: list[TokenRecord],
	label_split_df_override: pd.DataFrame | None = None,
	label_split_summary_override: pd.DataFrame | None = None,
) -> tuple[str, str, str]:
	ensure_directory(output_dir)
	model_slug = _sanitize_model_for_path_component(model)
	scenario_slug = _sanitize_model_for_path_component(scenario) if scenario else "single_prompt"

	tokens_df = pd.DataFrame([r.__dict__ for r in records])
	tokens_csv = os.path.join(output_dir, f"{model_slug}_{scenario_slug}_token_probabilities.csv")
	tokens_df.to_csv(tokens_csv, index=False)

	if label_split_df_override is None:
		label_split_df, label_split_summary_df = build_label_probability_split(records, labels=("STAY", "MOVE"))
	else:
		label_split_df = label_split_df_override
		label_split_summary_df = label_split_summary_override
		if label_split_summary_df is None:
			label_split_summary_df = summarize_label_probability_split(label_split_df)

	label_split_csv = os.path.join(output_dir, f"{model_slug}_{scenario_slug}_stay_move_probability_split.csv")
	label_split_df.to_csv(label_split_csv, index=False)
	label_split_summary_csv = os.path.join(output_dir, f"{model_slug}_{scenario_slug}_stay_move_probability_split_summary.csv")
	label_split_summary_df.to_csv(label_split_summary_csv, index=False)

	return tokens_csv, label_split_csv, label_split_summary_csv


def main() -> None:
	args = parse_args()

	if args.num_samples < 1:
		raise ValueError("--num-samples must be >= 1")
	if args.top_logprobs < 1:
		raise ValueError("--top-logprobs must be >= 1")

	prompt = args.prompt if args.prompt is not None else DEFAULT_PROMPT
	repeats_per_context = args.repeats_per_context if args.repeats_per_context is not None else args.num_samples
	if repeats_per_context < 1:
		raise ValueError("--repeats-per-context must be >= 1")

	scenario_keys: list[str] | None = None
	effective_agent_role = args.agent_role
	if args.scenario:
		if args.scenario.lower() == "all":
			scenario_keys = sorted(CONTEXT_SCENARIOS.keys())
			effective_agent_role = "both"
		else:
			if args.scenario not in CONTEXT_SCENARIOS:
				available = ", ".join(sorted(CONTEXT_SCENARIOS.keys()))
				raise ValueError(f"Unknown --scenario '{args.scenario}'. Available: {available}, or 'all'")
			scenario_keys = [args.scenario]

	api_key: str | None = None
	if Use_ONLINE_API:
		model = args.llm_model if args.llm_model else LLM_MODEL
		candidate_urls = [args.llm_url] if args.llm_url else [ONLINE_API_URL]
		api_key = ONLINE_API_KEY
	else:
		if args.llm_model:
			model = args.llm_model
		else:
			with requests.Session() as detect_session:
				model = detect_local_model_name(detect_session, args.timeout)

		if args.llm_url:
			if not _is_local_url(args.llm_url):
				raise ValueError("--llm-url must point to localhost/127.0.0.1 for local-only execution")
			candidate_urls = [args.llm_url]
		else:
			candidate_urls = LOCAL_OLLAMA_URLS

		for candidate_url in candidate_urls:
			if not _is_local_url(candidate_url):
				raise ValueError(f"Non-local URL is not allowed: {candidate_url}")

	records: list[TokenRecord] = []
	url_used = ""
	label_split_df_override: pd.DataFrame | None = None
	label_split_summary_override: pd.DataFrame | None = None
	outputs_payload: dict[str, Any]
	resume_outputs: dict[str, dict[str, str]] = {}

	if args.resume:
		resume_outputs = _read_per_scenario_progress_manifest(args.output_dir, model)
		if len(resume_outputs) > 0:
			print(
				f"[progress] Resume enabled: found {len(resume_outputs)} completed scenario(s) in manifest.",
				flush=True,
			)
		if scenario_keys is not None:
			scenario_keys = [scenario_key for scenario_key in scenario_keys if scenario_key not in resume_outputs]
			if len(scenario_keys) == 0:
				outputs_payload = {"per_scenario": resume_outputs}
				result = {
					"success": True,
					"model": model,
					"url": "",
					"num_samples": args.num_samples,
					"scenario": args.scenario,
					"agent_role": effective_agent_role,
					"grid_size": SCHELLING_GRID_SIZE,
					"non_wall_context_elements": NON_WALL_CONTEXT_ELEMENTS,
					"repeats_per_context": repeats_per_context,
					"prompt": prompt,
					"outputs": outputs_payload,
					"resume": True,
					"message": "All requested scenarios are already completed.",
				}
				print(json.dumps(result, indent=2))
				return
		else:
			if args.scenario and args.scenario in resume_outputs:
				outputs_payload = {"per_scenario": resume_outputs}
				outputs_payload.update(resume_outputs[args.scenario])
				result = {
					"success": True,
					"model": model,
					"url": "",
					"num_samples": args.num_samples,
					"scenario": args.scenario,
					"agent_role": effective_agent_role,
					"grid_size": SCHELLING_GRID_SIZE,
					"non_wall_context_elements": NON_WALL_CONTEXT_ELEMENTS,
					"repeats_per_context": repeats_per_context,
					"prompt": prompt,
					"outputs": outputs_payload,
					"resume": True,
					"message": f"Scenario '{args.scenario}' is already completed.",
				}
				print(json.dumps(result, indent=2))
				return

	with requests.Session() as session:
		if scenario_keys is not None:
			per_scenario_outputs: dict[str, dict[str, str]] = dict(resume_outputs)
			for scenario_key in scenario_keys:
				scenario_records, scenario_label_split_df, used_url = evaluate_scenario_permutations(
					scenario_key=scenario_key,
					agent_role=effective_agent_role,
					repeats_per_context=repeats_per_context,
					model=model,
					urls=candidate_urls,
					temperature=args.temperature,
					max_tokens=args.max_tokens,
					top_logprobs=args.top_logprobs,
					timeout=args.timeout,
					session=session,
					api_key=api_key,
				)
				url_used = used_url
				records.extend(scenario_records)

				scenario_label_split_summary_df = summarize_label_probability_split(scenario_label_split_df)
				tokens_csv, label_split_csv, label_split_summary_csv = save_outputs(
					args.output_dir,
					model,
					scenario_key,
					scenario_records,
					label_split_df_override=scenario_label_split_df,
					label_split_summary_override=scenario_label_split_summary_df,
				)

				per_scenario_outputs[scenario_key] = {
					"token_csv": tokens_csv,
					"stay_move_split_csv": label_split_csv,
					"stay_move_split_summary_csv": label_split_summary_csv,
				}

				manifest_path = _write_per_scenario_progress_manifest(
					args.output_dir,
					model,
					per_scenario_outputs,
				)
				print(
					f"[progress][scenario={scenario_key}] Saved outputs and updated manifest: {manifest_path}",
					flush=True,
				)

			outputs_payload = {"per_scenario": per_scenario_outputs}

			if len(scenario_keys) == 1:
				only_key = scenario_keys[0]
				outputs_payload.update(per_scenario_outputs[only_key])
		else:
			for sample_idx in range(args.num_samples):
				response_json, elapsed, used_url = request_with_logprobs(
					urls=candidate_urls,
					model=model,
					prompt=prompt,
					temperature=args.temperature,
					max_tokens=args.max_tokens,
					top_logprobs=args.top_logprobs,
					timeout=args.timeout,
					session=session,
					api_key=api_key,
				)
				url_used = used_url
				response_text, sample_records = extract_token_records(response_json, sample_idx)
				records.extend(sample_records)
				_ = response_text

			# Single-prompt mode keeps the original single-output behavior
			tokens_csv, label_split_csv, label_split_summary_csv = save_outputs(
				args.output_dir,
				model,
				None,
				records,
				label_split_df_override=label_split_df_override,
				label_split_summary_override=label_split_summary_override,
			)
			outputs_payload = {
				"token_csv": tokens_csv,
				"stay_move_split_csv": label_split_csv,
				"stay_move_split_summary_csv": label_split_summary_csv,
			}

	if len(records) == 0:
		raise RuntimeError("No token probability records were extracted")

	result = {
		"success": True,
		"model": model,
		"url": url_used,
		"num_samples": args.num_samples,
		"scenario": args.scenario,
		"agent_role": effective_agent_role,
		"grid_size": SCHELLING_GRID_SIZE,
		"non_wall_context_elements": NON_WALL_CONTEXT_ELEMENTS,
		"repeats_per_context": repeats_per_context,
		"prompt": prompt,
		"outputs": outputs_payload,
	}
	print(json.dumps(result, indent=2))


if __name__ == "__main__":
	main()