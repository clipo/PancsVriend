"""Token probability extraction for LLM responses.

This script supports two execution modes:

1) Single-prompt mode (default; no --scenario)
	 Sends one prompt (optionally repeated) to a chat-completions endpoint and
	 extracts token-level probabilities from returned logprobs.

2) Scenario mode (--scenario <name>|all)
	 Enumerates all valid local 3x3 Schelling neighborhoods for a 10x10 grid,
	 where walls (#) only appear in positions possible from real boundary cells,
	 and queries the model for each context.

	 Use --scenario all to iterate every scenario in context_scenarios.py.

Outputs are saved with model-prefixed filenames:
	- <model>_token_probabilities.csv
	- <model>_stay_move_probability_split.csv
	- <model>_stay_move_probability_split_summary.csv

If the endpoint/model does not return token logprobs, the script fails with a
clear error message.

Comprehensive CLI command reference
-----------------------------------

Core generation controls:
	--prompt TEXT
			Prompt text. If omitted, uses DEFAULT_PROMPT from this file.
	--num-samples INT
			Number of repeated generations for the same prompt (default: 1).
	--temperature FLOAT
			Sampling temperature (default: 0.3).
	--max-tokens INT
			Maximum generated tokens per sample (default: 24).
	--top-logprobs INT
			Number of top token alternatives to request per generated token
			(default: 20).
	--timeout INT
			Request timeout in seconds (default: 30).

Model / endpoint controls:
	--llm-model TEXT
			Model name override. If omitted, this file's configured model is used.
	--llm-url TEXT
			Endpoint URL override (typically an OpenAI-compatible/Ollama chat endpoint).
	--logprob-api-structure {ollama,openai}
			Explicit request/response structure for logprob extraction.
			- ollama (default): use Ollama-native payload + native logprobs parser.
			- openai: use OpenAI-style chat-completions payload + OpenAI parser.
			Strict mode: no cross-structure fallback.

Scenario controls:
	--scenario TEXT
			Scenario key from context_scenarios.py, or "all" for every scenario.
			If omitted, runs single-prompt mode.
	--agent-role {type_a,type_b,both}
			Which role(s) to evaluate in scenario mode (default: both).
	--repeats-per-context INT
			Repeated generations per neighborhood context in scenario mode.
			If omitted, falls back to --num-samples.
	--resume
			Resume scenario runs by skipping scenarios already listed in progress
			manifests.
	--processes INT
			Number of parallel worker processes for API requests (default: 1).

Output controls:
	--output-dir PATH
			Output root directory override.
			Writes to <output-dir>/<llm_model_slug> when provided.
			Otherwise writes to <workspace>/llm_log_probs/<llm_model_slug>.

Example commands
----------------

Single prompt, 3 samples:
	python llm_utility_approximation/llm_token_probabilities.py \
			--prompt "Respond with exactly: MOVE" \
			--num-samples 3 \
			--llm-model "phi4:latest"

Single prompt with custom endpoint and tighter sampling:
	python llm_utility_approximation/llm_token_probabilities.py \
			--llm-url "http://127.0.0.1:11434/v1/chat/completions" \
			--logprob-api-structure openai \
			--temperature 0.1 \
			--max-tokens 8 \
			--top-logprobs 30

Scenario run for one scenario and both roles:
	python llm_utility_approximation/llm_token_probabilities.py \
			--scenario baseline \
			--agent-role both \
			--repeats-per-context 2

All scenarios with resume + parallel workers + custom output:
	python llm_utility_approximation/llm_token_probabilities.py \
			--scenario all \
			--resume \
			--logprob-api-structure ollama \
			--processes 8 \
			--output-dir "./llm_log_probs"

Notes:
	- Online/local routing is controlled by Use_ONLINE_API in this file.
	- For available scenario keys, inspect CONTEXT_SCENARIOS in context_scenarios.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.parse import quote

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
online_ollama_url = "https://chat.binghamton.edu/ollama/api/chat"
online_openai_url = "https://chat.binghamton.edu/ollama/v1/chat"
ONLINE_API_KEY = "sk-571df6eec7f5495faef553ab5cb2c67a"
Use_ONLINE_API = True
TIMEOUT_RETRY_ATTEMPTS = 5
MEANINGFUL_ANSWER_MAX_REASKS = 5

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


def _default_online_url_for_structure(logprob_api_structure: str) -> str:
	if logprob_api_structure == "openai":
		return online_openai_url.rstrip("/")
	if logprob_api_structure == "ollama":
		return online_ollama_url.rstrip("/")
	raise ValueError(f"Unsupported --logprob-api-structure: {logprob_api_structure}")


def _default_local_urls_for_structure(logprob_api_structure: str) -> list[str]:
	if logprob_api_structure == "openai":
		openai_urls = [
			url
			for url in LOCAL_OLLAMA_URLS
			if url.rstrip("/").endswith("/v1/chat/completions") or url.rstrip("/").endswith("/chat/completions")
		]
		return openai_urls if len(openai_urls) > 0 else list(LOCAL_OLLAMA_URLS)
	if logprob_api_structure == "ollama":
		ollama_urls = [url for url in LOCAL_OLLAMA_URLS if url.rstrip("/").endswith("/api/chat")]
		return ollama_urls if len(ollama_urls) > 0 else list(LOCAL_OLLAMA_URLS)
	raise ValueError(f"Unsupported --logprob-api-structure: {logprob_api_structure}")

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
		default=0.3,
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
		"--logprob-api-structure",
		type=str,
		choices=["ollama", "openai"],
		default="ollama",
		help="Explicit API structure for request+strict parser behavior (default: ollama)",
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
		default=None,
		help="Optional output root directory override. Outputs are written to <output-dir>/<llm_model_slug>; if omitted, writes to <workspace>/llm_log_probs/<llm_model_slug>",
	)
	parser.add_argument(
		"--resume",
		action="store_true",
		help="Resume scenario runs by skipping scenarios already listed in the per-scenario progress manifest",
	)
	parser.add_argument(
		"--processes",
		type=int,
		default=None,
		help="Number of parallel processes for API requests (default: 1; sequential)",
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


def _first_token_only_records(records: list[TokenRecord]) -> list[TokenRecord]:
	return [record for record in records if int(record.token_index) == 0]


def _return_first_token_records_or_raise(
	text: str,
	records: list[TokenRecord],
	error_message: str,
) -> tuple[str, list[TokenRecord]]:
	first_token_records = _first_token_only_records(records)
	if first_token_records:
		return text, first_token_records
	raise RuntimeError(error_message)


def extract_token_records(
	response_json: dict[str, Any],
	sample_index: int,
	logprob_api_structure: str,
) -> tuple[str, list[TokenRecord]]:
	if logprob_api_structure == "ollama":
		text, records = _extract_native_ollama_logprobs(response_json, sample_index)
		return _return_first_token_records_or_raise(
			text,
			records,
			"Strict parser mode 'ollama' expected native Ollama logprobs, "
			"but response did not contain supported first-token native Ollama logprobs fields."
		)

	if logprob_api_structure == "openai":
		choices = response_json.get("choices")
		if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
			raise RuntimeError("Strict parser mode 'openai' expected a valid 'choices[0]' object")

		choice = choices[0]
		text = _extract_message_content(choice).strip()
		records = _parse_chat_content_logprobs(choice, sample_index)
		return _return_first_token_records_or_raise(
			text,
			records,
			"Strict parser mode 'openai' expected OpenAI chat-content logprobs fields, "
			"but none were found for the first generated token in the response."
		)

	raise ValueError(f"Unsupported --logprob-api-structure: {logprob_api_structure}")


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
	request_seed: int | None = None,
	logprob_api_structure: str = "ollama",
	extra_ollama_options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float, str]:
	headers = {"Content-Type": "application/json"}
	if api_key:
		headers["Authorization"] = f"Bearer {api_key}"

	last_error: Exception | None = None
	for url in urls:
		for attempt in range(1, TIMEOUT_RETRY_ATTEMPTS + 1):
			start = time.time()
			try:
				if logprob_api_structure == "ollama":
					options_payload: dict[str, Any] = {
						"temperature": temperature,
						"num_predict": max_tokens,
					}
					if isinstance(extra_ollama_options, dict):
						for option_key, option_value in extra_ollama_options.items():
							if not isinstance(option_key, str) or option_key.strip() == "":
								continue
							options_payload[option_key] = option_value

					payload: dict[str, Any] = {
						"model": model,
						"messages": [{"role": "user", "content": prompt}],
						"stream": False,
						"logprobs": True,
						"top_logprobs": top_logprobs,
						"options": options_payload,
					}
					if request_seed is not None:
						payload["options"]["seed"] = int(request_seed)
				elif logprob_api_structure == "openai":
					payload = {
						"model": model,
						"messages": [{"role": "user", "content": prompt}],
						"stream": False,
						"temperature": temperature,
						"max_tokens": max_tokens,
						"logprobs": True,
						"top_logprobs": top_logprobs,
					}
					if request_seed is not None:
						payload["seed"] = int(request_seed)
				else:
					raise ValueError(f"Unsupported --logprob-api-structure: {logprob_api_structure}")

				response = session.post(url, headers=headers, json=payload, timeout=timeout)
				elapsed = time.time() - start
				response.raise_for_status()
				return response.json(), elapsed, url
			except requests.exceptions.Timeout as exc:
				last_error = exc
				if attempt < TIMEOUT_RETRY_ATTEMPTS:
					continue
			except Exception as exc:
				last_error = exc
				break

	raise RuntimeError(
		f"Failed to query endpoint(s) using '{logprob_api_structure}' request structure. "
		"Verify endpoint compatibility, URL, model, and authentication settings."
	) from last_error


def ensure_directory(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def _write_per_scenario_progress_manifest(
	output_dir: str,
	model: str,
	per_scenario_outputs: dict[str, dict[str, str]],
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
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
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
		validated: dict[str, dict[str, str]] = {}
		for scenario_key, value in outputs.items():
			if isinstance(scenario_key, str) and isinstance(value, dict):
				validated[scenario_key] = {
					k: v for k, v in value.items() if isinstance(k, str) and isinstance(v, str)
				}
		return validated, validated_params

	return {}, validated_params


def _sanitize_model_for_path_component(name: str) -> str:
	# Windows-disallowed chars for a path segment: < > : " / \ | ? * and controls.
	sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '-', name.strip())
	sanitized = sanitized.rstrip(' .')
	sanitized = re.sub(r'-{2,}', '-', sanitized)
	return sanitized or "unknown-model"


def _extract_response_fingerprint(
	response_json: dict[str, Any],
	requested_model: str,
	request_url: str,
) -> dict[str, Any]:
	served_model = response_json.get("model")
	if not isinstance(served_model, str) or not served_model.strip():
		served_model = requested_model

	system_fingerprint = response_json.get("system_fingerprint")
	if not isinstance(system_fingerprint, str) or not system_fingerprint.strip():
		system_fingerprint = None

	created = response_json.get("created")
	created_at = response_json.get("created_at")

	return {
		"provider_source": "response",
		"request_url": request_url,
		"requested_model": requested_model,
		"served_model": served_model,
		"system_fingerprint": system_fingerprint,
		"created": created,
		"created_at": created_at,
	}


def _candidate_ollama_show_endpoints(url: str) -> list[str]:
	parsed = urlparse(url)
	if parsed.scheme not in {"http", "https"} or not parsed.netloc:
		return []

	base = f"{parsed.scheme}://{parsed.netloc}"
	path = parsed.path.rstrip("/")
	endpoints: list[str] = []

	if path.endswith("/api/chat"):
		endpoints.append(f"{base}{path[: -len('/api/chat')]}/api/show")
	if path.endswith("/v1/chat/completions"):
		endpoints.append(f"{base}{path[: -len('/v1/chat/completions')]}/api/show")
	if path.endswith("/chat/completions"):
		endpoints.append(f"{base}{path[: -len('/chat/completions')]}/api/show")

	endpoints.append(f"{base}/api/show")

	seen: set[str] = set()
	unique: list[str] = []
	for endpoint in endpoints:
		if endpoint not in seen:
			seen.add(endpoint)
			unique.append(endpoint)
	return unique


def _candidate_openai_model_endpoints(url: str, model: str) -> list[str]:
	parsed = urlparse(url)
	if parsed.scheme not in {"http", "https"} or not parsed.netloc:
		return []

	base = f"{parsed.scheme}://{parsed.netloc}"
	path = parsed.path.rstrip("/")
	encoded_model = quote(model, safe="")
	endpoints: list[str] = []

	if path.endswith("/chat/completions"):
		prefix = path[: -len('/chat/completions')]
		endpoints.append(f"{base}{prefix}/models/{encoded_model}")
	elif path.endswith("/v1"):
		endpoints.append(f"{base}{path}/models/{encoded_model}")

	if "/v1" in path:
		v1_index = path.find("/v1")
		v1_prefix = path[: v1_index + len('/v1')]
		endpoints.append(f"{base}{v1_prefix}/models/{encoded_model}")

	endpoints.append(f"{base}/v1/models/{encoded_model}")

	seen: set[str] = set()
	unique: list[str] = []
	for endpoint in endpoints:
		if endpoint not in seen:
			seen.add(endpoint)
			unique.append(endpoint)
	return unique


def fetch_model_fingerprint_from_provider(
	urls: list[str],
	model: str,
	timeout: int,
	session: requests.Session,
	api_key: str | None = None,
) -> dict[str, Any]:
	headers: dict[str, str] = {"Content-Type": "application/json"}
	if api_key:
		headers["Authorization"] = f"Bearer {api_key}"

	last_error: str | None = None
	for request_url in urls:
		for endpoint in _candidate_ollama_show_endpoints(request_url):
			try:
				response = session.post(
					endpoint,
					headers=headers,
					json={"model": model},
					timeout=timeout,
				)
				response.raise_for_status()
				data = response.json()
				if not isinstance(data, dict):
					continue

				model_digest = data.get("digest")
				if not isinstance(model_digest, str) or not model_digest.strip():
					model_digest = None

				served_model = data.get("model")
				if not isinstance(served_model, str) or not served_model.strip():
					served_model = model

				return {
					"provider_source": "ollama_api_show",
					"request_url": request_url,
					"fingerprint_endpoint": endpoint,
					"requested_model": model,
					"served_model": served_model,
					"system_fingerprint": None,
					"model_digest": model_digest,
					"model_modified_at": data.get("modified_at"),
					"model_details": data.get("details", {}),
				}
			except Exception as exc:
				last_error = str(exc)

		for endpoint in _candidate_openai_model_endpoints(request_url, model):
			try:
				response = session.get(endpoint, headers=headers, timeout=timeout)
				response.raise_for_status()
				data = response.json()
				if not isinstance(data, dict):
					continue

				served_model = data.get("id")
				if not isinstance(served_model, str) or not served_model.strip():
					served_model = model

				return {
					"provider_source": "openai_models_endpoint",
					"request_url": request_url,
					"fingerprint_endpoint": endpoint,
					"requested_model": model,
					"served_model": served_model,
					"system_fingerprint": None,
					"model_digest": None,
					"model_created": data.get("created"),
					"model_owned_by": data.get("owned_by"),
				}
			except Exception as exc:
				last_error = str(exc)

	return {
		"provider_source": "unavailable",
		"request_url": urls[0] if len(urls) > 0 else "",
		"fingerprint_endpoint": None,
		"requested_model": model,
		"served_model": model,
		"system_fingerprint": None,
		"model_digest": None,
		"error": last_error,
	}


def _resolve_model_output_dir(model: str) -> str:
	model_slug = _sanitize_model_for_path_component(model)
	return str(REPO_ROOT / "llm_log_probs" / model_slug)


def _resolve_output_dir_for_model(model: str, output_dir_arg: str | None) -> str:
	"""Resolve output directory with model namespacing.

	When --output-dir is provided, treat it as a root directory and save outputs
	inside a model-specific folder: <output-dir>/<model_slug>.
	If the provided directory already ends with <model_slug>, keep it unchanged.
	"""
	model_slug = _sanitize_model_for_path_component(model)
	if isinstance(output_dir_arg, str) and output_dir_arg.strip():
		provided_root = Path(output_dir_arg.strip())
		if provided_root.name == model_slug:
			return str(provided_root)
		return str(provided_root / model_slug)
	return _resolve_model_output_dir(model)


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


def _token_implies_label_from_first_token(fragment: str, label: str) -> bool:
	"""Return True when a first-token fragment implies a target label.

	Examples for STAY: S, ST, STA, STAY, STAYING
	Examples for MOVE: M, MO, MOV, MOVE, MOVED
	"""
	if not fragment:
		return False
	label = label.upper()
	return label.startswith(fragment) or fragment.startswith(label)


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


def _resolve_process_count(requested_processes: int | None, num_tasks: int) -> int:
	if num_tasks <= 1:
		return 1
	if requested_processes is None:
		return 1
	if requested_processes < 1:
		raise ValueError("--processes must be >= 1")
	return min(requested_processes, cpu_count(), num_tasks)


def _query_until_meaningful_labels(
	task: dict[str, Any],
	session: requests.Session,
) -> tuple[str, list[TokenRecord], str, float, dict[str, float], int, bool, dict[str, Any]]:
	"""Query endpoint and re-ask up to configured limit until STAY/MOVE total probability is > 0."""
	max_reasks = int(task.get("max_meaningful_reasks", MEANINGFUL_ANSWER_MAX_REASKS))
	if max_reasks < 0:
		max_reasks = 0

	attempt_index = 0
	last_response_text = ""
	last_sample_records: list[TokenRecord] = []
	last_used_url = ""
	last_elapsed = 0.0
	last_label_metrics: dict[str, float] = {
		"stay_probability": 0.0,
		"move_probability": 0.0,
		"total_labeled_probability": 0.0,
		"stay_logprob": float("-inf"),
		"move_logprob": float("-inf"),
		"stay_share": 0.0,
		"move_share": 0.0,
	}
	last_response_fingerprint: dict[str, Any] = {
		"provider_source": "response",
		"request_url": "",
		"requested_model": task["model"],
		"served_model": task["model"],
		"system_fingerprint": None,
		"created": None,
		"created_at": None,
	}

	while True:
		response_json, elapsed, used_url = request_with_logprobs(
			urls=task["urls"],
			model=task["model"],
			prompt=task["prompt"],
			temperature=task["temperature"],
			max_tokens=task["max_tokens"],
			top_logprobs=task["top_logprobs"],
			timeout=task["timeout"],
			session=session,
			api_key=task.get("api_key"),
			request_seed=task.get("request_seed"),
			logprob_api_structure=str(task.get("logprob_api_structure", "ollama")),
		)
		last_response_fingerprint = _extract_response_fingerprint(
			response_json=response_json,
			requested_model=task["model"],
			request_url=used_url,
		)

		response_text, sample_records = extract_token_records(
			response_json,
			int(task["sample_index"]),
			str(task.get("logprob_api_structure", "ollama")),
		)
		token_candidates = _token_candidates_from_sample_records(sample_records)
		label_metrics = _compute_label_metrics(token_candidates, labels=("STAY", "MOVE"))

		last_response_text = response_text
		last_sample_records = sample_records
		last_used_url = used_url
		last_elapsed = elapsed
		last_label_metrics = label_metrics

		if float(label_metrics.get("total_labeled_probability", 0.0)) > 0:
			return (
				last_response_text,
				last_sample_records,
				last_used_url,
				last_elapsed,
				last_label_metrics,
				attempt_index,
				False,
				last_response_fingerprint,
			)

		if attempt_index >= max_reasks:
			return (
				last_response_text,
				last_sample_records,
				last_used_url,
				last_elapsed,
				last_label_metrics,
				attempt_index,
				True,
				last_response_fingerprint,
			)

		attempt_index += 1


def _execute_single_scenario_request(
	task: dict[str, Any],
	session: requests.Session | None = None,
) -> tuple[list[TokenRecord], dict[str, Any], str, dict[str, Any]]:
	if session is None:
		with requests.Session() as local_session:
			return _execute_single_scenario_request(task, session=local_session)

	response_text, sample_records, used_url, elapsed, label_metrics, meaningful_reasks_used, meaningful_error, response_fingerprint = _query_until_meaningful_labels(
		task=task,
		session=session,
	)

	context_row = {
		"scenario": task["scenario"],
		"agent_role": task["agent_role"],
		"agent_label": task["agent_label"],
		"opposite_label": task["opposite_label"],
		"arrangement_index": task["arrangement_index"],
		"arrangement_code": task["arrangement_code"],
		"context": task["context"],
		"repeat_index": task["repeat_index"],
		"sample_index": int(task["sample_index"]),
		"num_similar": task["num_similar"],
		"num_opposite": task["num_opposite"],
		"num_empty": task["num_empty"],
		"num_wall": task["num_wall"],
		"ratio_similar": task["ratio_similar"],
		"response_text": response_text,
		"request_url": used_url,
		"elapsed_seconds": elapsed,
		"meaningful_reasks_used": int(meaningful_reasks_used),
		"meaningful_error": bool(meaningful_error),
		"meaningful_error_type": "zero_stay_move_probability_after_reasks" if meaningful_error else "",
		"response_served_model": response_fingerprint.get("served_model"),
		"response_system_fingerprint": response_fingerprint.get("system_fingerprint"),
		**label_metrics,
	}

	return sample_records, context_row, used_url, response_fingerprint


def _execute_single_scenario_request_worker(task: dict[str, Any]) -> tuple[list[TokenRecord], dict[str, Any], str, dict[str, Any]]:
	return _execute_single_scenario_request(task, session=None)


def _execute_single_prompt_request_worker(task: dict[str, Any]) -> tuple[int, str, list[TokenRecord], str, int, bool, dict[str, Any]]:
	with requests.Session() as session:
		response_text, sample_records, used_url, _elapsed, _label_metrics, reasks, meaningful_error, response_fingerprint = _query_until_meaningful_labels(
			task=task,
			session=session,
		)
		return int(task["sample_index"]), response_text, sample_records, used_url, int(reasks), bool(meaningful_error), response_fingerprint


def evaluate_scenario_permutations(
	scenario_key: str,
	agent_role: str,
	repeats_per_context: int,
	model: str,
	urls: list[str],
	logprob_api_structure: str,
	temperature: float,
	max_tokens: int,
	top_logprobs: int,
	timeout: int,
	session: requests.Session,
	processes: int | None = None,
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
	last_response_fingerprint: dict[str, Any] = {
		"provider_source": "response",
		"request_url": "",
		"requested_model": model,
		"served_model": model,
		"system_fingerprint": None,
		"created": None,
		"created_at": None,
	}
	global_sample_idx = 0
	all_neighbor_arrangements = generate_all_valid_schelling_neighbors(SCHELLING_GRID_SIZE)

	request_tasks: list[dict[str, Any]] = []
	for role_name, role_label, opposite_label in role_configs:
		for arrangement_idx, neighbors in enumerate(all_neighbor_arrangements):
			context_grid = generate_neighbor_context(neighbors)
			prompt = build_scenario_prompt(scenario_key, context_grid, role_label, opposite_label)
			arrangement_code = "".join(neighbors)
			num_same = int(sum(1 for item in neighbors if item == "S"))
			num_opposite = int(sum(1 for item in neighbors if item == "O"))
			num_empty = int(sum(1 for item in neighbors if item == "E"))
			num_wall = int(sum(1 for item in neighbors if item == "#"))
			num_non_wall = 8 - num_wall
			ratio_similar = (num_same / num_non_wall) if num_non_wall > 0 else 0.0

			for repeat_index in range(repeats_per_context):
				request_tasks.append(
					{
						"urls": urls,
						"model": model,
						"logprob_api_structure": logprob_api_structure,
						"prompt": prompt,
						"temperature": temperature,
						"max_tokens": max_tokens,
						"top_logprobs": top_logprobs,
						"timeout": timeout,
						"max_meaningful_reasks": MEANINGFUL_ANSWER_MAX_REASKS,
						"api_key": api_key,
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
					}
				)
				global_sample_idx += 1

	total_requests = len(request_tasks)
	processed_requests = 0
	process_count = _resolve_process_count(processes, total_requests)
	progress_start_time = time.time()
	last_progress_log_time = progress_start_time

	print(
		f"[progress] Starting scenario '{scenario_key}' with {len(role_configs)} role(s), "
		f"{len(all_neighbor_arrangements)} contexts, {repeats_per_context} repeat(s) each "
		f"({total_requests} total requests, processes={process_count}).",
		flush=True,
	)

	if process_count > 1:
		with Pool(process_count) as pool:
			for sample_records, context_row, used_url, response_fingerprint in pool.imap_unordered(
				_execute_single_scenario_request_worker,
				request_tasks,
			):
				all_records.extend(sample_records)
				context_rows.append(context_row)
				last_url_used = used_url
				if response_fingerprint:
					last_response_fingerprint = response_fingerprint
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
						f"({percent_complete:.2f}%) | elapsed={elapsed_seconds:.1f}s | eta={eta_display}",
						flush=True,
					)
					last_progress_log_time = now
	else:
		for task in request_tasks:
			sample_records, context_row, used_url, response_fingerprint = _execute_single_scenario_request(task, session=session)
			all_records.extend(sample_records)
			context_rows.append(context_row)
			last_url_used = used_url
			if response_fingerprint:
				last_response_fingerprint = response_fingerprint
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
					f"({percent_complete:.2f}%) | elapsed={elapsed_seconds:.1f}s | eta={eta_display}",
					flush=True,
				)
				last_progress_log_time = now

	print("[progress] Scenario evaluation complete.", flush=True)

	context_rows.sort(key=lambda row: int(row.get("sample_index", -1)))
	all_records.sort(key=lambda rec: (rec.sample_index, rec.token_index, rec.top_rank))
	context_df = pd.DataFrame(context_rows)
	if len(context_df) > 0:
		context_df.attrs["last_response_fingerprint"] = last_response_fingerprint
	return all_records, context_df, last_url_used


def _probability_of_label_from_candidates(token_candidates: list[list[tuple[str, float]]], label: str) -> float:
	"""Estimate label probability from first-token alternatives only.

	This intentionally ignores token indices >= 1 and treats first-token fragments
	that match a label prefix (or extend beyond the full label) as label-implying.
	"""
	if len(token_candidates) == 0:
		return 0.0

	label = label.upper()
	first_position_candidates = token_candidates[0]
	probability = 0.0
	for token_text, token_prob in first_position_candidates:
		if token_prob <= 0:
			continue
		fragment = _normalize_token_fragment(token_text)
		if _token_implies_label_from_first_token(fragment, label):
			probability += float(token_prob)

	return float(probability)


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
				"num_meaningful_errors",
				"mean_meaningful_reasks_used",
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
		num_meaningful_errors = (
			int(label_split_df["meaningful_error"].sum())
			if "meaningful_error" in label_split_df.columns
			else 0
		)
		mean_meaningful_reasks_used = (
			float(label_split_df["meaningful_reasks_used"].mean())
			if "meaningful_reasks_used" in label_split_df.columns
			else 0.0
		)
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
					"num_meaningful_errors": num_meaningful_errors,
					"mean_meaningful_reasks_used": mean_meaningful_reasks_used,
				}
			]
		)
		return summary_df

	meaningful_error_column_present = "meaningful_error" in label_split_df.columns
	meaningful_reasks_column_present = "meaningful_reasks_used" in label_split_df.columns

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
			num_meaningful_errors=("meaningful_error", "sum") if meaningful_error_column_present else ("sample_index", lambda s: 0),
			mean_meaningful_reasks_used=("meaningful_reasks_used", "mean") if meaningful_reasks_column_present else ("sample_index", lambda s: 0.0),
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
	if args.processes is not None and args.processes < 1:
		raise ValueError("--processes must be >= 1")

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
		candidate_urls = [args.llm_url] if args.llm_url else [_default_online_url_for_structure(args.logprob_api_structure)]
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
			candidate_urls = _default_local_urls_for_structure(args.logprob_api_structure)

		for candidate_url in candidate_urls:
			if not _is_local_url(candidate_url):
				raise ValueError(f"Non-local URL is not allowed: {candidate_url}")

	output_dir = _resolve_output_dir_for_model(model, args.output_dir)
	ensure_directory(output_dir)

	records: list[TokenRecord] = []
	url_used = ""
	meaningful_error_count = 0
	meaningful_reasks_total = 0
	label_split_df_override: pd.DataFrame | None = None
	label_split_summary_override: pd.DataFrame | None = None
	outputs_payload: dict[str, Any]
	resume_outputs: dict[str, dict[str, str]] = {}
	resume_run_parameters: dict[str, Any] = {}
	current_run_parameters: dict[str, Any] = {
		"llm_model": model,
		"candidate_urls": candidate_urls,
		"logprob_api_structure": args.logprob_api_structure,
		"scenario_request": args.scenario,
		"effective_agent_role": effective_agent_role,
		"repeats_per_context": repeats_per_context,
		"num_samples": args.num_samples,
		"temperature": args.temperature,
		"max_tokens": args.max_tokens,
		"top_logprobs": args.top_logprobs,
		"timeout_seconds": args.timeout,
		"processes": args.processes,
		"output_dir": output_dir,
		"use_online_api": bool(Use_ONLINE_API),
		"grid_size": SCHELLING_GRID_SIZE,
		"non_wall_context_elements": NON_WALL_CONTEXT_ELEMENTS,
		"max_meaningful_reasks": MEANINGFUL_ANSWER_MAX_REASKS,
	}

	if args.resume:
		resume_outputs, resume_run_parameters = _read_per_scenario_progress_manifest(output_dir, model)

		if len(resume_outputs) > 0 and len(resume_run_parameters) > 0:
			comparable_keys = [
				"llm_model",
				"candidate_urls",
				"logprob_api_structure",
				"effective_agent_role",
				"repeats_per_context",
				"num_samples",
				"temperature",
				"max_tokens",
				"top_logprobs",
				"timeout_seconds",
				"processes",
				"use_online_api",
				"grid_size",
				"non_wall_context_elements",
				"max_meaningful_reasks",
			]
			mismatches: list[str] = []
			for key in comparable_keys:
				if key not in resume_run_parameters:
					continue
				if resume_run_parameters.get(key) != current_run_parameters.get(key):
					mismatches.append(
						f"{key}: resume={resume_run_parameters.get(key)!r}, current={current_run_parameters.get(key)!r}"
					)

			if len(mismatches) > 0:
				details = "\n  - " + "\n  - ".join(mismatches)
				raise ValueError(
					"Resume parameter mismatch detected. To ensure exact replication, "
					"resume must use the same parameters stored in the manifest." + details
				)

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
		provider_model_fingerprint = fetch_model_fingerprint_from_provider(
			urls=candidate_urls,
			model=model,
			timeout=args.timeout,
			session=session,
			api_key=api_key,
		)
		current_run_parameters["provider_model_fingerprint"] = provider_model_fingerprint
		response_model_fingerprint: dict[str, Any] | None = None

		if scenario_keys is not None:
			manifest_run_parameters: dict[str, Any] = {
				**current_run_parameters,
				"resume": bool(args.resume),
			}
			per_scenario_outputs: dict[str, dict[str, str]] = dict(resume_outputs)
			for scenario_key in scenario_keys:
				scenario_records, scenario_label_split_df, used_url = evaluate_scenario_permutations(
					scenario_key=scenario_key,
					agent_role=effective_agent_role,
					repeats_per_context=repeats_per_context,
					model=model,
					urls=candidate_urls,
					logprob_api_structure=args.logprob_api_structure,
					temperature=args.temperature,
					max_tokens=args.max_tokens,
					top_logprobs=args.top_logprobs,
					timeout=args.timeout,
					session=session,
					processes=args.processes,
					api_key=api_key,
				)
				url_used = used_url
				records.extend(scenario_records)
				scenario_response_fingerprint = scenario_label_split_df.attrs.get("last_response_fingerprint")
				if response_model_fingerprint is None and isinstance(scenario_response_fingerprint, dict):
					response_model_fingerprint = scenario_response_fingerprint
				if "meaningful_error" in scenario_label_split_df.columns:
					meaningful_error_count += int(scenario_label_split_df["meaningful_error"].sum())
				if "meaningful_reasks_used" in scenario_label_split_df.columns:
					meaningful_reasks_total += int(scenario_label_split_df["meaningful_reasks_used"].sum())

				scenario_label_split_summary_df = summarize_label_probability_split(scenario_label_split_df)
				tokens_csv, label_split_csv, label_split_summary_csv = save_outputs(
					output_dir,
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

				if response_model_fingerprint is not None:
					manifest_run_parameters["response_model_fingerprint"] = response_model_fingerprint

				manifest_path = _write_per_scenario_progress_manifest(
					output_dir,
					model,
					per_scenario_outputs,
					run_parameters=manifest_run_parameters,
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
			total_prompt_requests = args.num_samples
			process_count = _resolve_process_count(args.processes, total_prompt_requests)

			if process_count > 1:
				prompt_tasks = [
					{
						"urls": candidate_urls,
						"model": model,
						"logprob_api_structure": args.logprob_api_structure,
						"prompt": prompt,
						"temperature": args.temperature,
						"max_tokens": args.max_tokens,
						"top_logprobs": args.top_logprobs,
						"timeout": args.timeout,
						"max_meaningful_reasks": MEANINGFUL_ANSWER_MAX_REASKS,
						"api_key": api_key,
						"sample_index": sample_idx,
					}
					for sample_idx in range(args.num_samples)
				]

				with Pool(process_count) as pool:
					for _sample_idx, response_text, sample_records, used_url, reasks_used, meaningful_error, response_fingerprint in pool.imap_unordered(
						_execute_single_prompt_request_worker,
						prompt_tasks,
					):
						url_used = used_url
						records.extend(sample_records)
						if response_model_fingerprint is None and isinstance(response_fingerprint, dict):
							response_model_fingerprint = response_fingerprint
						meaningful_reasks_total += int(reasks_used)
						if meaningful_error:
							meaningful_error_count += 1
						_ = response_text

				records.sort(key=lambda rec: (rec.sample_index, rec.token_index, rec.top_rank))
			else:
				for sample_idx in range(args.num_samples):
					task = {
						"urls": candidate_urls,
						"model": model,
						"logprob_api_structure": args.logprob_api_structure,
						"prompt": prompt,
						"temperature": args.temperature,
						"max_tokens": args.max_tokens,
						"top_logprobs": args.top_logprobs,
						"timeout": args.timeout,
						"max_meaningful_reasks": MEANINGFUL_ANSWER_MAX_REASKS,
						"api_key": api_key,
						"sample_index": sample_idx,
					}
					response_text, sample_records, used_url, _elapsed, _label_metrics, reasks_used, meaningful_error, response_fingerprint = _query_until_meaningful_labels(
						task=task,
						session=session,
					)
					url_used = used_url
					records.extend(sample_records)
					if response_model_fingerprint is None and isinstance(response_fingerprint, dict):
						response_model_fingerprint = response_fingerprint
					meaningful_reasks_total += int(reasks_used)
					if meaningful_error:
						meaningful_error_count += 1
					_ = response_text

			# Single-prompt mode keeps the original single-output behavior
			tokens_csv, label_split_csv, label_split_summary_csv = save_outputs(
				output_dir,
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

		if response_model_fingerprint is not None:
			current_run_parameters["response_model_fingerprint"] = response_model_fingerprint

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
		"max_meaningful_reasks": MEANINGFUL_ANSWER_MAX_REASKS,
		"meaningful_error_count": meaningful_error_count,
		"meaningful_reasks_total": meaningful_reasks_total,
		"prompt": prompt,
		"outputs": outputs_payload,
	}
	print(json.dumps(result, indent=2))


if __name__ == "__main__":
	main()