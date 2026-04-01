"""Compare OpenAI-style vs Ollama logprob MOVE/STAY probabilities.

This script samples neighborhood contexts from valid 3x3 Schelling arrangements,
queries two provider entry points at temperature 0, and compares first-token
STAY/MOVE probabilities extracted from returned token logprobs.

Outputs:
- CSV table with per-context probabilities and deltas
- JSON summary with aggregate statistics

Example:
    python llm_utility_approximation/compare_openai_ollama_logprobs.py \
        --openai-model "mixtral:8x22b-instruct" \
        --ollama-model "mixtral:8x22b-instruct" \
        --num-contexts 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

import config as cfg
from context_scenarios import CONTEXT_SCENARIOS
from llm_token_probabilities import (
	LOCAL_OLLAMA_URLS,
	TIMEOUT_RETRY_ATTEMPTS,
	build_scenario_prompt,
	extract_token_records,
	generate_all_valid_schelling_neighbors,
	generate_neighbor_context,
	request_with_logprobs,
	_token_candidates_from_sample_records,
	_compute_label_metrics,
)


def _default_openai_url() -> str:
	env_url = os.environ.get("OPENAI_URL")
	if env_url:
		return env_url
	return "https://chat.binghamton.edu/ollama/v1/chat/completions"


def _default_ollama_url() -> str:
	env_url = os.environ.get("OLLAMA_URL")
	if env_url:
		return env_url
	return "https://chat.binghamton.edu/ollama/api/chat"


def _normalize_path(url: str) -> str:
	return urlparse(str(url).strip()).path.rstrip("/").lower()


def _validate_provider_urls(openai_url: str, ollama_url: str) -> None:
	openai_path = _normalize_path(openai_url)
	ollama_path = _normalize_path(ollama_url)

	if not openai_path.endswith("/chat/completions"):
		raise ValueError(
			"--openai-url must be an OpenAI-style chat completions endpoint ending in '/chat/completions'."
		)
	if openai_path.endswith("/api/chat"):
		raise ValueError("--openai-url cannot be an Ollama native '/api/chat' endpoint.")

	if not ollama_path.endswith("/api/chat"):
		raise ValueError(
			"--ollama-url must be an Ollama native endpoint ending in '/api/chat'."
		)
	if ollama_path.endswith("/chat/completions"):
		raise ValueError("--ollama-url cannot be an OpenAI-style '/chat/completions' endpoint.")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Compare OpenAI vs Ollama logprob MOVE/STAY probabilities")
	parser.add_argument("--scenario", type=str, default="baseline", help="Scenario key from context_scenarios.py")
	parser.add_argument("--agent-role", type=str, choices=["type_a", "type_b"], default="type_a")
	parser.add_argument("--num-contexts", type=int, default=10, help="Number of unique neighborhood contexts to analyze")
	parser.add_argument("--seed", type=int, default=42, help="Sampling seed for context selection")
	parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (recommended 0)")
	parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens to request for logprobs (should cover at least the first token)")
	parser.add_argument("--top-logprobs", type=int, default=20)
	parser.add_argument("--timeout", type=int, default=30)

	parser.add_argument(
		"--openai-url",
		type=str,
		default=_default_openai_url(),
		help="OpenAI-compatible chat completions endpoint (e.g., https://api.openai.com/v1/chat/completions)",
	)
	parser.add_argument("--openai-model", type=str, default=cfg.OLLAMA_MODEL)
	parser.add_argument("--openai-api-key", type=str, default=os.environ.get("OPENAI_API_KEY") or cfg.OLLAMA_API_KEY)

	parser.add_argument("--ollama-url", type=str, default=_default_ollama_url())
	parser.add_argument("--ollama-model", type=str, default=cfg.OLLAMA_MODEL)
	parser.add_argument("--ollama-api-key", type=str, default=os.environ.get("OLLAMA_API_KEY") or cfg.OLLAMA_API_KEY)

	parser.add_argument("--output-dir", type=str, default="llm_log_probs/comparisons")
	return parser.parse_args()


def _candidate_mass_first_token(sample_records: list[Any]) -> float:
	token_candidates = _token_candidates_from_sample_records(sample_records)
	if len(token_candidates) == 0:
		return 0.0
	return float(sum(max(0.0, float(prob)) for _, prob in token_candidates[0]))


def _query_provider(
	provider_name: str,
	url: str,
	model: str,
	api_key: str | None,
	logprob_api_structure: str,
	prompt: str,
	temperature: float,
	max_tokens: int,
	top_logprobs: int,
	timeout: int,
	session: requests.Session,
) -> dict[str, Any]:
	response_json, elapsed, used_url = request_with_logprobs(
		urls=[url],
		model=model,
		prompt=prompt,
		temperature=temperature,
		max_tokens=max_tokens,
		top_logprobs=top_logprobs,
		timeout=timeout,
		session=session,
		api_key=api_key,
		logprob_api_structure=logprob_api_structure,
	)

	response_text, records = extract_token_records(
		response_json,
		sample_index=0,
		logprob_api_structure=logprob_api_structure,
	)
	token_candidates = _token_candidates_from_sample_records(records)
	label_metrics = _compute_label_metrics(token_candidates, labels=("STAY", "MOVE"))

	result = {
		"provider": provider_name,
		"request_url": used_url,
		"elapsed_seconds": float(elapsed),
		"response_text": response_text,
		"candidate_mass": _candidate_mass_first_token(records),
		"stay_probability": float(label_metrics["stay_probability"]),
		"move_probability": float(label_metrics["move_probability"]),
		"total_labeled_probability": float(label_metrics["total_labeled_probability"]),
		"stay_share": float(label_metrics["stay_share"]),
		"move_share": float(label_metrics["move_share"]),
	}
	return result


def _summarize(df: pd.DataFrame) -> dict[str, Any]:
	if len(df) == 0:
		return {"num_rows": 0}

	delta_move = df["delta_move_probability"].tolist()
	delta_stay = df["delta_stay_probability"].tolist()
	abs_delta_move = [abs(x) for x in delta_move]
	abs_delta_stay = [abs(x) for x in delta_stay]

	return {
		"num_rows": int(len(df)),
		"mean_delta_move_probability": float(statistics.fmean(delta_move)),
		"mean_delta_stay_probability": float(statistics.fmean(delta_stay)),
		"mean_abs_delta_move_probability": float(statistics.fmean(abs_delta_move)),
		"mean_abs_delta_stay_probability": float(statistics.fmean(abs_delta_stay)),
		"max_abs_delta_move_probability": float(max(abs_delta_move)),
		"max_abs_delta_stay_probability": float(max(abs_delta_stay)),
		"openai_mean_move_probability": float(df["openai_move_probability"].mean()),
		"ollama_mean_move_probability": float(df["ollama_move_probability"].mean()),
		"openai_mean_stay_probability": float(df["openai_stay_probability"].mean()),
		"ollama_mean_stay_probability": float(df["ollama_stay_probability"].mean()),
		"openai_mean_candidate_mass": float(df["openai_candidate_mass"].mean()),
		"ollama_mean_candidate_mass": float(df["ollama_candidate_mass"].mean()),
	}


def main() -> None:
	args = parse_args()

	if args.scenario not in CONTEXT_SCENARIOS:
		raise ValueError(f"Unknown scenario '{args.scenario}'. Available: {sorted(CONTEXT_SCENARIOS.keys())}")
	if args.num_contexts < 1:
		raise ValueError("--num-contexts must be >= 1")
	if args.top_logprobs < 1:
		raise ValueError("--top-logprobs must be >= 1")
	_validate_provider_urls(args.openai_url, args.ollama_url)

	scenario_info = CONTEXT_SCENARIOS[args.scenario]
	if args.agent_role == "type_a":
		agent_label = str(scenario_info["type_a"])
		opposite_label = str(scenario_info["type_b"])
	else:
		agent_label = str(scenario_info["type_b"])
		opposite_label = str(scenario_info["type_a"])

	arrangements = generate_all_valid_schelling_neighbors()
	if args.num_contexts > len(arrangements):
		raise ValueError(
			f"Requested {args.num_contexts} contexts but only {len(arrangements)} available valid contexts"
		)

	rng = random.Random(args.seed)
	selected_indices = sorted(rng.sample(range(len(arrangements)), k=args.num_contexts))

	rows: list[dict[str, Any]] = []
	errors: list[dict[str, Any]] = []

	with requests.Session() as session:
		for rank, arrangement_idx in enumerate(selected_indices):
			neighbors = arrangements[arrangement_idx]
			context = generate_neighbor_context(neighbors)
			prompt = build_scenario_prompt(args.scenario, context, agent_label, opposite_label)

			arrangement_code = "".join(neighbors)
			num_similar = int(sum(1 for token in neighbors if token == "S"))
			num_opposite = int(sum(1 for token in neighbors if token == "O"))
			num_empty = int(sum(1 for token in neighbors if token == "E"))
			num_wall = int(sum(1 for token in neighbors if token == "#"))

			try:
				openai_result = _query_provider(
					provider_name="openai",
					url=args.openai_url,
					model=args.openai_model,
					api_key=args.openai_api_key,
					logprob_api_structure="openai",
					prompt=prompt,
					temperature=args.temperature,
					max_tokens=args.max_tokens,
					top_logprobs=args.top_logprobs,
					timeout=args.timeout,
					session=session,
				)

				ollama_result = _query_provider(
					provider_name="ollama",
					url=args.ollama_url,
					model=args.ollama_model,
					api_key=args.ollama_api_key,
					logprob_api_structure="ollama",
					prompt=prompt,
					temperature=args.temperature,
					max_tokens=args.max_tokens,
					top_logprobs=args.top_logprobs,
					timeout=args.timeout,
					session=session,
				)
			except Exception as exc:
				errors.append(
					{
						"context_rank": rank,
						"arrangement_index": arrangement_idx,
						"arrangement_code": arrangement_code,
						"error": f"{type(exc).__name__}: {exc}",
					}
				)
				continue

			delta_move_probability = float(openai_result["move_probability"] - ollama_result["move_probability"])
			delta_stay_probability = float(openai_result["stay_probability"] - ollama_result["stay_probability"])

			rows.append(
				{
					"context_rank": rank,
					"arrangement_index": arrangement_idx,
					"scenario": args.scenario,
					"agent_role": args.agent_role,
					"agent_label": agent_label,
					"opposite_label": opposite_label,
					"arrangement_code": arrangement_code,
					"context": context,
					"num_similar": num_similar,
					"num_opposite": num_opposite,
					"num_empty": num_empty,
					"num_wall": num_wall,
					"openai_move_probability": float(openai_result["move_probability"]),
					"openai_stay_probability": float(openai_result["stay_probability"]),
					"openai_total_labeled_probability": float(openai_result["total_labeled_probability"]),
					"openai_move_share": float(openai_result["move_share"]),
					"openai_stay_share": float(openai_result["stay_share"]),
					"openai_candidate_mass": float(openai_result["candidate_mass"]),
					"openai_elapsed_seconds": float(openai_result["elapsed_seconds"]),
					"openai_request_url": str(openai_result["request_url"]),
					"ollama_move_probability": float(ollama_result["move_probability"]),
					"ollama_stay_probability": float(ollama_result["stay_probability"]),
					"ollama_total_labeled_probability": float(ollama_result["total_labeled_probability"]),
					"ollama_move_share": float(ollama_result["move_share"]),
					"ollama_stay_share": float(ollama_result["stay_share"]),
					"ollama_candidate_mass": float(ollama_result["candidate_mass"]),
					"ollama_elapsed_seconds": float(ollama_result["elapsed_seconds"]),
					"ollama_request_url": str(ollama_result["request_url"]),
					"delta_move_probability": delta_move_probability,
					"delta_stay_probability": delta_stay_probability,
					"abs_delta_move_probability": abs(delta_move_probability),
					"abs_delta_stay_probability": abs(delta_stay_probability),
				}
			)


	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

	results_df = pd.DataFrame(rows)
	if len(results_df) > 0:
		results_df = results_df.sort_values("abs_delta_move_probability", ascending=False)

	errors_df = pd.DataFrame(errors)
	summary = _summarize(results_df)
	summary.update(
		{
			"scenario": args.scenario,
			"agent_role": args.agent_role,
			"num_contexts_requested": int(args.num_contexts),
			"num_contexts_succeeded": int(len(results_df)),
			"num_contexts_failed": int(len(errors_df)),
			"seed": int(args.seed),
			"temperature": float(args.temperature),
			"top_logprobs": int(args.top_logprobs),
			"max_tokens": int(args.max_tokens),
			"openai_url": args.openai_url,
			"openai_model": args.openai_model,
			"ollama_url": args.ollama_url,
			"ollama_model": args.ollama_model,
			"timeout": int(args.timeout),
			"timeout_retry_attempts": int(TIMEOUT_RETRY_ATTEMPTS),
		}
	)

	results_path = output_dir / f"openai_vs_ollama_logprob_delta_{timestamp}.csv"
	errors_path = output_dir / f"openai_vs_ollama_logprob_delta_errors_{timestamp}.csv"
	summary_path = output_dir / f"openai_vs_ollama_logprob_delta_summary_{timestamp}.json"

	results_df.to_csv(results_path, index=False)
	errors_df.to_csv(errors_path, index=False)
	with open(summary_path, "w", encoding="utf-8") as handle:
		json.dump(summary, handle, indent=2)

	print("\n=== OpenAI vs Ollama logprob comparison complete ===")
	print(f"Results CSV: {results_path}")
	print(f"Errors CSV:  {errors_path}")
	print(f"Summary:     {summary_path}")
	print(f"Succeeded:   {summary['num_contexts_succeeded']} / {summary['num_contexts_requested']}")

	if int(summary["num_contexts_succeeded"]) == 0:
		raise RuntimeError(
			"No contexts succeeded. Verify endpoint compatibility and that each provider uses the correct endpoint type: "
			"--openai-url must end with '/chat/completions' and --ollama-url must end with '/api/chat'."
		)

	if len(results_df) > 0:
		display_cols = [
			"context_rank",
			"arrangement_code",
			"openai_move_probability",
			"ollama_move_probability",
			"delta_move_probability",
			"openai_stay_probability",
			"ollama_stay_probability",
			"delta_stay_probability",
		]
		top = results_df[display_cols].head(15)
		print("\nTop 15 contexts by |delta_move_probability|:")
		print(top.to_string(index=False))

		print("\nAggregate summary:")
		for key in [
			"mean_abs_delta_move_probability",
			"mean_abs_delta_stay_probability",
			"max_abs_delta_move_probability",
			"max_abs_delta_stay_probability",
			"openai_mean_move_probability",
			"ollama_mean_move_probability",
			"openai_mean_stay_probability",
			"ollama_mean_stay_probability",
			"openai_mean_candidate_mass",
			"ollama_mean_candidate_mass",
		]:
			value = summary.get(key)
			if isinstance(value, (float, int)) and math.isfinite(float(value)):
				print(f"- {key}: {float(value):.6f}")


if __name__ == "__main__":
	main()