#!/usr/bin/env python3
"""Check STAY/MOVE probability consistency on a balanced neighborhood.

Purpose:
- Build one approximately 50/50 local neighborhood context.
- Query Mixtral (or a provided model) repeatedly.
- Evaluate whether estimated decision probabilities are stable enough that
  repeated sampling is or is not necessary.

This script intentionally reuses the same probability extraction path as
`llm_utility_approximation/llm_token_probabilities.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import requests
from context_scenarios import CONTEXT_SCENARIOS

try:
	from llm_utility_approximation import llm_token_probabilities as ltp
except ImportError:
	import llm_token_probabilities as ltp


ONLINE_API_URL = "https://chat.binghamton.edu/ollama/v1/chat/completions"

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Check LLM STAY/MOVE probability consistency on a balanced neighborhood"
	)
	parser.add_argument(
		"--scenario",
		type=str,
		default="ethnic_asian_hispanic",
		help="Scenario key from context_scenarios.py (default: baseline)",
	)
	parser.add_argument(
		"--tries",
		type=int,
		default=100,
		help="Number of repeated requests per agent role (default: 50)",
	)
	parser.add_argument(
		"--llm-model",
		type=str,
		# default="mixtral:8x22b-instruct",
        # default="llama3.3:latest",
		default="gemma3:27b",
		help="Model name to request (default: mixtral:8x22b-instruct)",
	)
	parser.add_argument(
		"--llm-url",
		type=str,
		default=None,
		help="Endpoint override (if omitted, uses llm_token_probabilities defaults)",
	)
	parser.add_argument(
		"--llm-api-key",
		type=str,
		default=None,
		help="API key override",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=0.0,
		help="Sampling temperature (default: 0.3)",
	)
	parser.add_argument(
		"--max-tokens",
		type=int,
		default=24,
		help="Max generation tokens (default: 24)",
	)
	parser.add_argument(
		"--top-logprobs",
		type=int,
		default=20,
		help="Top-k alternatives per generated token (default: 30)",
	)
	parser.add_argument(
		"--timeout",
		type=int,
		default=30,
		help="Request timeout in seconds (default: 30)",
	)
	parser.add_argument(
		"--empty-slots",
		type=int,
		default=2,
		choices=[0, 1, 2],
		help="Number of E cells in 8-neighbor ring (default: 0)",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=17,
		help="Seed for neighborhood layout shuffling (default: 7)",
	)
	parser.add_argument(
		"--request-seed-mode",
		type=str,
		choices=["none", "fixed", "varying"],
		default="varying",
		help="Request-level seed mode: none, fixed, or varying per sample_index (default: none)",
	)
	parser.add_argument(
		"--request-seed-base",
		type=int,
		default=12345,
		help="Base request seed used for fixed/varying modes (default: 12345)",
	)
	parser.add_argument(
		"--std-threshold",
		type=float,
		default=0.08,
		help="Max allowed stddev of move_probability to call a role consistent (default: 0.08)",
	)
	parser.add_argument(
		"--agreement-threshold",
		type=float,
		default=0.75,
		help="Min majority-label agreement share to call a role consistent (default: 0.75)",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="llm_log_probs/consistency_checks",
		help="Directory for CSV/JSON outputs",
	)
	parser.add_argument(
		"--prompt-nonce-per-trial",
		action="store_true",
		default=True,
		help="Append a unique nonce to each trial prompt to reduce provider caching artifacts (default: enabled)",
	)
	parser.add_argument(
		"--no-prompt-nonce-per-trial",
		action="store_false",
		dest="prompt_nonce_per_trial",
		help="Disable per-trial prompt nonce",
	)
	return parser.parse_args()


def _generate_balanced_neighbor_symbols(empty_slots: int, seed: int) -> list[str]:
	if empty_slots < 0 or empty_slots > 2:
		raise ValueError("--empty-slots must be 0, 1, or 2")

	non_empty = 8 - empty_slots
	num_similar = non_empty // 2
	num_opposite = non_empty - num_similar

	symbols = ["S"] * num_similar + ["O"] * num_opposite + ["E"] * empty_slots
	rng = random.Random(seed)
	rng.shuffle(symbols)
	return symbols


def _neighbors_to_context(neighbors: list[str]) -> str:
	if len(neighbors) != 8:
		raise ValueError(f"Expected 8 neighbors, got {len(neighbors)}")
	return "\n".join(
		[
			f"{neighbors[0]} {neighbors[1]} {neighbors[2]}",
			f"{neighbors[3]} X {neighbors[4]}",
			f"{neighbors[5]} {neighbors[6]} {neighbors[7]}",
		]
	)


def _context_to_slug(context: str) -> str:
	flattened = "".join(ch for ch in context.upper() if ch.isalnum() or ch == "#")
	return flattened or "unknowncontext"


def _temperature_to_slug(temperature: float) -> str:
	raw = f"{temperature:.6f}".rstrip("0").rstrip(".")
	if raw == "":
		raw = "0"
	return raw.replace("-", "m").replace(".", "p")


def _request_seed_for_call(mode: str, base: int, sample_index: int) -> int | None:
	if mode == "none":
		return None
	if mode == "fixed":
		return int(base)
	if mode == "varying":
		return int(base) + int(sample_index)
	raise ValueError(f"Unsupported request seed mode: {mode}")


def _role_consistency_summary(
	role_rows: list[dict[str, Any]],
	std_threshold: float,
	agreement_threshold: float,
) -> dict[str, Any]:
	move_probs = [float(row["move_probability"]) for row in role_rows]
	stay_probs = [float(row["stay_probability"]) for row in role_rows]
	label_votes = [str(row["response_label"]) for row in role_rows]

	move_std = stdev(move_probs) if len(move_probs) > 1 else 0.0
	stay_std = stdev(stay_probs) if len(stay_probs) > 1 else 0.0

	move_count = int(sum(1 for label in label_votes if label == "MOVE"))
	stay_count = int(sum(1 for label in label_votes if label == "STAY"))
	majority_count = max(move_count, stay_count)
	majority_label = "MOVE" if move_count >= stay_count else "STAY"
	agreement = majority_count / len(label_votes) if len(label_votes) > 0 else 0.0

	is_consistent = (move_std <= std_threshold) and (agreement >= agreement_threshold)

	return {
		"num_trials": len(role_rows),
		"mean_move_probability": mean(move_probs) if move_probs else math.nan,
		"mean_stay_probability": mean(stay_probs) if stay_probs else math.nan,
		"min_move_probability": min(move_probs) if move_probs else math.nan,
		"max_move_probability": max(move_probs) if move_probs else math.nan,
		"min_stay_probability": min(stay_probs) if stay_probs else math.nan,
		"max_stay_probability": max(stay_probs) if stay_probs else math.nan,
		"std_move_probability": move_std,
		"std_stay_probability": stay_std,
		"move_votes": move_count,
		"stay_votes": stay_count,
		"majority_label": majority_label,
		"majority_agreement": agreement,
		"consistent": bool(is_consistent),
	}


def _resolve_api_settings(args: argparse.Namespace) -> tuple[list[str], str | None]:
	if args.llm_url:
		urls = [args.llm_url]
	elif bool(ltp.Use_ONLINE_API):
		urls = [str(ltp.online_ollama_url)]
	else:
		urls = list(ltp.LOCAL_OLLAMA_URLS)

	api_key = args.llm_api_key
	if api_key is None and bool(ltp.Use_ONLINE_API) and not args.llm_url:
		api_key = str(ltp.ONLINE_API_KEY)

	return urls, api_key


def main() -> None:
	args = parse_args()
	if args.tries < 1:
		raise ValueError("--tries must be >= 1")
	if args.scenario not in CONTEXT_SCENARIOS:
		available = ", ".join(sorted(CONTEXT_SCENARIOS.keys()))
		raise ValueError(f"Unknown scenario '{args.scenario}'. Available: {available}")

	scenario = CONTEXT_SCENARIOS[args.scenario]
	neighbors = _generate_balanced_neighbor_symbols(args.empty_slots, args.seed)
	context = _neighbors_to_context(neighbors)
	urls, api_key = _resolve_api_settings(args)

	role_configs = [
		("type_a", str(scenario["type_a"]), str(scenario["type_b"])),
		("type_b", str(scenario["type_b"]), str(scenario["type_a"])),
	]

	rows: list[dict[str, Any]] = []
	sample_index = 0

	with requests.Session() as session:
		for role_name, role_label, opposite_label in role_configs:
			base_prompt = ltp.build_scenario_prompt(
				scenario_key=args.scenario,
				context=context,
				agent_label=role_label,
				opposite_label=opposite_label,
			)

			for trial_index in range(args.tries):
				prompt = base_prompt
				if args.prompt_nonce_per_trial:
					prompt = f"{base_prompt}\n\n[trial_nonce:{role_name}:{trial_index}]"

				request_seed = _request_seed_for_call(
					mode=args.request_seed_mode,
					base=args.request_seed_base,
					sample_index=sample_index,
				)

				task = {
					"urls": urls,
					"model": args.llm_model,
					"prompt": prompt,
					"temperature": args.temperature,
					"max_tokens": args.max_tokens,
					"top_logprobs": args.top_logprobs,
					"timeout": args.timeout,
					"sample_index": sample_index,
					"max_meaningful_reasks": int(ltp.MEANINGFUL_ANSWER_MAX_REASKS),
					"api_key": api_key,
					"request_seed": request_seed,
				}
				(
					response_text,
					_sample_records,
					used_url,
					elapsed,
					label_metrics,
					meaningful_reasks_used,
					meaningful_error,
					_response_fingerprint,
				) = ltp._query_until_meaningful_labels(task=task, session=session)

				response_text_clean = response_text.strip().upper()
				if response_text_clean.startswith("MOVE"):
					response_label = "MOVE"
				elif response_text_clean.startswith("STAY"):
					response_label = "STAY"
				else:
					response_label = "OTHER"

				rows.append(
					{
						"scenario": args.scenario,
						"agent_role": role_name,
						"prompt_nonce_per_trial": bool(args.prompt_nonce_per_trial),
						"agent_label": role_label,
						"opposite_label": opposite_label,
						"trial_index": trial_index,
						"sample_index": sample_index,
						"request_seed": request_seed,
						"context": context,
						"arrangement_code": "".join(neighbors),
						"response_text": response_text,
						"response_label": response_label,
						"request_url": used_url,
						"elapsed_seconds": elapsed,
						"meaningful_reasks_used": int(meaningful_reasks_used),
						"meaningful_error": bool(meaningful_error),
						"stay_probability": float(label_metrics.get("stay_probability", 0.0)),
						"move_probability": float(label_metrics.get("move_probability", 0.0)),
						"total_labeled_probability": float(label_metrics.get("total_labeled_probability", 0.0)),
						"stay_share": float(label_metrics.get("stay_share", 0.0)),
						"move_share": float(label_metrics.get("move_share", 0.0)),
					}
				)
				sample_index += 1

	rows_by_role: dict[str, list[dict[str, Any]]] = {"type_a": [], "type_b": []}
	for row in rows:
		rows_by_role[str(row["agent_role"])].append(row)

	summary_by_role: dict[str, dict[str, Any]] = {}
	for role_name in ["type_a", "type_b"]:
		summary_by_role[role_name] = _role_consistency_summary(
			role_rows=rows_by_role[role_name],
			std_threshold=args.std_threshold,
			agreement_threshold=args.agreement_threshold,
		)

	overall_consistent = all(summary_by_role[role]["consistent"] for role in ["type_a", "type_b"])
	recommendation = (
		"Probabilities look stable in this balanced case; one-shot estimates may be acceptable."
		if overall_consistent
		else "Probabilities vary enough in this balanced case; keep multiple repeats when estimating decision probabilities."
	)

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	model_slug = ltp._sanitize_model_for_path_component(args.llm_model)
	context_slug = _context_to_slug(context)
	temperature_slug = _temperature_to_slug(args.temperature)
	stem = f"{model_slug}_{args.scenario}_{context_slug}_T{temperature_slug}_consistency"

	csv_path = output_dir / f"{stem}_trials.csv"
	json_path = output_dir / f"{stem}_summary.json"

	fieldnames = list(rows[0].keys()) if rows else []
	with csv_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	summary_payload: dict[str, Any] = {
		"model": args.llm_model,
		"scenario": args.scenario,
		"tries_per_role": args.tries,
		"temperature": args.temperature,
		"max_tokens": args.max_tokens,
		"top_logprobs": args.top_logprobs,
		"timeout": args.timeout,
		"request_seed_settings": {
			"mode": args.request_seed_mode,
			"base": args.request_seed_base,
		},
		"temperature_slug": temperature_slug,
		"context": context,
		"context_slug": context_slug,
		"arrangement_code": "".join(neighbors),
		"empty_slots": args.empty_slots,
		"consistency_criteria": {
			"std_threshold": args.std_threshold,
			"agreement_threshold": args.agreement_threshold,
		},
		"summary_by_role": summary_by_role,
		"overall_consistent": overall_consistent,
		"recommendation": recommendation,
		"outputs": {
			"trials_csv": str(csv_path),
			"summary_json": str(json_path),
		},
	}

	with json_path.open("w", encoding="utf-8") as handle:
		json.dump(summary_payload, handle, indent=2)

	print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
	main()
