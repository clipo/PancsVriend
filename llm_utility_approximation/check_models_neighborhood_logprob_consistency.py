#!/usr/bin/env python3
"""Check logprob consistency across neighborhood configurations for multiple models.

This script intentionally targets one fixed chat endpoint:
https://chat.binghamton.edu/ollama/api/chat

It sweeps sampled valid Schelling neighborhoods, runs repeated trials for both
agent roles, and summarizes consistency at temperature 1.0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import requests

try:
	from context_scenarios import CONTEXT_SCENARIOS
except ModuleNotFoundError:
	REPO_ROOT = Path(__file__).resolve().parent.parent
	if str(REPO_ROOT) not in sys.path:
		sys.path.insert(0, str(REPO_ROOT))
	from context_scenarios import CONTEXT_SCENARIOS

try:
	import llm_token_probabilities as ltp
except ModuleNotFoundError:
	from llm_utility_approximation import llm_token_probabilities as ltp


CHAT_URL = "https://chat.binghamton.edu/ollama/api/chat"
MODELS = ['gemma3:27b', 'gemma3:4b', 'gemma3:latest', 'granite3.1-dense:latest', 'granite3.1-moe:latest', 'hermes3:latest', 'llama3.1:405B', 'llama3.1:70B', 'llama3.2:latest', 'llama3.3:latest', 'mistral:instruct', 'mixtral:8x22b', 'mixtral:8x22b-instruct', 'phi4:latest', 'qwen2.5-coder:32B', 'qwen2.5-coder:latest', 'qwq:latest']


def _status(message: str) -> None:
	timestamp = datetime.now().strftime("%H:%M:%S")
	print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Run temperature=1.0 neighborhood consistency checks for a list of models "
			"on the fixed Binghamton chat endpoint."
		)
	)
	parser.add_argument(
		"--llm-api-key",
		type=str,
		default=str(ltp.ONLINE_API_KEY) if bool(ltp.Use_ONLINE_API) else None,
		help="API key for endpoint authentication",
	)
	parser.add_argument(
		"--scenario",
		type=str,
		default="baseline",
		help="Scenario key from context_scenarios.py",
	)
	parser.add_argument(
		"--models",
		type=str,
		nargs="*",
		default=MODELS,
		help="Explicit list of model names to evaluate",
	)
	parser.add_argument(
		"--models-file",
		type=str,
		default=str(Path(__file__).resolve().parent / "hosted_models_list.txt"),
		help="Newline-delimited model list file used when --models is not provided",
	)
	parser.add_argument(
		"--num-contexts",
		type=int,
		default=100,
		help="Number of neighborhood configurations to sample",
	)
	parser.add_argument(
		"--context-seed",
		type=int,
		default=42,
		help="Seed used to sample neighborhood configurations",
	)
	parser.add_argument(
		"--tries",
		type=int,
		default=10,
		help="Trials per role per context",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=1.0,
		help="Sampling temperature (default: 1.0)",
	)
	parser.add_argument(
		"--max-tokens",
		type=int,
		default=24,
		help="Max generation tokens",
	)
	parser.add_argument(
		"--top-logprobs",
		type=int,
		default=20,
		help="Top-k alternatives per token",
	)
	parser.add_argument(
		"--timeout",
		type=int,
		default=30,
		help="HTTP timeout in seconds",
	)
	parser.add_argument(
		"--request-seed-base",
		type=int,
		default=12345,
		help="Base request seed; request seed = base + sample_index",
	)
	parser.add_argument(
		"--std-threshold",
		type=float,
		default=0.08,
		help="Max role stddev(move_probability) to call role consistent",
	)
	parser.add_argument(
		"--agreement-threshold",
		type=float,
		default=0.75,
		help="Min role majority agreement to call role consistent",
	)
	parser.add_argument(
		"--prompt-nonce-per-trial",
		action="store_true",
		default=True,
		help="Append per-trial nonce to prompt (default: enabled)",
	)
	parser.add_argument(
		"--no-prompt-nonce-per-trial",
		action="store_false",
		dest="prompt_nonce_per_trial",
		help="Disable per-trial nonce",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="llm_log_probs/consistency_checks/neighborhood_sweep",
		help="Directory for output artifacts",
	)
	return parser.parse_args()


def _load_models_from_file(path: Path) -> list[str]:
	if not path.exists():
		raise FileNotFoundError(f"Model list file not found: {path}")
	models: list[str] = []
	with path.open("r", encoding="utf-8") as handle:
		for raw in handle:
			name = raw.strip()
			if not name:
				continue
			if name.startswith("#"):
				continue
			models.append(name)
	return models


def _resolve_models(args: argparse.Namespace) -> list[str]:
	if args.models and len(args.models) > 0:
		models = [m.strip() for m in args.models if m.strip()]
		if len(models) == 0:
			raise ValueError("--models was provided but no non-empty names were found")
		return sorted(set(models))

	file_models = _load_models_from_file(Path(args.models_file).resolve())
	if len(file_models) == 0:
		raise ValueError(f"No model names found in --models-file: {args.models_file}")
	return sorted(set(file_models))


def _request_seed_for_call(base: int, sample_index: int) -> int:
	return int(base) + int(sample_index)


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


def _arrangement_stats(neighbors: list[str]) -> dict[str, int]:
	return {
		"num_similar": int(sum(1 for token in neighbors if token == "S")),
		"num_opposite": int(sum(1 for token in neighbors if token == "O")),
		"num_empty": int(sum(1 for token in neighbors if token == "E")),
		"num_wall": int(sum(1 for token in neighbors if token == "#")),
	}


def _pick_contexts(num_contexts: int, context_seed: int) -> list[tuple[int, list[str], str, str, dict[str, int]]]:
	all_arrangements = ltp.generate_all_valid_schelling_neighbors(ltp.SCHELLING_GRID_SIZE)
	if num_contexts > len(all_arrangements):
		raise ValueError(
			f"Requested {num_contexts} contexts but only {len(all_arrangements)} available valid contexts"
		)

	rng = random.Random(context_seed)
	selected_indices = sorted(rng.sample(range(len(all_arrangements)), k=num_contexts))
	selected: list[tuple[int, list[str], str, str, dict[str, int]]] = []
	for rank, arrangement_idx in enumerate(selected_indices, start=1):
		neighbors = all_arrangements[arrangement_idx]
		arrangement_code = "".join(neighbors)
		context = ltp.generate_neighbor_context(neighbors)
		selected.append((rank, neighbors, arrangement_code, context, _arrangement_stats(neighbors)))
	return selected


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
	if len(rows) == 0:
		with path.open("w", newline="", encoding="utf-8") as handle:
			handle.write("")
		return
	fieldnames = list(rows[0].keys())
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _run_model(
	model: str,
	args: argparse.Namespace,
	scenario: dict[str, Any],
	selected_contexts: list[tuple[int, list[str], str, str, dict[str, int]]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
	urls = [CHAT_URL]
	role_configs = [
		("type_a", str(scenario["type_a"]), str(scenario["type_b"])),
		("type_b", str(scenario["type_b"]), str(scenario["type_a"])),
	]

	trial_rows: list[dict[str, Any]] = []
	context_rows: list[dict[str, Any]] = []
	sample_index = 0
	total_contexts = len(selected_contexts)
	total_trials = total_contexts * 2 * int(args.tries)
	model_start = time.perf_counter()
	_status(
		f"Model {model}: starting {total_contexts} context(s), {args.tries} trial(s) per role "
		f"({total_trials} total requests expected)"
	)

	with requests.Session() as session:
		for context_rank, neighbors, arrangement_code, context, context_stats in selected_contexts:
			_status(f"Model {model}: context {context_rank}/{total_contexts} started")
			context_role_rows: dict[str, list[dict[str, Any]]] = {"type_a": [], "type_b": []}
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
						prompt = (
							f"{base_prompt}\n\n"
							f"[trial_nonce:model={model};ctx={context_rank};role={role_name};trial={trial_index}]"
						)

					request_seed = _request_seed_for_call(args.request_seed_base, sample_index)
					task = {
						"urls": urls,
						"model": model,
						"prompt": prompt,
						"temperature": args.temperature,
						"max_tokens": args.max_tokens,
						"top_logprobs": args.top_logprobs,
						"timeout": args.timeout,
						"sample_index": sample_index,
						"max_meaningful_reasks": int(ltp.MEANINGFUL_ANSWER_MAX_REASKS),
						"api_key": args.llm_api_key,
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

					row = {
						"model": model,
						"scenario": args.scenario,
						"temperature": float(args.temperature),
						"context_rank": int(context_rank),
						"arrangement_code": arrangement_code,
						"context": context,
						"agent_role": role_name,
						"prompt_nonce_per_trial": bool(args.prompt_nonce_per_trial),
						"agent_label": role_label,
						"opposite_label": opposite_label,
						"trial_index": trial_index,
						"sample_index": sample_index,
						"request_seed": request_seed,
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
						"num_similar": context_stats["num_similar"],
						"num_opposite": context_stats["num_opposite"],
						"num_empty": context_stats["num_empty"],
						"num_wall": context_stats["num_wall"],
					}
					trial_rows.append(row)
					context_role_rows[role_name].append(row)
					sample_index += 1

			role_summary: dict[str, dict[str, Any]] = {}
			for role_name in ["type_a", "type_b"]:
				role_summary[role_name] = _role_consistency_summary(
					role_rows=context_role_rows[role_name],
					std_threshold=args.std_threshold,
					agreement_threshold=args.agreement_threshold,
				)

			context_consistent = all(bool(role_summary[role]["consistent"]) for role in ["type_a", "type_b"])
			context_row = {
				"model": model,
				"scenario": args.scenario,
				"temperature": float(args.temperature),
				"context_rank": int(context_rank),
				"arrangement_code": arrangement_code,
				"num_similar": context_stats["num_similar"],
				"num_opposite": context_stats["num_opposite"],
				"num_empty": context_stats["num_empty"],
				"num_wall": context_stats["num_wall"],
				"type_a_consistent": bool(role_summary["type_a"]["consistent"]),
				"type_b_consistent": bool(role_summary["type_b"]["consistent"]),
				"context_consistent": bool(context_consistent),
				"type_a_mean_move_probability": float(role_summary["type_a"]["mean_move_probability"]),
				"type_b_mean_move_probability": float(role_summary["type_b"]["mean_move_probability"]),
				"type_a_std_move_probability": float(role_summary["type_a"]["std_move_probability"]),
				"type_b_std_move_probability": float(role_summary["type_b"]["std_move_probability"]),
				"type_a_majority_agreement": float(role_summary["type_a"]["majority_agreement"]),
				"type_b_majority_agreement": float(role_summary["type_b"]["majority_agreement"]),
				"type_a_majority_label": str(role_summary["type_a"]["majority_label"]),
				"type_b_majority_label": str(role_summary["type_b"]["majority_label"]),
			}
			context_rows.append(context_row)
			completed_trials = sample_index
			progress_pct = (completed_trials / total_trials * 100.0) if total_trials > 0 else 0.0
			_status(
				f"Model {model}: context {context_rank}/{total_contexts} complete "
				f"({completed_trials}/{total_trials} requests, {progress_pct:.1f}%)"
			)

	context_consistency_flags = [bool(r["context_consistent"]) for r in context_rows]
	model_pass_rate = (
		sum(1 for flag in context_consistency_flags if flag) / len(context_consistency_flags)
		if len(context_consistency_flags) > 0
		else 0.0
	)
	model_summary = {
		"model": model,
		"status": "ok",
		"error": None,
		"scenario": args.scenario,
		"temperature": float(args.temperature),
		"num_contexts": int(len(context_rows)),
		"tries_per_role_per_context": int(args.tries),
		"contexts_consistent": int(sum(1 for flag in context_consistency_flags if flag)),
		"contexts_inconsistent": int(sum(1 for flag in context_consistency_flags if not flag)),
		"context_consistency_rate": float(model_pass_rate),
		"mean_context_type_a_std_move_probability": float(
			mean([float(r["type_a_std_move_probability"]) for r in context_rows]) if len(context_rows) > 0 else math.nan
		),
		"mean_context_type_b_std_move_probability": float(
			mean([float(r["type_b_std_move_probability"]) for r in context_rows]) if len(context_rows) > 0 else math.nan
		),
	}
	model_elapsed = time.perf_counter() - model_start
	_status(f"Model {model}: completed in {model_elapsed:.1f}s")
	return model_summary, trial_rows, context_rows


def _print_model_wise_output(model_summaries: list[dict[str, Any]]) -> None:
	print("\n=== Model-wise Output ===")
	for row in model_summaries:
		model = str(row.get("model", "<unknown-model>"))
		status = str(row.get("status", "unknown")).upper()
		if status != "OK":
			error = str(row.get("error", "unknown error"))
			print(f"- {model}: ERROR | {error}")
			continue

		rate = float(row.get("context_consistency_rate", 0.0))
		consistent = int(row.get("contexts_consistent", 0))
		inconsistent = int(row.get("contexts_inconsistent", 0))
		num_contexts = int(row.get("num_contexts", 0))
		print(
			f"- {model}: OK | consistency_rate={rate:.3f} "
			f"| contexts_consistent={consistent}/{num_contexts} "
			f"| contexts_inconsistent={inconsistent}"
		)


def _print_combined_summary(
	model_summaries: list[dict[str, Any]],
	successful_model_rows: list[dict[str, Any]],
	mean_model_consistency_rate: float,
	args: argparse.Namespace,
	run_dir: Path,
	report_json_path: Path,
) -> None:
	print("\n=== Combined Summary ===")
	print(f"- Models requested: {len(args.models) if args.models else 0}")
	print(f"- Models evaluated: {len(model_summaries)}")
	print(f"- Models successful: {len(successful_model_rows)}")
	print(f"- Models with errors: {len([r for r in model_summaries if r.get('status') != 'ok'])}")
	if not math.isnan(mean_model_consistency_rate):
		print(f"- Mean model context consistency rate: {mean_model_consistency_rate:.3f}")
	else:
		print("- Mean model context consistency rate: n/a")
	print(f"- Contexts per model: {int(args.num_contexts)}")
	print(f"- Trials per role per context: {int(args.tries)}")
	print(f"- Temperature: {float(args.temperature):.1f}")
	print(f"- Output directory: {run_dir}")
	print(f"- Report JSON: {report_json_path}")


def main() -> None:
	args = parse_args()
	run_start = time.perf_counter()
	_status("Starting neighborhood logprob consistency sweep")
	_status(
		f"Scenario={args.scenario} | models={len(args.models) if args.models else 0} "
		f"| contexts={args.num_contexts} | tries={args.tries} | temperature={args.temperature}"
	)
	if args.scenario not in CONTEXT_SCENARIOS:
		raise ValueError(f"Unknown --scenario '{args.scenario}'. Available: {sorted(CONTEXT_SCENARIOS.keys())}")
	if args.num_contexts < 1:
		raise ValueError("--num-contexts must be >= 1")
	if args.tries < 1:
		raise ValueError("--tries must be >= 1")

	requested_models = _resolve_models(args)
	selected_models = list(requested_models)
	_status(f"Resolved {len(selected_models)} model(s) for evaluation")

	if len(selected_models) == 0:
		raise ValueError("No models selected for evaluation after filtering/validation")

	scenario = CONTEXT_SCENARIOS[args.scenario]
	selected_contexts = _pick_contexts(args.num_contexts, args.context_seed)

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = output_dir / f"temp1p0_neighborhood_consistency_{run_stamp}"
	run_dir.mkdir(parents=True, exist_ok=True)

	all_trial_rows: list[dict[str, Any]] = []
	all_context_rows: list[dict[str, Any]] = []
	model_summaries: list[dict[str, Any]] = []

	for model_idx, model in enumerate(selected_models, start=1):
		_status(f"[{model_idx}/{len(selected_models)}] Running model {model}")
		try:
			model_summary, trial_rows, context_rows = _run_model(
				model=model,
				args=args,
				scenario=scenario,
				selected_contexts=selected_contexts,
			)
			model_summaries.append(model_summary)
			all_trial_rows.extend(trial_rows)
			all_context_rows.extend(context_rows)
			_status(f"[{model_idx}/{len(selected_models)}] Model {model} finished successfully")
		except Exception as exc:
			_status(f"[{model_idx}/{len(selected_models)}] Model {model} failed: {exc}")
			model_summaries.append(
				{
					"model": model,
					"status": "error",
					"error": str(exc),
					"scenario": args.scenario,
					"temperature": float(args.temperature),
				}
			)

	model_csv_rows = []
	for row in model_summaries:
		model_csv_rows.append(
			{
				"model": row.get("model"),
				"status": row.get("status"),
				"context_consistency_rate": row.get("context_consistency_rate"),
				"contexts_consistent": row.get("contexts_consistent"),
				"contexts_inconsistent": row.get("contexts_inconsistent"),
				"num_contexts": row.get("num_contexts"),
				"tries_per_role_per_context": row.get("tries_per_role_per_context"),
				"error": row.get("error"),
			}
		)

	trial_csv_path = run_dir / "trials.csv"
	context_csv_path = run_dir / "context_summary.csv"
	model_csv_path = run_dir / "model_summary.csv"
	report_json_path = run_dir / "report.json"

	_write_csv(trial_csv_path, all_trial_rows)
	_write_csv(context_csv_path, all_context_rows)
	_write_csv(model_csv_path, model_csv_rows)

	successful_model_rows = [r for r in model_summaries if r.get("status") == "ok"]
	if len(successful_model_rows) > 0:
		mean_model_consistency_rate = mean(float(r.get("context_consistency_rate", 0.0)) for r in successful_model_rows)
	else:
		mean_model_consistency_rate = math.nan

	report_payload: dict[str, Any] = {
		"generated_at": datetime.now().isoformat(timespec="seconds"),
		"llm_url": CHAT_URL,
		"scenario": args.scenario,
		"temperature": float(args.temperature),
		"tries_per_role_per_context": int(args.tries),
		"num_contexts": int(args.num_contexts),
		"context_seed": int(args.context_seed),
		"request_seed_mode": "varying",
		"request_seed_base": int(args.request_seed_base),
		"prompt_nonce_per_trial": bool(args.prompt_nonce_per_trial),
		"std_threshold": float(args.std_threshold),
		"agreement_threshold": float(args.agreement_threshold),
		"requested_models": requested_models,
		"selected_models": selected_models,
		"models_evaluated": len(model_summaries),
		"models_successful": len(successful_model_rows),
		"models_with_errors": len([r for r in model_summaries if r.get("status") != "ok"]),
		"mean_model_context_consistency_rate": float(mean_model_consistency_rate)
		if not math.isnan(mean_model_consistency_rate)
		else None,
		"outputs": {
			"trials_csv": str(trial_csv_path),
			"context_summary_csv": str(context_csv_path),
			"model_summary_csv": str(model_csv_path),
			"report_json": str(report_json_path),
		},
		"model_summaries": model_summaries,
	}

	with report_json_path.open("w", encoding="utf-8") as handle:
		json.dump(report_payload, handle, indent=2)
	_status(f"Saved outputs to {run_dir}")

	_print_model_wise_output(model_summaries)
	_print_combined_summary(
		model_summaries=model_summaries,
		successful_model_rows=successful_model_rows,
		mean_model_consistency_rate=mean_model_consistency_rate,
		args=args,
		run_dir=run_dir,
		report_json_path=report_json_path,
	)
	total_elapsed = time.perf_counter() - run_start
	_status(f"Run complete in {total_elapsed:.1f}s")

	print(
		json.dumps(
			{
				"status": "ok",
				"models_evaluated": len(model_summaries),
				"models_successful": len(successful_model_rows),
				"num_contexts": int(args.num_contexts),
				"temperature": float(args.temperature),
				"output_dir": str(run_dir),
				"report_json": str(report_json_path),
			},
			indent=2,
		)
	)


if __name__ == "__main__":
	main()
