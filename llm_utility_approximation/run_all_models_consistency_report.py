#!/usr/bin/env python3
"""Run fixed/varying seed consistency checks for all online Ollama models.

This script:
1) Discovers available models from the provider backing --llm-url.
2) Runs check_llm_decision_prob_consistency.py twice per model:
   - fixed request seed
   - varying request seed
3) Compares outcomes and emits a final report indicating whether each model is
   deterministic enough for log-probability-based estimation in Schelling runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

try:
	from llm_utility_approximation import check_llm_decision_prob_consistency as consistency_check
	from llm_utility_approximation import llm_token_probabilities as ltp
except ImportError:
	import check_llm_decision_prob_consistency as consistency_check
	import llm_token_probabilities as ltp

MODELS = ['gemma3:27b', 'gemma3:4b', 'gemma3:latest', 'granite3.1-dense:latest', 'granite3.1-moe:latest', 'hermes3:latest', 'llama3.1:405B', 'llama3.1:70B', 'llama3.2:latest', 'llama3.3:latest', 'mistral:instruct', 'mixtral:8x22b', 'mixtral:8x22b-instruct', 'phi4:latest', 'qwen2.5-coder:32B', 'qwen2.5-coder:latest', 'qwq:latest']

@dataclass
class CheckRunResult:
	mode: str
	success: bool
	summary: dict[str, Any] | None
	error: str | None
	stdout: str
	stderr: str


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Discover all models at an online Ollama/OpenAI-compatible endpoint, "
			"run fixed/varying seed consistency checks, and generate a final "
			"determinism report for Schelling log-probability estimation."
		)
	)
	parser.add_argument(
		"--llm-url",
		type=str,
		default=str(ltp.online_ollama_url),
		help="Chat endpoint URL used for consistency checks",
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
		"--tries",
		type=int,
		default=100,
		help="Trials per role for each mode",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=1.0,
		help="Sampling temperature for each request",
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
		help="Top-k token alternatives",
	)
	parser.add_argument(
		"--timeout",
		type=int,
		default=30,
		help="HTTP timeout in seconds",
	)
	parser.add_argument(
		"--empty-slots",
		type=int,
		default=2,
		choices=[0, 1, 2],
		help="Number of empty slots in balanced neighborhood",
	)
	parser.add_argument(
		"--layout-seed",
		type=int,
		default=17,
		help="Seed for neighborhood arrangement",
	)
	parser.add_argument(
		"--fixed-seed-base",
		type=int,
		default=12345,
		help="Request seed value used in fixed mode",
	)
	parser.add_argument(
		"--varying-seed-base",
		type=int,
		default=12345,
		help="Base request seed for varying mode (seed = base + sample_index)",
	)
	parser.add_argument(
		"--std-threshold",
		type=float,
		default=0.08,
		help="Role-level consistency std threshold passed to underlying checker",
	)
	parser.add_argument(
		"--agreement-threshold",
		type=float,
		default=0.75,
		help="Role-level majority agreement threshold passed to checker",
	)
	parser.add_argument(
		"--cross-seed-max-mean-delta",
		type=float,
		default=0.01,
		help="Maximum allowed abs delta in mean move probability across modes",
	)
	parser.add_argument(
		"--cross-seed-max-std-delta",
		type=float,
		default=0.01,
		help="Maximum allowed abs delta in std move probability across modes",
	)
	parser.add_argument(
		"--cross-seed-max-agreement-delta",
		type=float,
		default=0.1,
		help="Maximum allowed abs delta in majority agreement across modes",
	)
	parser.add_argument(
		"--cross-temp-max-mean-delta",
		type=float,
		default=0.01,
		help="Maximum allowed abs delta in mean move probability across temperatures",
	)
	parser.add_argument(
		"--cross-temp-max-std-delta",
		type=float,
		default=0.01,
		help="Maximum allowed abs delta in std move probability across temperatures",
	)
	parser.add_argument(
		"--cross-temp-max-agreement-delta",
		type=float,
		default=0.1,
		help="Maximum allowed abs delta in majority agreement across temperatures",
	)
	parser.add_argument(
		"--model-name-contains",
		type=str,
		default=None,
		help="Optional substring filter for discovered model names",
	)
	parser.add_argument(
		"--exclude-model-name-contains",
		type=str,
		default=None,
		help="Optional substring exclusion for model names",
	)
	parser.add_argument(
		"--max-models",
		type=int,
		default=None,
		help="Optional cap on number of models to evaluate",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="llm_log_probs/consistency_checks/all_models",
		help="Directory where aggregate report artifacts are written",
	)
	parser.add_argument(
		"--prompt-nonce-per-trial",
		action="store_true",
		default=True,
		help="Append unique nonce to each trial prompt (default: enabled)",
	)
	parser.add_argument(
		"--no-prompt-nonce-per-trial",
		action="store_false",
		dest="prompt_nonce_per_trial",
		help="Disable per-trial prompt nonce",
	)
	return parser.parse_args()


def _auth_headers(api_key: str | None) -> dict[str, str]:
	headers: dict[str, str] = {"Content-Type": "application/json"}
	if api_key:
		headers["Authorization"] = f"Bearer {api_key}"
	return headers


def _candidate_model_list_endpoints(llm_url: str) -> list[tuple[str, str]]:
	parsed = urlparse(llm_url)
	if parsed.scheme not in {"http", "https"} or not parsed.netloc:
		raise ValueError(f"Invalid --llm-url: {llm_url}")

	base = f"{parsed.scheme}://{parsed.netloc}"
	path = parsed.path.rstrip("/")
	prefix = path
	if path.endswith("/api/chat"):
		prefix = path[: -len("/api/chat")]
	elif path.endswith("/v1/chat/completions"):
		prefix = path[: -len("/v1/chat/completions")]
	elif path.endswith("/chat/completions"):
		prefix = path[: -len("/chat/completions")]
	elif path.endswith("/v1"):
		prefix = path[: -len("/v1")]

	candidates: list[tuple[str, str]] = []
	if prefix:
		candidates.append((f"{base}{prefix}/api/tags", "ollama_tags"))
		candidates.append((f"{base}{prefix}/v1/models", "openai_models"))
	candidates.append((f"{base}/api/tags", "ollama_tags"))
	candidates.append((f"{base}/v1/models", "openai_models"))

	seen: set[str] = set()
	unique: list[tuple[str, str]] = []
	for endpoint, endpoint_type in candidates:
		if endpoint not in seen:
			seen.add(endpoint)
			unique.append((endpoint, endpoint_type))
	return unique


def discover_models(llm_url: str, api_key: str | None, timeout: int) -> tuple[list[str], str]:
	headers = _auth_headers(api_key)
	last_error = "No endpoint attempted"

	with requests.Session() as session:
		for endpoint, endpoint_type in _candidate_model_list_endpoints(llm_url):
			try:
				resp = session.get(endpoint, headers=headers, timeout=timeout)
				resp.raise_for_status()
				data = resp.json()
				models: list[str] = []

				if endpoint_type == "ollama_tags":
					items = data.get("models", []) if isinstance(data, dict) else []
					if isinstance(items, list):
						for item in items:
							if isinstance(item, dict):
								name = item.get("name")
								if isinstance(name, str) and name.strip():
									models.append(name.strip())
				else:
					items = data.get("data", []) if isinstance(data, dict) else []
					if isinstance(items, list):
						for item in items:
							if isinstance(item, dict):
								name = item.get("id")
								if isinstance(name, str) and name.strip():
									models.append(name.strip())

				models = sorted(set(models))
				if len(models) > 0:
					return models, endpoint
				last_error = f"No models found at {endpoint}"
			except Exception as exc:
				last_error = f"{endpoint}: {exc}"

	raise RuntimeError(f"Failed to discover models from provider. Last error: {last_error}")


def _extract_json_stdout(stdout: str) -> dict[str, Any]:
	text = stdout.strip()
	if not text:
		raise ValueError("Empty stdout from consistency script")
	start = text.find("{")
	end = text.rfind("}")
	if start < 0 or end < 0 or end <= start:
		raise ValueError("Could not locate JSON object in stdout")
	return json.loads(text[start : end + 1])


def _parse_temperature_values() -> list[float]:
	return [0.0, 0.3, 1.0]


def run_mode_for_model(
	model: str,
	mode: str,
	seed_base: int,
	temperature: float,
	args: argparse.Namespace,
	mode_output_dir: Path,
) -> CheckRunResult:
	cmd = [
		sys.executable,
		str(Path(consistency_check.__file__).resolve()),
		"--llm-model",
		model,
		"--llm-url",
		args.llm_url,
		"--scenario",
		args.scenario,
		"--tries",
		str(args.tries),
		"--temperature",
		str(temperature),
		"--max-tokens",
		str(args.max_tokens),
		"--top-logprobs",
		str(args.top_logprobs),
		"--timeout",
		str(args.timeout),
		"--empty-slots",
		str(args.empty_slots),
		"--seed",
		str(args.layout_seed),
		"--request-seed-mode",
		mode,
		"--request-seed-base",
		str(seed_base),
		"--std-threshold",
		str(args.std_threshold),
		"--agreement-threshold",
		str(args.agreement_threshold),
		"--output-dir",
		str(mode_output_dir),
	]

	if args.llm_api_key:
		cmd.extend(["--llm-api-key", args.llm_api_key])

	if args.prompt_nonce_per_trial:
		cmd.append("--prompt-nonce-per-trial")
	else:
		cmd.append("--no-prompt-nonce-per-trial")

	process = subprocess.run(cmd, capture_output=True, text=True)
	if process.returncode != 0:
		return CheckRunResult(
			mode=mode,
			success=False,
			summary=None,
			error=(process.stderr.strip() or process.stdout.strip() or f"Exit code {process.returncode}"),
			stdout=process.stdout,
			stderr=process.stderr,
		)

	try:
		summary = _extract_json_stdout(process.stdout)
		return CheckRunResult(
			mode=mode,
			success=True,
			summary=summary,
			error=None,
			stdout=process.stdout,
			stderr=process.stderr,
		)
	except Exception as exc:
		return CheckRunResult(
			mode=mode,
			success=False,
			summary=None,
			error=f"Failed to parse summary JSON: {exc}",
			stdout=process.stdout,
			stderr=process.stderr,
		)


def _role_delta_metrics(
	left_role: dict[str, Any],
	right_role: dict[str, Any],
	left_label_key: str,
	right_label_key: str,
) -> dict[str, Any]:
	mean_delta = abs(float(right_role.get("mean_move_probability", 0.0)) - float(left_role.get("mean_move_probability", 0.0)))
	std_delta = abs(float(right_role.get("std_move_probability", 0.0)) - float(left_role.get("std_move_probability", 0.0)))
	agreement_delta = abs(float(right_role.get("majority_agreement", 0.0)) - float(left_role.get("majority_agreement", 0.0)))

	left_majority = str(left_role.get("majority_label", ""))
	right_majority = str(right_role.get("majority_label", ""))
	labels_match = left_majority == right_majority

	return {
		"mean_delta": mean_delta,
		"std_delta": std_delta,
		"agreement_delta": agreement_delta,
		"labels_match": labels_match,
		left_label_key: left_majority,
		right_label_key: right_majority,
	}


def compare_fixed_varying(
	fixed_summary: dict[str, Any],
	varying_summary: dict[str, Any],
	args: argparse.Namespace,
) -> dict[str, Any]:
	roles = ["type_a", "type_b"]
	max_mean_delta = 0.0
	max_std_delta = 0.0
	max_agreement_delta = 0.0
	majority_label_match = True
	role_deltas: dict[str, Any] = {}

	for role in roles:
		fixed_role = fixed_summary.get("summary_by_role", {}).get(role, {})
		vary_role = varying_summary.get("summary_by_role", {}).get(role, {})
		delta = _role_delta_metrics(
			left_role=fixed_role,
			right_role=vary_role,
			left_label_key="majority_label_fixed",
			right_label_key="majority_label_varying",
		)
		if not bool(delta["labels_match"]):
			majority_label_match = False

		max_mean_delta = max(max_mean_delta, float(delta["mean_delta"]))
		max_std_delta = max(max_std_delta, float(delta["std_delta"]))
		max_agreement_delta = max(max_agreement_delta, float(delta["agreement_delta"]))

		role_deltas[role] = {
			"mean_move_probability_abs_delta": delta["mean_delta"],
			"std_move_probability_abs_delta": delta["std_delta"],
			"majority_agreement_abs_delta": delta["agreement_delta"],
			"majority_label_fixed": delta["majority_label_fixed"],
			"majority_label_varying": delta["majority_label_varying"],
			"majority_label_match": delta["labels_match"],
		}

	fixed_consistent = bool(fixed_summary.get("overall_consistent"))
	vary_consistent = bool(varying_summary.get("overall_consistent"))

	deterministic_enough = (
		fixed_consistent
		and vary_consistent
		and majority_label_match
		and max_mean_delta <= float(args.cross_seed_max_mean_delta)
		and max_std_delta <= float(args.cross_seed_max_std_delta)
		and max_agreement_delta <= float(args.cross_seed_max_agreement_delta)
	)

	return {
		"fixed_overall_consistent": fixed_consistent,
		"varying_overall_consistent": vary_consistent,
		"majority_label_match_all_roles": majority_label_match,
		"max_mean_move_probability_abs_delta": max_mean_delta,
		"max_std_move_probability_abs_delta": max_std_delta,
		"max_majority_agreement_abs_delta": max_agreement_delta,
		"role_deltas": role_deltas,
		"deterministic_enough_for_schelling_logprob_estimation": deterministic_enough,
	}


def compare_across_temperatures(
	per_temperature_fixed_summaries: dict[str, dict[str, Any]],
	args: argparse.Namespace,
) -> dict[str, Any]:
	temp_keys = list(per_temperature_fixed_summaries.keys())
	if len(temp_keys) <= 1:
		return {
			"enabled": False,
			"reference_temperature": temp_keys[0] if len(temp_keys) == 1 else None,
			"temperature_pairs": {},
			"max_mean_move_probability_abs_delta": 0.0,
			"max_std_move_probability_abs_delta": 0.0,
			"max_majority_agreement_abs_delta": 0.0,
			"majority_label_match_all_pairs": True,
			"temperature_invariant_enough": True,
		}

	reference = temp_keys[0]
	roles = ["type_a", "type_b"]
	max_mean_delta = 0.0
	max_std_delta = 0.0
	max_agreement_delta = 0.0
	majority_label_match_all_pairs = True
	temperature_pairs: dict[str, Any] = {}

	ref_summary = per_temperature_fixed_summaries[reference]
	for other in temp_keys[1:]:
		other_summary = per_temperature_fixed_summaries[other]
		role_deltas: dict[str, Any] = {}
		for role in roles:
			ref_role = ref_summary.get("summary_by_role", {}).get(role, {})
			other_role = other_summary.get("summary_by_role", {}).get(role, {})
			delta = _role_delta_metrics(
				left_role=ref_role,
				right_role=other_role,
				left_label_key="majority_label_reference",
				right_label_key="majority_label_other",
			)
			if not bool(delta["labels_match"]):
				majority_label_match_all_pairs = False

			max_mean_delta = max(max_mean_delta, float(delta["mean_delta"]))
			max_std_delta = max(max_std_delta, float(delta["std_delta"]))
			max_agreement_delta = max(max_agreement_delta, float(delta["agreement_delta"]))

			role_deltas[role] = {
				"mean_move_probability_abs_delta": delta["mean_delta"],
				"std_move_probability_abs_delta": delta["std_delta"],
				"majority_agreement_abs_delta": delta["agreement_delta"],
				"majority_label_reference": delta["majority_label_reference"],
				"majority_label_other": delta["majority_label_other"],
				"majority_label_match": delta["labels_match"],
			}

		temperature_pairs[f"{reference}_vs_{other}"] = {
			"reference_temperature": reference,
			"other_temperature": other,
			"role_deltas": role_deltas,
		}

	temp_invariant_enough = (
		majority_label_match_all_pairs
		and max_mean_delta <= float(args.cross_temp_max_mean_delta)
		and max_std_delta <= float(args.cross_temp_max_std_delta)
		and max_agreement_delta <= float(args.cross_temp_max_agreement_delta)
	)

	return {
		"enabled": True,
		"reference_temperature": reference,
		"temperature_pairs": temperature_pairs,
		"max_mean_move_probability_abs_delta": max_mean_delta,
		"max_std_move_probability_abs_delta": max_std_delta,
		"max_majority_agreement_abs_delta": max_agreement_delta,
		"majority_label_match_all_pairs": majority_label_match_all_pairs,
		"temperature_invariant_enough": temp_invariant_enough,
	}


def apply_model_filters(models: list[str], args: argparse.Namespace) -> list[str]:
	filtered = list(models)
	if args.model_name_contains:
		needle = args.model_name_contains.lower()
		filtered = [m for m in filtered if needle in m.lower()]
	if args.exclude_model_name_contains:
		needle = args.exclude_model_name_contains.lower()
		filtered = [m for m in filtered if needle not in m.lower()]
	if args.max_models is not None:
		filtered = filtered[: args.max_models]
	return filtered


def build_global_recommendation(num_total: int, num_passed: int) -> str:
	if num_total <= 0:
		return "No models were evaluated. Unable to determine suitability for Schelling log-probability estimation."
	share = num_passed / num_total
	if share >= 0.8:
		return (
			f"{num_passed}/{num_total} models met deterministic-enough criteria. "
			"Log-probability-based estimation appears broadly suitable for Schelling simulations on this provider."
		)
	if share >= 0.5:
		return (
			f"{num_passed}/{num_total} models met deterministic-enough criteria. "
			"Suitability is mixed; restrict Schelling log-probability estimation to passing models."
		)
	return (
		f"Only {num_passed}/{num_total} models met deterministic-enough criteria. "
		"Use caution: log-probability-based estimation may be unreliable for many models without additional averaging."
	)


def _build_error_model_result(
	model: str,
	error: str,
	temperature_values: list[float],
	fixed_run: CheckRunResult | None = None,
	varying_run: CheckRunResult | None = None,
	fixed_summary: dict[str, Any] | None = None,
	varying_summary: dict[str, Any] | None = None,
	varying_temp_runs: dict[str, Any] | None = None,
) -> dict[str, Any]:
	result: dict[str, Any] = {
		"model": model,
		"status": "error",
		"error": error,
		"comparison": {
			"deterministic_enough_for_schelling_logprob_estimation": False,
		},
		"temperature_comparison": {
			"enabled": len(temperature_values) > 1,
			"temperature_invariant_enough": False,
		},
	}

	if fixed_run is not None:
		result["fixed_run"] = {
			"success": fixed_run.success,
			"error": fixed_run.error,
		}
	if varying_run is not None:
		result["varying_run"] = {
			"success": varying_run.success,
			"error": varying_run.error,
		}
	if fixed_summary is not None:
		result["fixed_summary"] = fixed_summary
	if varying_summary is not None:
		result["varying_summary"] = varying_summary
	if varying_temp_runs is not None:
		result["varying_temp_runs"] = varying_temp_runs

	return result


def write_reports(output_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
	output_dir.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	json_path = output_dir / f"all_models_consistency_report_{timestamp}.json"
	csv_path = output_dir / f"all_models_consistency_report_{timestamp}.csv"
	md_path = output_dir / f"all_models_consistency_report_{timestamp}.md"

	with json_path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2)

	rows = payload.get("model_results", [])
	fieldnames = [
		"model",
		"status",
		"deterministic_enough_for_schelling_logprob_estimation",
		"temperature_invariant_enough",
		"fixed_overall_consistent",
		"varying_overall_consistent",
		"majority_label_match_all_roles",
		"max_mean_move_probability_abs_delta",
		"max_std_move_probability_abs_delta",
		"max_majority_agreement_abs_delta",
		"max_temperature_mean_move_probability_abs_delta",
		"max_temperature_std_move_probability_abs_delta",
		"max_temperature_majority_agreement_abs_delta",
		"error",
	]
	with csv_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			comparison = row.get("comparison", {}) if isinstance(row, dict) else {}
			temp_cmp = row.get("temperature_comparison", {}) if isinstance(row, dict) else {}
			writer.writerow(
				{
					"model": row.get("model"),
					"status": row.get("status"),
					"deterministic_enough_for_schelling_logprob_estimation": comparison.get(
						"deterministic_enough_for_schelling_logprob_estimation"
					),
					"temperature_invariant_enough": temp_cmp.get("temperature_invariant_enough"),
					"fixed_overall_consistent": comparison.get("fixed_overall_consistent"),
					"varying_overall_consistent": comparison.get("varying_overall_consistent"),
					"majority_label_match_all_roles": comparison.get("majority_label_match_all_roles"),
					"max_mean_move_probability_abs_delta": comparison.get("max_mean_move_probability_abs_delta"),
					"max_std_move_probability_abs_delta": comparison.get("max_std_move_probability_abs_delta"),
					"max_majority_agreement_abs_delta": comparison.get("max_majority_agreement_abs_delta"),
					"max_temperature_mean_move_probability_abs_delta": temp_cmp.get("max_mean_move_probability_abs_delta"),
					"max_temperature_std_move_probability_abs_delta": temp_cmp.get("max_std_move_probability_abs_delta"),
					"max_temperature_majority_agreement_abs_delta": temp_cmp.get("max_majority_agreement_abs_delta"),
					"error": row.get("error"),
				}
			)

	passing_models = payload.get("passing_models", [])
	failing_models = payload.get("failing_models", [])
	lines = [
		"# All-Model Consistency Report",
		"",
		f"Generated at: {payload.get('generated_at')}",
		f"Scenario: {payload.get('scenario')}",
		f"Temperatures: {payload.get('temperatures')}",
		f"Total models evaluated: {payload.get('models_evaluated')}",
		f"Models passing deterministic-enough criterion: {payload.get('models_passing')}",
		"",
		"## Final recommendation",
		payload.get("global_recommendation", ""),
		"",
		"## Passing models",
	]
	if isinstance(passing_models, list) and len(passing_models) > 0:
		lines.extend([f"- {m}" for m in passing_models])
	else:
		lines.append("- None")
	lines.append("")
	lines.append("## Failing models")
	if isinstance(failing_models, list) and len(failing_models) > 0:
		lines.extend([f"- {m}" for m in failing_models])
	else:
		lines.append("- None")
	lines.append("")
	lines.append("## Criteria")
	criteria = payload.get("criteria", {})
	for key, value in criteria.items():
		lines.append(f"- {key}: {value}")

	with md_path.open("w", encoding="utf-8") as handle:
		handle.write("\n".join(lines) + "\n")

	return {
		"report_json": str(json_path),
		"report_csv": str(csv_path),
		"report_md": str(md_path),
	}


def main() -> None:
	args = parse_args()
	if args.tries < 1:
		raise ValueError("--tries must be >= 1")
	temperature_values = _parse_temperature_values()

	all_models, discovery_endpoint = discover_models(args.llm_url, args.llm_api_key, args.timeout)
	print(f"Discovered {len(all_models)} models at {discovery_endpoint}: {all_models}")

	# selected_models = apply_model_filters(all_models, args)
	print(f"Using the following models for consistency checks: {MODELS}")
	selected_models = apply_model_filters(MODELS, args)

	output_dir = Path(args.output_dir)
	per_model_root = output_dir / "per_model"
	model_results: list[dict[str, Any]] = []

	for model in selected_models:
		model_slug = ltp._sanitize_model_for_path_component(model)
		model_output_dir = per_model_root / model_slug

		# Seed-mode comparison is done once at the base temperature.
		fixed_dir = model_output_dir / "fixed_seed"
		vary_dir = model_output_dir / "varying_seed"

		fixed_run = run_mode_for_model(
			model=model,
			mode="fixed",
			seed_base=args.fixed_seed_base,
			temperature=float(args.temperature),
			args=args,
			mode_output_dir=fixed_dir,
		)
		varying_run = run_mode_for_model(
			model=model,
			mode="varying",
			seed_base=args.varying_seed_base,
			temperature=float(args.temperature),
			args=args,
			mode_output_dir=vary_dir,
		)

		if not fixed_run.success or not varying_run.success:
			error_parts = []
			if not fixed_run.success:
				error_parts.append(f"fixed: {fixed_run.error}")
			if not varying_run.success:
				error_parts.append(f"varying: {varying_run.error}")
			model_results.append(
				_build_error_model_result(
					model=model,
					error=" | ".join(error_parts),
					temperature_values=temperature_values,
					fixed_run=fixed_run,
					varying_run=varying_run,
				)
			)
			continue

		comparison = compare_fixed_varying(fixed_run.summary or {}, varying_run.summary or {}, args)

		# Varying-temperature check uses the same fixed request seed across all temperatures.
		varying_temp_runs: dict[str, Any] = {}
		per_temperature_fixed_summaries: dict[str, dict[str, Any]] = {}
		temp_errors: list[str] = []

		for temp in temperature_values:
			temp_key = str(temp)
			temp_slug = temp_key.replace("-", "m").replace(".", "p")
			temp_dir = model_output_dir / "varying_temp" / f"T{temp_slug}"
			temp_run = run_mode_for_model(
				model=model,
				mode="fixed",
				seed_base=args.fixed_seed_base,
				temperature=temp,
				args=args,
				mode_output_dir=temp_dir,
			)

			if not temp_run.success:
				temp_errors.append(f"T={temp_key}: {temp_run.error}")
				varying_temp_runs[temp_key] = {
					"status": "error",
					"error": temp_run.error,
				}
				continue

			varying_temp_runs[temp_key] = {
				"status": "ok",
				"fixed_summary": temp_run.summary,
			}
			per_temperature_fixed_summaries[temp_key] = temp_run.summary or {}

		if len(temp_errors) > 0:
			model_results.append(
				_build_error_model_result(
					model=model,
					error=" | ".join(temp_errors),
					temperature_values=temperature_values,
					fixed_summary=fixed_run.summary,
					varying_summary=varying_run.summary,
					varying_temp_runs=varying_temp_runs,
				)
			)
			continue

		temperature_comparison = compare_across_temperatures(
			per_temperature_fixed_summaries=per_temperature_fixed_summaries,
			args=args,
		)

		comparison["deterministic_enough_for_schelling_logprob_estimation"] = bool(
			comparison.get("deterministic_enough_for_schelling_logprob_estimation")
			and temperature_comparison.get("temperature_invariant_enough", True)
		)

		model_results.append(
			{
				"model": model,
				"status": "ok",
				"error": None,
				"fixed_summary": fixed_run.summary,
				"varying_summary": varying_run.summary,
				"comparison": comparison,
				"varying_temp_runs": varying_temp_runs,
				"temperature_comparison": temperature_comparison,
			}
		)

	passing_models = [
		str(item["model"])
		for item in model_results
		if item.get("status") == "ok"
		and bool(item.get("comparison", {}).get("deterministic_enough_for_schelling_logprob_estimation"))
	]
	failing_models = [
		str(item["model"])
		for item in model_results
		if not bool(item.get("comparison", {}).get("deterministic_enough_for_schelling_logprob_estimation"))
	]

	payload: dict[str, Any] = {
		"generated_at": datetime.now().isoformat(timespec="seconds"),
		"llm_url": args.llm_url,
		"model_discovery_endpoint": discovery_endpoint,
		"scenario": args.scenario,
		"tries_per_role": args.tries,
		"temperature": float(args.temperature),
		"temperatures": temperature_values,
		"max_tokens": args.max_tokens,
		"top_logprobs": args.top_logprobs,
		"timeout": args.timeout,
		"empty_slots": args.empty_slots,
		"layout_seed": args.layout_seed,
		"fixed_seed_base": args.fixed_seed_base,
		"varying_seed_base": args.varying_seed_base,
		"models_discovered": len(all_models),
		"models_selected": len(selected_models),
		"models_evaluated": len(model_results),
		"models_passing": len(passing_models),
		"passing_models": passing_models,
		"failing_models": failing_models,
		"criteria": {
			"role_std_threshold": args.std_threshold,
			"role_agreement_threshold": args.agreement_threshold,
			"cross_seed_max_mean_delta": args.cross_seed_max_mean_delta,
			"cross_seed_max_std_delta": args.cross_seed_max_std_delta,
			"cross_seed_max_agreement_delta": args.cross_seed_max_agreement_delta,
			"cross_temp_max_mean_delta": args.cross_temp_max_mean_delta,
			"cross_temp_max_std_delta": args.cross_temp_max_std_delta,
			"cross_temp_max_agreement_delta": args.cross_temp_max_agreement_delta,
		},
		"global_recommendation": build_global_recommendation(len(model_results), len(passing_models)),
		"model_results": model_results,
	}

	report_paths = write_reports(output_dir, payload)
	payload["report_paths"] = report_paths

	print(json.dumps({
		"status": "ok",
		"models_discovered": len(all_models),
		"models_selected": len(selected_models),
		"models_evaluated": len(model_results),
		"models_passing": len(passing_models),
		"global_recommendation": payload["global_recommendation"],
		"report_paths": report_paths,
	}, indent=2))


if __name__ == "__main__":
	main()
