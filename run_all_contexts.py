"""Unified multi-scenario runner for LLM experiments.

Usage Guide
-----------
Run from repo root after activating the project venv:

	python run_all_contexts.py --runs 10 --processes 5 --llm-model phi4:latest

Key options:
* --scenarios <names...>  Limit execution to specific scenarios (defaults to all).
* --runs N                Target number of runs per scenario (falls back to config when resuming).
* --processes P           Parallel worker count (omit for auto min(cpu_count, runs)).
* --no-parallel           Force sequential execution.
* --llm-model/url/api-key Override the LLM endpoint configuration.

The script checks for existing experiments whose config matches the requested
scenario and model. Incomplete experiments are resumed using their stored
parameters and resume seeds; otherwise a fresh experiment directory is created.
Results for each scenario are summarized at the end of the run.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import config as cfg
from context_scenarios import CONTEXT_SCENARIOS
from llm_runner import (
	_analyze_run_status,
	check_existing_experiment,
	run_llm_experiment,
)


def _sorted_experiment_dirs(root: Path) -> Iterable[Path]:
	if not root.exists():
		return []
	dirs: List[Path] = [p for p in root.iterdir() if p.is_dir()]
	return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)


def _find_resume_candidate(
	scenario: str,
	llm_model: str,
	target_runs: Optional[int] = None,
) -> Tuple[Optional[str], bool]:
	experiments_root = Path("experiments")
	for exp_dir in _sorted_experiment_dirs(experiments_root):
		config_path = exp_dir / "config.json"
		if not config_path.exists():
			continue
		try:
			with config_path.open("r", encoding="utf-8") as fh:
				config_data = json.load(fh)
		except Exception as exc:
			print(f"Skipping {exp_dir.name}: unable to read config.json ({exc})")
			continue

		if config_data.get("scenario") != scenario:
			continue

		configured_model = config_data.get("llm_model") or cfg.OLLAMA_MODEL
		if configured_model != llm_model:
			continue

		exists, _, output_dir, existing_run_ids = check_existing_experiment(exp_dir.name)
		if not exists:
			continue

		configured_runs = config_data.get("n_runs")
		planned_runs: Optional[int]
		if isinstance(configured_runs, int) and configured_runs > 0:
			planned_runs = configured_runs
		else:
			planned_runs = None

		requested_runs = target_runs if isinstance(target_runs, int) and target_runs > 0 else None
		if planned_runs is None and requested_runs is None:
			planned_runs = len(existing_run_ids) or None
		elif requested_runs is not None:
			planned_runs = max(planned_runs or 0, requested_runs)

		if planned_runs is None or planned_runs <= 0:
			planned_runs = len(existing_run_ids) or 0

		if planned_runs <= 0:
			return exp_dir.name, False

		raw_max_steps = config_data.get("max_steps", 1000)
		try:
			max_steps = int(raw_max_steps)
		except (TypeError, ValueError):
			max_steps = 1000

		if planned_runs < len(existing_run_ids):
			planned_runs = len(existing_run_ids)

		completed = 0
		has_incomplete = False
		for run_id in range(planned_runs):
			status = _analyze_run_status(output_dir, run_id, max_steps)
			state = status.get("status")
			if state in {"converged", "reached_max"}:
				completed += 1
			elif state in {"aborted", "missing"}:
				has_incomplete = True

		if has_incomplete or completed < planned_runs:
			return exp_dir.name, False

		return exp_dir.name, True

	return None, False


def _validate_scenarios(user_scenarios: Optional[List[str]]) -> List[str]:
	if not user_scenarios:
		return list(CONTEXT_SCENARIOS.keys())

	unknown = sorted(set(user_scenarios) - set(CONTEXT_SCENARIOS.keys()))
	if unknown:
		valid = ", ".join(sorted(CONTEXT_SCENARIOS.keys()))
		names = ", ".join(unknown)
		raise ValueError(f"Unknown scenario(s): {names}. Valid scenarios: {valid}")
	return user_scenarios


def _format_summary_row(scenario: str, output_dir: Optional[str], run_count: int) -> str:
	target_dir = output_dir or "<failed>"
	return f"{scenario:<30} -> {target_dir} ({run_count} runs)"


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Run or resume LLM experiments across all context scenarios",
	)
	parser.add_argument("--runs", type=int, default=None, help="Target number of runs per scenario")
	parser.add_argument("--max-steps", type=int, default=None, help="Maximum steps per simulation")
	parser.add_argument("--llm-model", type=str, default=None, help="LLM model identifier")
	parser.add_argument("--llm-url", type=str, default=None, help="Override LLM endpoint URL")
	parser.add_argument("--llm-api-key", type=str, default=None, help="Override LLM API key")
	parser.add_argument("--processes", type=int, default=None, help="Parallel process count")
	parser.add_argument("--no-parallel", action="store_true", help="Force sequential execution")
	parser.add_argument(
		"--scenarios",
		nargs="*",
		help="Subset of scenarios to run; defaults to all",
	)

	args = parser.parse_args()

	try:
		scenarios = _validate_scenarios(args.scenarios)
	except ValueError as exc:
		parser.error(str(exc))

	llm_model = args.llm_model or cfg.OLLAMA_MODEL
	llm_url = args.llm_url or cfg.OLLAMA_URL
	llm_api_key = args.llm_api_key or cfg.OLLAMA_API_KEY

	parallel = not args.no_parallel
	if args.processes == 1:
		parallel = False

	summary: List[Tuple[str, Optional[str], int]] = []

	for scenario in scenarios:
		print("=" * 80)
		print(f"Scenario: {scenario}")

		resume_candidate, fully_completed = _find_resume_candidate(scenario, llm_model, args.runs)
		if resume_candidate:
			if fully_completed:
				print(f"Found completed experiment '{resume_candidate}' – reusing results")
			else:
				print(f"Found incomplete experiment '{resume_candidate}' – resuming")
		else:
			print("No matching experiment found – starting a new one")

		output_dir, results = run_llm_experiment(
			scenario=scenario,
			n_runs=args.runs,
			max_steps=args.max_steps,
			llm_model=llm_model,
			llm_url=llm_url,
			llm_api_key=llm_api_key,
			parallel=parallel,
			n_processes=args.processes,
			resume_experiment=resume_candidate,
		)

		run_count = len(results) if results is not None else 0
		summary.append((scenario, output_dir, run_count))

	print("=" * 80)
	print("Completed workflow summary:")
	for scenario, out_dir, count in summary:
		print(_format_summary_row(scenario, out_dir, count))


if __name__ == "__main__":
	main()

