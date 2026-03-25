"""Unified multi-scenario runner for LLM experiments.

Usage Guide
-----------
Run from repo root after activating the project venv:

	python run_all_contexts.py --runs 10 --processes 5 --llm-model phi4:latest

Key options:
* --scenarios <names...>  Limit execution to specific scenarios (defaults to all).
* --runs N                Target number of runs per scenario (falls back to config when resuming).
* --processes P           Parallel worker count (omit for auto min(cpu_count, runs)).
* --save-every-steps N    Persist states/move logs every N steps (default 1, all detail retained).
* --no-parallel           Force sequential execution.
* --new                   Always start new experiments (skip resume detection).
* --use-log-probs         Use precomputed MOVE/STAY shares from summary CSVs.
* --log-probs-root PATH   Optional root/model directory for log-prob summary files.
* --llm-model/url/api-key Override the LLM endpoint configuration.

The script checks for existing experiments whose config matches the requested
scenario and model. Incomplete experiments are resumed using their stored
parameters and resume seeds; otherwise a fresh experiment directory is created.
Results for each scenario are summarized at the end of the run.
"""

import argparse
import json
from datetime import datetime
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


def _default_manifest_path() -> Path:
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	return Path("experiments") / "manifests" / f"{timestamp}_run_manifest.json"


def _build_manifest_records(summary: List[Tuple[str, Optional[str], int]]) -> List[dict]:
	records: List[dict] = []
	for scenario, output_dir, run_count in summary:
		status = "success" if output_dir else "failed"
		experiment_name = Path(output_dir).name if output_dir else None
		effective_config = _load_effective_experiment_config(output_dir)
		records.append({
			"scenario": scenario,
			"status": status,
			"output_dir": output_dir,
			"experiment_name": experiment_name,
			"run_count": run_count,
			"effective_config": effective_config,
		})
	return records


def _load_effective_experiment_config(output_dir: Optional[str]) -> Optional[dict]:
	if not output_dir:
		return None
	config_path = Path(output_dir) / "config.json"
	if not config_path.exists():
		return None
	try:
		return json.loads(config_path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		return None


def _parser_defaults(parser: argparse.ArgumentParser) -> dict:
	defaults = {}
	for action in parser._actions:
		if not action.dest or action.dest == "help":
			continue
		defaults[action.dest] = action.default
	return defaults


def _build_effective_args(
	parser: argparse.ArgumentParser,
	args: argparse.Namespace,
	scenarios: List[str],
	llm_model: str,
	llm_url: str,
	llm_api_key: Optional[str],
	parallel: bool,
) -> dict:
	defaults = _parser_defaults(parser)
	effective = dict(defaults)
	effective.update(vars(args))
	effective.update({
		"scenarios": scenarios,
		"llm_model": llm_model,
		"llm_url": llm_url,
		"llm_api_key_last4": llm_api_key[-4:] if llm_api_key else None,
		"parallel": parallel,
	})
	return effective


def _selected_experiments(records: List[dict]) -> dict:
	selected = {}
	for record in records:
		if record.get("status") != "success":
			continue
		scenario = record.get("scenario")
		experiment_name = record.get("experiment_name")
		if scenario and experiment_name:
			selected[str(scenario)] = str(experiment_name)
	return selected


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
	parser.add_argument("--save-every-steps", type=int, default=None, help="Persist states/move logs every N steps (default: 1)")
	parser.add_argument("--no-parallel", action="store_true", help="Force sequential execution")
	parser.add_argument(
		"--use-log-probs",
		action="store_true",
		help="Use precomputed scenario log-probability summary CSVs instead of live LLM API calls",
	)
	parser.add_argument(
		"--log-probs-root",
		type=str,
		default=None,
		help="Optional root/model directory containing log-prob summary CSV files",
	)
	parser.add_argument(
		"--new",
		action="store_true",
		help="Start new experiments only; skip checking for resumable/completed experiments",
	)
	parser.add_argument(
		"--scenarios",
		nargs="*",
		help="Subset of scenarios to run; defaults to all",
	)
	parser.add_argument(
		"--manifest-file",
		type=str,
		default=None,
		help="Optional output path for run manifest JSON (default: experiments/manifests/<timestamp>_run_manifest.json)",
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

		if args.new:
			resume_candidate = None
			print("--new set: skipping resume detection and starting a new experiment")
		else:
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
			use_log_probs=args.use_log_probs,
			log_probs_root=args.log_probs_root,
			save_every_steps=args.save_every_steps,
		)

		run_count = len(results) if results is not None else 0
		summary.append((scenario, output_dir, run_count))

	print("=" * 80)
	print("Completed workflow summary:")
	for scenario, out_dir, count in summary:
		print(_format_summary_row(scenario, out_dir, count))

	manifest_records = _build_manifest_records(summary)
	manifest_path = Path(args.manifest_file) if args.manifest_file else _default_manifest_path()
	manifest_path.parent.mkdir(parents=True, exist_ok=True)

	manifest_payload = {
		"manifest_version": 1,
		"created_at": datetime.now().isoformat(timespec="seconds"),
		"llm_model": llm_model,
		"scenarios_requested": scenarios,
		"effective_args": _build_effective_args(
			parser=parser,
			args=args,
			scenarios=scenarios,
			llm_model=llm_model,
			llm_url=llm_url,
			llm_api_key=llm_api_key,
			parallel=parallel,
		),
		"experiments": manifest_records,
		"selected_experiments": _selected_experiments(manifest_records),
	}

	manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
	print(f"Run manifest written: {manifest_path}")


if __name__ == "__main__":
	main()

