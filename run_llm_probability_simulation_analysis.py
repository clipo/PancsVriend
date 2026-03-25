#!/usr/bin/env python3
"""Run token probabilities -> simulations -> scenario analysis in one CLI.

Example:
    python run_llm_probability_simulation_analysis.py \
        --llm-model phi4:latest \
        --token-args "--scenario all --processes 10 --temperature 0.3" \
        --contexts-args "--runs 100 --processes 10 --use-log-probs --save-every-steps 10" \
        --analysis-args "--include-movement"

Each stage also allows pass-through arguments:
- --token-args    -> llm_utility_approximation/llm_token_probabilities.py
- --contexts-args -> run_all_contexts.py
- --analysis-args -> analysis_tools/run_all_scenario_analysis.py


"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent
TOKEN_SCRIPT = REPO_ROOT / "llm_utility_approximation" / "llm_token_probabilities.py"
CONTEXTS_SCRIPT = REPO_ROOT / "run_all_contexts.py"
ANALYSIS_SCRIPT = REPO_ROOT / "analysis_tools" / "run_all_scenario_analysis.py"


def _sanitize_model_for_path_component(name: str) -> str:
    sanitized = "".join("-" if c in '<>:"/\\|?*' or ord(c) < 32 else c for c in name.strip())
    sanitized = sanitized.rstrip(" .")
    while "--" in sanitized:
        sanitized = sanitized.replace("--", "-")
    return sanitized or "unknown-model"


def _parse_passthrough_args(raw: str | None) -> list[str]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    return shlex.split(raw)


def _contains_flag(args: Iterable[str], flag: str) -> bool:
    args_list = list(args)
    for idx, token in enumerate(args_list):
        if token == flag:
            return True
        if token.startswith(f"{flag}="):
            return True
        if idx > 0 and args_list[idx - 1] == flag:
            return True
    return False


def _get_flag_value(args: Iterable[str], flag: str) -> str | None:
    args_list = list(args)
    for idx, token in enumerate(args_list):
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
        if token == flag and idx + 1 < len(args_list):
            return args_list[idx + 1]
    return None


def _default_manifest_path() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path("experiments") / "manifests" / f"{timestamp}_run_manifest.json")


def _run_command(command: list[str], dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"\n[run] {printable}")
    if dry_run:
        return
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run token probabilities, run_all_contexts, and scenario analysis in one command.",
    )
    parser.add_argument(
        "--llm-model",
        required=True,
        help="LLM model name used across token-probabilities, contexts, and analysis steps",
    )
    parser.add_argument(
        "--token-args",
        type=str,
        default="",
        help="Quoted passthrough args for llm_token_probabilities.py",
    )
    parser.add_argument(
        "--contexts-args",
        type=str,
        default="",
        help="Quoted passthrough args for run_all_contexts.py",
    )
    parser.add_argument(
        "--analysis-args",
        type=str,
        default="",
        help="Quoted passthrough args for run_all_scenario_analysis.py",
    )
    parser.add_argument(
        "--manifest-file",
        type=str,
        default=None,
        help="Explicit simulation manifest JSON path used by analysis (required with --skip-contexts).",
    )
    parser.add_argument(
        "--token-output-root",
        type=str,
        default="llm_log_probs",
        help="Root directory for per-model probability outputs when token args do not set --output-dir",
    )
    parser.add_argument(
        "--skip-token-probs",
        action="store_true",
        help="Skip llm_token_probabilities.py stage",
    )
    parser.add_argument(
        "--skip-contexts",
        action="store_true",
        help="Skip run_all_contexts.py stage",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip run_all_scenario_analysis.py stage",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to next stage if a command fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    token_base_args = _parse_passthrough_args(args.token_args)
    contexts_base_args = _parse_passthrough_args(args.contexts_args)
    analysis_base_args = _parse_passthrough_args(args.analysis_args)

    model = args.llm_model
    model_slug = _sanitize_model_for_path_component(model)
    default_token_output_dir = str(Path(args.token_output_root) / model_slug)

    contexts_manifest_arg = _get_flag_value(contexts_base_args, "--manifest-file")
    analysis_manifest_arg = _get_flag_value(analysis_base_args, "--manifest-file")

    if args.skip_contexts:
        manifest_path = args.manifest_file or analysis_manifest_arg
        if not manifest_path:
            parser.error("--skip-contexts requires a specified manifest via --manifest-file (or in --analysis-args).")
    else:
        manifest_path = (
            args.manifest_file
            or analysis_manifest_arg
            or contexts_manifest_arg
        )

    if not args.skip_contexts and not manifest_path:
        manifest_path = _default_manifest_path()

    print("\n" + "=" * 90)
    print(f"Model: {model}")

    if not args.skip_token_probs:
        token_cmd = [sys.executable, str(TOKEN_SCRIPT), "--llm-model", model]
        token_cmd.extend(token_base_args)
        if not _contains_flag(token_base_args, "--output-dir"):
            token_cmd.extend(["--output-dir", default_token_output_dir])
        try:
            _run_command(token_cmd, dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            print(f"[error] Token probability stage failed with exit code {exc.returncode}.")
            if not args.continue_on_error:
                raise

    if not args.skip_contexts:
        contexts_cmd = [sys.executable, str(CONTEXTS_SCRIPT)]
        contexts_cmd.extend(contexts_base_args)
        if not _contains_flag(contexts_base_args, "--llm-model"):
            contexts_cmd.extend(["--llm-model", model])
        if manifest_path and not _contains_flag(contexts_base_args, "--manifest-file"):
            contexts_cmd.extend(["--manifest-file", manifest_path])
        try:
            _run_command(contexts_cmd, dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            print(f"[error] Context simulation stage failed with exit code {exc.returncode}.")
            if not args.continue_on_error:
                raise

    if not args.skip_analysis:
        analysis_cmd = [sys.executable, str(ANALYSIS_SCRIPT)]
        analysis_cmd.extend(analysis_base_args)
        if not _contains_flag(analysis_base_args, "--llm-model"):
            analysis_cmd.extend(["--llm-model", model])
        if manifest_path and not _contains_flag(analysis_base_args, "--manifest-file"):
            analysis_cmd.extend(["--manifest-file", manifest_path])
        if not _contains_flag(analysis_base_args, "--output-folder"):
            analysis_cmd.extend(["--output-folder", f"reports_{model_slug}"])
        try:
            _run_command(analysis_cmd, dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            print(f"[error] Scenario analysis stage failed with exit code {exc.returncode}.")
            if not args.continue_on_error:
                raise

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
