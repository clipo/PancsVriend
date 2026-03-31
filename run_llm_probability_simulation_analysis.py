#!/usr/bin/env python3
"""Guide: llm_probability_simulation_analysis pipeline.

Run token probabilities -> simulations -> scenario analysis in one CLI.

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

Root-path guide:
- Token stage root can be set via `token_args.token_output_root` in YAML.
- If `token_args.token_output_root` is missing, fallback is top-level `token_log_probs_root`.
- Context simulation uses `contexts_args.log_probs_root` when provided.
- If `contexts_args.log_probs_root` is missing, fallback is the effective token root.


"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parent
TOKEN_SCRIPT = REPO_ROOT / "llm_utility_approximation" / "llm_token_probabilities.py"
CONTEXTS_SCRIPT = REPO_ROOT / "run_all_contexts.py"
ANALYSIS_SCRIPT = REPO_ROOT / "analysis_tools" / "run_all_scenario_analysis.py"


def _default_run_id(model_slug: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}_{model_slug}"


def _resolve_run_layout(run_root: str, run_id: str, llm_model: str) -> dict[str, str]:
    model_slug = _sanitize_model_for_path_component(llm_model)
    root = Path(run_root)
    run_dir = root / run_id
    manifest_dir = run_dir / "manifest"
    analysis_dir = run_dir / "analysis"
    plots_dir = run_dir / "plots"
    experiments_dir = run_dir / "experiments"

    return {
        "run_root": str(root),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest_dir": str(manifest_dir),
        "analysis_dir": str(analysis_dir),
        "plots_dir": str(plots_dir),
        "experiments_dir": str(experiments_dir),
        "token_output_root": str(run_dir / "log_probs"),
        "default_manifest_file": str(manifest_dir / f"{run_id}_run_manifest.json"),
        "model_slug": model_slug,
    }


PLOT_FILE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
    ".eps",
    ".webp",
}


def _move_plots_to_dedicated_folder(analysis_dir: str, plots_dir: str) -> int:
    source_root = Path(analysis_dir)
    if not source_root.exists():
        return 0

    target_root = Path(plots_dir)
    target_root.mkdir(parents=True, exist_ok=True)
    moved_count = 0

    for root, _dirs, files in os.walk(source_root):
        root_path = Path(root)
        for file_name in files:
            src = root_path / file_name
            if src.suffix.lower() not in PLOT_FILE_EXTENSIONS:
                continue
            relative = src.relative_to(source_root)
            dst = target_root / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dst)
            moved_count += 1

    return moved_count


def _resolve_cli_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


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


def _contains_cli_flag(argv: list[str], flag: str) -> bool:
    for token in argv:
        if token == flag:
            return True
        if token.startswith(f"{flag}="):
            return True
    return False


def _passthrough_map_to_cli_args(arg_map: dict[str, Any]) -> list[str]:
    cli_args: list[str] = []
    for key, value in arg_map.items():
        flag = f"--{str(key).replace('_', '-')}"
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cli_args.append(flag)
            continue
        if isinstance(value, list):
            if len(value) == 0:
                continue
            cli_args.append(flag)
            cli_args.extend(str(item) for item in value)
            continue
        cli_args.extend([flag, str(value)])
    return cli_args


def _load_yaml_config(config_yaml_path: str) -> dict[str, Any]:
    resolved_path = _resolve_cli_path(config_yaml_path)
    config_path = Path(resolved_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config YAML root must be a mapping/object: {config_path}")
    return payload


def _merge_profile_config(config_payload: dict[str, Any], profile_name: str | None) -> dict[str, Any]:
    merged: dict[str, Any] = {k: v for k, v in config_payload.items() if k != "profiles"}
    if profile_name is None:
        return merged

    profiles_obj = config_payload.get("profiles")
    if not isinstance(profiles_obj, dict):
        raise ValueError("Config YAML does not contain a valid 'profiles' mapping")
    selected = profiles_obj.get(profile_name)
    if not isinstance(selected, dict):
        available = ", ".join(sorted(str(k) for k in profiles_obj.keys()))
        raise ValueError(f"Unknown config profile '{profile_name}'. Available: {available}")

    merged.update(selected)
    return merged


def _apply_config_to_args(
    parsed_args: argparse.Namespace,
    config_values: dict[str, Any],
    cli_argv: list[str],
) -> argparse.Namespace:
    top_level_fields = [
        "run_root",
        "run_id",
        "llm_model",
        "manifest_file",
        "token_log_probs_root",
        "token_output_root",
        "skip_token_probs",
        "skip_contexts",
        "skip_analysis",
        "continue_on_error",
        "dry_run",
    ]

    for field_name in top_level_fields:
        flag = f"--{field_name.replace('_', '-')}"
        if _contains_cli_flag(cli_argv, flag):
            continue
        if field_name in config_values:
            setattr(parsed_args, field_name, config_values.get(field_name))

    passthrough_fields = ["token_args", "contexts_args", "analysis_args"]
    for field_name in passthrough_fields:
        flag = f"--{field_name.replace('_', '-')}"
        if _contains_cli_flag(cli_argv, flag):
            continue
        if field_name not in config_values:
            continue

        raw_value = config_values.get(field_name)
        if isinstance(raw_value, dict):
            arg_map = dict(raw_value)
            if field_name == "token_args":
                token_root_from_token_args = arg_map.pop("token_output_root", None)
                if token_root_from_token_args is None:
                    token_root_from_token_args = arg_map.pop("token_log_probs_root", None)
                if token_root_from_token_args is not None:
                    setattr(parsed_args, "token_args_token_output_root", token_root_from_token_args)
            cli_list = _passthrough_map_to_cli_args(arg_map)
            setattr(parsed_args, field_name, shlex.join(cli_list))
        elif isinstance(raw_value, str):
            setattr(parsed_args, field_name, raw_value)
        elif raw_value is None:
            setattr(parsed_args, field_name, "")
        else:
            raise ValueError(f"{field_name} in config must be a string, map, or null")

    return parsed_args


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


def _resolve_token_output_paths(
    model_slug: str,
    explicit_output_dir: str | None,
    token_root_fallback: str,
) -> tuple[str, str]:
    """Return (token_output_root, token_output_dir_for_token_script).

    `explicit_output_dir` follows llm_token_probabilities semantics where the
    value can be either <root> or <root>/<model_slug>.
    """
    if explicit_output_dir:
        resolved_output_dir = _resolve_cli_path(explicit_output_dir)
        output_dir_path = Path(resolved_output_dir)
        if output_dir_path.name == model_slug:
            return str(output_dir_path.parent), str(output_dir_path)
        return str(output_dir_path), str(output_dir_path / model_slug)

    resolved_root = _resolve_cli_path(token_root_fallback)
    return resolved_root, str(Path(resolved_root) / model_slug)


def _run_command(command: list[str], dry_run: bool, cwd: str | None = None) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    if cwd:
        print(f"\n[run][cwd={cwd}] {printable}")
    else:
        print(f"\n[run] {printable}")
    if dry_run:
        return
    command_cwd = cwd if cwd is not None else str(REPO_ROOT)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = str(REPO_ROOT)
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{repo_pythonpath}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = repo_pythonpath
    subprocess.run(command, cwd=command_cwd, check=True, env=env)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run token probabilities, run_all_contexts, and scenario analysis in one command.",
    )
    parser.add_argument(
        "--config-yaml",
        type=str,
        default=None,
        help="Optional YAML config file to populate pipeline arguments",
    )
    parser.add_argument(
        "--config-profile",
        type=str,
        default=None,
        help="Optional profile name under config YAML 'profiles' to apply",
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default="experiments_with_log_probs",
        help="Root folder for isolated log-prob experiment runs",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run folder name under --run-root (default: auto timestamp + model)",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
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
        "--token-log-probs-root",
        type=str,
        default=None,
        help="Root directory for token-probability outputs when token args do not set --output-dir (preferred alias; same purpose as --token-output-root)",
    )
    parser.add_argument(
        "--token-output-root",
        type=str,
        default=None,
        help="Deprecated alias for --token-log-probs-root (default: <run>/log_probs)",
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
    cli_argv = sys.argv[1:]
    args = parser.parse_args()

    if args.config_yaml:
        try:
            config_payload = _load_yaml_config(args.config_yaml)
            merged_config = _merge_profile_config(config_payload, args.config_profile)
            args = _apply_config_to_args(args, merged_config, cli_argv)
        except (OSError, ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))

    if not args.llm_model:
        parser.error("--llm-model is required (or set 'llm_model' in --config-yaml).")

    token_base_args = _parse_passthrough_args(args.token_args)
    contexts_base_args = _parse_passthrough_args(args.contexts_args)
    analysis_base_args = _parse_passthrough_args(args.analysis_args)

    model = args.llm_model
    model_slug = _sanitize_model_for_path_component(model)
    run_id = args.run_id or _default_run_id(model_slug)
    run_layout = _resolve_run_layout(args.run_root, run_id, model)

    token_root_from_token_args = getattr(args, "token_args_token_output_root", None)
    token_log_probs_root_arg = token_root_from_token_args or args.token_log_probs_root or args.token_output_root
    if token_log_probs_root_arg is None:
        token_log_probs_root_arg = run_layout["token_output_root"]

    explicit_token_output_dir = _get_flag_value(token_base_args, "--output-dir")
    default_token_output_root, default_token_output_dir = _resolve_token_output_paths(
        model_slug=model_slug,
        explicit_output_dir=explicit_token_output_dir,
        token_root_fallback=token_log_probs_root_arg,
    )

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
        manifest_path = run_layout["default_manifest_file"]

    if manifest_path:
        manifest_path = _resolve_cli_path(manifest_path)

    if not args.dry_run:
        Path(run_layout["manifest_dir"]).mkdir(parents=True, exist_ok=True)
        Path(run_layout["experiments_dir"]).mkdir(parents=True, exist_ok=True)
        Path(run_layout["analysis_dir"]).mkdir(parents=True, exist_ok=True)
        Path(run_layout["plots_dir"]).mkdir(parents=True, exist_ok=True)

        source_config_path: str | None = _resolve_cli_path(args.config_yaml) if args.config_yaml else None
        if source_config_path:
            copied_source_config = Path(run_layout["run_dir"]) / "run_config_source.yaml"
            shutil.copy2(source_config_path, copied_source_config)

        effective_config_payload = {
            "config_yaml": source_config_path,
            "config_profile": args.config_profile,
            "run_root": args.run_root,
            "run_id": args.run_id,
            "llm_model": args.llm_model,
            "token_args": args.token_args,
            "contexts_args": args.contexts_args,
            "analysis_args": args.analysis_args,
            "manifest_file": args.manifest_file,
            "token_log_probs_root": args.token_log_probs_root,
            "token_output_root": args.token_output_root,
            "token_args_token_output_root": token_root_from_token_args,
            "skip_token_probs": args.skip_token_probs,
            "skip_contexts": args.skip_contexts,
            "skip_analysis": args.skip_analysis,
            "continue_on_error": args.continue_on_error,
            "dry_run": args.dry_run,
            "resolved": {
                "manifest_path": manifest_path,
                "default_token_output_root": default_token_output_root,
                "default_token_output_dir": default_token_output_dir,
                "token_base_args_list": token_base_args,
                "contexts_base_args_list": contexts_base_args,
                "analysis_base_args_list": analysis_base_args,
            },
        }
        effective_config_path = Path(run_layout["run_dir"]) / "run_config_effective.yaml"
        effective_config_path.write_text(
            yaml.safe_dump(effective_config_payload, sort_keys=False),
            encoding="utf-8",
        )

        metadata_path = Path(run_layout["run_dir"]) / "run_layout_manifest.json"
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "llm_model": model,
            "config_yaml": source_config_path,
            "config_profile": args.config_profile,
            "run_layout": run_layout,
            "stages": {
                "skip_token_probs": args.skip_token_probs,
                "skip_contexts": args.skip_contexts,
                "skip_analysis": args.skip_analysis,
            },
            "passthrough": {
                "token_args": args.token_args,
                "contexts_args": args.contexts_args,
                "analysis_args": args.analysis_args,
            },
            "resolved_paths": {
                "token_output_root": default_token_output_root,
                "token_output_dir": default_token_output_dir,
                "manifest_path": manifest_path,
                "analysis_output_dir": run_layout["analysis_dir"],
                "plots_output_dir": run_layout["plots_dir"],
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print(f"Model: {model}")
    print(f"Run root: {run_layout['run_root']}")
    print(f"Run id: {run_layout['run_id']}")
    print(f"Manifest dir: {run_layout['manifest_dir']}")
    print(f"Experiments dir: {run_layout['experiments_dir']}")
    print(f"Analysis dir: {run_layout['analysis_dir']}")
    print(f"Plots dir: {run_layout['plots_dir']}")

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
        if _contains_flag(contexts_base_args, "--use-log-probs") and not _contains_flag(contexts_base_args, "--log-probs-root"):
            contexts_cmd.extend(["--log-probs-root", default_token_output_root])
        if manifest_path and not _contains_flag(contexts_base_args, "--manifest-file"):
            contexts_cmd.extend(["--manifest-file", manifest_path])
        try:
            _run_command(contexts_cmd, dry_run=args.dry_run, cwd=run_layout["run_dir"])
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
        analysis_output_folder = _get_flag_value(analysis_base_args, "--output-folder")
        if analysis_output_folder is None:
            analysis_output_folder = run_layout["analysis_dir"]
        resolved_analysis_output_folder = _resolve_cli_path(analysis_output_folder)
        if not _contains_flag(analysis_base_args, "--output-folder"):
            analysis_cmd.extend(["--output-folder", resolved_analysis_output_folder])
        try:
            _run_command(analysis_cmd, dry_run=args.dry_run, cwd=run_layout["run_dir"])
            if not args.dry_run:
                moved_plot_files = _move_plots_to_dedicated_folder(
                    analysis_dir=resolved_analysis_output_folder,
                    plots_dir=run_layout["plots_dir"],
                )
                print(f"[post] Moved {moved_plot_files} plot file(s) into dedicated plots folder.")
        except subprocess.CalledProcessError as exc:
            print(f"[error] Scenario analysis stage failed with exit code {exc.returncode}.")
            if not args.continue_on_error:
                raise

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
