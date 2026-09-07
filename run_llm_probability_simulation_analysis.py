#!/usr/bin/env python3
"""Guide: llm_probability_simulation_analysis pipeline.

Run value-function build -> simulations -> scenario analysis in one CLI.

Example:
    python run_llm_probability_simulation_analysis.py \
        --config-yaml configs/vf_run_gemma_lp.yaml --config-profile production

Each stage allows pass-through arguments:
- --vf-build-args -> value_functions/sampling/build_value_function.py
- --contexts-args -> run_all_contexts.py
- --analysis-args -> analysis_tools/run_all_scenario_analysis.py
- --rank-stability-args -> value_functions/comparison/vf_rank_stability.py
- --cross-model-args -> value_functions/comparison/cross_model_vf_comparison.py

Cross-model stage (added 2026-09-06): the last stage regenerates the
cross-model figures and tables (value_functions/comparison/cross_model_vf_comparison.py
--run-root <run_root>: cross_model_bump_*.png, cross_model_level_*.png,
cross_model_{chance_tests,pairwise_tests,rankings}.csv under
<run_root>/cross_model/, i.e. experiments_with_llama_cpp/cross_model/ for the
production root) from every model's newest full run OF THE
SAME TABLE FAMILY under this run's --run-root, so the comparison is
refreshed whenever one of its models' pipelines completes and never has to
be run by hand. Only the exact logprob tables (-vf-lp) are a result: the
sampled -vf-r3 tables carry the llama-server batch-numerics artifact, so a
-vf-r3 run records the stage as skipped (cross_model_args --family r3 forces
the superseded cross_model_sampled_* set). Exact tables carry no sampling
error, so the multi-split sufficiency ruler does not apply to them: the
rank-stability stage runs --exact (s = 0) and no multisplit_* directory is
consulted. Models finishing
concurrently (run_vf_prod10k_queue.sh waves) serialise on
<run_root>/.cross_model_figures.lock. `cross_model_args` in the yaml passes
extra flags (--out-dir, --dpi); `skip_cross_model: true` disables the stage.
A `cross_model` block in run_layout_manifest.json records the command.

Rank-stability stage (added 2026-09-04): after the scenario analysis, the DI
ordering of this run's scenarios is classified against the value function's
multi-split RULER (value_functions/comparison/vf_rank_stability.py). The stage derives
everything from the run itself — label from the value-function template
(vf_<label>__{scenario}__<style>.json), the production board from
contexts_args, the ruler as the newest multisplit_<label>*/ directory in
value_functions/results/sampled/multisplit/ whose multisplit_status.json
matches that board and max_steps (rng_scheme 'keyed' preferred over
'shared'). `rank_stability_args` in the yaml overrides any of --label,
--multisplit, --tie-mult, ...; `skip_rank_stability: true` disables the
stage. Output: <run_dir>/analysis/rank_stability/ and a `rank_stability`
block in run_layout_manifest.json. The verdict (SETTLED / FIXABLE /
UNMEASURED) is a result, not a failure; only a missing ruler or a crash
fails the stage.

Value-function guide:
- The vf-build stage samples P(MOVE | context) from a live llama.cpp server
  (build_value_function.py --config <sampling yaml>); it only runs when
  `vf_build_args` provides a `config` and `skip_vf_build` is false. Skip it
  when the artifacts already exist under the canonical store
  (value_functions/results/sampled/).
- When `contexts_args.value_function` is set, the resolved artifacts (JSON +
  figures) are FROZEN into <run_dir>/value_functions/ together with a
  composition-heatmap grid rendered from the frozen copies, and the contexts
  stage reads the frozen copies — each run folder carries the exact decision
  tables it simulated from, for side-by-side comparison with the results.
- The token-probability stage was removed 2026-08-25 (superseded by value
  functions). Old `token_args`/`skip_token_probs`/`token_*_root` config keys
  are ignored with a warning.

Analysis-stage extras:
- Scenario hierarchy/significance tables are produced by
    analysis_tools/run_all_scenario_analysis.py and written to analysis output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass  # Older Pythons or non-tty streams

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parent
VF_BUILD_SCRIPT = REPO_ROOT / "value_functions" / "sampling" / "build_value_function.py"
CONTEXTS_SCRIPT = REPO_ROOT / "run_all_contexts.py"
ANALYSIS_SCRIPT = REPO_ROOT / "analysis_tools" / "run_all_scenario_analysis.py"
RANK_STABILITY_SCRIPT = REPO_ROOT / "value_functions" / "comparison" / "vf_rank_stability.py"
CROSS_MODEL_SCRIPT = REPO_ROOT / "value_functions" / "comparison" / "cross_model_vf_comparison.py"
# Canonical store of value-function artifacts AND their multi-split rulers
# (multisplit_<label>[_suffix]/). A ruler is a property of the table and the
# board it was measured on, so it lives with the table, never inside a
# production run folder.
VF_STORE = REPO_ROOT / "value_functions" / "results" / "sampled"
# Rulers moved into a subfolder on 2026-09-07 so the store root holds only
# vf_*.json tables; keep both names so the error text still points at the store.
MULTISPLIT_STORE = VF_STORE / "multisplit"
BOARD_KEYS = ("grid_size", "num_type_a", "num_type_b", "max_steps")

DEPRECATED_TOKEN_KEYS = (
    "token_args",
    "skip_token_probs",
    "token_log_probs_root",
    "token_output_root",
)


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
        "value_functions_dir": str(run_dir / "value_functions"),
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


def _extract_flag_value(argv: list[str], flag: str) -> str | None:
    """Return the value passed for `flag` in a shlex-split arg list.

    Supports both `--flag value` and `--flag=value`. Returns None if absent.
    """
    prefix = f"{flag}="
    for i, token in enumerate(argv):
        if token == flag:
            return argv[i + 1] if i + 1 < len(argv) else None
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def _freeze_scenario_file(contexts_base_args: list[str], run_dir: str | Path) -> Path | None:
    """Copy the run's --scenario-file into run_dir for self-contained provenance.

    The repo copy of scenarios_*.py is mutable and (currently) untracked, so a later
    edit would silently change the prompts a resumed run sends vs. the runs already on
    disk. A frozen copy makes that class of drift impossible and also matches the
    scenario_file path config.json records at the run-dir level. Returns the frozen
    path, or None if no --scenario-file was passed. Prints a warning (and returns
    None) if the argument resolves to a missing file.
    """
    scenario_file_arg = _extract_flag_value(contexts_base_args, "--scenario-file")
    if not scenario_file_arg:
        return None
    resolved = _resolve_cli_path(scenario_file_arg)
    if not os.path.exists(resolved):
        print(f"WARNING: --scenario-file '{scenario_file_arg}' resolved to "
              f"'{resolved}' which does not exist; not frozen.")
        return None
    frozen = Path(run_dir) / Path(resolved).name
    shutil.copy2(resolved, frozen)
    return frozen


def _replace_flag_value(argv: list[str], flag: str, new_value: str) -> list[str]:
    """Return argv with the value of `flag` replaced (space and equals forms)."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == flag and i + 1 < len(argv):
            out.extend([flag, new_value])
            i += 2
            continue
        if token.startswith(f"{flag}="):
            out.append(f"{flag}={new_value}")
            i += 1
            continue
        out.append(token)
        i += 1
    return out


def _render_vf_heatmaps(frozen_jsons: list[Path], out_path: Path) -> None:
    """Composition-surface heatmap grid — the exact lookup table the simulation
    uses — rendered FROM THE FROZEN artifacts. plot_value_functions.fig_heatmaps
    is reused so this rendering cannot drift from the sweep figures."""
    for extra in (REPO_ROOT / "value_functions" / "sampling", REPO_ROOT / "prompt_refinement"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    try:
        from sampling_common import load_value_function
        from plot_value_functions import SCENARIO_ORDER, fig_heatmaps
    except ImportError as exc:
        print(f"WARNING: composition-heatmap rendering unavailable ({exc}); "
              f"frozen JSONs written without the grid figure.")
        return
    found = {}
    for path in frozen_jsons:
        vf = load_value_function(str(path))
        found[vf["meta"].get("scenario", path.stem)] = vf
    order = [s for s in SCENARIO_ORDER if s in found]
    order += sorted(k for k in found if k not in order)
    fig_heatmaps(found, order, str(out_path))
    print(f"[freeze] composition heatmaps -> {out_path}")


def _write_table_hashes(vf_dir: Path) -> None:
    """value_functions/TABLE_HASHES.json: one-glance table identity per run.

    Two runs used the same decision table iff their hashes match — no diffing
    of frozen artifacts needed. The hash covers ONLY `compositions` (the
    numbers the simulation actually reads), so cosmetic meta edits do not
    change identity, while any top-up does (counts merge into the cells).
    Also surfaces n_samples and meta.calibration (requested vs achieved
    precision — absent on artifacts built before 2026-09-02), the other two
    at-a-glance version markers. A separate file, not run_layout_manifest:
    the frozen dir is static after launch, so this can never race with a
    manifest rewrite.
    """
    import hashlib
    entries = {}
    for p in sorted(vf_dir.glob("vf_*.json")):
        vf = json.loads(p.read_text())
        comp = vf.get("compositions", {})
        entries[p.name] = {
            "table_sha256": hashlib.sha256(
                json.dumps(comp, sort_keys=True).encode()).hexdigest(),
            "n_samples": sum(c.get("n_samples", 0)
                             for rows in comp.values() for c in rows),
            "calibration": vf.get("meta", {}).get("calibration"),
        }
    (vf_dir / "TABLE_HASHES.json").write_text(json.dumps(entries, indent=1))
    print(f"[freeze] table hashes -> {vf_dir / 'TABLE_HASHES.json'}")


def _freeze_value_functions(
    contexts_base_args: list[str],
    value_functions_dir: str,
    dry_run: bool,
) -> list[str]:
    """Freeze the run's value-function artifacts into <run_dir>/value_functions/.

    Same rationale as _freeze_scenario_file: the repo-level artifact store is
    mutable (top-ups, half-data rebuilds), so the run folder keeps byte-exact
    copies of the decision tables it simulated from, plus their existing
    figures and a composition-heatmap grid for side-by-side comparison with
    the results. Returns contexts args rewritten so the contexts stage reads
    the frozen copies (and config.json records the in-run-dir path).
    """
    vf_arg = _extract_flag_value(contexts_base_args, "--value-function")
    if not vf_arg:
        return contexts_base_args

    # Absolute: the contexts stage runs with cwd inside the run dir, so a
    # run_root-relative path would only resolve via llm_runner's repo-root
    # anchor fallback (and not at all for an absolute run_root elsewhere).
    vf_dir = Path(_resolve_cli_path(str(value_functions_dir)))
    resolved = Path(_resolve_cli_path(vf_arg))  # keeps '{scenario}' literal
    if "{scenario}" in vf_arg:
        sources = sorted(resolved.parent.glob(resolved.name.replace("{scenario}", "*")))
        frozen_target = str(vf_dir / resolved.name)
    elif resolved.is_dir():
        sources = sorted(resolved.glob("vf_*.json"))
        frozen_target = str(vf_dir)
    else:
        sources = [resolved] if resolved.exists() else []
        frozen_target = str(vf_dir / resolved.name)
    if not sources:
        raise FileNotFoundError(
            f"--value-function {vf_arg!r} matched no artifact (resolved: {resolved})")

    if not dry_run:
        vf_dir.mkdir(parents=True, exist_ok=True)
        for src in sources:
            shutil.copy2(src, vf_dir / src.name)
            for ext in (".png", ".pdf", ".svg"):
                figure = src.with_suffix(ext)
                if figure.exists():
                    shutil.copy2(figure, vf_dir / figure.name)
        _render_vf_heatmaps(sorted(vf_dir.glob("vf_*.json")),
                            vf_dir / "value_function_heatmaps.png")
        _write_table_hashes(vf_dir)

    print(f"[freeze] {len(sources)} value-function artifact(s) -> {vf_dir}")
    return _replace_flag_value(contexts_base_args, "--value-function", frozen_target)


def _vf_label_and_style(vf_arg: str | None) -> tuple[str | None, str | None]:
    """('qwen3.6-27b-chat-grammar', 'R3_dual_count') from a value-function
    template or artifact path 'vf_<label>__<scenario>__<style>.json'."""
    if not vf_arg:
        return None, None
    m = re.match(r"vf_(.+?)__(.+?)__(.+)\.json$", Path(vf_arg).name)
    return (m.group(1), m.group(3)) if m else (None, None)


def _production_board(contexts_args: list[str]) -> dict[str, str | None]:
    """The board the contexts stage simulates on, as recorded in its flags."""
    return {k: _get_flag_value(contexts_args, f"--{k.replace('_', '-')}")
            for k in BOARD_KEYS}


def _resolve_ruler(label: str, board: dict[str, str | None]) -> tuple[Path | None, list[dict]]:
    """Newest multi-split ruler for `label` whose recorded board matches.

    Scans MULTISPLIT_STORE/multisplit_<label>*/multisplit_status.json
    (value_functions/results/sampled/multisplit/ since 2026-09-07). A ruler is only
    comparable to a production batch measured on the same board and step cap
    (a 10x10 ruler runs ~2x optimistic against a 20x20 batch), and only on the
    current tau scale (tau_definition == 'se'). Among matches, a 'keyed'
    rng_scheme beats 'shared' (a status file without the key is 'shared'),
    then the newest wins. Returns (chosen dir or None, every candidate seen).
    """
    seen: list[dict] = []
    for st in sorted(MULTISPLIT_STORE.glob(f"multisplit_{label}*/multisplit_status.json")):
        try:
            d = json.loads(st.read_text())
        except (OSError, ValueError):
            continue
        d["_dir"] = st.parent
        d["_mtime"] = st.stat().st_mtime
        # Only a half-split ruler is a sufficiency ruler; other keep fractions
        # are the scaling diagnostic (status file: is_sufficiency_check False).
        d["_board_ok"] = (d.get("tau_definition") == "se"
                          and d.get("is_sufficiency_check", True)
                          and all(board.get(k) is None or str(d.get(k)) == str(board[k])
                                  for k in BOARD_KEYS))
        seen.append(d)
    ok = [d for d in seen if d["_board_ok"]]
    if not ok:
        return None, seen
    ok.sort(key=lambda d: (d.get("rng_scheme", "shared") == "keyed", d["_mtime"]))
    return ok[-1]["_dir"], seen


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
    deprecated_present = [k for k in DEPRECATED_TOKEN_KEYS if k in config_values]
    if deprecated_present:
        print(f"[deprecated] the token-probability stage was removed 2026-08-25 "
              f"(superseded by value functions); ignoring config key(s): "
              f"{', '.join(deprecated_present)}")

    top_level_fields = [
        "run_root",
        "run_id",
        "llm_model",
        "manifest_file",
        "skip_vf_build",
        "skip_contexts",
        "skip_analysis",
        "skip_rank_stability",
        "skip_cross_model",
        "continue_on_error",
        "dry_run",
    ]

    for field_name in top_level_fields:
        flag = f"--{field_name.replace('_', '-')}"
        if _contains_cli_flag(cli_argv, flag):
            continue
        if field_name in config_values:
            setattr(parsed_args, field_name, config_values.get(field_name))

    passthrough_fields = ["vf_build_args", "contexts_args", "analysis_args",
                          "rank_stability_args", "cross_model_args"]
    for field_name in passthrough_fields:
        flag = f"--{field_name.replace('_', '-')}"
        if _contains_cli_flag(cli_argv, flag):
            continue
        if field_name not in config_values:
            continue

        raw_value = config_values.get(field_name)
        if isinstance(raw_value, dict):
            cli_list = _passthrough_map_to_cli_args(dict(raw_value))
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
        description="Run value-function build, run_all_contexts, and scenario analysis in one command.",
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
        help="LLM model name/label used across the vf-build, contexts, and analysis steps",
    )
    parser.add_argument(
        "--vf-build-args",
        type=str,
        default="",
        help="Quoted passthrough args for value_functions/sampling/build_value_function.py "
             "(the stage only runs when these include --config)",
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
        "--rank-stability-args",
        type=str,
        default="",
        help="Quoted passthrough args for value_functions/comparison/vf_rank_stability.py; "
             "--label, --multisplit, --production-experiments and --out-dir are "
             "derived from the run when absent",
    )
    parser.add_argument(
        "--cross-model-args",
        type=str,
        default="",
        help="Quoted passthrough args for value_functions/comparison/cross_model_vf_comparison.py "
             "(--run-root is this run's --run-root unless given here)",
    )
    parser.add_argument(
        "--manifest-file",
        type=str,
        default=None,
        help="Explicit simulation manifest JSON path used by analysis (if omitted with --skip-contexts, defaults to manifest/<run_id>_run_manifest.json when --run-id is provided).",
    )
    parser.add_argument(
        "--skip-vf-build",
        action="store_true",
        help="Skip the build_value_function.py sampling stage (use pre-built artifacts)",
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
        "--skip-rank-stability",
        action="store_true",
        help="Skip the rank-stability classification stage",
    )
    parser.add_argument(
        "--skip-cross-model",
        action="store_true",
        help="Skip the final cross-model figure/table regeneration stage",
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

    vf_build_base_args = _parse_passthrough_args(args.vf_build_args)
    contexts_base_args = _parse_passthrough_args(args.contexts_args)
    analysis_base_args = _parse_passthrough_args(args.analysis_args)
    rank_stability_base_args = _parse_passthrough_args(args.rank_stability_args)
    cross_model_base_args = _parse_passthrough_args(args.cross_model_args)
    # Captured before _freeze_value_functions rewrites --value-function to the
    # frozen copy; the label is parsed from the template's file name either way.
    vf_template_arg = _get_flag_value(contexts_base_args, "--value-function")
    vf_label, vf_style = _vf_label_and_style(vf_template_arg)
    production_board = _production_board(contexts_base_args)

    model = args.llm_model
    explicit_run_id = args.run_id
    model_slug = _sanitize_model_for_path_component(model)
    run_id = args.run_id or _default_run_id(model_slug)
    run_layout = _resolve_run_layout(args.run_root, run_id, model)

    contexts_manifest_arg = _get_flag_value(contexts_base_args, "--manifest-file")
    analysis_manifest_arg = _get_flag_value(analysis_base_args, "--manifest-file")

    if args.skip_contexts:
        manifest_path = (
            args.manifest_file
            or analysis_manifest_arg
            or contexts_manifest_arg
        )
        if not manifest_path and explicit_run_id:
            manifest_path = run_layout["default_manifest_file"]
        if not manifest_path:
            parser.error(
                "--skip-contexts requires --manifest-file, --analysis-args/--contexts-args with --manifest-file, "
                "or an explicit --run-id to derive the default manifest path."
            )
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

        _freeze_scenario_file(contexts_base_args, run_layout["run_dir"])

        # Freeze the vf sampling config alongside the run config: it documents
        # how the artifacts this run simulates from were (or would be) sampled.
        vf_build_config_arg = _extract_flag_value(vf_build_base_args, "--config")
        if vf_build_config_arg:
            resolved_vf_config = _resolve_cli_path(vf_build_config_arg)
            if os.path.exists(resolved_vf_config):
                shutil.copy2(resolved_vf_config,
                             Path(run_layout["run_dir"]) / "vf_build_config_source.yaml")
            else:
                print(f"WARNING: vf_build_args --config '{vf_build_config_arg}' resolved to "
                      f"'{resolved_vf_config}' which does not exist; not frozen.")

        effective_config_payload = {
            "config_yaml": source_config_path,
            "config_profile": args.config_profile,
            "run_root": args.run_root,
            "run_id": args.run_id,
            "llm_model": args.llm_model,
            "vf_build_args": args.vf_build_args,
            "contexts_args": args.contexts_args,
            "analysis_args": args.analysis_args,
            "rank_stability_args": args.rank_stability_args,
            "cross_model_args": args.cross_model_args,
            "manifest_file": args.manifest_file,
            "skip_vf_build": args.skip_vf_build,
            "skip_contexts": args.skip_contexts,
            "skip_analysis": args.skip_analysis,
            "skip_rank_stability": args.skip_rank_stability,
            "skip_cross_model": args.skip_cross_model,
            "continue_on_error": args.continue_on_error,
            "dry_run": args.dry_run,
            "resolved": {
                "manifest_path": manifest_path,
                "value_functions_dir": run_layout["value_functions_dir"],
                "vf_build_base_args_list": vf_build_base_args,
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
                "skip_vf_build": args.skip_vf_build,
                "skip_contexts": args.skip_contexts,
                "skip_analysis": args.skip_analysis,
                "skip_rank_stability": args.skip_rank_stability,
                "skip_cross_model": args.skip_cross_model,
            },
            "passthrough": {
                "vf_build_args": args.vf_build_args,
                "contexts_args": args.contexts_args,
                "analysis_args": args.analysis_args,
                "rank_stability_args": args.rank_stability_args,
                "cross_model_args": args.cross_model_args,
            },
            "resolved_paths": {
                "value_functions_dir": run_layout["value_functions_dir"],
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
    print(f"Value functions dir: {run_layout['value_functions_dir']}")

    if not args.skip_vf_build:
        if _contains_flag(vf_build_base_args, "--config"):
            vf_build_cmd = [sys.executable, str(VF_BUILD_SCRIPT)]
            vf_build_cmd.extend(vf_build_base_args)
            try:
                # cwd stays at the repo root so artifacts land in the canonical
                # store (value_functions/results/sampled/); the run
                # folder then freezes copies of what it uses.
                _run_command(vf_build_cmd, dry_run=args.dry_run)
            except subprocess.CalledProcessError as exc:
                print(f"[error] Value-function build stage failed with exit code {exc.returncode}.")
                if not args.continue_on_error:
                    raise
        else:
            print("[skip] vf-build stage: vf_build_args has no --config; "
                  "assuming pre-built artifacts.")

    if not args.skip_contexts:
        contexts_base_args = _freeze_value_functions(
            contexts_base_args, run_layout["value_functions_dir"], args.dry_run)
        contexts_cmd = [sys.executable, str(CONTEXTS_SCRIPT)]
        contexts_cmd.extend(contexts_base_args)
        if not _contains_flag(contexts_base_args, "--llm-model"):
            contexts_cmd.extend(["--llm-model", model])
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

    if not args.skip_rank_stability:
        _run_rank_stability_stage(args, run_layout, rank_stability_base_args,
                                  vf_label, production_board, vf_template_arg)

    if not args.skip_cross_model:
        _run_cross_model_stage(args, run_layout, cross_model_base_args)

    print("\nPipeline completed.")


def _tables_are_exact(vf_template: str | None) -> bool:
    """True when every artifact the --value-function template resolves to
    declares meta.source == 'logprob' (checked on the SOURCE artifacts, so it
    also holds for a dry run, before anything is frozen)."""
    if not vf_template:
        return False
    resolved = Path(_resolve_cli_path(vf_template))
    files = (sorted(resolved.parent.glob(resolved.name.replace("{scenario}", "*")))
             if "{scenario}" in vf_template else
             sorted(resolved.glob("vf_*.json")) if resolved.is_dir() else
             [resolved] if resolved.exists() else [])
    return _all_logprob(files)


def _all_logprob(files) -> bool:
    if not files:
        return False
    srcs = set()
    for f in files:
        try:
            srcs.add(json.loads(f.read_text()).get("meta", {}).get("source", "sampled"))
        except (OSError, ValueError):
            return False
    return srcs == {"logprob"}


def _frozen_tables_are_exact(value_functions_dir) -> bool:
    """True when every frozen vf_*.json declares meta.source == 'logprob'."""
    files = sorted(Path(value_functions_dir).glob("vf_*.json")) if value_functions_dir else []
    return _all_logprob(files)



def _run_rank_stability_stage(args, run_layout, rs_args, vf_label, board,
                              vf_template=None) -> None:
    """Classify the DI ordering of this run against the value function's ruler.

    Non-fatal outcomes are verdicts (exit 0 SETTLED / 4 FIXABLE / 6
    UNMEASURED) and are recorded, not raised. A missing ruler or a checker
    crash fails the stage (raises unless --continue-on-error), because the
    orchestrated run is not finished until its ordering has been classified.
    """
    print("\n" + "=" * 90 + "\nRank-stability stage")
    label = _get_flag_value(rs_args, "--label") or vf_label
    ms_dir = _get_flag_value(rs_args, "--multisplit")
    # EXACT tables (2026-09-05): logprob-derived value functions carry no
    # sampling error, so there is no ruler to resolve — the frozen artifacts
    # say so in meta.source, or the yaml forces it with --exact.
    exact = (_contains_flag(rs_args, "--exact")
             or _frozen_tables_are_exact(run_layout["value_functions_dir"])
             or _tables_are_exact(vf_template))
    record: dict[str, Any] = {"label": label, "board": board, "exact_tables": exact}
    failure: str | None = None
    if not label:
        failure = ("cannot derive the value-function label: contexts_args has no "
                   "vf_<label>__{scenario}__<style>.json --value-function and "
                   "rank_stability_args gives no --label")
    elif exact:
        record["ruler"] = None
    elif ms_dir is None:
        chosen, seen = _resolve_ruler(label, board)
        record["rulers_seen"] = [
            {"dir": str(d["_dir"]), "board_ok": d["_board_ok"],
             "rng_scheme": d.get("rng_scheme", "shared"),
             "grid_size": d.get("grid_size"), "max_steps": d.get("max_steps"),
             "splits": d.get("splits")} for d in seen]
        if chosen is None:
            failure = (f"no multi-split ruler for {label!r} matching board {board} "
                       f"under {MULTISPLIT_STORE} (saw {len(seen)}; run "
                       f"run_vf_multisplit_backfill.sh on the production board first)")
        else:
            ms_dir = str(chosen)
    if failure is None:
        out_dir = (_get_flag_value(rs_args, "--out-dir")
                   or str(Path(run_layout["analysis_dir"]) / "rank_stability"))
        cmd = [sys.executable, str(RANK_STABILITY_SCRIPT)]
        pairs = [("--label", label), ("--production-experiments", run_layout["experiments_dir"]),
                 ("--out-dir", out_dir)]
        if not exact:
            pairs.append(("--multisplit", ms_dir))
        for flag, val in pairs:
            if not _contains_flag(rs_args, flag):
                cmd.extend([flag, str(val)])
        if exact and not _contains_flag(rs_args, "--exact"):
            cmd.append("--exact")
        cmd.extend(rs_args)
        print("[cmd]", shlex.join(cmd))
        record.update({"ruler": ms_dir, "out_dir": out_dir, "command": shlex.join(cmd)})
        if args.dry_run:
            record["status"] = "dry_run"
        else:
            rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
            verdict = {0: "SETTLED", 4: "FIXABLE", 6: "UNMEASURED"}.get(rc)
            record["exit_code"] = rc
            if verdict is None:
                failure = f"vf_rank_stability.py exited {rc}"
            else:
                record["status"] = "classified"
                record["verdict"] = verdict
                status_file = Path(out_dir) / "rank_status.json"
                if status_file.exists():
                    st = json.loads(status_file.read_text())
                    record["decision"] = st.get("decision")
                    record["ruler_rng_scheme"] = st.get("ruler_rng_scheme")
                print(f"[rank-stability] verdict: {verdict}")
    if failure is not None:
        record["status"] = "failed"
        record["error"] = failure
        print(f"[error] Rank-stability stage: {failure}")
    if not args.dry_run:
        manifest_path = Path(run_layout["run_dir"]) / "run_layout_manifest.json"
        try:
            meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        meta["rank_stability"] = record
        manifest_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if failure is not None and not args.continue_on_error:
        raise SystemExit(3)


def _record_stage(run_layout, key: str, record: dict[str, Any]) -> None:
    manifest_path = Path(run_layout["run_dir"]) / "run_layout_manifest.json"
    try:
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    meta[key] = record
    manifest_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _table_family(model_slug: str) -> str:
    """'lp' for <model>-vf-lp (exact logprob tables), else 'r3' (sampled)."""
    m = re.search(r"-vf-([a-z0-9]+)$", model_slug)
    fam = m.group(1) if m else "r3"
    if fam not in ("r3", "lp"):
        print(f"[cross-model] unknown table family suffix '-vf-{fam}' in {model_slug!r}; using r3")
        fam = "r3"
    return fam


def _run_cross_model_stage(args, run_layout, cm_args) -> None:
    """Regenerate the cross-model comparison from every model under run_root.

    The figures compare the newest FULL run of each model (see
    cross_model_vf_comparison.load_models), so this run only enters the
    comparison once it is as large as that model's other runs — a smoke test
    never displaces a production run. Concurrent pipelines (if two are ever run at once)
    take a file lock so two processes never write the same PNG at once.
    """
    import fcntl

    print("\n" + "=" * 90 + "\nCross-model stage")
    run_root = _get_flag_value(cm_args, "--run-root") or run_layout["run_root"]
    # Table family from the model slug (-vf-lp exact / -vf-r3 sampled). Only
    # the exact tables are a result (2026-09-06: sampled tables carry the
    # batch-numerics artifact); a sampled-table run refreshes nothing unless
    # cross_model_args asks for --family explicitly.
    explicit = _get_flag_value(cm_args, "--family")
    family = explicit or _table_family(run_layout["model_slug"])
    if family != "lp" and not explicit:
        note = (f"sampled tables ({run_layout['model_slug']}) are superseded by the exact "
                f"logprob tables and are not a result; not regenerating cross_model_*")
        print(f"[skip] Cross-model stage: {note}")
        if not args.dry_run:
            _record_stage(run_layout, "cross_model",
                          {"family": family, "status": "skipped", "reason": note})
        return
    cmd = [sys.executable, str(CROSS_MODEL_SCRIPT)]
    if not _contains_flag(cm_args, "--run-root"):
        cmd.extend(["--run-root", str(Path(run_root).resolve())])
    if not _contains_flag(cm_args, "--family"):
        cmd.extend(["--family", family])
    cmd.extend(cm_args)
    out_dir = _get_flag_value(cm_args, "--out-dir") or str(
        Path(run_root).resolve() / "cross_model")
    record: dict[str, Any] = {"run_root": str(run_root), "family": family,
                              "out_dir": out_dir, "command": shlex.join(cmd)}
    print("[cmd]", shlex.join(cmd))
    failure: str | None = None
    if args.dry_run:
        record["status"] = "dry_run"
    else:
        lock_path = Path(run_root) / ".cross_model_figures.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)     # released when the file closes
            rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
        record["exit_code"] = rc
        if rc == 0:
            record["status"] = "regenerated"
            print(f"[cross-model] figures and tables refreshed under {out_dir}")
        else:
            failure = f"cross_model_vf_comparison.py exited {rc}"
    if failure is not None:
        record["status"] = "failed"
        record["error"] = failure
        print(f"[error] Cross-model stage: {failure}")
    if not args.dry_run:
        _record_stage(run_layout, "cross_model", record)
    if failure is not None and not args.continue_on_error:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
