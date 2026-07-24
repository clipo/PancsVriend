#!/usr/bin/env python3
"""Template-driven ntfy stage notifications, auto-derived from the run config.

Replaces the hand-written parameter text in the runner/queue shell scripts
("20 runs x 100 steps ..." while the config said 10) with messages rendered
from the SAME YAML the experiment actually runs, so counts can never drift.

Three parts:
  ExperimentContext  — the experiment parameters, read from a run YAML profile
  TEMPLATES          — event -> message templates (THE customization point)
  Notifier           — renders a template and hands it to ./ntfy.sh (transport
                       stays in ntfy.sh; per-run/ETA pings stay in
                       watch_progress.sh, which is log-driven)

Shell usage (see run_a3_model.sh):
    .venv/bin/python notifications.py send production_started \
        --config configs/llama_cpp_run_gemma31b_a3.yaml --label gemma-4-31b-a3 \
        [--profile production] [--set port=8081 --set elapsed_min=266] [--dry-run]
    .venv/bin/python notifications.py describe --config <yaml> [--profile ...]

`describe` prints the one-line experiment summary (for embedding in queue
messages via $(...)).

Customizing:
  - wording/emoji/priority        -> edit TEMPLATES
  - new event                     -> add a TEMPLATES entry, call `send <event>`
  - new parameter in messages     -> add a field in ExperimentContext.from_yaml,
                                     reference {field} in any template

A notifier failure must never kill an experiment: every path degrades to a
stderr warning, missing placeholders render as '?', and the CLI always exits 0.
"""

import argparse
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass, field

import yaml

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# TEMPLATES — the customization point.
# Placeholders: any ExperimentContext field ({label}, {runs}, {max_steps},
# {n_scenarios}, {style}, {scenario_file}, {processes}, {temperature}, {desc},
# {profile}) plus per-call extras passed via Notifier.send(...)/--set
# ({port}, {elapsed_min}, {log}, {msg}, ...). Missing values render as '?'.
# ---------------------------------------------------------------------------
TEMPLATES = {
    "launching": dict(
        title="{label} launching",
        body="Server up on port {port} (-np {processes}). {desc}. Running smoke test next.",
        tags="rocket", priority="default"),
    "launch_failed": dict(
        title="{label} launch FAILED",
        body="{msg}",
        tags="rotating_light", priority="high"),
    "smoke_failed": dict(
        title="{label} smoke test FAILED",
        body="Smoke test failed. Check {log}.",
        tags="rotating_light", priority="high"),
    "production_started": dict(
        title="{label} production STARTED",
        body="{desc} @ T={temperature}. Per-scenario pings follow.",
        tags="checkered_flag", priority="default"),
    "production_complete": dict(
        title="✅ {label} DONE in {elapsed_min} min",
        body=("{desc}\n"
              "Results: experiments_with_llama_cpp/"),
        tags="tada", priority="max"),
    "results_pushed": dict(
        title="{label} results pushed",
        body="{desc} — commit on master.",
        tags="package", priority="low"),
    "production_failed": dict(
        title="{label} production FAILED",
        body="Production run failed after smoke passed. Check {log}.",
        tags="rotating_light", priority="high"),
}


def _warn(msg):
    print(f"[notifications] WARNING: {msg}", file=sys.stderr)


class _SafeDict(dict):
    """format_map source that renders missing placeholders as '?' (and warns)
    instead of raising — a bad template must not crash a multi-hour run."""

    def __missing__(self, key):
        _warn(f"no value for placeholder '{{{key}}}'")
        return "?"


def _normalize_style(llm_style):
    """Same normalization rule as llm_runner.resolve_llm_style (inlined so this
    module stays light — no requests/numpy import chain); None -> 'legacy'."""
    if llm_style is None or str(llm_style).strip() == "":
        return "legacy"
    style = str(llm_style).strip().lower().replace(" ", "+").replace("_", "+")
    while "++" in style:
        style = style.replace("++", "+")
    return style


def _scenario_names(scenario_file):
    """Keys of CONTEXT_SCENARIOS in a scenario file, loaded directly with
    importlib (path fallback mirrors llm_runner.apply_scenario_file)."""
    path = str(scenario_file)
    if not os.path.isabs(path) and not os.path.exists(path):
        candidate = os.path.join(REPO_ROOT, path)
        if os.path.exists(candidate):
            path = candidate
    try:
        spec = importlib.util.spec_from_file_location("_notif_scenarios", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return list(getattr(module, "CONTEXT_SCENARIOS", {}) or {})
    except Exception as exc:
        _warn(f"could not load scenarios from {scenario_file}: {exc}")
        return []


@dataclass
class ExperimentContext:
    """The parameters of one experiment run, as read from its run YAML."""
    label: str = "?"
    config_path: str | None = None
    profile: str = "production"
    runs: int | None = None
    max_steps: int | None = None
    style: str = "?"
    scenario_file: str = "?"        # basename without .py, e.g. "scenarios_a3"
    scenario_names: list = field(default_factory=list)
    n_scenarios: int | None = None
    processes: int | str | None = None
    temperature: float | None = None

    @classmethod
    def from_yaml(cls, cfg_path, profile="production", label=None):
        """Read the SAME profile block the run uses (profiles.<profile>.
        contexts_args, falling back to top-level contexts_args)."""
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh) or {}
        block = (cfg.get("profiles") or {}).get(profile) or cfg
        ca = block.get("contexts_args") or cfg.get("contexts_args") or {}
        scen_path = ca.get("scenario_file") or "context_scenarios.py"
        # scenarios: [] means "all scenarios in scenario_file"
        names = list(ca.get("scenarios") or []) or _scenario_names(scen_path)
        return cls(
            label=label or cfg.get("llm_model") or "?",
            config_path=str(cfg_path),
            profile=profile,
            runs=ca.get("runs"),
            max_steps=ca.get("max_steps"),
            style=_normalize_style(ca.get("llm_style")),
            scenario_file=os.path.splitext(os.path.basename(str(scen_path)))[0],
            scenario_names=names,
            n_scenarios=len(names) or None,
            processes=ca.get("processes"),
            temperature=ca.get("temperature"),
        )

    @property
    def desc(self):
        """One-line summary, e.g.
        '10 runs x 100 steps x 6 scenarios (scenarios_a3, chat+grammar)'."""
        vals = _SafeDict(runs=self.runs, max_steps=self.max_steps,
                         n_scenarios=self.n_scenarios,
                         scenario_file=self.scenario_file, style=self.style)
        vals = {k: ("?" if v is None else v) for k, v in vals.items()}
        return ("{runs} runs x {max_steps} steps x {n_scenarios} scenarios "
                "({scenario_file}, {style})").format_map(vals)


class Notifier:
    """Renders TEMPLATES[event] from an ExperimentContext (+extras) and sends
    it through ./ntfy.sh. Never raises."""

    def __init__(self, ctx=None, dry_run=False):
        self.ctx = ctx or ExperimentContext()
        self.dry_run = dry_run

    def send(self, event, **extra):
        tmpl = TEMPLATES.get(event)
        if tmpl is None:
            _warn(f"unknown event '{event}' (known: {', '.join(TEMPLATES)})")
            return False
        values = _SafeDict(vars(self.ctx))
        values["desc"] = self.ctx.desc
        values.update({k: v for k, v in extra.items() if v is not None})
        title = str(tmpl["title"]).format_map(values)
        body = str(tmpl["body"]).format_map(values)
        if self.dry_run:
            print(f"[dry-run] title: {title}\n[dry-run] body: {body}\n"
                  f"[dry-run] tags={tmpl['tags']} priority={tmpl['priority']}")
            return True
        try:
            subprocess.run(
                [os.path.join(REPO_ROOT, "ntfy.sh"), title, body,
                 str(tmpl["tags"]), str(tmpl["priority"])],
                cwd=REPO_ROOT, check=False, timeout=30)
            return True
        except Exception as exc:
            _warn(f"send failed for '{event}': {exc}")
            return False


def _build_context(args):
    if args.config:
        try:
            return ExperimentContext.from_yaml(args.config, args.profile,
                                               getattr(args, "label", None))
        except Exception as exc:
            _warn(f"could not read config {args.config}: {exc}")
    return ExperimentContext(label=getattr(args, "label", None) or "?",
                             profile=args.profile)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_send = sub.add_parser("send", help="render an event template and send it")
    p_send.add_argument("event", help=f"one of: {', '.join(TEMPLATES)}")
    p_send.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        dest="extras", help="extra template values (repeatable)")
    p_send.add_argument("--dry-run", action="store_true",
                        help="print the rendered message instead of sending")
    p_desc = sub.add_parser("describe", help="print the one-line experiment summary")
    for p in (p_send, p_desc):
        p.add_argument("--config", help="run YAML (omit for label-only events)")
        p.add_argument("--profile", default="production")
        p.add_argument("--label", help="experiment label shown in titles")

    args = parser.parse_args(argv)
    ctx = _build_context(args)

    if args.cmd == "describe":
        print(ctx.desc)
        return 0

    extra = {}
    for item in args.extras:
        if "=" not in item:
            _warn(f"ignoring malformed --set '{item}' (need KEY=VALUE)")
            continue
        key, value = item.split("=", 1)
        extra[key] = value
    Notifier(ctx, dry_run=args.dry_run).send(args.event, **extra)
    return 0  # never fail the calling run


if __name__ == "__main__":
    sys.exit(main())
