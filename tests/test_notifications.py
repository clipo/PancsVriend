"""Tests for notifications.py — config-derived ntfy stage messages.

Regression intent: stage pings must derive runs/steps/scenarios/style from the
run YAML actually being executed, never from hardcoded text (the "20 runs" ping
on a 10-run experiment). Rendering must degrade ('?') instead of raising —
a notifier bug must not kill a multi-hour run.
"""

import os

import pytest

import notifications as N

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(REPO, "configs", "llama_cpp_run_gemma31b_a3_chat.yaml")


def test_from_yaml_production_profile():
    ctx = N.ExperimentContext.from_yaml(CFG, "production", label="gemma-a3-chat")
    assert ctx.label == "gemma-a3-chat"
    assert ctx.runs == 10
    assert ctx.max_steps == 100
    assert ctx.style == "chat+grammar"
    assert ctx.n_scenarios == 6            # scenarios: [] -> all in scenarios_a3.py
    assert "baseline" in ctx.scenario_names
    assert ctx.processes == 4
    assert ctx.desc == "10 runs x 100 steps x 6 scenarios (scenarios_a3, chat+grammar)"


def test_from_yaml_smoke_profile():
    ctx = N.ExperimentContext.from_yaml(CFG, "smoke_test")
    assert ctx.runs == 2
    assert ctx.max_steps == 2
    assert ctx.n_scenarios == 1            # scenarios: [baseline]
    assert ctx.label == "gemma-4-31B-it-Q5_K_M-a3-chat"  # falls back to llm_model


def test_style_normalization():
    assert N._normalize_style(None) == "legacy"
    assert N._normalize_style("") == "legacy"
    assert N._normalize_style("chat grammar") == "chat+grammar"
    assert N._normalize_style("completions_grammar") == "completions+grammar"


def test_render_every_event_dry_run(capsys):
    ctx = N.ExperimentContext.from_yaml(CFG, label="m")
    notifier = N.Notifier(ctx, dry_run=True)
    for event in N.TEMPLATES:
        assert notifier.send(event, port=8081, elapsed_min=42,
                             log="logs/x.log", msg="boom") is True
    out = capsys.readouterr().out
    # config-derived values reach the rendered text
    assert "10 runs x 100 steps x 6 scenarios" in out
    assert "chat+grammar" in out
    assert "?" not in out                  # all placeholders resolved


def test_missing_placeholder_degrades_not_raises(capsys):
    notifier = N.Notifier(N.ExperimentContext(label="m"), dry_run=True)
    assert notifier.send("production_complete") is True   # no elapsed_min given
    captured = capsys.readouterr()
    assert "?" in captured.out
    assert "elapsed_min" in captured.err


def test_unknown_event_is_soft_failure(capsys):
    assert N.Notifier(dry_run=True).send("no_such_event") is False
    assert "unknown event" in capsys.readouterr().err


def test_cli_describe_and_send_always_exit_zero(capsys):
    assert N.main(["describe", "--config", CFG]) == 0
    assert "10 runs x 100 steps x 6 scenarios (scenarios_a3, chat+grammar)" \
        in capsys.readouterr().out
    # bad config path must still exit 0 (label-only fallback)
    assert N.main(["send", "launch_failed", "--config", "/nope.yaml",
                   "--label", "x", "--set", "msg=binary missing",
                   "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "x launch FAILED" in out
    assert "binary missing" in out
