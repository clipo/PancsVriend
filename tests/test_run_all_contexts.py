import builtins
import json
import sys

import pytest

import run_all_contexts as rac


def test_validate_scenarios_invalid():
    with pytest.raises(ValueError) as excinfo:
        rac._validate_scenarios(["not-a-scenario"])
    assert "Unknown scenario" in str(excinfo.value)


def test_find_resume_candidate_detects_incomplete(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp_dir = tmp_path / "experiments" / "exp_one"
    exp_dir.mkdir(parents=True)
    config = {
        "scenario": "baseline",
        "llm_model": "phi4:latest",
        "n_runs": 2,
        "max_steps": 500,
    }
    (exp_dir / "config.json").write_text(json.dumps(config))

    statuses = {
        0: {"status": "converged"},
        1: {"status": "aborted"},
    }

    def fake_check(name):
        assert name == "exp_one"
        return True, 1, str(exp_dir), {0}

    def fake_status(output_dir, run_id, max_steps):
        assert output_dir == str(exp_dir)
        assert max_steps == config["max_steps"]
        return statuses[run_id]

    monkeypatch.setattr(rac, "check_existing_experiment", fake_check)
    monkeypatch.setattr(rac, "_analyze_run_status", fake_status)

    found_name, complete = rac._find_resume_candidate("baseline", "phi4:latest")
    assert found_name == "exp_one"
    assert not complete


def test_find_resume_candidate_returns_complete_when_done(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp_dir = tmp_path / "experiments" / "exp_two"
    exp_dir.mkdir(parents=True)
    config = {
        "scenario": "baseline",
        "llm_model": "phi4:latest",
        "n_runs": 2,
        "max_steps": 100,
    }
    (exp_dir / "config.json").write_text(json.dumps(config))

    def fake_check(name):
        assert name == "exp_two"
        return True, 2, str(exp_dir), {0, 1}

    def fake_status(output_dir, run_id, max_steps):
        return {"status": "converged"}

    monkeypatch.setattr(rac, "check_existing_experiment", fake_check)
    monkeypatch.setattr(rac, "_analyze_run_status", fake_status)

    found_name, complete = rac._find_resume_candidate("baseline", "phi4:latest")
    assert found_name == "exp_two"
    assert complete


def test_main_invokes_run_llm_experiment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    scenarios_seen = []

    def fake_find(scenario, llm_model, target_runs):
        if scenario == "baseline":
            return "resume-exp", False
        return None, False

    def fake_run_llm_experiment(**kwargs):
        scenarios_seen.append((kwargs["scenario"], kwargs["resume_experiment"]))
        out_dir = f"experiments/{kwargs['scenario']}"
        return out_dir, [{"run_id": 0}, {"run_id": 1}]

    printed = []

    def fake_print(*args, **kwargs):
        printed.append(" ".join(str(a) for a in args))

    monkeypatch.setattr(rac, "_find_resume_candidate", fake_find)
    monkeypatch.setattr(rac, "run_llm_experiment", fake_run_llm_experiment)
    monkeypatch.setattr(builtins, "print", fake_print)

    monkeypatch.setattr(sys, "argv", [
        "run_all_contexts.py",
        "--runs",
        "3",
        "--processes",
        "2",
        "--llm-model",
        "phi4:latest",
        "--scenarios",
        "baseline",
        "ethnic_asian_hispanic",
    ])

    rac.main()

    assert scenarios_seen == [("baseline", "resume-exp"), ("ethnic_asian_hispanic", None)]
    assert any("baseline" in line and "experiments/baseline" in line for line in printed)
    assert any("ethnic_asian_hispanic" in line for line in printed)
