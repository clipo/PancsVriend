"""The cross-model comparison is a pipeline stage, not a hand-run script.

    .venv/bin/python -m pytest tests/test_cross_model_stage.py -q

Covers:
- cross_model_vf_comparison.pick_run: newest of the LARGEST runs per model, so
  a smoke test (2 runs) under the same model slug never displaces the 10k run
- the orchestrator runs cross_model_vf_comparison.py as its final stage with
  --run-root = the run's root for exact-table (-vf-lp) runs, records "skipped"
  for sampled-table (-vf-r3) runs; --skip-cross-model / skip_cross_model omit
  it; cross_model_args pass through and an explicit --run-root/--family wins
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "value_functions" / "comparison"))

import cross_model_vf_comparison as cm  # noqa: E402
import run_llm_probability_simulation_analysis as orch  # noqa: E402


def _summary(root: Path, run_name: str, n_rows: int) -> Path:
    p = root / run_name / "analysis" / "run_summary_by_run.csv"
    p.parent.mkdir(parents=True)
    p.write_text("run_id,scenario,scenario_key,dissimilarity_index\n"
                 + "".join(f"{i},baseline,llm_baseline,0.1\n" for i in range(n_rows)))
    return p


class TestPickRun:
    def test_largest_run_beats_newer_smoke_test(self, tmp_path):
        prod = _summary(tmp_path, "run_20260905_140659_olmo-2-32b-vf-r3", 60)
        smoke = _summary(tmp_path, "run_20260906_090000_olmo-2-32b-vf-r3", 2)
        chosen, skipped = cm.pick_run([str(smoke), str(prod)])
        assert chosen == str(prod)
        assert skipped == [(str(smoke), 2)]

    def test_newest_among_equal_sizes(self, tmp_path):
        old = _summary(tmp_path, "run_20260901_000000_olmo-2-32b-vf-r3", 60)
        new = _summary(tmp_path, "run_20260905_000000_olmo-2-32b-vf-r3", 60)
        chosen, skipped = cm.pick_run([str(new), str(old)])
        assert chosen == str(new)
        assert skipped == [(str(old), 60)]

    def test_empty(self):
        assert cm.pick_run([]) == (None, [])

    def test_families_never_mix(self, tmp_path):
        _summary(tmp_path, "run_20260905_000000_olmo-2-32b-vf-r3", 6)
        _summary(tmp_path, "run_20260906_000000_olmo-2-32b-vf-lp", 4)
        r3 = cm.load_models(tmp_path, "r3")
        lp = cm.load_models(tmp_path)                     # exact tables are the default
        assert len(r3["olmo-2-32b"]) == 6 and len(lp["olmo-2-32b"]) == 4
        assert r3["olmo-2-32b"]["_source"].iloc[0].endswith(
            "run_20260905_000000_olmo-2-32b-vf-r3/analysis/run_summary_by_run.csv")

    def test_load_models_uses_pick_run(self, tmp_path, capsys):
        _summary(tmp_path, "run_20260905_140659_olmo-2-32b-vf-lp", 60)
        _summary(tmp_path, "run_20260906_090000_olmo-2-32b-vf-lp", 2)
        data = cm.load_models(tmp_path)                   # default family = exact (lp)
        assert list(data) == ["olmo-2-32b"]
        assert len(data["olmo-2-32b"]) == 60
        assert "scenario" in data["olmo-2-32b"].columns      # scenario_key renamed
        assert "skipped run_20260906_090000_olmo-2-32b-vf-lp" in capsys.readouterr().out


def _dry_run(tmp_path, *extra):
    cmd = [sys.executable, str(REPO_ROOT / "run_llm_probability_simulation_analysis.py"),
           "--llm-model", "unit-vf-lp", "--run-root", str(tmp_path / "root"),
           "--run-id", "run_unit", "--dry-run", "--skip-vf-build", "--skip-contexts",
           "--skip-analysis", "--skip-rank-stability", *extra]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    return res.stdout


class TestOrchestratorStage:
    def test_final_stage_runs_cross_model_on_run_root(self, tmp_path):
        out = _dry_run(tmp_path)
        assert "Cross-model stage" in out
        cmd_line = [l for l in out.splitlines() if "cross_model_vf_comparison.py" in l][0]
        assert f"--run-root {tmp_path / 'root'}" in cmd_line
        assert out.index("Cross-model stage") < out.index("Pipeline completed.")

    def test_family_from_model_suffix(self, tmp_path):
        assert orch._table_family("gemma-4-31b-vf-lp") == "lp"
        assert orch._table_family("gemma-4-31b-vf-r3") == "r3"
        assert orch._table_family("gemma-4-31b-it-q5") == "r3"
        out = _dry_run(tmp_path)                       # unit-vf-lp
        assert "--family lp" in [l for l in out.splitlines() if "cross_model_vf_comparison.py" in l][0]

    def test_sampled_table_run_is_skipped(self, tmp_path):
        cmd = [sys.executable, str(REPO_ROOT / "run_llm_probability_simulation_analysis.py"),
               "--llm-model", "unit-vf-r3", "--run-root", str(tmp_path / "root2"),
               "--run-id", "run_unit", "--dry-run", "--skip-vf-build", "--skip-contexts",
               "--skip-analysis", "--skip-rank-stability"]
        res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        assert "cross_model_vf_comparison.py" not in res.stdout
        assert "[skip] Cross-model stage" in res.stdout
        # ... unless the yaml asks for the superseded set explicitly
        res = subprocess.run(cmd + ["--cross-model-args", "--family r3"],
                             cwd=str(REPO_ROOT), capture_output=True, text=True)
        assert "--family r3" in [l for l in res.stdout.splitlines() if "cross_model_vf_comparison.py" in l][0]

    def test_skip_flag(self, tmp_path):
        out = _dry_run(tmp_path, "--skip-cross-model")
        assert "cross_model_vf_comparison.py" not in out

    def test_passthrough_and_explicit_root(self, tmp_path):
        out = _dry_run(tmp_path, "--cross-model-args", "--run-root /elsewhere --dpi 72")
        cmd_line = [l for l in out.splitlines() if "cross_model_vf_comparison.py" in l][0]
        assert "--run-root /elsewhere" in cmd_line and "--dpi 72" in cmd_line
        assert str(tmp_path / "root") not in cmd_line

    def test_yaml_keys(self, tmp_path):
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text("llm_model: unit-vf-r3\nskip_cross_model: true\n"
                             "cross_model_args:\n  dpi: 72\n")
        args = orch._build_parser().parse_args(["--config-yaml", str(yaml_path)])
        args = orch._apply_config_to_args(args, orch._load_yaml_config(str(yaml_path)), [])
        assert args.skip_cross_model is True
        assert args.cross_model_args == "--dpi 72"

    def test_failure_is_recorded_and_raises(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "root" / "run_unit"
        run_dir.mkdir(parents=True)
        layout = orch._resolve_run_layout(str(tmp_path / "root"), "run_unit", "unit-vf-lp")
        monkeypatch.setattr(orch, "CROSS_MODEL_SCRIPT", REPO_ROOT / "does_not_exist.py")

        class A:
            dry_run = False
            continue_on_error = False
        with pytest.raises(SystemExit):
            orch._run_cross_model_stage(A(), layout, [])
        import json
        rec = json.loads((run_dir / "run_layout_manifest.json").read_text())["cross_model"]
        assert rec["status"] == "failed" and rec["exit_code"] != 0
        assert (tmp_path / "root" / ".cross_model_figures.lock").exists()
