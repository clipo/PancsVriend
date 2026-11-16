"""The census goodness of fit is a pipeline stage, after the cross-model stage.

    .venv/bin/python -m pytest tests/test_census_gof_stage.py -q

Covers:
- the orchestrator runs census_goodness_of_fit.py after the cross-model stage,
  with --out-dir <run_root>/cross_model, for exact-table (-vf-lp) runs
- a sampled-table (-vf-r3) run records it as skipped
- --skip-census-gof / skip_census_gof omit it; the cross-model stage is unaffected
- an explicit --out-dir in cross_model_args wins
- a failure is recorded in run_layout_manifest.json and raises; "no complete
  census" is recorded as its own status (nothing_to_test), flags any earlier
  table, and does not fail the pipeline
- census_goodness_of_fit.py on synthetic inputs: one row per model, PASS for a
  census drawn from its table, csv + png written, seed and reps recorded;
  no census, or only a partial one, exits NOTHING_TO_TEST and writes nothing
"""
import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "prompt_refinement"), str(REPO / "value_functions" / "comparison")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_llm_probability_simulation_analysis as orch  # noqa: E402

SCRIPT = "census_goodness_of_fit.py"


def _dry_run(tmp_path, model="unit-vf-lp", *extra):
    cmd = [sys.executable, str(REPO / "run_llm_probability_simulation_analysis.py"),
           "--llm-model", model, "--run-root", str(tmp_path / "root"),
           "--run-id", "run_unit", "--dry-run", "--skip-vf-build", "--skip-contexts",
           "--skip-analysis", "--skip-rank-stability", *extra]
    res = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    return res.stdout


def test_runs_after_cross_model_on_run_root(tmp_path):
    out = _dry_run(tmp_path)
    line = [l for l in out.splitlines() if SCRIPT in l][0]
    assert f"--out-dir {tmp_path / 'root' / 'cross_model'}" in line
    assert out.index("Cross-model stage") < out.index("Census goodness-of-fit stage") \
        < out.index("Pipeline completed.")


def test_sampled_table_run_is_skipped(tmp_path):
    out = _dry_run(tmp_path, "unit-vf-r3")
    assert SCRIPT not in out and "[skip] Census goodness-of-fit stage" in out


def test_skip_flag_leaves_cross_model(tmp_path):
    out = _dry_run(tmp_path, "unit-vf-lp", "--skip-census-gof")
    assert SCRIPT not in out and "cross_model_vf_comparison.py" in out


def test_runs_even_when_cross_model_is_skipped(tmp_path):
    out = _dry_run(tmp_path, "unit-vf-lp", "--skip-cross-model")
    assert SCRIPT in out and "cross_model_vf_comparison.py" not in out


def test_explicit_out_dir(tmp_path):
    out = _dry_run(tmp_path, "unit-vf-lp", "--cross-model-args", "--out-dir /elsewhere")
    assert "--out-dir /elsewhere" in [l for l in out.splitlines() if SCRIPT in l][0]


def test_yaml_key(tmp_path):
    y = tmp_path / "cfg.yaml"
    y.write_text("llm_model: unit-vf-lp\nskip_census_gof: true\n")
    args = orch._build_parser().parse_args(["--config-yaml", str(y)])
    args = orch._apply_config_to_args(args, orch._load_yaml_config(str(y)), [])
    assert args.skip_census_gof is True


def test_failure_is_recorded_and_raises(tmp_path, monkeypatch):
    run_dir = tmp_path / "root" / "run_unit"
    run_dir.mkdir(parents=True)
    layout = orch._resolve_run_layout(str(tmp_path / "root"), "run_unit", "unit-vf-lp")
    monkeypatch.setattr(orch, "CENSUS_GOF_SCRIPT", REPO / "does_not_exist.py")

    class A:
        dry_run = False
        continue_on_error = False
    with pytest.raises(SystemExit):
        orch._run_census_gof_stage(A(), layout, [])
    rec = json.loads((run_dir / "run_layout_manifest.json").read_text())["census_gof"]
    assert rec["status"] == "failed" and rec["exit_code"] != 0


def test_nothing_to_test_is_its_own_status(tmp_path, monkeypatch):
    import census_goodness_of_fit as gof
    assert orch.CENSUS_GOF_NOTHING_TO_TEST == gof.NOTHING_TO_TEST
    run_dir = tmp_path / "root" / "run_unit"
    run_dir.mkdir(parents=True)
    (tmp_path / "root" / "cross_model").mkdir()
    (tmp_path / "root" / "cross_model" / "census_goodness_of_fit.csv").write_text("old\n")
    layout = orch._resolve_run_layout(str(tmp_path / "root"), "run_unit", "unit-vf-lp")
    fake = tmp_path / "fake_gof.py"
    fake.write_text(f"import sys; sys.exit({gof.NOTHING_TO_TEST})\n")
    monkeypatch.setattr(orch, "CENSUS_GOF_SCRIPT", fake)

    class A:
        dry_run = False
        continue_on_error = False
    orch._run_census_gof_stage(A(), layout, [])            # does not raise
    rec = json.loads((run_dir / "run_layout_manifest.json").read_text())["census_gof"]
    assert rec["status"] == "nothing_to_test" and rec["existing_table_left_unchanged"] is True


def test_script_on_synthetic_census(tmp_path, monkeypatch):
    import census_goodness_of_fit as gof
    import sanity_vs_exact as sve
    style, label = "R3_dual_count", "unit-a-chat-grammar"
    for d in ("exact", "tables", "raw"):
        (tmp_path / d).mkdir()
    rng = np.random.default_rng(0)
    for scen in sve.SCENARIOS:
        cells = [(s, o, float(rng.uniform(0.05, 0.95))) for o in range(1, 6) for s in range(o + 1)]
        (tmp_path / "exact" / f"vf_{label}-lp__{scen}__{style}.json").write_text(json.dumps(
            {"compositions": {"red": [{"n_similar": s, "n_occupied": o, "p_move_effective": p,
                                       "ci95": [p, p]} for s, o, p in cells]}}))
        (tmp_path / "tables" / f"vf_{label}-sanity__{scen}__{style}.json").write_text(json.dumps(
            {"meta": {"created": "U"}, "compositions": {"red": []}}))
        with gzip.open(tmp_path / "raw" / f"vf_{label}-sanity__{scen}__{style}_raw.jsonl.gz", "wt") as fh:
            for s, o, p in cells:
                for i, mv in enumerate(rng.random(100) < p):
                    fh.write(json.dumps({"agent_role": "red", "n_similar": s, "n_occupied": o,
                                         "sample": i, "parse": "MOVE" if mv else "STAY"}) + "\n")
    monkeypatch.setattr(sve, "LOGPROB_TABLES_DIR", tmp_path / "exact")
    monkeypatch.setattr(sve, "SANITY_TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(sve, "SANITY_DIR", tmp_path)
    monkeypatch.setattr(gof, "SANITY_TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(sys, "argv", [SCRIPT, "--out-dir", str(tmp_path / "out"), "--dpi", "40"])
    assert gof.main() == 0
    import csv
    rows = list(csv.DictReader(open(tmp_path / "out" / "census_goodness_of_fit.csv")))
    assert len(rows) == 1 and rows[0]["label"] == label and rows[0]["verdict"] == "PASS"
    assert (rows[0]["seed"], rows[0]["bootstrap_reps"]) == ("0", "10000")
    assert (tmp_path / "out" / "census_goodness_of_fit.png").stat().st_size > 0


def test_script_without_any_census(tmp_path, monkeypatch):
    import census_goodness_of_fit as gof
    (tmp_path / "tables").mkdir()
    monkeypatch.setattr(gof, "SANITY_TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(sys, "argv", [SCRIPT, "--out-dir", str(tmp_path / "out")])
    assert gof.main() == gof.NOTHING_TO_TEST
    assert not (tmp_path / "out").exists()


def test_script_with_only_a_partial_census(tmp_path, monkeypatch):
    import census_goodness_of_fit as gof
    import sanity_vs_exact as sve
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "vf_unit-b-chat-grammar-sanity__baseline__R3_dual_count.json").write_text("{}")
    monkeypatch.setattr(gof, "SANITY_TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(sve, "SANITY_TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(sys, "argv", [SCRIPT, "--out-dir", str(tmp_path / "out")])
    assert gof.main() == gof.NOTHING_TO_TEST           # one scenario of six: not complete
    assert not (tmp_path / "out").exists()
