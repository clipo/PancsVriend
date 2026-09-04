"""A run's record on disk: per-step counts + per-step frames, or per-move.

By default a run writes move_logs/step_moves_run_<id>.csv (one row per step)
and states/states_run_<id>.npz with frame 0 = initial grid, frame k+1 = grid
after step k. With full_move_log=True (live-LLM runs, --full-move-log) it
writes the older per-move agent_moves_run_<id>.json.gz with one frame per
record. run_files reads both and hands back the same per-step tables, which
is what these tests pin: the two formats must agree with each other and with
what the run measured live, and the 12k legacy logs must still load.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg  # noqa: E402
import run_files  # noqa: E402
from Agent import Agent  # noqa: E402
from base_simulation import Simulation, _load_single_run_result  # noqa: E402
from llm_runner import _analyze_run_status  # noqa: E402


def mechanical(agent, r, c, grid):
    return agent.random_response(r, c, grid)


def _run(tmp_path, size, n_a, n_b, monkeypatch, run_id=0, max_steps=40, full_move_log=False):
    monkeypatch.setattr(cfg, "GRID_SIZE", size)
    monkeypatch.setattr(cfg, "NUM_TYPE_A", n_a)
    monkeypatch.setattr(cfg, "NUM_TYPE_B", n_b)
    monkeypatch.delenv("FULL_MOVE_LOG", raising=False)
    sim = Simulation(run_id=run_id, agent_factory=Agent, decision_func=mechanical,
                     random_seed=run_id, full_move_log=full_move_log)
    sim.run_single_simulation(output_dir=str(tmp_path), max_steps=max_steps)
    return sim


def _task(tmp_path, run_id=0):
    return (run_id, str(tmp_path), cfg.NO_MOVE_THRESHOLD)


# ---------------------------------------------------------------------------
# Per-step format (the default)
# ---------------------------------------------------------------------------

def test_default_run_writes_the_per_step_format(tmp_path, monkeypatch):
    sim = _run(tmp_path, 10, 40, 40, monkeypatch)
    assert Path(run_files.step_log_path(tmp_path, 0)).exists()
    assert not Path(run_files.move_log_path(tmp_path, 0)).exists()

    step_log = pd.read_csv(run_files.step_log_path(tmp_path, 0))
    assert list(step_log.columns) == list(run_files.STEP_LOG_COLUMNS)
    assert step_log["step"].tolist() == list(range(sim.step + 1))
    # every decision has exactly one reason
    reasons = step_log[list(run_files.REASONS)].sum(axis=1)
    assert (reasons == step_log["decisions"]).all()
    assert (step_log["moved"] == step_log["successful_move"]).all()
    assert (step_log["parse_failed"] == 0).all()


def test_one_frame_per_step_plus_the_initial_grid(tmp_path, monkeypatch):
    """frame 0 is the initial grid, frame k+1 the grid after step k."""
    sim = _run(tmp_path, 10, 40, 40, monkeypatch)
    frames = run_files.load_frames(tmp_path, 0)
    assert frames.shape[0] == sim.step + 2
    assert len(sim.states) == sim.step + 2

    steps, grids = run_files.load_step_frames(tmp_path, 0)
    assert steps == list(range(sim.step + 1))
    assert np.array_equal(grids[-1], frames[-1])
    assert np.array_equal(run_files.load_final_grid(tmp_path, 0), sim._grid_to_int())


def test_rebuilt_metrics_match_the_live_ones(tmp_path, monkeypatch):
    """Reconstructing from disk must reproduce what the run itself measured."""
    sim = _run(tmp_path, 10, 40, 40, monkeypatch)
    rebuilt = _load_single_run_result(_task(tmp_path))

    live = {int(m["step"]): m for m in sim.metrics_history}
    assert rebuilt["final_step"] == sim.step
    assert rebuilt["converged"] == sim.converged
    assert rebuilt["convergence_step"] == sim.convergence_step
    checked = 0
    for row in rebuilt["metrics_history"]:
        step = int(row["step"])
        assert step in live
        for metric in ("clusters", "share", "ghetto_rate", "dissimilarity_index"):
            assert row[metric] == pytest.approx(live[step][metric], abs=1e-9)
        checked += 1
    assert checked == len(live) > 1


# ---------------------------------------------------------------------------
# Full (per-move) format: --full-move-log and live-LLM runs
# ---------------------------------------------------------------------------

def test_full_move_log_writes_one_state_frame_per_move_record(tmp_path, monkeypatch):
    """The 1:1 pairing the per-move readers depend on.

    update_agents() calls log_agent_move() and log_state_per_move() together on
    each of its branches, so the two lists advance in lockstep. If a future edit
    adds a branch and forgets one of the pair, frame i stops being record i's
    grid and every rebuilt metric silently shifts.
    """
    sim = _run(tmp_path, 10, 40, 40, monkeypatch, full_move_log=True)
    assert len(sim.states) == len(sim.agent_move_log)
    assert not Path(run_files.step_log_path(tmp_path, 0)).exists()

    records = run_files.load_move_log_json(tmp_path, 0)
    frames = run_files.load_frames(tmp_path, 0)
    assert len(records) == frames.shape[0]
    assert records and all("grid" not in r for r in records)


def test_full_move_log_env_var_switches_the_format(tmp_path, monkeypatch):
    monkeypatch.setenv("FULL_MOVE_LOG", "1")
    monkeypatch.setattr(cfg, "GRID_SIZE", 10)
    monkeypatch.setattr(cfg, "NUM_TYPE_A", 40)
    monkeypatch.setattr(cfg, "NUM_TYPE_B", 40)
    sim = Simulation(run_id=0, agent_factory=Agent, decision_func=mechanical, random_seed=0)
    assert sim.full_move_log
    sim.run_single_simulation(output_dir=str(tmp_path), max_steps=5)
    assert Path(run_files.move_log_path(tmp_path, 0)).exists()


def test_the_two_formats_describe_the_same_run(tmp_path, monkeypatch):
    """Same run_id, both formats: identical step log, frames and metrics."""
    step_dir, full_dir = tmp_path / "step", tmp_path / "full"
    _run(step_dir, 10, 40, 40, monkeypatch, run_id=3)
    _run(full_dir, 10, 40, 40, monkeypatch, run_id=3, full_move_log=True)

    from_step = run_files.load_step_log(step_dir, 3)
    from_full = run_files.load_step_log(full_dir, 3)
    pd.testing.assert_frame_equal(from_step, from_full, check_dtype=False)

    steps_s, grids_s = run_files.load_step_frames(step_dir, 3)
    steps_f, grids_f = run_files.load_step_frames(full_dir, 3)
    assert steps_s == steps_f
    assert all(np.array_equal(a, b) for a, b in zip(grids_s, grids_f))

    assert (_load_single_run_result((3, str(step_dir), cfg.NO_MOVE_THRESHOLD))
            == _load_single_run_result((3, str(full_dir), cfg.NO_MOVE_THRESHOLD)))


# ---------------------------------------------------------------------------
# Legacy logs already on disk
# ---------------------------------------------------------------------------

def test_legacy_move_logs_with_an_inline_grid_still_load(tmp_path, monkeypatch):
    """Move logs written before 2026-09-01 carry a grid copy in every record."""
    _run(tmp_path, 10, 40, 40, monkeypatch, full_move_log=True)
    records = run_files.load_move_log_json(tmp_path, 0)
    frames = run_files.load_frames(tmp_path, 0)

    legacy_dir = tmp_path / "legacy"
    (legacy_dir / "move_logs").mkdir(parents=True)     # no states/ at all
    for i, record in enumerate(records):
        record["grid"] = frames[i].tolist()
    with gzip.open(legacy_dir / "move_logs" / "agent_moves_run_0.json.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps(records, separators=(",", ":")))

    from_npz = _load_single_run_result(_task(tmp_path))
    # No states file: the inline copy has to carry it on its own.
    from_inline = _load_single_run_result(_task(legacy_dir))
    assert from_inline["metrics_history"] == from_npz["metrics_history"]


def test_list_run_ids_sees_both_formats(tmp_path, monkeypatch):
    _run(tmp_path, 10, 40, 40, monkeypatch, run_id=1, max_steps=3)
    _run(tmp_path, 10, 40, 40, monkeypatch, run_id=2, max_steps=3, full_move_log=True)
    (tmp_path / "move_logs" / "agent_moves_run_7.csv").write_text("step,moved\n0,True\n")
    assert run_files.list_run_ids(tmp_path) == [1, 2, 7]


def test_reanalysis_ignores_the_ambient_grid_size(tmp_path, monkeypatch):
    """A 10x10 run re-analysed under a 20x20 config must still be read as 10x10.

    The grid shape used to come from cfg.GRID_SIZE, so this case padded 300
    cells with None and returned metrics for a three-quarters-empty grid —
    plausible numbers, no error. It now comes from the data.
    """
    sim = _run(tmp_path, 10, 40, 40, monkeypatch)
    as_recorded = _load_single_run_result(_task(tmp_path))

    monkeypatch.setattr(cfg, "GRID_SIZE", 20)
    monkeypatch.setattr(cfg, "NUM_TYPE_A", 160)
    monkeypatch.setattr(cfg, "NUM_TYPE_B", 160)
    under_wrong_config = _load_single_run_result(_task(tmp_path))

    assert under_wrong_config["metrics_history"] == as_recorded["metrics_history"]
    live = {int(m["step"]): m for m in sim.metrics_history}
    final = under_wrong_config["metrics_history"][-1]
    assert final["share"] == pytest.approx(live[int(final["step"])]["share"], abs=1e-9)


# ---------------------------------------------------------------------------
# Resume bookkeeping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("full_move_log", [False, True])
def test_resume_status_is_read_the_same_from_both_formats(tmp_path, monkeypatch, full_move_log):
    """An aborted run resumes from its last grid at last_step + 1."""
    sim = _run(tmp_path, 10, 40, 40, monkeypatch, max_steps=3, full_move_log=full_move_log)
    assert not sim.converged
    status = _analyze_run_status(str(tmp_path), 0, max_steps=1000)
    assert status["status"] == "aborted"
    assert status["last_step"] == 2
    assert status["next_step"] == 3
    assert np.array_equal(status["seed_grid"], sim._grid_to_int())

    at_cap = _analyze_run_status(str(tmp_path), 0, max_steps=3)
    assert at_cap["status"] == "reached_max"
    assert at_cap["seed_grid"] is None


def test_resume_status_reports_convergence(tmp_path, monkeypatch):
    sim = _run(tmp_path, 10, 40, 40, monkeypatch, max_steps=200)
    status = _analyze_run_status(str(tmp_path), 0, max_steps=200)
    assert status["status"] == ("converged" if sim.converged else "reached_max")
    assert status["last_step"] == sim.step
    if sim.converged:
        assert status["convergence_step"] == sim.convergence_step


@pytest.mark.parametrize("full_move_log", [False, True])
def test_resumed_run_keeps_the_steps_before_the_abort(tmp_path, monkeypatch, full_move_log):
    """Resume used to overwrite the files with the resumed segment only."""
    sim = _run(tmp_path, 10, 40, 40, monkeypatch, full_move_log=full_move_log)
    frames = run_files.load_frames(tmp_path, 0)
    step_log = run_files.load_step_log(tmp_path, 0)
    assert sim.step >= 4

    # A periodic save after step 1 looks like this; the process then dies.
    if full_move_log:
        records = [r for r in run_files.load_move_log_json(tmp_path, 0) if r["step"] < 2]
        sim.agent_move_log, sim.states = records, list(frames[:len(records)])
    else:
        sim.step_log, sim.states = step_log[step_log["step"] < 2].to_dict("records"), list(frames[:3])
    sim.save_agent_move_log(str(tmp_path))
    sim.save_states(str(tmp_path))
    seed = run_files.load_final_grid(tmp_path, 0)
    assert np.array_equal(seed, frames[len(sim.states) - 1])

    resumed = Simulation(run_id=0, agent_factory=Agent, decision_func=mechanical, random_seed=0,
                         initial_int_grid=seed, initial_step=2, full_move_log=full_move_log)
    resumed.preload_record(str(tmp_path))
    resumed.run_single_simulation(output_dir=str(tmp_path), max_steps=40)

    after = run_files.load_step_log(tmp_path, 0)
    assert after["step"].tolist() == list(range(resumed.step + 1))
    pd.testing.assert_frame_equal(after.iloc[:2], step_log.iloc[:2], check_dtype=False)
    new_frames = run_files.load_frames(tmp_path, 0)
    assert np.array_equal(new_frames[:len(sim.states)], frames[:len(sim.states)])
    steps, grids = run_files.load_step_frames(tmp_path, 0)
    assert steps == list(range(resumed.step + 1))
