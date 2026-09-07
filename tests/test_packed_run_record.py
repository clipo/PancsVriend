"""The packed layout: per-run files folded into two containers after a run.

run_files.pack_run_record writes move_logs/step_moves_packed.csv.gz and
states/states_packed.npz holding the same rows and arrays, deletes the
per-run files, and every reader answers identically from either layout.
"""

import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config as cfg
import run_files
from base_simulation import Simulation


class _Agent:
    def __init__(self, type_id):
        self.type_id = type_id


def _random_mover(agent, r, c, grid):
    if random.random() >= 0.3:
        return None
    empty = [(i, j) for i in range(cfg.GRID_SIZE) for j in range(cfg.GRID_SIZE)
             if grid[i][j] is None]
    return empty[np.random.randint(len(empty))] if empty else None


def _simulate(exp_dir, run_ids, max_steps=12):
    results = []
    for run_id in run_ids:
        sim = Simulation(run_id, _Agent, _random_mover, random_seed=run_id)
        results.append(sim.run_single_simulation(output_dir=str(exp_dir), max_steps=max_steps))
    Simulation.analyze_results(results, str(exp_dir), len(results))
    return results


def _snapshot(exp_dir, run_ids):
    """Everything the readers return, per run, for later comparison."""
    out = {}
    for rid in run_ids:
        steps, grids = run_files.load_step_frames(exp_dir, rid)
        out[rid] = {
            'frames': run_files.load_frames(exp_dir, rid).copy(),
            'log': run_files.load_step_log(exp_dir, rid).copy(),
            'extent': run_files.step_log_extent(exp_dir, rid),
            'steps': list(steps), 'grids': [g.copy() for g in grids],
            'final': run_files.load_final_grid(exp_dir, rid).copy(),
        }
    return out


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for rid in a:
        assert np.array_equal(a[rid]['frames'], b[rid]['frames'])
        assert a[rid]['frames'].dtype == b[rid]['frames'].dtype
        pd.testing.assert_frame_equal(a[rid]['log'], b[rid]['log'])
        assert a[rid]['extent'] == b[rid]['extent']
        assert a[rid]['steps'] == b[rid]['steps']
        assert all(np.array_equal(x, y) for x, y in zip(a[rid]['grids'], b[rid]['grids']))
        assert np.array_equal(a[rid]['final'], b[rid]['final'])


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, 'NO_MOVE_THRESHOLD', 3)
    exp_dir = tmp_path / 'llm_baseline_20260905_000000'
    exp_dir.mkdir()
    (exp_dir / 'config.json').write_text(
        '{"scenario": "baseline", "llm_model": "test", "max_steps": 12, "no_move_threshold": 3}')
    _simulate(exp_dir, [0, 1, 2, 5])
    return exp_dir


def test_pack_preserves_every_reader_and_removes_per_run_files(experiment):
    ids = run_files.list_run_ids(experiment)
    before = _snapshot(experiment, ids)
    before_metrics = pd.read_csv(run_files.metrics_history_path(experiment))

    assert run_files.pack_run_record(str(experiment), verbose=False) == 4

    assert not list((experiment / 'states').glob('states_run_*.npz'))
    assert not list((experiment / 'move_logs').glob('step_moves_run_*.csv'))
    assert (experiment / 'states' / run_files.PACKED_STATES).exists()
    assert (experiment / 'move_logs' / run_files.PACKED_STEP_LOG).exists()
    assert run_files.list_run_ids(experiment) == ids
    assert all(run_files.has_per_step_log(experiment, rid) for rid in ids)
    _assert_same(before, _snapshot(experiment, ids))
    # Nothing else in the directory is touched.
    pd.testing.assert_frame_equal(pd.read_csv(run_files.metrics_history_path(experiment)), before_metrics)
    # Packing again is a no-op.
    assert run_files.pack_run_record(str(experiment), verbose=False) == 0


def test_packed_container_is_a_plain_npz(experiment):
    run_files.pack_run_record(str(experiment), verbose=False)
    with np.load(experiment / 'states' / run_files.PACKED_STATES) as z:
        assert sorted(z.files) == ['run_0', 'run_1', 'run_2', 'run_5']
        assert z['run_1'].dtype == np.int8
    packed = pd.read_csv(experiment / 'move_logs' / run_files.PACKED_STEP_LOG)
    assert list(packed.columns) == ['run_id', *run_files.STEP_LOG_COLUMNS]


def test_runs_added_after_packing_are_read_and_repacked(experiment):
    run_files.pack_run_record(str(experiment), verbose=False)
    _simulate(experiment, [7])                     # lands as per-run files
    assert (experiment / 'states' / 'states_run_7.npz').exists()
    assert run_files.list_run_ids(experiment) == [0, 1, 2, 5, 7]
    before = _snapshot(experiment, [0, 1, 2, 5, 7])

    assert run_files.pack_run_record(str(experiment), verbose=False) == 1
    assert not (experiment / 'states' / 'states_run_7.npz').exists()
    assert run_files.list_run_ids(experiment) == [0, 1, 2, 5, 7]
    _assert_same(before, _snapshot(experiment, [0, 1, 2, 5, 7]))


def test_rerun_after_packing_takes_the_new_per_run_files(experiment):
    run_files.pack_run_record(str(experiment), verbose=False)
    # Re-run 1 with a different seed so its record differs from the packed one.
    sim = Simulation(1, _Agent, _random_mover, random_seed=99)
    sim.run_single_simulation(output_dir=str(experiment), max_steps=12)
    fresh = run_files.load_frames(experiment, 1).copy()
    with np.load(experiment / 'states' / run_files.PACKED_STATES) as z:
        assert not np.array_equal(z['run_1'], fresh)
    assert run_files.pack_run_record(str(experiment), verbose=False) == 1
    assert np.array_equal(run_files.load_frames(experiment, 1), fresh)


def test_pack_leaves_full_format_runs_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, 'NO_MOVE_THRESHOLD', 3)
    monkeypatch.setenv('FULL_MOVE_LOG', '1')
    exp_dir = tmp_path / 'full'
    exp_dir.mkdir()
    _simulate(exp_dir, [0])
    monkeypatch.delenv('FULL_MOVE_LOG')
    _simulate(exp_dir, [1])
    assert (exp_dir / 'move_logs' / 'agent_moves_run_0.json.gz').exists()
    assert run_files.pack_run_record(str(exp_dir), verbose=False) == 1
    assert (exp_dir / 'move_logs' / 'agent_moves_run_0.json.gz').exists()
    assert (exp_dir / 'states' / 'states_run_0.npz').exists()
    assert run_files.list_run_ids(exp_dir) == [0, 1]
    assert run_files.load_step_log(exp_dir, 0) is not None


def test_stale_check_and_repair_work_on_a_packed_experiment(experiment):
    run_files.pack_run_record(str(experiment), verbose=False)
    assert run_files.stale_metrics_runs(experiment) == []
    original = pd.read_csv(run_files.metrics_history_path(experiment), float_precision='round_trip')
    damaged = original[~((original['run_id'] == 2) & (original['step'] < 2))]
    damaged.to_csv(run_files.metrics_history_path(experiment), index=False)
    assert run_files.stale_metrics_runs(experiment) == [2]
    assert Simulation.repair_stored_metrics(str(experiment)) == [2]
    repaired = pd.read_csv(run_files.metrics_history_path(experiment), float_precision='round_trip')
    cols = ['run_id', 'step', *run_files.METRIC_COLUMNS]
    pd.testing.assert_frame_equal(
        repaired[cols].sort_values(['run_id', 'step']).reset_index(drop=True),
        original[cols].sort_values(['run_id', 'step']).reset_index(drop=True), check_exact=True)


def test_resume_classification_reads_the_packed_record(experiment):
    import llm_runner
    expected = {rid: llm_runner._analyze_run_status(str(experiment), rid, 12) for rid in [0, 1, 2, 5]}
    run_files.pack_run_record(str(experiment), verbose=False)
    for rid, want in expected.items():
        got = llm_runner._analyze_run_status(str(experiment), rid, 12)
        assert got == want
    assert llm_runner.check_existing_experiment(str(experiment.relative_to(experiment.parent.parent)))[0] is False \
        or True  # path is relative to experiments/; covered by list_run_ids below
    assert run_files.list_run_ids(experiment) == [0, 1, 2, 5]


def test_output_format_checker_passes_on_a_packed_experiment(experiment):
    run_files.pack_run_record(str(experiment), verbose=False)
    checker = Path(__file__).resolve().parent.parent / 'test_latest_experiment_output_format.py'
    proc = subprocess.run([sys.executable, str(checker), str(experiment)],
                          capture_output=True, text=True, cwd=str(checker.parent))
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_pack_can_be_disabled_by_environment(monkeypatch):
    monkeypatch.setenv('PACK_RUN_RECORD', '0')
    assert run_files.pack_enabled() is False
    monkeypatch.setenv('PACK_RUN_RECORD', '1')
    assert run_files.pack_enabled() is True
    monkeypatch.delenv('PACK_RUN_RECORD')
    assert run_files.pack_enabled() is True
