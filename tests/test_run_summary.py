"""Tests for the per-run summary CSV and the shared convergence definition."""

import gzip
import json
import os

import numpy as np
import pandas as pd
import pytest

import config as cfg
import run_files
from base_simulation import Simulation, convergence_from_step_moves
from run_summary import (
    METRIC_COLUMNS,
    RUN_SUMMARY_FILENAME,
    SUMMARY_COLUMNS,
    build_run_summary,
    combine_run_summaries,
    write_run_summary,
)


# --- the shared convergence definition --------------------------------------

def test_convergence_reports_first_step_of_window():
    # Steps 5-9 are the no-move window; the first of them is the answer.
    step_moves = {0: 3, 1: 2, 2: 1, 3: 4, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    converged, first, detected = convergence_from_step_moves(step_moves, 5)
    assert converged is True
    assert first == 5
    assert detected == 9
    assert detected == first + 5 - 1


def test_convergence_requires_the_streak_to_reach_the_end():
    # Converged early, then an agent moved again: not converged.
    step_moves = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 2, 6: 0, 7: 0}
    converged, first, detected = convergence_from_step_moves(step_moves, 5)
    assert converged is False
    assert first is None and detected is None


def test_convergence_needs_a_full_window():
    converged, first, _ = convergence_from_step_moves({0: 0, 1: 0, 2: 0}, 5)
    assert converged is False and first is None


def test_convergence_on_empty_log():
    assert convergence_from_step_moves({}, 5) == (False, None, None)


def test_run_step_records_first_step_of_window(monkeypatch):
    """The live loop and the log rebuild must agree on the same step."""
    monkeypatch.setattr(cfg, 'NO_MOVE_THRESHOLD', 5)

    class _Agent:
        def __init__(self, type_id):
            self.type_id = type_id

    sim = Simulation(run_id=0, agent_factory=_Agent, decision_func=lambda *a: None)
    sim.decision_func = lambda agent, r, c, grid: None  # nobody ever moves

    while not sim.converged and sim.step < 20:
        sim.run_step()

    assert sim.converged is True
    assert sim.convergence_step == 0
    assert sim.step == sim.convergence_step + cfg.NO_MOVE_THRESHOLD - 1

    rebuilt = convergence_from_step_moves({step: 0 for step in range(sim.step + 1)},
                                          cfg.NO_MOVE_THRESHOLD)
    assert rebuilt[1] == sim.convergence_step


# --- the summary CSV ---------------------------------------------------------

def _metrics_row(run_id, step, **overrides):
    row = {
        'run_id': run_id,
        'step': step,
        'clusters': 10 + step,
        'switch_rate': 0.5,
        'distance': 1.5,
        'mix_deviation': 0.2,
        'share': 0.5,
        'ghetto_rate': 0.1,
        'dissimilarity_index': 0.3 + step / 100.0,
    }
    row.update(overrides)
    return row


def _write_experiment(tmp_path, name='llm_baseline_20260901_120000', scenario='baseline',
                      max_steps=1000, metrics=None, convergence=None):
    exp_dir = tmp_path / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'config.json').write_text(json.dumps({
        'scenario': scenario,
        'llm_model': 'test-model:latest',
        'max_steps': max_steps,
        'no_move_threshold': 5,
    }), encoding='utf-8')
    if metrics is not None:
        pd.DataFrame(metrics).to_csv(exp_dir / 'metrics_history.csv', index=False)
    if convergence is not None:
        pd.DataFrame(convergence).to_csv(exp_dir / 'convergence_summary.csv', index=False)
    return exp_dir


def _write_move_log(exp_dir, run_id, moves_per_step, grid=None, fmt='full'):
    """A run in which `moves_per_step[i]` agents moved at step i.

    fmt='full' writes the legacy per-move JSON with an inline grid copy in
    every record (what the 12k runs on disk look like); fmt='step' writes the
    per-step CSV plus one frame per step in states_run_<id>.npz.
    """
    move_logs = exp_dir / 'move_logs'
    move_logs.mkdir(exist_ok=True)
    grid = grid if grid is not None else np.zeros((10, 10), dtype=int).tolist()
    if fmt == 'step':
        rows = [dict(run_files.new_step_row(step), decisions=max(n, 1), moved=n, successful_move=n)
                for step, n in enumerate(moves_per_step)]
        pd.DataFrame(rows, columns=list(run_files.STEP_LOG_COLUMNS)).to_csv(
            run_files.step_log_path(exp_dir, run_id), index=False)
        (exp_dir / 'states').mkdir(exist_ok=True)
        np.savez_compressed(run_files.states_path(exp_dir, run_id),
                            states=np.array([grid] * (len(moves_per_step) + 1)))
        return
    records = []
    for step, n_moves in enumerate(moves_per_step):
        for i in range(max(n_moves, 1)):
            records.append({'step': step, 'moved': i < n_moves, 'grid': grid})
    with gzip.open(move_logs / f'agent_moves_run_{run_id}.json.gz', 'wt', encoding='utf-8') as fh:
        json.dump(records, fh)


def test_summary_columns_and_final_step_invariant(tmp_path):
    metrics = [_metrics_row(0, step) for step in range(9)]
    convergence = [{'run_id': 0, 'converged': True, 'convergence_step': 4, 'final_step': 8}]
    exp_dir = _write_experiment(tmp_path, metrics=metrics, convergence=convergence)

    df = write_run_summary(str(exp_dir))

    assert list(df.columns) == SUMMARY_COLUMNS
    assert (exp_dir / RUN_SUMMARY_FILENAME).exists()

    row = df.iloc[0]
    assert row['run_id'] == 0
    assert row['scenario'] == 'baseline'
    assert row['converged'] is True or row['converged'] == True  # noqa: E712
    assert row['convergence_step'] == 4
    assert row['final_step'] == 8
    # The whole point: final step is the last of the five no-move steps.
    assert row['final_step'] == row['convergence_step'] + cfg.NO_MOVE_THRESHOLD - 1
    assert row['n_steps'] == 9
    assert row['stop_reason'] == 'converged'
    assert row['llm_model'] == 'test-model:latest'
    assert row['metrics_source'] == 'live'


def test_metrics_are_taken_from_the_final_step(tmp_path):
    metrics = [_metrics_row(0, step) for step in range(9)]
    convergence = [{'run_id': 0, 'converged': True, 'convergence_step': 4, 'final_step': 8}]
    exp_dir = _write_experiment(tmp_path, metrics=metrics, convergence=convergence)

    row = build_run_summary(str(exp_dir)).iloc[0]

    last = metrics[-1]
    for column in METRIC_COLUMNS:
        assert row[column] == pytest.approx(last[column])


def test_non_converged_run_has_no_convergence_step(tmp_path):
    metrics = [_metrics_row(0, step) for step in range(1000)]
    convergence = [{'run_id': 0, 'converged': False, 'convergence_step': None, 'final_step': 999}]
    exp_dir = _write_experiment(tmp_path, metrics=metrics, convergence=convergence, max_steps=1000)

    row = build_run_summary(str(exp_dir)).iloc[0]

    assert bool(row['converged']) is False
    assert pd.isna(row['convergence_step'])
    assert row['final_step'] == 999
    assert row['stop_reason'] == 'max_steps'


@pytest.mark.parametrize('fmt', ['full', 'step'])
def test_placeholder_row_is_rebuilt_from_the_move_log(tmp_path, fmt):
    """The .npz resume path writes converged=True, final_step='unknown'."""
    convergence = [{'run_id': 0, 'converged': True, 'convergence_step': '', 'final_step': 'unknown'}]
    exp_dir = _write_experiment(tmp_path, metrics=[_metrics_row(0, 8)], convergence=convergence)
    # Steps 0-3 had movement, steps 4-8 did not.
    _write_move_log(exp_dir, 0, [2, 3, 1, 2, 0, 0, 0, 0, 0], fmt=fmt)

    row = build_run_summary(str(exp_dir)).iloc[0]

    assert bool(row['converged']) is True
    assert row['convergence_step'] == 4
    assert row['final_step'] == 8
    assert row['metrics_source'] == 'rebuilt'
    assert row['stop_reason'] == 'converged'


@pytest.mark.parametrize('stored_convergence_step', [8, None])
def test_legacy_last_of_window_row_is_corrected(tmp_path, stored_convergence_step):
    """Pre-2026-09-01 live rows stored the LAST no-move step, so they came out
    with convergence_step == final_step; the .npz resume placeholder left the
    step empty instead. Both shift onto the first step of the window."""
    convergence = [{'run_id': 0, 'converged': True,
                    'convergence_step': stored_convergence_step, 'final_step': 8}]
    exp_dir = _write_experiment(tmp_path, metrics=[_metrics_row(0, 8)], convergence=convergence)

    row = build_run_summary(str(exp_dir)).iloc[0]

    assert row['convergence_step'] == 4
    assert row['final_step'] == 8


def test_final_step_is_the_last_simulated_step_at_the_cap(tmp_path):
    """The live loop leaves self.step one past the last simulated step when a
    run stops at max_steps, so the stored final_step (10) overshoots."""
    metrics = [_metrics_row(0, step) for step in range(10)]
    convergence = [{'run_id': 0, 'converged': False, 'convergence_step': None, 'final_step': 10}]
    exp_dir = _write_experiment(tmp_path, metrics=metrics, convergence=convergence, max_steps=10)

    row = build_run_summary(str(exp_dir)).iloc[0]

    assert row['final_step'] == 9
    assert row['n_steps'] == 10
    assert row['stop_reason'] == 'max_steps'
    assert row['clusters'] == metrics[-1]['clusters']


def test_dissimilarity_is_recomputed_when_column_is_absent(tmp_path):
    """Experiments predating the metric's addition to Metrics.py."""
    metrics = [{k: v for k, v in _metrics_row(0, 8).items() if k != 'dissimilarity_index'}]
    convergence = [{'run_id': 0, 'converged': True, 'convergence_step': 4, 'final_step': 8}]
    exp_dir = _write_experiment(tmp_path, metrics=metrics, convergence=convergence)

    # Fully segregated final grid: type 0 on the top half, type 1 on the bottom.
    grid = np.full((10, 10), -1, dtype=np.int8)
    grid[:5, :] = 0
    grid[5:, :] = 1
    states_dir = exp_dir / 'states'
    states_dir.mkdir()
    np.savez_compressed(states_dir / 'states_run_0.npz', states=np.array([grid]))

    row = build_run_summary(str(exp_dir)).iloc[0]

    assert row['dissimilarity_index'] is not None
    assert row['dissimilarity_index'] > 0.5


def test_analyze_results_writes_the_summary(tmp_path):
    """The automatic hook: every runner gets run_summary.csv for free."""
    results = [{
        'run_id': 0,
        'scenario': 'race_white_black',
        'converged': True,
        'convergence_step': 4,
        'final_step': 8,
        'metrics_history': [_metrics_row(0, step) for step in range(9)],
    }]
    output_dir = tmp_path / 'exp'
    output_dir.mkdir()

    Simulation.analyze_results(results, str(output_dir), len(results))

    df = pd.read_csv(output_dir / RUN_SUMMARY_FILENAME)
    assert list(df.columns) == SUMMARY_COLUMNS
    assert len(df) == 1
    assert df.iloc[0]['scenario'] == 'race_white_black'
    assert df.iloc[0]['convergence_step'] == 4
    assert df.iloc[0]['final_step'] == 8


def test_combine_run_summaries_spans_scenarios(tmp_path):
    for name, scenario in (('llm_baseline_1', 'baseline'), ('llm_race_1', 'race_white_black')):
        _write_experiment(
            tmp_path, name=name, scenario=scenario,
            metrics=[_metrics_row(run_id, step) for run_id in (0, 1) for step in range(9)],
            convergence=[{'run_id': run_id, 'converged': True, 'convergence_step': 4, 'final_step': 8}
                         for run_id in (0, 1)],
        )

    out_path = tmp_path / 'combined.csv'
    combined = combine_run_summaries(
        [str(tmp_path / 'llm_baseline_1'), str(tmp_path / 'llm_race_1')], out_path=out_path)

    assert out_path.exists()
    assert len(combined) == 4
    assert set(combined['scenario']) == {'baseline', 'race_white_black'}
    assert set(combined['experiment']) == {'llm_baseline_1', 'llm_race_1'}
    assert list(combined.columns) == SUMMARY_COLUMNS


def test_missing_run_data_does_not_raise(tmp_path):
    exp_dir = _write_experiment(tmp_path, metrics=None, convergence=None)
    df = write_run_summary(str(exp_dir))
    assert df.empty
    assert os.path.exists(exp_dir / RUN_SUMMARY_FILENAME)


def test_analysis_step_writes_combined_summary(tmp_path, monkeypatch):
    """The run_all_scenario_analysis step: per-experiment files + the roll-up."""
    import sys
    from pathlib import Path
    # run_all_scenario_analysis imports experiment_list_for_analysis as a
    # top-level module; it is normally invoked from analysis_tools/.
    analysis_tools_dir = str(Path(__file__).resolve().parent.parent / 'analysis_tools')
    if analysis_tools_dir not in sys.path:
        sys.path.insert(0, analysis_tools_dir)

    import analysis_tools.run_all_scenario_analysis as runner
    import experiment_list_for_analysis as experiment_list

    experiments_root = tmp_path / 'experiments'
    for folder, scenario in (('llm_baseline_x', 'baseline'), ('llm_race_x', 'race_white_black')):
        _write_experiment(
            experiments_root, name=folder, scenario=scenario,
            metrics=[_metrics_row(0, step) for step in range(9)],
            convergence=[{'run_id': 0, 'converged': True, 'convergence_step': 4, 'final_step': 8}],
        )

    monkeypatch.setattr(experiment_list, 'SCENARIOS',
                        {'llm_baseline': 'llm_baseline_x', 'race_white_black': 'llm_race_x'},
                        raising=False)

    reports_dir = tmp_path / 'reports'
    combined = runner.build_run_summaries_for_selection(
        reports_dir, experiments_dir=experiments_root, verbose=False)

    assert (experiments_root / 'llm_baseline_x' / RUN_SUMMARY_FILENAME).exists()
    assert (experiments_root / 'llm_race_x' / RUN_SUMMARY_FILENAME).exists()

    out_path = reports_dir / 'run_summary_by_run_all_scenarios.csv'
    assert out_path.exists()
    assert len(combined) == 2
    # scenario_key distinguishes llm_baseline from a mechanical baseline, which
    # the per-experiment config.json scenario ('baseline') cannot.
    assert set(combined['scenario_key']) == {'llm_baseline', 'race_white_black'}
    assert list(combined.columns) == ['run_id', 'scenario_key'] + SUMMARY_COLUMNS[1:]


def test_scenario_falls_back_to_the_folder_name(tmp_path):
    """The mechanical baseline's config.json predates the 'scenario' key."""
    exp_dir = tmp_path / 'baseline_20250729_174459'
    exp_dir.mkdir()
    (exp_dir / 'config.json').write_text(json.dumps({'max_steps': 1000, 'no_move_threshold': 5}),
                                         encoding='utf-8')
    pd.DataFrame([_metrics_row(0, 8)]).to_csv(exp_dir / 'metrics_history.csv', index=False)
    pd.DataFrame([{'run_id': 0, 'converged': True, 'convergence_step': 4, 'final_step': 8}]).to_csv(
        exp_dir / 'convergence_summary.csv', index=False)

    row = build_run_summary(str(exp_dir)).iloc[0]

    assert row['scenario'] == 'baseline'
    assert row['llm_model'] == ''  # no LLM was involved; do not invent one


def test_scenario_fallback_strips_the_llm_prefix(tmp_path):
    from run_summary import _scenario_from_experiment_name

    assert _scenario_from_experiment_name('llm_race_white_black_20251119_142602') == 'race_white_black'
    assert _scenario_from_experiment_name('baseline_20250729_174459') == 'baseline'


@pytest.mark.parametrize('fmt', ['full', 'step'])
def test_run_capped_mid_streak_reports_the_max_step(tmp_path, fmt):
    """3 no-move steps at the cap is not convergence, but final_step is still
    the last step simulated — the lower of "convergence + threshold - 1" and
    the cap."""
    metrics = [_metrics_row(0, step) for step in range(1000)]
    convergence = [{'run_id': 0, 'converged': False, 'convergence_step': None, 'final_step': 1000}]
    exp_dir = _write_experiment(tmp_path, metrics=metrics, convergence=convergence, max_steps=1000)
    # Steps 997-999 had no movement: a streak of 3, short of the threshold of 5.
    _write_move_log(exp_dir, 0, [1] * 997 + [0, 0, 0], fmt=fmt)

    row = build_run_summary(str(exp_dir)).iloc[0]

    assert bool(row['converged']) is False
    assert pd.isna(row['convergence_step'])
    assert row['final_step'] == 999
    assert row['n_steps'] == 1000
    assert row['stop_reason'] == 'max_steps'


def test_rebuilt_rows_are_reused_instead_of_reparsing_the_move_log(tmp_path):
    """Rebuilding a placeholder run parses its whole move log; that answer is
    deterministic, so the analysis pipeline should not redo it every pass."""
    convergence = [{'run_id': 0, 'converged': True, 'convergence_step': '', 'final_step': 'unknown'}]
    exp_dir = _write_experiment(tmp_path, metrics=None, convergence=convergence)
    _write_move_log(exp_dir, 0, [2, 3, 1, 2, 0, 0, 0, 0, 0])

    first = write_run_summary(str(exp_dir), verbose=False).iloc[0]
    assert first['metrics_source'] == 'rebuilt'
    assert first['convergence_step'] == 4

    # Delete the move log: a reuse pass must not need it any more.
    (exp_dir / 'move_logs' / 'agent_moves_run_0.json.gz').unlink()
    second = write_run_summary(str(exp_dir), verbose=False).iloc[0]
    assert second['convergence_step'] == 4
    assert second['final_step'] == 8
    assert second['metrics_source'] == 'rebuilt'

    # ...but --force, with no log to read, honestly reports nothing recovered.
    forced = write_run_summary(str(exp_dir), verbose=False, reuse_existing=False).iloc[0]
    assert forced['stop_reason'] == 'incomplete'
