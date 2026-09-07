"""The stored metrics_history is a cache of per-frame metrics; the analysis
pipeline verifies it against the run record instead of regenerating it.

Covers run_files.stale_metrics_runs / step_log_extent,
Simulation.repair_stored_metrics, the stored-column path of
dissimilarity_index_over_time, and plot_style.steps_to_fraction_of_final.
"""

import gzip
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config as cfg
import run_files
from base_simulation import Simulation

_ANALYSIS_TOOLS = str(Path(__file__).resolve().parent.parent / 'analysis_tools')
if _ANALYSIS_TOOLS not in sys.path:
    sys.path.insert(0, _ANALYSIS_TOOLS)


class _Agent:
    def __init__(self, type_id):
        self.type_id = type_id


def _random_mover(agent, r, c, grid):
    """Move to a random empty cell with probability 0.3 (seeded per run)."""
    if random.random() >= 0.3:
        return None
    empty = [(i, j) for i in range(cfg.GRID_SIZE) for j in range(cfg.GRID_SIZE)
             if grid[i][j] is None]
    return empty[np.random.randint(len(empty))] if empty else None


def _simulate(exp_dir, run_ids, max_steps=12):
    """A small per-step-format experiment with real files and analysis output."""
    results = []
    for run_id in run_ids:
        sim = Simulation(run_id, _Agent, _random_mover, random_seed=run_id)
        results.append(sim.run_single_simulation(output_dir=str(exp_dir), max_steps=max_steps))
    Simulation.analyze_results(results, str(exp_dir), len(results))
    return results


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, 'NO_MOVE_THRESHOLD', 3)
    exp_dir = tmp_path / 'llm_baseline_20260905_000000'
    exp_dir.mkdir()
    (exp_dir / 'config.json').write_text(
        '{"scenario": "baseline", "llm_model": "test", "max_steps": 12, "no_move_threshold": 3}')
    _simulate(exp_dir, [0, 1, 2])
    return exp_dir


def _read_metrics(exp_dir):
    return pd.read_csv(run_files.metrics_history_path(exp_dir), float_precision='round_trip')


def _write_metrics(exp_dir, df):
    df.to_csv(run_files.metrics_history_path(exp_dir), index=False)


# --- step_log_extent / stale_metrics_runs -----------------------------------

def test_step_log_extent_matches_pandas_reader(experiment):
    log = run_files.load_step_log(experiment, 1)
    assert run_files.step_log_extent(experiment, 1) == \
        (len(log), int(log['step'].min()), int(log['step'].max()))
    assert run_files.step_log_extent(experiment, 99) is None


def test_fresh_experiment_is_complete(experiment):
    assert run_files.stale_metrics_runs(experiment) == []


def test_front_truncated_history_is_stale(experiment):
    """A live-LLM run resumed after an abort stores only its post-resume rows:
    max step still matches the step log, so only count/first-step catch it."""
    df = _read_metrics(experiment)
    keep = ~((df['run_id'] == 1) & (df['step'] < 2))
    _write_metrics(experiment, df[keep])
    assert run_files.stale_metrics_runs(experiment) == [1]


def test_missing_run_extra_row_and_nan_metric_are_stale(experiment):
    df = _read_metrics(experiment)
    df = df[df['run_id'] != 2]                                   # run 2 absent
    df.loc[(df['run_id'] == 0) & (df['step'] == 0), 'share'] = np.nan
    dup = df[(df['run_id'] == 1) & (df['step'] == 0)]
    df = pd.concat([df, dup], ignore_index=True)                 # duplicated row
    _write_metrics(experiment, df)
    assert run_files.stale_metrics_runs(experiment) == [0, 1, 2]


def test_missing_metric_column_means_rebuild_everything(experiment):
    df = _read_metrics(experiment).drop(columns=['dissimilarity_index'])
    _write_metrics(experiment, df)
    assert run_files.stale_metrics_runs(experiment) is None
    os.remove(run_files.metrics_history_path(experiment))
    assert run_files.stale_metrics_runs(experiment) is None


def test_run_ids_argument_limits_the_check(experiment):
    df = _read_metrics(experiment)
    _write_metrics(experiment, df[df['run_id'] != 2])
    assert run_files.stale_metrics_runs(experiment, run_ids=[0, 1]) == []
    assert run_files.stale_metrics_runs(experiment, run_ids=[2]) == [2]


# --- repair_stored_metrics ---------------------------------------------------

def test_repair_is_a_no_op_on_a_complete_experiment(experiment):
    path = run_files.metrics_history_path(experiment)
    before = gzip.open(path, 'rb').read()
    assert Simulation.repair_stored_metrics(str(experiment)) == []
    assert gzip.open(path, 'rb').read() == before


def test_repair_rebuilds_only_the_stale_run_and_restores_its_rows(experiment):
    path = run_files.metrics_history_path(experiment)
    original = _read_metrics(experiment)
    summary_before = pd.read_csv(experiment / 'run_summary.csv')

    damaged = original[~((original['run_id'] == 1) & (original['step'] < 2))]
    _write_metrics(experiment, damaged)

    assert Simulation.repair_stored_metrics(str(experiment)) == [1]

    repaired = _read_metrics(experiment)
    cols = ['run_id', 'step', *run_files.METRIC_COLUMNS]
    pd.testing.assert_frame_equal(
        repaired[cols].sort_values(['run_id', 'step']).reset_index(drop=True),
        original[cols].sort_values(['run_id', 'step']).reset_index(drop=True),
        check_exact=True)
    assert run_files.stale_metrics_runs(experiment) == []
    # The per-run bookkeeping is unchanged by the repair.
    summary_after = pd.read_csv(experiment / 'run_summary.csv')
    pd.testing.assert_frame_equal(summary_after, summary_before)


def test_repair_rebuilds_everything_without_a_usable_file(experiment):
    original = _read_metrics(experiment)
    os.remove(run_files.metrics_history_path(experiment))
    assert Simulation.repair_stored_metrics(str(experiment)) == [0, 1, 2]
    rebuilt = _read_metrics(experiment)
    cols = ['run_id', 'step', *run_files.METRIC_COLUMNS]
    pd.testing.assert_frame_equal(
        rebuilt[cols].sort_values(['run_id', 'step']).reset_index(drop=True),
        original[cols].sort_values(['run_id', 'step']).reset_index(drop=True),
        check_exact=True)


# --- the pipeline's repair step and the ANOVA from the roll-up ---------------

def test_repair_step_verifies_by_default_and_rebuilds_only_when_forced(experiment, tmp_path, monkeypatch):
    import experiment_list_for_analysis as experiment_list
    from analysis_tools import run_all_scenario_analysis as runner

    monkeypatch.setattr(experiment_list, 'SCENARIOS', {'llm_baseline': experiment.name}, raising=False)
    path = run_files.metrics_history_path(experiment)
    before = gzip.open(path, 'rb').read()
    calls = []
    monkeypatch.setattr(Simulation, 'load_and_analyze_results',
                        lambda *a, **k: calls.append(('full', a, k)))

    rebuilt = runner.repair_selected_experiments(experiments_dir=experiment.parent, verbose=False)
    assert rebuilt == {experiment.name: []}               # complete: nothing rebuilt
    assert calls == []                                    # and no full rebuild
    assert gzip.open(path, 'rb').read() == before

    runner.repair_selected_experiments(experiments_dir=experiment.parent, force=True, verbose=False)
    assert len(calls) == 1 and calls[0][2] == {'force_recompute': True}


def test_anova_reads_the_run_summary_roll_up(experiment, tmp_path):
    from analysis_tools import anova_by_metric
    from analysis_tools.output_paths import set_reports_dir
    from run_summary import combine_run_summaries

    reports = tmp_path / 'reports'
    combined = combine_run_summaries([str(experiment)], verbose=False)
    combined.insert(1, 'scenario_key', 'llm_baseline')
    reports.mkdir()
    combined.to_csv(reports / 'run_summary_by_run.csv', index=False)
    set_reports_dir(reports)

    df = anova_by_metric.anova_by_metric(reports_dir=reports)
    assert list(df['scenario'].unique()) == ['llm_baseline'] and len(df) == 3
    table = pd.read_csv(reports / 'anova_results_by_metric.csv')
    assert list(table['metric']) == anova_by_metric.METRICS
    assert (table['status'] == 'skipped').all()           # one group: nothing to compare
    assert (reports / 'anova_results_by_metric.md').exists()


# --- dissimilarity_index_over_time reads the stored column -------------------

def test_stored_di_timeseries_matches_frames_and_flags_stale_runs(experiment):
    from analysis_tools import dissimilarity_index_over_time as di

    stored, todo = di.stored_run_timeseries(experiment, [0, 1, 2], 'llm_baseline')
    assert todo == []
    assert list(stored.columns) == ['scenario', 'experiment', 'run_id', 'step', 'dissimilarity_index']
    frames = pd.concat([di.compute_run_timeseries(experiment, r, 'llm_baseline', True)
                        for r in (0, 1, 2)], ignore_index=True)
    key = ['run_id', 'step']
    pd.testing.assert_frame_equal(
        stored.sort_values(key).reset_index(drop=True),
        frames.sort_values(key).reset_index(drop=True), check_exact=True)

    df = _read_metrics(experiment)
    _write_metrics(experiment, df[~((df['run_id'] == 1) & (df['step'] < 2))])
    stored, todo = di.stored_run_timeseries(experiment, [0, 1, 2], 'llm_baseline')
    assert todo == [1]
    assert sorted(stored['run_id'].unique()) == [0, 2]

    os.remove(run_files.metrics_history_path(experiment))
    stored, todo = di.stored_run_timeseries(experiment, [0, 1, 2], 'llm_baseline')
    assert stored is None and todo == [0, 1, 2]


def test_di_process_experiment_output_is_unchanged_by_the_stored_path(experiment, tmp_path, monkeypatch):
    from analysis_tools import dissimilarity_index_over_time as di
    from analysis_tools.output_paths import set_reports_dir

    set_reports_dir(tmp_path / 'reports')
    ts_stored, final_stored = di.process_experiment(experiment.parent, 'llm_baseline', experiment.name, recompute=True)
    os.remove(run_files.metrics_history_path(experiment))       # force the frame path
    ts_frames, final_frames = di.process_experiment(experiment.parent, 'llm_baseline', experiment.name, recompute=True)
    pd.testing.assert_frame_equal(ts_stored, ts_frames, check_exact=True)
    pd.testing.assert_frame_equal(final_stored, final_frames, check_exact=True)


# --- steps_to_fraction_of_final ---------------------------------------------

def _reference_steps(df, metric):
    """The per-run loop convergence_patterns_and_speed carried until 2026-09-05."""
    out = []
    for run_id in df['run_id'].unique():
        run = df[df['run_id'] == run_id].sort_values('step')
        final_val, initial_val = run[metric].iloc[-1], run[metric].iloc[0]
        if final_val == initial_val:
            continue
        target = initial_val + 0.9 * (final_val - initial_val)
        hits = run[run[metric] >= target] if final_val > initial_val else run[run[metric] <= target]
        if not hits.empty:
            out.append(int(hits['step'].iloc[0]))
    return out


@pytest.mark.parametrize('frame', [
    pd.DataFrame({'run_id': [1, 1, 1], 'step': [0, 1, 2], 'm': [0.0, 1.0, np.nan]}),
    pd.DataFrame({'run_id': [1, 1, 1], 'step': [0, 1, 2], 'm': [np.nan, 0.5, 1.0]}),
    pd.DataFrame({'run_id': [1, 1, 1, 1], 'step': [0, 1, 2, 3], 'm': [0.0, np.nan, 0.95, 1.0]}),
    pd.DataFrame({'run_id': [2, 1, 2, 1, 2], 'step': [2, 1, 0, 0, 1], 'm': [1.0, 0.5, 0.0, 0.0, 0.9]}),
    pd.DataFrame({'run_id': [1], 'step': [0], 'm': [0.3]}),
    pd.DataFrame({'run_id': [1, 1], 'step': [0, 1], 'm': [0.3, 0.3]}),
    pd.DataFrame({'run_id': [1, 1, 1], 'step': [0, 1, 2], 'm': [1.0, 0.05, 0.0]}),
    pd.DataFrame({'run_id': [1, 1, 1, 1], 'step': [0, 1, 2, 3], 'm': [0.0, 2.0, 0.5, 1.0]}),
    pd.DataFrame({'run_id': [1, 1, 1], 'step': [0, 1, 2], 'm': [0, 9, 10]}),
    pd.DataFrame({'run_id': [5, 5, 3, 3, 9, 9], 'step': [0, 1, 0, 1, 0, 1], 'm': [0, 1, 0, 1, 0, 1]}),
], ids=['nan_last', 'nan_first', 'nan_mid', 'unsorted', 'single', 'flat',
        'decreasing', 'overshoot', 'int', 'run_order'])
def test_steps_to_fraction_matches_the_reference_loop(frame):
    from plot_style import steps_to_fraction_of_final
    assert steps_to_fraction_of_final(frame, 'm') == _reference_steps(frame, 'm')


def test_steps_to_fraction_on_simulated_history(experiment):
    from plot_style import steps_to_fraction_of_final
    df = _read_metrics(experiment)
    for metric in run_files.METRIC_COLUMNS:
        assert steps_to_fraction_of_final(df, metric) == _reference_steps(df, metric)


# --- initial-grid metrics in run_summary (the paired chance scenario) --------

def test_run_summary_carries_the_initial_grid_metrics(experiment):
    from Metrics import calculate_all_metrics
    from run_summary import INITIAL_COLUMNS, METRIC_COLUMNS, SUMMARY_COLUMNS, write_run_summary

    assert [c[len('initial_'):] for c in INITIAL_COLUMNS] == METRIC_COLUMNS
    assert all(c in SUMMARY_COLUMNS for c in INITIAL_COLUMNS)
    df = pd.read_csv(experiment / 'run_summary.csv').set_index('run_id')
    for rid in (0, 1, 2):
        frame0 = run_files.load_frames(experiment, rid)[0]
        expected = calculate_all_metrics(frame0)
        for metric in METRIC_COLUMNS:
            assert df.loc[rid, f'initial_{metric}'] == pytest.approx(expected[metric], abs=1e-12)
    # A summary written before the columns existed gets them filled on the next pass.
    df.drop(columns=INITIAL_COLUMNS).reset_index().to_csv(experiment / 'run_summary.csv', index=False)
    rebuilt = write_run_summary(str(experiment), verbose=False).set_index('run_id')
    assert rebuilt.loc[1, 'initial_dissimilarity_index'] == pytest.approx(df.loc[1, 'initial_dissimilarity_index'])
