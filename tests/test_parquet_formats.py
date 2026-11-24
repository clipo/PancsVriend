"""metrics_history / packed step log as Parquet (2026-11-23).

* A reader gets from Parquet the exact stored float64 values, whatever CSV
  parser setting it passes -- the values a 'round_trip' CSV read gives. The
  default CSV parser is up to an ulp off for many of them (user decision
  2026-11-24: exact values, last digits may differ from old CSV outputs).
* A folder written before the change keeps CSV for containers it does not
  have yet, so it never mixes formats; a new folder gets Parquet.
* read_metrics_file refuses CSV-only options for Parquet instead of ignoring them.
"""
import numpy as np
import pandas as pd
import pytest

import run_files


def _frame(n=5000, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "clusters": rng.integers(0, 90, n),
        "switch_rate": rng.random(n) / 3,              # many values the default parser misreads
        "share": rng.random(n) * 7 / 9,
        "ghetto_rate": rng.integers(0, 200, n),
        "dissimilarity_index": np.r_[rng.random(n - 2), [0.0, np.nan]],
        "step": np.arange(n),
        "run_id": np.repeat([0, 1], [n // 2, n - n // 2]),
    })


def _bits(frame):
    return {c: (frame[c].to_numpy().view(np.int64) if frame[c].dtype == np.float64
                else frame[c].to_numpy()) for c in frame.columns}


@pytest.mark.parametrize("kwargs", [{}, {"float_precision": "round_trip"},
                                    {"usecols": ["share", "run_id", "step"]},
                                    {"usecols": ["run_id", "switch_rate"], "float_precision": "round_trip"}])
def test_parquet_reads_give_the_exact_stored_values(tmp_path, kwargs):
    frame = _frame()
    csv, parquet = tmp_path / "m.csv.gz", tmp_path / "m.parquet"
    frame.to_csv(csv, index=False)
    exact = pd.read_csv(csv, float_precision="round_trip")
    exact.to_parquet(parquet, index=False)
    want = pd.read_csv(csv, **{**kwargs, "float_precision": "round_trip"})
    got = run_files.read_metrics_file(str(parquet), **kwargs)
    assert list(got.columns) == list(want.columns)
    assert list(got.dtypes) == list(want.dtypes)
    w, g = _bits(want), _bits(got)
    for col in want.columns:
        assert np.array_equal(w[col], g[col], equal_nan=False) or \
            np.array_equal(want[col].to_numpy(), got[col].to_numpy(), equal_nan=True), col
    if not kwargs:   # and they are not what the default CSV parser gives
        assert (_bits(pd.read_csv(csv))["switch_rate"] != g["switch_rate"]).any()


def test_parquet_rejects_csv_only_options(tmp_path):
    path = tmp_path / "m.parquet"
    _frame(10).to_parquet(path, index=False)
    with pytest.raises(NotImplementedError):
        run_files.read_metrics_file(str(path), chunksize=4)


def test_new_folder_gets_parquet_and_old_folder_keeps_csv(tmp_path):
    new, old = tmp_path / "new", tmp_path / "old"
    (new / "move_logs").mkdir(parents=True)
    (old / "move_logs").mkdir(parents=True)
    assert run_files.metrics_history_path(new).endswith("metrics_history.parquet")
    assert run_files.packed_step_log_path(new).endswith("step_moves_packed.parquet")
    _frame(10).to_csv(old / "metrics_history.csv.gz", index=False)   # written before the change
    assert run_files.metrics_history_path(old).endswith("metrics_history.csv.gz")
    assert run_files.packed_step_log_path(old).endswith("step_moves_packed.csv.gz")
