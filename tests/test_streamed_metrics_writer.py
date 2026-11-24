"""Simulation._write_metrics_streamed must reproduce the one-shot table.

The old analyze_results built ONE DataFrame from every run's metric dicts, so
pandas typed each column from all rows together: a column that is a plain int
0 in one run (Metrics returns int 0 in degenerate cases) and float elsewhere
is float64 everywhere and prints "0.0". The streamed writer (2026-11-23) types
columns from per-run all-int flags; these tests pin it to the one-shot result
for both formats, using staged files and in-memory histories together.
"""
import gzip

import numpy as np
import pandas as pd

import run_files
from base_simulation import Simulation


def _history(run_id, n, int_zero_switch=False):
    rows = []
    for step in range(n):
        rows.append({
            "clusters": 10 + step,
            "switch_rate": 0 if (int_zero_switch and step == 1) else 0.1 * step + 1 / 3,
            "distance": float(step) / 7,
            "mix_deviation": np.float64(0.25) + step,
            "share": 0 if int_zero_switch else 0.5,
            "ghetto_rate": step,
            "dissimilarity_index": 0.1 + step / 9,
            "step": step,
            "run_id": run_id,
        })
    return rows


def _results(tmp_path):
    out = []
    for rid, n, flag in [(2, 3, False), (0, 4, True), (1, 2, False)]:
        hist = _history(rid, n, flag)
        if rid == 1:                               # an in-memory history
            out.append({"run_id": rid, "metrics_history": hist})
            continue
        path, cols, all_int = run_files.stage_metrics(str(tmp_path), rid, hist)
        out.append({"run_id": rid, "metrics_history": [], "metrics_file": path,
                    "metrics_columns": cols, "metrics_all_int": all_int})
    out.append({"run_id": 9, "metrics_history": []})          # resume placeholder
    expected = pd.DataFrame([row for rid, n, flag in [(2, 3, False), (0, 4, True), (1, 2, False)]
                             for row in _history(rid, n, flag)])
    expected = expected.sort_values(["run_id", "step"], kind="stable").reset_index(drop=True)
    return out, expected


def test_streamed_csv_text_equals_one_shot(tmp_path):
    results, expected = _results(tmp_path)
    path = str(tmp_path / "metrics_history.csv.gz")
    assert Simulation._write_metrics_streamed(results, path)
    assert gzip.open(path, "rt").read() == expected.to_csv(index=False)
    assert expected["switch_rate"].dtype == np.float64        # the int 0 was upcast


def test_streamed_parquet_values_equal_one_shot(tmp_path):
    results, expected = _results(tmp_path)
    path = str(tmp_path / "metrics_history.parquet")
    assert Simulation._write_metrics_streamed(results, path)
    got = pd.read_parquet(path)
    assert list(got.columns) == list(expected.columns)
    assert list(got.dtypes) == list(expected.dtypes)
    for col in expected.columns:
        a, b = got[col].to_numpy(), expected[col].to_numpy()
        if a.dtype == np.float64:
            a, b = a.view(np.int64), b.view(np.int64)          # bit for bit
        assert np.array_equal(a, b), col
