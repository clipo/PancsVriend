"""Placebo campaign: a composition-blind value function must leave the grid a
random allocation, and the pipeline must say so.

p_move = 0.5 in every (n_similar, n_occupied) cell, both roles, through the
ordinary llm_runner path (keyed RNG, per-step record). A uniformly chosen
agent moving to a uniformly chosen empty cell is a symmetric transition
between configurations, so the uniform distribution over configurations is
preserved at every step: the final grid must have the same distribution as
the initial grid for every metric, and both must match the independent
random-allocation null. This is the known-answer test of the simulation
loop, the frame-0 bookkeeping behind the paired chance test, and the metrics
(2026-09-05; the seven-metric table on 500 x 200 gave paired-t p 0.45-0.99,
KS p 0.11-1.00).

Runs are seeded by run_id, so the numbers are deterministic: a failure is a
change in the code, not an unlucky draw. Marked slow (~1-4 min): skipped
unless pytest is given --runslow.
"""

import contextlib
import io
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import run_files
from Metrics import calculate_all_metrics
from run_summary import METRIC_COLUMNS

N_RUNS, MAX_STEPS, P_MOVE = 300, 150, 0.5
BOARD = {"grid_size": 20, "num_type_a": 160, "num_type_b": 160}
ALPHA = 0.01          # deterministic data, so this is a tolerance on known values, not a false-positive rate


def _placebo_policy():
    return {(role, f"comp:{s},{o}"): P_MOVE
            for role in ("type_a", "type_b") for o in range(9) for s in range(o + 1)}


@pytest.mark.slow
def test_composition_blind_dynamics_leave_a_random_allocation(tmp_path):
    import llm_runner as L
    from value_functions.comparison.vf_rank_stability import metric_null
    from DissimilarityIndex import random_baseline

    grid_cfg = L.grid_config_from_args(BOARD["grid_size"], BOARD["num_type_a"], BOARD["num_type_b"])
    out = str(tmp_path / "placebo")
    args = [(rid, "baseline", "placebo-p05", None, None, out, MAX_STEPS, None, None, None, None,
             0.3, None, None, _placebo_policy(), "placebo", grid_cfg) for rid in range(N_RUNS)]
    with contextlib.redirect_stdout(io.StringIO()):
        with mp.get_context("fork").Pool(4) as pool:
            results = list(pool.imap(L.run_single_simulation, args))
    assert not any(r["converged"] for r in results)          # half the agents move every step

    initial = {m: [] for m in METRIC_COLUMNS}
    final = {m: [] for m in METRIC_COLUMNS}
    for rid in range(N_RUNS):
        frames = run_files.load_frames(out, rid)
        a, b = calculate_all_metrics(frames[0]), calculate_all_metrics(frames[-1])
        for m in METRIC_COLUMNS:
            initial[m].append(a[m]); final[m].append(b[m])

    null = metric_null(BOARD)["metrics"]
    chance_di = random_baseline(BOARD["grid_size"], BOARD["num_type_a"], BOARD["num_type_b"])["mean"]
    problems = []
    for m in METRIC_COLUMNS:
        i, f = np.asarray(initial[m], float), np.asarray(final[m], float)
        chance = chance_di if m == "dissimilarity_index" else null[m]["mean"]
        p_paired = stats.ttest_rel(f, i).pvalue
        p_ks = stats.ks_2samp(i, f).pvalue
        p_null = stats.ttest_1samp(f, chance).pvalue
        if min(p_paired, p_ks, p_null) < ALPHA:
            problems.append(f"{m}: paired-t p={p_paired:.3g} KS p={p_ks:.3g} final-vs-null p={p_null:.3g} "
                            f"(initial {i.mean():.4f}, final {f.mean():.4f}, chance {chance:.4f})")
    assert not problems, "placebo dynamics changed the distribution:\n  " + "\n  ".join(problems)
