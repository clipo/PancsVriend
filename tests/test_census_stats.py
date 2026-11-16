"""census_stats: the statistics behind the census checks.

    .venv/bin/python -m pytest tests/test_census_stats.py -q

Covers:
- the vectorised two-sided exact p-values equal scipy's binomtest
- Holm adjustment
- the composite null p-value (max over the exact table's [lo, hi])
- the goodness-of-fit test (deviance_omnibus): reproducible under its fixed
  seed, does not reject data drawn from the table, rejects a shifted table
- power / sample sizing: monotone, floor and cap respected
- the exact null expectation of census Wilson misses
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "prompt_refinement"), str(REPO / "value_functions" / "comparison")):
    if p not in sys.path:
        sys.path.insert(0, p)

import census_stats as cs  # noqa: E402


@pytest.mark.parametrize("n,p0", [(100, 0.5), (300, 0.00104), (500, 0.99941),
                                  (336, 0.1021), (1200, 0.0035)])
def test_two_sided_pvals_match_scipy(n, p0):
    v = cs._two_sided_pvals(n, p0)
    for k in range(0, n + 1, max(1, n // 50)):
        assert v[k] == pytest.approx(binomtest(k, n, p0).pvalue, abs=1e-9)


def test_holm():
    assert cs.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert cs.holm([0.2]) == [0.2]
    assert cs.holm([0.6, 0.6]) == [1.0, 1.0]               # capped at 1


def test_composite_null():
    # a point interval is the ordinary exact test
    assert cs.composite_binom_p(30, 100, 0.2, 0.2, 0.2) == pytest.approx(
        binomtest(30, 100, 0.2).pvalue)
    # an interval containing the observed proportion cannot reject
    assert cs.composite_binom_p(10, 100, 0.09, 0.11, 0.1) == 1.0
    # the max over the interval is at least the point p-value
    assert cs.composite_binom_p(15, 100, 0.08, 0.12, 0.1) >= binomtest(15, 100, 0.1).pvalue
    # a table that says "never" (to 1e-8) is contradicted by a single MOVE
    assert cs.composite_binom_p(1, 100, 0.0, 1e-8, 0.0) < 1e-5
    assert cs.composite_binom_p(0, 0, 0.1, 0.1, 0.1) == 1.0
    # the interval, not the point, is the null: these reject at the point value
    assert binomtest(1, 100, 0.0).pvalue == 0.0
    assert cs.composite_binom_p(1, 100, 0.0, 0.02, 0.0) == 1.0
    assert binomtest(12, 100, 0.1).pvalue < 1.0
    assert cs.composite_binom_p(12, 100, 0.08, 0.14, 0.1) == 1.0


def _table(m=300, seed=1):
    rng = np.random.default_rng(seed)
    P = np.concatenate([rng.uniform(0.05, 0.95, m // 2), np.zeros(m // 4), np.ones(m - m // 2 - m // 4)])
    return P, np.full(m, 100), P.copy(), P.copy()


def test_goodness_of_fit_reproducible_and_calibrated():
    P, N, LO, HI = _table()
    K = np.random.default_rng(7).binomial(N, P)
    a = cs.deviance_omnibus(K, N, P, LO, HI, reps=2000)
    b = cs.deviance_omnibus(K, N, P, LO, HI, reps=2000)
    assert a == b                                           # fixed seed: a reported result
    assert a["p_mc"] > 0.01 and a["G"] >= 0
    assert a["reps"] == 2000 and a["seed"] == cs.GOF_SEED
    assert 0 < a["p_mc"] <= 1 and a["p_mc_se"] <= 0.012


def test_goodness_of_fit_rejects_a_wrong_table():
    P, N, LO, HI = _table()
    shifted = np.clip(P + 0.08 * ((P > 0) & (P < 1)), 0, 1)  # every interior cell off by 0.08
    K = np.random.default_rng(7).binomial(N, shifted)
    assert cs.deviance_omnibus(K, N, P, LO, HI, reps=2000)["p_mc"] < 0.01


def test_goodness_of_fit_saturated_cells_are_exact():
    # p = 0 / 1 cells reproduced exactly contribute nothing
    P = np.array([0.0, 1.0, 0.0]); N = np.array([100, 100, 100])
    r = cs.deviance_omnibus(np.array([0, 100, 0]), N, P, P, P, reps=500)
    assert r["G"] == 0 and r["p_mc"] == 1.0


def test_goodness_of_fit_uses_the_interval():
    P = np.array([0.0, 0.5]); N = np.array([100, 100])
    LO, HI = np.array([0.0, 0.5]), np.array([0.02, 0.5])
    assert cs.deviance_omnibus(np.array([1, 50]), N, P, LO, HI, reps=200)["G"] == 0   # inside
    assert cs.deviance_omnibus(np.array([5, 50]), N, P, LO, HI, reps=200)["G"] > 0    # outside


def test_reported_constants():
    # the paper's numbers depend on these: changing them changes the reported p
    assert (cs.GOF_REPS, cs.GOF_SEED) == (10000, 0)
    r = cs.deviance_omnibus([5], [10], [0.5], [0.5], [0.5])
    assert (r["reps"], r["seed"]) == (10000, 0)


def test_return_null():
    P, N, LO, HI = _table(60)
    r = cs.deviance_omnibus(np.random.default_rng(3).binomial(N, P), N, P, LO, HI, reps=300,
                            return_null=True)
    assert r["null"].shape == (300,)


def test_power_and_sizing():
    assert cs.power(300, 0.1, 2.0, 0.05) < cs.power(1200, 0.1, 2.0, 0.05)
    assert cs.power(0, 0.1, 2.0, 0.05) == 0.0
    n, pw = cs.size_fresh(0.1021, 1, 2.0, 0.8, 0.05, 300, 1200)
    assert n == 300 and pw >= 0.8                           # interior cell: the floor suffices
    n, pw = cs.size_fresh(0.00104, 1, 2.0, 0.8, 0.05, 300, 1200)
    assert n == 1200 and pw < 0.8                           # near-0 cell: capped, BEYOND_CAP


def test_expected_misses():
    mean, tail = cs.expected_misses([100, 100], [0.0, 1.0])
    assert mean == pytest.approx(0.0, abs=1e-9) and tail(0) == 1.0
    mean, _ = cs.expected_misses([100] * 20, [0.5] * 20)
    assert 0.3 < mean < 2.0                                 # ~5% of 20, Wilson at n=100
