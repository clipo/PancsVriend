"""Statistical helpers for comparing reference and sample-derived probabilities.

Centralizing the test here keeps call sites readable and provides a single
swap point if the framing needs to change later (e.g., bootstrap, two-sample
proportion test).
"""

from __future__ import annotations

import math

from scipy.stats import binomtest


def binomial_two_sided_p(k: int, n: int, p_null: float) -> float:
    """Two-sided binomial test p-value for observed `k` successes in `n` trials
    against null probability `p_null`.

    Returns NaN when the test is ill-posed: `n <= 0`, or `p_null` is NaN/out of range.
    `p_null` is clamped defensively into [0, 1] to tolerate tiny float overshoot.
    """
    if n is None or n <= 0:
        return float("nan")
    if k is None or k < 0 or k > n:
        return float("nan")
    if p_null is None:
        return float("nan")
    try:
        p = float(p_null)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(p):
        return float("nan")
    if p < 0.0:
        p = 0.0
    elif p > 1.0:
        p = 1.0
    return float(binomtest(int(k), int(n), p, alternative="two-sided").pvalue)
