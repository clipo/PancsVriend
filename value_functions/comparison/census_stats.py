"""Statistics for checking an exact P(MOVE) table against a sequential census.

Used by census_goodness_of_fit.py (the paper result) and sanity_vs_exact.py
(the per-cell check). The design was reviewed twice on 2026-11-16 before it replaced the
pooled-escalation verdict; the reasons are in sanity_vs_exact.py's docstring.
Two checks with two different jobs (user decision 2026-11-16):

  GOODNESS OF FIT   — REPORTED IN THE PAPER. All cells, census draws only, no
                selection: deviance G against a parametric bootstrap from the
                exact table. One number per model.
  PER-CELL CHECK    — INTERNAL SANITY CHECK, not a paper result: does each
                cell the census flagged hold up on fresh draws, i.e. is the
                extracted table a good replacement for sampling cell by cell?
                The family F = cells whose census Wilson interval misses
                p_exact. Each gets ONE round of FRESH draws (never-issued seeds,
                n fixed from census data alone), tested by a two-sided exact
                binomial test, Holm across F. Each cell counted once.
  Descriptive   the census miss count against its EXACT null expectation
                (Poisson-binomial over cells; Wilson at n=100 over-covers p=0/1
                cells and under-covers p~1e-3, so "5% of 540 = 27" is wrong).

The exact table carries a mass-bound interval ci95 = [lo, hi] around
p_move_effective (probability mass outside the enumerated token paths), so the
null for a cell is composite: p in [lo, hi]. The cell p-value is the largest
over that interval, and a p_exact of exactly 0 or 1 no longer turns a single
contrary reply into p = 0 unless the interval itself excludes it.

Denominators are VALID replies (move + stay): p_exact is conditional on a
parseable reply.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom, binomtest

from sampling_common import wilson_ci

CI_EPS = 1e-9          # floating-point slack on Wilson limits at k = 0 or k = n
P_CLIP = 1e-12         # keep logs finite for p_exact at exactly 0 or 1
# The goodness-of-fit test is a REPORTED result: fixed bootstrap size and seed,
# shared by sanity_vs_exact.py and census_goodness_of_fit.py so both print the
# same p-value. Monte Carlo standard error of p at 10,000 reps is <= 0.005.
GOF_REPS = 10000
GOF_SEED = 0


# --------------------------------------------------------------------------- stage 1

def wilson_miss(k: int, n: int, p: float) -> bool:
    lo, hi = wilson_ci(k, n)
    return not (lo - CI_EPS <= p <= hi + CI_EPS)


def miss_prob(n: int, p: float) -> float:
    """P(Wilson interval at n misses p | true value p), exactly."""
    return float(sum(binom.pmf(x, n, p) for x in range(n + 1) if wilson_miss(x, n, p)))


def expected_misses(N, P, reps: int = 20000, seed: int = 0) -> tuple[float, callable]:
    """Poisson-binomial null of the census miss count: (mean, tail(obs) -> P(X >= obs))."""
    pm = np.array([miss_prob(int(n), float(p)) for n, p in zip(N, P)])
    sims = (np.random.default_rng(seed).random((reps, len(pm))) < pm).sum(1)
    return float(pm.sum()), (lambda obs: float((sims >= obs).mean()))


def _deviance(K, N, LO, HI):
    """Binomial deviance per row, against the composite null p in [lo, hi]
    (the interval point nearest the observed proportion minimises it)."""
    ph = K / N
    p = np.clip(np.clip(ph, LO, HI), P_CLIP, 1 - P_CLIP)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(K > 0, K * np.log(ph / p), 0.0)
        b = np.where(N - K > 0, (N - K) * np.log((1 - ph) / (1 - p)), 0.0)
    return 2 * (a + b).sum(-1)


def deviance_omnibus(K, N, P, LO, HI, reps: int = GOF_REPS, seed: int = GOF_SEED,
                     return_null: bool = False) -> dict:
    """THE GOODNESS-OF-FIT TEST (the census result reported in the paper).

    Over ALL census cells: the binomial deviance G — how far every cell's
    sampled count is from the exact table, summed — against a parametric
    bootstrap: `reps` censuses simulated from the exact table itself, each
    scored the same way. p = share of simulated censuses at least as far off as
    the real one. No cell is selected, so the null is exact up to Monte Carlo
    error. return_null adds the simulated G values (for the figure)."""
    K, N, P, LO, HI = (np.asarray(x, dtype=float) for x in (K, N, P, LO, HI))
    g0 = float(_deviance(K, N, LO, HI))
    sim = np.random.default_rng(seed).binomial(N.astype(int), np.clip(P, 0, 1), size=(reps, len(P)))
    gs = _deviance(sim.astype(float), N, LO, HI)
    p = float((np.sum(gs >= g0) + 1) / (reps + 1))
    out = {"G": round(g0, 2), "G_null_mean": round(float(gs.mean()), 2),
           "G_null_q95": round(float(np.quantile(gs, 0.95)), 2),
           "p_mc": p, "p_mc_se": round(math.sqrt(p * (1 - p) / reps), 4), "reps": reps, "seed": seed}
    if return_null:
        out["null"] = gs
    return out


# --------------------------------------------------------------------------- cell tests

def composite_binom_p(k: int, n: int, lo: float, hi: float, p_exact: float) -> float:
    """Two-sided exact binomial p-value, maximised over the null interval [lo, hi]."""
    if n == 0:
        return 1.0
    lo, hi = max(0.0, min(lo, p_exact)), min(1.0, max(hi, p_exact))
    grid = {lo, hi, p_exact, min(max(k / n, lo), hi)}
    return float(max(binomtest(k, n, q).pvalue for q in grid))


def holm(ps: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control)."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    m, adj, run = len(ps), [1.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


def _two_sided_pvals(n: int, p0: float) -> np.ndarray:
    """Exact two-sided ("minlike", as scipy) p-value for every k in 0..n at once."""
    pmf = binom.pmf(np.arange(n + 1), n, p0)
    s = np.sort(pmf)
    cs = np.cumsum(s)
    idx = np.searchsorted(s, pmf * (1 + 1e-7), side="right")
    return np.minimum(cs[idx - 1], 1.0)


def alt_p(p0: float, odds_ratio: float) -> float:
    p0 = min(max(p0, P_CLIP), 1 - P_CLIP)
    lg = math.log(p0 / (1 - p0)) + math.log(odds_ratio)
    return 1 / (1 + math.exp(-lg))


def power(n: int, p0: float, odds_ratio: float, alpha: float) -> float:
    """Power of the two-sided exact test at n against p0 shifted by an odds ratio."""
    if n <= 0:
        return 0.0
    rej = _two_sided_pvals(n, p0) <= alpha
    return float(binom.pmf(np.arange(n + 1), n, alt_p(p0, odds_ratio))[rej].sum())


def size_fresh(p0: float, direction: int, odds_ratio: float, target: float, alpha: float,
               floor: int, cap: int, step: int = 50) -> tuple[int, float]:
    """Smallest n in [floor, cap] (multiples of `step`) giving `target` power
    against the stated alternative in the census deviation's direction
    (direction +1: odds x OR, -1: odds / OR). Sizing uses census data only,
    which leaves the fresh draws' null distribution untouched. Returns (n, power)."""
    orr = odds_ratio if direction >= 0 else 1 / odds_ratio
    n = floor
    while True:
        pw = power(n, p0, orr, alpha)
        if pw >= target or n >= cap:
            return n, pw
        n = min(cap, n + step)
