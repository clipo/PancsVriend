"""The generalized dissimilarity index must still be the 10x10 one it replaced.

The tract map went from a hardcoded 10x10 array to get_census_tract()'s
size // 3 edge bands (remainder to the centre) so the model could move to a
20x20 grid, and tract_map() is the vectorised form of that rule. The safety
argument is that the two agree cell for cell at every size and are a no-op at
10x10, so that is what these tests pin — against the per-cell reference
implementation in DissimilarityIndex, not against recorded numbers, so the
two can never drift apart silently.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DissimilarityIndex import (  # noqa: E402
    band_widths,
    compute_dissimilarity,
    compute_dissimilarity_from_int_grid,
    compute_dissimilarity_index,
    get_census_tract,
    max_configuration,
    random_baseline,
    theoretical_max,
    tract_map,
    tract_sizes,
    two_cluster_di,
)


def _pop(size, density=0.8, share_a=0.5):
    """Agent counts stated explicitly, never taken from config.

    Other test modules monkeypatch config.GRID_SIZE / NUM_TYPE_A, and the
    reference samplers fall back to config when the counts are omitted, so a
    test that relied on that default would pass or fail on suite ORDER.
    """
    n = int(round(density * size * size))
    n_a = int(round(share_a * n))
    return n_a, n - n_a


class _Agent:
    def __init__(self, type_id):
        self.type_id = type_id


def _object_grid(int_grid):
    """int grid (-1 empty, 0/1) -> the Agent-or-None grid the simulation uses."""
    h, w = int_grid.shape
    grid = np.empty((h, w), dtype=object)
    for r in range(h):
        for c in range(w):
            v = int(int_grid[r, c])
            grid[r, c] = None if v == -1 else _Agent(v)
    return grid


def test_tract_map_reproduces_the_hardcoded_10x10_map():
    expected = np.array([[get_census_tract(r, c) for c in range(10)]
                         for r in range(10)])
    assert np.array_equal(tract_map(10), expected)
    assert band_widths(10) == (3, 4, 3)
    assert tract_sizes(10).tolist() == [9, 12, 9, 12, 16, 12, 9, 12, 9]


@pytest.mark.parametrize("size, widths", [
    (10, (3, 4, 3)), (20, (6, 8, 6)), (30, (10, 10, 10)),
    (40, (13, 14, 13)), (50, (16, 18, 16)),
])
def test_tract_map_generalizes(size, widths):
    tracts = tract_map(size)
    assert band_widths(size) == widths
    # tract_map is the vectorised get_census_tract: they must agree per cell.
    expected = np.array([[get_census_tract(r, c, size) for c in range(size)]
                         for r in range(size)])
    assert np.array_equal(tracts, expected)
    assert sum(widths) == size
    # Always exactly nine tracts, and every cell belongs to one of them.
    assert sorted(np.unique(tracts).tolist()) == list(range(9))
    assert tract_sizes(size).sum() == size * size


@pytest.mark.parametrize("size", [10, 20, 30])
def test_matches_the_reference_on_random_grids(size):
    """The reference is now size-agnostic too, so pin agreement beyond 10x10."""
    rng = np.random.default_rng(size)
    for _ in range(200):
        vals = rng.integers(-1, 2, size=(size, size)).astype(np.int8)
        grid = _object_grid(vals)
        expected = compute_dissimilarity_index(grid)
        assert compute_dissimilarity_from_int_grid(vals) == pytest.approx(expected, abs=1e-12)


def test_matches_the_frozen_reference_on_random_10x10_grids():
    rng = np.random.default_rng(0)
    for _ in range(500):
        vals = rng.integers(-1, 2, size=(10, 10)).astype(np.int8)
        grid = _object_grid(vals)
        expected = compute_dissimilarity_index(grid)
        assert compute_dissimilarity(grid) == pytest.approx(expected, abs=1e-12)
        assert compute_dissimilarity_from_int_grid(vals) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("fill, expected", [(-1, 0.0), (0, 1.0), (1, 1.0)])
def test_degenerate_grids(fill, expected):
    """Empty grid -> 0; one group present and the other absent -> 1."""
    vals = np.full((10, 10), fill, dtype=np.int8)
    assert compute_dissimilarity_from_int_grid(vals) == expected
    assert compute_dissimilarity(_object_grid(vals)) == expected


def test_non_square_grid_is_rejected():
    with pytest.raises(ValueError, match="square"):
        compute_dissimilarity_from_int_grid(np.zeros((10, 20), dtype=np.int8))


@pytest.mark.parametrize("size", [10, 20, 30])
def test_maximum_is_attained_by_two_contiguous_blocks(size):
    """The ceiling is real: it is measured off a witness grid, not asserted."""
    n_a, n_b = _pop(size)
    value, grid = max_configuration(size, n_a, n_b)
    assert value == pytest.approx(1.0)
    assert compute_dissimilarity_from_int_grid(grid) == pytest.approx(1.0)
    assert theoretical_max(size, n_a, n_b) == pytest.approx(1.0)


def test_reference_values_do_not_cache_across_config_changes(monkeypatch):
    """The samplers fall back to config for the agent counts; that fallback must
    not be baked into the cache key, or the first caller would fix the answer
    for the whole process."""
    import config
    monkeypatch.setattr(config, "GRID_SIZE", 10)
    monkeypatch.setattr(config, "NUM_TYPE_A", 40)
    monkeypatch.setattr(config, "NUM_TYPE_B", 40)
    dense = random_baseline(20)
    monkeypatch.setattr(config, "NUM_TYPE_A", 20)
    monkeypatch.setattr(config, "NUM_TYPE_B", 20)
    sparse = random_baseline(20)
    assert dense["n_a"] == 160 and sparse["n_a"] == 80
    assert sparse["mean"] > dense["mean"]      # fewer agents per tract, noisier


def test_random_baseline_falls_as_the_grid_grows():
    """The null mean scales like 1/sqrt(agents per tract).

    Quadrupling the cells per tract (10x10 -> 20x20 at fixed density) should
    halve it; this is the reason DI levels are not comparable across grid sizes
    and the reference lines exist.
    """
    small = random_baseline(10, *_pop(10))["mean"]
    large = random_baseline(20, *_pop(20))["mean"]
    assert small / large == pytest.approx(2.0, rel=0.05)


def test_two_cluster_city_sits_well_below_the_ceiling():
    """A segregated city whose boundary misses the tract lines cannot score 1."""
    tc = two_cluster_di(20, *_pop(20))
    assert 0.6 < tc["mean"] < 0.9
    assert tc["max"] < 1.0
    assert tc["mean"] > random_baseline(20, *_pop(20))["p95"]


def test_di_rises_monotonically_with_the_gap_between_the_cities():
    """How much empty space separates the two cities is not a cosmetic choice.

    A buffer deletes exactly the mixed cells nearest the cut, so DI climbs with
    it. The default marginalises over the whole range and must therefore land
    strictly between the touching and maximally-separated extremes.
    """
    n_a, n_b = _pop(20)                       # 160 + 160, so 80 vacancies
    touching = two_cluster_di(20, n_a, n_b, gap=0)["mean"]
    one_cell = two_cluster_di(20, n_a, n_b, gap=20)["mean"]
    widest = two_cluster_di(20, n_a, n_b, gap=80)["mean"]
    marginal = two_cluster_di(20, n_a, n_b)["mean"]      # gap="random" default
    assert touching < one_cell < widest < 1.0
    assert touching < marginal < widest


def test_the_touching_extreme_really_touches():
    """gap=0 scatters vacancies inside each city, some of them on the boundary.

    That could in principle separate the two cities by accident, which would
    make the "touching" extreme not an extreme at all. It does not: they are
    orthogonally adjacent somewhere in every draw.
    """
    import numpy as np
    from DissimilarityIndex import tract_map  # noqa: F401  (keeps import list honest)
    n_a, n_b = _pop(20)
    rng = np.random.default_rng(0)
    n_cells, size = 400, 20
    rows, cols = np.divmod(np.arange(n_cells), size)
    yy, xx = rows + 0.5, cols + 0.5
    for _ in range(50):
        theta = rng.uniform(0, np.pi)
        order = np.argsort(xx * np.cos(theta) + yy * np.sin(theta)
                           + rng.uniform(0, 1e-9, n_cells))
        g = np.full(n_cells, -1, dtype=np.int8)
        g[rng.choice(order[:n_cells // 2], n_a, replace=False)] = 0
        g[rng.choice(order[n_cells // 2:], n_b, replace=False)] = 1
        g = g.reshape(size, size)
        a, b = g == 0, g == 1
        assert ((a[:, :-1] & b[:, 1:]).any() or (b[:, :-1] & a[:, 1:]).any()
                or (a[:-1] & b[1:]).any() or (b[:-1] & a[1:]).any())


def test_gap_wider_than_the_vacancies_is_rejected():
    with pytest.raises(ValueError, match="gap must be between"):
        two_cluster_di(20, *_pop(20), gap=81)
