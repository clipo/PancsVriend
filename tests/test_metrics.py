"""The six literature metrics in Metrics.py (the DI has its own test module)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import Metrics  # noqa: E402
from base_simulation import _MetricsMockAgent  # noqa: E402


def _grid(types):
    """Agent-object grid (None = empty) from an int grid (-1 empty, 0/1 types)."""
    grid = np.full(types.shape, None)
    for r in range(types.shape[0]):
        for c in range(types.shape[1]):
            if types[r, c] >= 0:
                grid[r, c] = _MetricsMockAgent(int(types[r, c]))
    return grid


def _pairwise_distance(grid):
    """The pre-2026-09-02 O(n^2) definition, kept as the oracle."""
    dists = []
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            agent = grid[r][c]
            if agent:
                min_dist = float("inf")
                for r2 in range(height):
                    for c2 in range(width):
                        target = grid[r2][c2]
                        if target and target.type_id != agent.type_id:
                            min_dist = min(min_dist, abs(r - r2) + abs(c - c2))
                if min_dist != float("inf"):
                    dists.append(min_dist)
    return np.mean(dists) if dists else 0


@pytest.mark.parametrize("size", [10, 20])
def test_distance_matches_the_pairwise_definition(size):
    rng = np.random.default_rng(size)
    for _ in range(10):
        types = rng.choice([-1, 0, 1], size=(size, size), p=[0.2, 0.4, 0.4])
        grid = _grid(types)
        assert Metrics.compute_distance(grid) == _pairwise_distance(grid)


def test_distance_edge_cases():
    empty = _grid(np.full((10, 10), -1))
    assert Metrics.compute_distance(empty) == 0

    one_type = _grid(np.where(np.arange(100).reshape(10, 10) % 3 == 0, 0, -1))
    assert Metrics.compute_distance(one_type) == 0     # nobody has an other-type agent

    corners = np.full((10, 10), -1)
    corners[0, 0], corners[9, 9] = 0, 1
    assert Metrics.compute_distance(_grid(corners)) == 18

    halves = np.full((10, 10), -1)
    halves[:, :5], halves[:, 5:] = 0, 1
    # column 4 is 1 away from column 5, column 0 is 5 away: mean over 1..5 twice
    assert Metrics.compute_distance(_grid(halves)) == pytest.approx(3.0)


def test_calculate_all_metrics_keys():
    types = np.random.default_rng(1).choice([-1, 0, 1], size=(10, 10))
    metrics = Metrics.calculate_all_metrics(_grid(types))
    assert set(metrics) == {"clusters", "switch_rate", "distance", "mix_deviation",
                            "share", "ghetto_rate", "dissimilarity_index"}
