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


# --- the pre-2026-09-05 loop implementations, kept as the oracle -------------
#
# Every metric in Metrics.py is now a whole-array computation on the int grid;
# these are the object-grid loops it replaced, verbatim, so the equivalence is
# pinned bit for bit rather than by intent.

def _ref_count_clusters(grid):
    visited = np.zeros(grid.shape, dtype=bool)
    clusters = 0

    def dfs(r, c, type_id):
        stack = [(r, c)]
        while stack:
            r0, c0 = stack.pop()
            if visited[r0][c0]:
                continue
            visited[r0][c0] = True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r1, c1 = r0 + dr, c0 + dc
                if 0 <= r1 < grid.shape[0] and 0 <= c1 < grid.shape[1]:
                    agent = grid[r1][c1]
                    if agent and agent.type_id == type_id and not visited[r1][c1]:
                        stack.append((r1, c1))

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            agent = grid[r][c]
            if agent and not visited[r][c]:
                dfs(r, c, agent.type_id)
                clusters += 1
    return clusters


def _ref_switch_rate(grid):
    switches, total = 0, 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            agent = grid[r][c]
            if agent:
                types = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    r1, c1 = r + dr, c + dc
                    if 0 <= r1 < grid.shape[0] and 0 <= c1 < grid.shape[1]:
                        n = grid[r1][c1]
                        if n:
                            types.append(n.type_id)
                if len(types) > 1:
                    total += len(types) - 1
                    switches += sum(1 for i in range(len(types) - 1) if types[i] != types[i + 1])
    return switches / total if total > 0 else 0


def _ref_mix_deviation(grid):
    deviations = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            agent = grid[r][c]
            if agent:
                like, unlike = 0, 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        r1, c1 = r + dr, c + dc
                        if 0 <= r1 < grid.shape[0] and 0 <= c1 < grid.shape[1]:
                            n = grid[r1][c1]
                            if n:
                                if n.type_id == agent.type_id:
                                    like += 1
                                else:
                                    unlike += 1
                total = like + unlike
                if total > 0:
                    deviations.append(abs(0.5 - like / total))
    return np.mean(deviations) if deviations else 0


def _ref_share(grid):
    like, unlike = 0, 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            agent = grid[r][c]
            if agent:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        r1, c1 = r + dr, c + dc
                        if 0 <= r1 < grid.shape[0] and 0 <= c1 < grid.shape[1]:
                            n = grid[r1][c1]
                            if n:
                                if n.type_id == agent.type_id:
                                    like += 1
                                else:
                                    unlike += 1
    total = like + unlike
    return like / total if total > 0 else 0


def _ref_ghetto_rate(grid):
    ghettos = 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            agent = grid[r][c]
            if agent:
                has_unlike = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        r1, c1 = r + dr, c + dc
                        if 0 <= r1 < grid.shape[0] and 0 <= c1 < grid.shape[1]:
                            n = grid[r1][c1]
                            if n and n.type_id != agent.type_id:
                                has_unlike = True
                                break
                if not has_unlike:
                    ghettos += 1
    return ghettos


_REFERENCE = {
    "clusters": (_ref_count_clusters, Metrics.count_clusters),
    "switch_rate": (_ref_switch_rate, Metrics.compute_switch_rate),
    "distance": (_pairwise_distance, Metrics.compute_distance),
    "mix_deviation": (_ref_mix_deviation, Metrics.compute_mix_deviation),
    "share": (_ref_share, Metrics.compute_share),
    "ghetto_rate": (_ref_ghetto_rate, Metrics.compute_ghetto_rate),
}


def _adversarial_grids():
    rng = np.random.default_rng(7)
    grids = {
        "empty": np.full((10, 10), -1),
        "all_one_type": np.zeros((10, 10), int),
        "1x1": np.array([[0]]),
        "1x2": np.array([[0, 1]]),
        "7x13": rng.integers(-1, 2, (7, 13)),
        "three_types": rng.integers(-1, 3, (10, 10)),
        "full_random": rng.integers(0, 2, (10, 10)),
        "sparse_random": np.where(rng.random((20, 20)) < 0.2, rng.integers(0, 2, (20, 20)), -1),
    }
    g = np.full((10, 10), -1); g[3, 3] = 0; grids["single_agent"] = g
    g = np.full((10, 10), -1); g[0, 0] = 0; g[9, 9] = 1; grids["two_far_apart"] = g
    g = np.full((5, 5), -1); g[2, 2] = 0; g[2, 3] = 1; grids["adjacent_pair"] = g
    g = np.full((10, 10), -1); g[0, 0] = 0; g[0, 9] = 1; g[9, 0] = 1; g[9, 9] = 0; grids["corners"] = g
    g = np.zeros((10, 10), int); g[:, 5:] = 1; grids["halves"] = g
    return grids


@pytest.mark.parametrize("name", sorted(_adversarial_grids()))
@pytest.mark.parametrize("metric", sorted(_REFERENCE))
def test_vectorised_metric_matches_the_loop_oracle(name, metric):
    types = _adversarial_grids()[name]
    reference, vectorised = _REFERENCE[metric]
    expected = reference(_grid(types))
    assert vectorised(types) == expected                # int grid in
    assert vectorised(_grid(types)) == expected         # object grid in


@pytest.mark.parametrize("seed", range(6))
def test_vectorised_metrics_match_on_random_boards(seed):
    rng = np.random.default_rng(seed)
    size = int(rng.choice([10, 20]))
    density = float(rng.choice([0.3, 0.8, 1.0]))
    types = np.where(rng.random((size, size)) < density, rng.integers(0, 2, (size, size)), -1)
    grid = _grid(types)
    expected = {name: ref(grid) for name, (ref, _) in _REFERENCE.items()}
    expected["dissimilarity_index"] = Metrics.compute_dissimilarity(grid)
    got = Metrics.calculate_all_metrics(types)
    assert got == expected                              # exact, not approx
    assert Metrics.calculate_all_metrics(grid) == expected


def test_as_int_grid_round_trips_and_rejects_overflow():
    types = np.array([[-1, 0], [1, -1]])
    assert Metrics.as_int_grid(_grid(types)).dtype == np.int8
    assert np.array_equal(Metrics.as_int_grid(_grid(types)), types)
    assert Metrics.as_int_grid(types.astype(np.int64)).dtype == np.int8
    with pytest.raises(OverflowError):
        Metrics.as_int_grid(np.array([[300]]))
