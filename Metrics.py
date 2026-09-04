import numpy as np
from scipy.ndimage import distance_transform_cdt

# The dissimilarity index is the paper's headline metric and lives in its own
# module together with its tract partition, its frozen 10x10 reference
# implementation and its reference values; the six metrics below are carried
# for comparability with the earlier literature. Re-exported here so
# `from Metrics import compute_dissimilarity_from_int_grid` keeps working.
from DissimilarityIndex import (  # noqa: F401
    compute_dissimilarity,
    compute_dissimilarity_from_int_grid,
    tract_map,
)


def calculate_all_metrics(grid):
    """All seven segregation metrics for one grid of Agent objects / None.

    The dissimilarity index used to be computed only downstream, from the
    saved states (analysis_tools/dissimilarity_index_over_time.py), so every
    consumer had to recompute it and the per-step series was reconstructed by
    mapping move-frames back onto steps. It is a per-grid statistic like the
    other six, so it is computed here with them (2026-08-25, batch C).
    """
    clusters = count_clusters(grid)
    switch_rate = compute_switch_rate(grid)
    distance = compute_distance(grid)
    mix_dev = compute_mix_deviation(grid)
    share = compute_share(grid)
    ghetto_rate = compute_ghetto_rate(grid)
    dissimilarity = compute_dissimilarity(grid)
    return {
        "clusters": clusters,
        "switch_rate": switch_rate,
        "distance": distance,
        "mix_deviation": mix_dev,
        "share": share,
        "ghetto_rate": ghetto_rate,
        "dissimilarity_index": dissimilarity
    }


def count_clusters(grid):
    visited = np.zeros(grid.shape, dtype=bool)
    clusters = 0

    def dfs(r, c, type_id):
        stack = [(r, c)]
        while stack:
            r0, c0 = stack.pop()
            if visited[r0][c0]:
                continue
            visited[r0][c0] = True
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
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

def compute_switch_rate(grid):
    switches, total = 0, 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            agent = grid[r][c]
            if agent:
                types = []
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    r1, c1 = r + dr, c + dc
                    if 0 <= r1 < grid.shape[0] and 0 <= c1 < grid.shape[1]:
                        n = grid[r1][c1]
                        if n:
                            types.append(n.type_id)
                if len(types) > 1:
                    total += len(types) - 1
                    switches += sum(1 for i in range(len(types)-1) if types[i] != types[i+1])
    return switches / total if total > 0 else 0

def compute_distance(grid):
    """Mean, over agents, of the taxicab distance to the nearest agent of the
    other type. Agents with no other-type agent on the grid are skipped; 0
    when nobody has one.

    distance_transform_cdt gives every cell its taxicab distance to the nearest
    zero of its input, so with "not other-type" as the input, agent cells read
    off exactly what the old O(n^2) pairwise scan computed (checked equal in
    tests/test_metrics.py).
    """
    types = np.full(grid.shape, -1, dtype=np.int8)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r][c]:
                types[r, c] = grid[r][c].type_id
    dists = []
    for type_id in np.unique(types[types >= 0]):
        other = (types >= 0) & (types != type_id)
        if not other.any():
            continue
        to_other = distance_transform_cdt(~other, metric="taxicab")
        dists.extend(to_other[types == type_id].tolist())
    return float(np.mean(dists)) if dists else 0


def compute_mix_deviation(grid):
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
                    deviation = abs(0.5 - like / total)
                    deviations.append(deviation)
    return np.mean(deviations) if deviations else 0

def compute_share(grid):
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

def compute_ghetto_rate(grid):
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
