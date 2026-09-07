import numpy as np
from scipy.ndimage import distance_transform_cdt, label

# The dissimilarity index is the paper's headline metric and lives in its own
# module together with its tract partition, its frozen 10x10 reference
# implementation and its reference values; the six metrics below are carried
# for comparability with the earlier literature. Re-exported here so
# `from Metrics import compute_dissimilarity_from_int_grid` keeps working.
from DissimilarityIndex import (  # noqa: F401
    as_int_grid,
    compute_dissimilarity,
    compute_dissimilarity_from_int_grid,
    tract_map,
)

# Every metric here is a function of the int grid (-1 empty, else type_id),
# computed with whole-array numpy operations. Until 2026-09-05 each one
# walked the grid of Agent objects in nested Python loops with per-cell
# attribute access — six passes over the same 8-neighbourhood, ~4.4 ms per
# 20x20 step against ~0.3 ms now, and 51-63% of a value-function run's time.
# The loop versions live on in tests/test_metrics.py as the oracle every
# function below is checked against, bit for bit, on real and adversarial
# grids. Callers may pass either representation; as_int_grid converts.

_NB8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
_NB4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]     # up, down, left, right: the switch-rate order
_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def calculate_all_metrics(grid):
    """All seven segregation metrics for one grid (Agent objects / None, or ints).

    The dissimilarity index used to be computed only downstream, from the
    saved states (analysis_tools/dissimilarity_index_over_time.py), so every
    consumer had to recompute it and the per-step series was reconstructed by
    mapping move-frames back onto steps. It is a per-grid statistic like the
    other six, so it is computed here with them (2026-08-25, batch C).
    """
    t = as_int_grid(grid)
    like, unlike = _neighbourhood_counts(t)
    occupied = t >= 0
    return {
        "clusters": _count_clusters(t, occupied),
        "switch_rate": _switch_rate(t, occupied),
        "distance": _distance(t, occupied),
        "mix_deviation": _mix_deviation(like, unlike, occupied),
        "share": _share(like, unlike, occupied),
        "ghetto_rate": _ghetto_rate(unlike, occupied),
        "dissimilarity_index": compute_dissimilarity_from_int_grid(t),
    }


def _neighbour(t, dr, dc):
    """t[r + dr, c + dc] at every cell; -1 (empty) beyond the edge."""
    out = np.full(t.shape, -1, dtype=t.dtype)
    rows, cols = t.shape
    r0, r1 = max(0, -dr), min(rows, rows - dr)
    c0, c1 = max(0, -dc), min(cols, cols - dc)
    out[r0:r1, c0:c1] = t[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return out


def _neighbourhood_counts(t):
    """Per cell, how many of its 8 neighbours are occupied by the same / the
    other type (walls and empties count as neither)."""
    like = np.zeros(t.shape, dtype=np.int16)
    unlike = np.zeros(t.shape, dtype=np.int16)
    for dr, dc in _NB8:
        n = _neighbour(t, dr, dc)
        present = n >= 0
        same = present & (n == t)
        like += same
        unlike += present & ~same
    return like, unlike


def _count_clusters(t, occupied):
    """4-connected same-type components, summed over the types present."""
    clusters = 0
    for type_id in np.unique(t[occupied]):
        clusters += label(t == type_id, structure=_CROSS)[1]
    return int(clusters)


def _switch_rate(t, occupied):
    """Over agents with more than one occupied 4-neighbour: type changes
    between consecutive occupied neighbours (in up, down, left, right order)
    divided by the number of consecutive pairs."""
    prev = np.full(t.shape, -1, dtype=t.dtype)
    have_prev = np.zeros(t.shape, dtype=bool)
    count = np.zeros(t.shape, dtype=np.int8)
    pairs = np.zeros(t.shape, dtype=np.int8)
    switches = np.zeros(t.shape, dtype=np.int8)
    for dr, dc in _NB4:
        n = _neighbour(t, dr, dc)
        present = n >= 0
        both = present & have_prev
        pairs += both
        switches += both & (n != prev)
        prev = np.where(present, n, prev)
        have_prev |= present
        count += present
    agents = occupied & (count > 1)
    total = int(pairs[agents].sum())
    return int(switches[agents].sum()) / total if total > 0 else 0


def _distance(t, occupied):
    """Mean, over agents, of the taxicab distance to the nearest agent of the
    other type. Agents with no other-type agent on the grid are skipped; 0
    when nobody has one.

    distance_transform_cdt gives every cell its taxicab distance to the nearest
    zero of its input, so with "not other-type" as the input, agent cells read
    off exactly what the old O(n^2) pairwise scan computed (checked equal in
    tests/test_metrics.py).
    """
    dists = []
    for type_id in np.unique(t[occupied]):
        other = occupied & (t != type_id)
        if not other.any():
            continue
        to_other = distance_transform_cdt(~other, metric="taxicab")
        dists.extend(to_other[t == type_id].tolist())
    return float(np.mean(dists)) if dists else 0


def _mix_deviation(like, unlike, occupied):
    """Mean over agents with at least one occupied neighbour of
    |0.5 - like / (like + unlike)|."""
    total = like + unlike
    agents = occupied & (total > 0)
    if not agents.any():
        return 0
    return np.mean(np.abs(0.5 - like[agents] / total[agents]))


def _share(like, unlike, occupied):
    """Same-type share of all agent-neighbour pairs."""
    like_total = int(like[occupied].sum())
    total = like_total + int(unlike[occupied].sum())
    return like_total / total if total > 0 else 0


def _ghetto_rate(unlike, occupied):
    """Agents with no other-type neighbour."""
    return int((occupied & (unlike == 0)).sum())


# --- per-metric entry points (either representation) -------------------------

def count_clusters(grid):
    t = as_int_grid(grid)
    return _count_clusters(t, t >= 0)


def compute_switch_rate(grid):
    t = as_int_grid(grid)
    return _switch_rate(t, t >= 0)


def compute_distance(grid):
    t = as_int_grid(grid)
    return _distance(t, t >= 0)


def compute_mix_deviation(grid):
    t = as_int_grid(grid)
    like, unlike = _neighbourhood_counts(t)
    return _mix_deviation(like, unlike, t >= 0)


def compute_share(grid):
    t = as_int_grid(grid)
    like, unlike = _neighbourhood_counts(t)
    return _share(like, unlike, t >= 0)


def compute_ghetto_rate(grid):
    t = as_int_grid(grid)
    _, unlike = _neighbourhood_counts(t)
    return _ghetto_rate(unlike, t >= 0)
