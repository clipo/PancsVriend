"""Dissimilarity index — the paper's headline segregation metric.

Everything about the dissimilarity index lives in this one file: the tract
partition, the index itself, the frozen 10x10 reference implementation kept for
equivalence testing, the reference VALUES that DI figures are annotated with
(random-allocation baseline and attainable maximum), and the plotting helpers.
The other six metrics in Metrics.py are carried for comparability with the
earlier literature; this one carries the paper, so it gets its own module
instead of being one function among seven.

Metrics.py imports the three computational functions from here, so existing
call sites (`from Metrics import compute_dissimilarity_from_int_grid`, etc.)
keep working unchanged.

matplotlib is imported lazily inside the plotting functions on purpose:
Metrics.py sits on the simulation's hot path and must not drag a plotting
stack into every run.

CLI — regenerate the tract-map figure and the stored maps/reference table:

    python DissimilarityIndex.py
    python DissimilarityIndex.py --sizes 10 20 30 40 50 --draws 20000
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

# 3x3 block partition: 9 tracts at every grid size. Fixed COUNT, growing
# tracts — see tract_map() for why the count is what is held constant.
N_TRACTS = 9

# Grid sizes the tract-map figure covers by default.
DEFAULT_SIZES = (10, 20, 30, 40, 50)

# Monte Carlo draws for the random-allocation baseline. The null is sampled
# exactly (multivariate hypergeometric on tract counts, not a shuffled grid),
# so 5000 already puts the standard error of the mean near 5e-4 — two orders
# below anything visible in a figure.
DEFAULT_DRAWS = 20000   # was 5000; see vf_rank_stability.metric_null
DEFAULT_SEED = 0


# ---------------------------------------------------------------------------
# The tract partition
# ---------------------------------------------------------------------------

def tract_map(size):
    """(size, size) int8 map of 9 tracts — a 3x3 block partition of the grid.

    Vectorised form of get_census_tract(): the edge bands are size // 3 wide
    and the centre band takes the remainder, so the corner tracts stay the
    smallest and the centre tract the largest at every grid size, exactly as
    in the original 10x10 map (3-4-3 -> 20: 6-8-6, 30: 10-10-10, 50: 16-18-16).

    What is held fixed across grid sizes is the tract COUNT and the tract
    SHAPE ordering (small corners, big centre), not the tract proportions;
    the null baseline is governed by 1/sqrt(agents per tract), see
    random_baseline(), and moves by well under 1% between partition rules.
    """
    edge = size // 3
    bands = np.zeros(size, dtype=np.int8)
    bands[edge:size - edge] = 1
    bands[size - edge:] = 2
    return (bands[:, None] * 3 + bands[None, :]).astype(np.int8)


def tract_sizes(size):
    """Cells per tract, tract 0..8, for a `size` x `size` grid."""
    return np.bincount(tract_map(size).ravel(), minlength=N_TRACTS)


def band_widths(size):
    """The three row (== column) band widths, e.g. (3, 4, 3) at size 10."""
    b = tract_map(size)[0]
    return tuple(int((b == k).sum()) for k in range(3))


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------

def compute_dissimilarity_from_int_grid(int_grid):
    """DI for an int grid (-1 empty, 0/1 type ids). 0.5*sum|a_i/A - b_i/B|.

    1.0 when one group is present and the other absent; 0.0 on an empty grid.
    """
    grid = np.asarray(int_grid)
    tracts = tract_map(grid.shape[0])
    if grid.shape != tracts.shape:
        raise ValueError(f"dissimilarity expects a square grid, got {grid.shape}")
    flat, tflat = grid.reshape(-1), tracts.reshape(-1)
    mask0, mask1 = flat == 0, flat == 1
    total0, total1 = int(mask0.sum()), int(mask1.sum())
    if total0 == 0 or total1 == 0:
        return 1.0 if (total0 + total1) > 0 else 0.0
    counts0 = np.bincount(tflat[mask0], minlength=N_TRACTS)
    counts1 = np.bincount(tflat[mask1], minlength=N_TRACTS)
    return float(0.5 * np.abs(counts0 / total0 - counts1 / total1).sum())


def as_int_grid(grid):
    """The int8 grid (-1 empty, else type_id) for either representation.

    Object grids of Agent / None are the simulation's live representation;
    int grids are what states_run_<id>.npz stores and what every metric is
    a function of. One converter, shared by Metrics.calculate_all_metrics,
    Simulation._grid_to_int and this module, replaces four copies of the
    same cell-by-cell loop (2026-09-05). Int inputs come back as int8 without
    a scan; a type_id outside int8 raises rather than wrapping.
    """
    grid = np.asarray(grid)
    if grid.dtype != object:
        if grid.dtype == np.int8:
            return grid
        if grid.size and (grid.min() < -128 or grid.max() > 127):
            raise OverflowError("type_id outside int8 range")
        return grid.astype(np.int8)
    int_grid = np.full(grid.shape, -1, dtype=np.int8)
    occupied = grid != None                    # noqa: E711 — elementwise on object arrays
    if occupied.any():
        int_grid[occupied] = [agent.type_id for agent in grid[occupied]]
    return int_grid


def compute_dissimilarity(grid):
    """DI for a grid of Agent objects / None (the simulation's representation)."""
    return compute_dissimilarity_from_int_grid(as_int_grid(grid))


# ---------------------------------------------------------------------------
# Reference implementation (scalar, per-cell)
# ---------------------------------------------------------------------------
# The original per-cell implementation, kept verbatim as the thing the
# vectorised code above is checked against — see
# tests/test_dissimilarity_index.py. get_census_tract() is the single
# definition of the tract partition; tract_map() must agree with it at every
# size. Nothing on the simulation's hot path calls these.

def compute_dissimilarity_index(grid):
    """
    Calculate the Dissimilarity Index for the Schelling segregation model.

    The Dissimilarity Index measures the proportion of one group that would need
    to move to achieve an even distribution across spatial units.

    Args:
        grid: 2D numpy array containing Agent objects or None for empty cells
              (must be square)

    Returns:
        float: Dissimilarity index value between 0 (perfect integration) and 1 (complete segregation)

    Raises:
        ValueError: If grid is not square
    """
    # Verify grid is square
    if grid.shape[0] != grid.shape[1]:
        raise ValueError(f"Grid must be square, but got {grid.shape[0]}x{grid.shape[1]}")

    grid_size = grid.shape[0]
    
    # Initialize counters for each census tract
    # tract_counts[tract_id] = [type_0_count, type_1_count]
    tract_counts = {i: [0, 0] for i in range(9)}

    # Count agents by type in each census tract
    total_type_0 = 0
    total_type_1 = 0

    for row in range(grid_size):
        for col in range(grid_size):
            agent = grid[row][col]
            if agent is not None:
                tract_id = get_census_tract(row, col, grid_size)
                if agent.type_id == 0:
                    tract_counts[tract_id][0] += 1
                    total_type_0 += 1
                elif agent.type_id == 1:
                    tract_counts[tract_id][1] += 1
                    total_type_1 += 1
    
    # Handle edge case: if either type has zero total agents
    if total_type_0 == 0 or total_type_1 == 0:
        # Complete segregation or only one type present
        return 1.0 if (total_type_0 > 0 or total_type_1 > 0) else 0.0
    
    # Calculate Dissimilarity Index
    # D = 0.5 * Σ |ai/A - bi/B|
    # where ai = type 0 agents in tract i, A = total type 0 agents
    #       bi = type 1 agents in tract i, B = total type 1 agents
    dissimilarity_sum = 0.0
    
    for tract_id in range(9):
        type_0_in_tract = tract_counts[tract_id][0]
        type_1_in_tract = tract_counts[tract_id][1]
        
        # Calculate proportions
        prop_type_0 = type_0_in_tract / total_type_0
        prop_type_1 = type_1_in_tract / total_type_1
        
        # Add absolute difference to sum
        dissimilarity_sum += abs(prop_type_0 - prop_type_1)
    
    # Final dissimilarity index
    dissimilarity_index = 0.5 * dissimilarity_sum
    
    return dissimilarity_index


def get_census_tract(row, col, grid_size=10):
    """
    Map grid coordinates to census tract ID (0-8).

    The grid is divided into 9 census tracts (3x3 layout). Extra cells from
    non-divisible grid sizes are allocated to the center tracts, keeping
    corner tracts smaller.

    Division logic:
    - edge_size = grid_size // 3
    - Section 0 (edge): [0, edge_size)
    - Section 1 (center): [edge_size, grid_size - edge_size)
    - Section 2 (edge): [grid_size - edge_size, grid_size)

    Example for 10x10 grid (edge_size=3, center_size=4):
    +-------+--------+-------+
    | Tract | Tract  | Tract |
    |   0   |   1    |   2   |
    | (3x3) | (3x4)  | (3x3) |
    +-------+--------+-------+
    | Tract | Tract  | Tract |
    |   3   |   4    |   5   |
    | (4x3) | (4x4)  | (4x3) |
    +-------+--------+-------+
    | Tract | Tract  | Tract |
    |   6   |   7    |   8   |
    | (3x3) | (3x4)  | (3x3) |
    +-------+--------+-------+

    Example for 20x20 grid (edge_size=6, center_size=8):
    - Corner tracts (0,2,6,8): 6x6 = 36 cells each
    - Edge tracts (1,3,5,7): 6x8 or 8x6 = 48 cells each
    - Center tract (4): 8x8 = 64 cells

    Args:
        row: Row index
        col: Column index
        grid_size: Size of the square grid (default 10)

    Returns:
        int: Census tract ID (0-8)
    """
    edge_size = grid_size // 3

    # Determine which row section (0, 1, or 2)
    if row < edge_size:
        row_section = 0
    elif row < grid_size - edge_size:
        row_section = 1
    else:
        row_section = 2

    # Determine which column section (0, 1, or 2)
    if col < edge_size:
        col_section = 0
    elif col < grid_size - edge_size:
        col_section = 1
    else:
        col_section = 2

    # Calculate tract ID from row and column sections
    tract_id = row_section * 3 + col_section

    return tract_id


# ---------------------------------------------------------------------------
# Reference values: where the scale's floor and ceiling actually sit
# ---------------------------------------------------------------------------
# A DI of 0.30 means "indistinguishable from random" on a 10x10 grid and
# "clearly segregated" on a 20x20 one, because the baseline halves with the
# grid. These functions are what the tract-map figure annotates each panel
# with; they are not drawn on the experiment result plots.

def default_population(size):
    """(n_a, n_b) for `size`, holding config.py's density and A:B ratio fixed.

    config.py is the 10x10 design (40 + 40 agents on 100 cells = 80% density,
    1:1); a 20x20 run at the same density is 160 + 160. Imported lazily so the
    simulation hot path does not pull config in through Metrics.
    """
    import config
    n_cells_cfg = config.GRID_SIZE ** 2
    n_agents_cfg = config.NUM_TYPE_A + config.NUM_TYPE_B
    density = n_agents_cfg / n_cells_cfg
    share_a = config.NUM_TYPE_A / n_agents_cfg
    n_agents = int(round(density * size * size))
    n_a = int(round(share_a * n_agents))
    return n_a, n_agents - n_a


def _resolve_population(size, n_a, n_b):
    if n_a is None or n_b is None:
        d_a, d_b = default_population(size)
        n_a = d_a if n_a is None else n_a
        n_b = d_b if n_b is None else n_b
    if n_a + n_b > size * size:
        raise ValueError(f"{n_a} + {n_b} agents do not fit on a {size}x{size} grid")
    return int(n_a), int(n_b)


# Each of the four samplers below is a thin wrapper that resolves the agent
# counts BEFORE calling its cached implementation. Caching on the public
# signature would key on n_a=None and freeze whatever config.py happened to say
# at the first call, so a later call under a different grid design would get the
# earlier answer back silently.

def random_baseline(size, n_a=None, n_b=None, draws=DEFAULT_DRAWS,
                    seed=DEFAULT_SEED):
    """Sampling distribution of DI under random allocation — the "no sorting" floor.

    The null is "each agent is equally likely to occupy any free cell", under
    which both groups have the same expected share of every tract, so the
    per-tract differences the index sums are centred on zero. What is left is
    sampling noise, whose scale is 1/sqrt(agents per tract) — which is why this
    baseline HALVES from 10x10 to 20x20 even though the tract count is fixed.

    Drawn by shuffling the agents over the cells, which is exactly the null, and
    then measured with compute_dissimilarity_from_int_grid — the same function
    the simulation and the analysis scripts use. Sampling in tract-count space
    instead (multivariate hypergeometric on the capacities) would be about 0.2s
    faster over the whole figure and would need a second implementation of the
    index; not a trade worth making.

    Returns a dict with mean, sd and the 5/50/95 percentiles.
    """
    n_a, n_b = _resolve_population(size, n_a, n_b)
    n_cells = size * size
    cells = np.array([-1] * (n_cells - n_a - n_b) + [0] * n_a + [1] * n_b,
                     dtype=np.int8)
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for i in range(draws):
        rng.shuffle(cells)
        values[i] = compute_dissimilarity_from_int_grid(cells.reshape(size, size))
    q5, q50, q95 = np.percentile(values, [5, 50, 95])
    return {"size": size, "n_a": n_a, "n_b": n_b, "draws": draws,
            "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "p5": float(q5), "median": float(q50), "p95": float(q95)}


def _tracts_connected(tracts):
    """4-connected on the 3x3 tract lattice — i.e. one contiguous cluster."""
    if not tracts:
        return False
    remaining = set(tracts)
    seen = {next(iter(remaining))}
    stack = list(seen)
    while stack:
        r, c = divmod(stack.pop(), 3)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r2, c2 = r + dr, c + dc
            if 0 <= r2 < 3 and 0 <= c2 < 3:
                nb = r2 * 3 + c2
                if nb in remaining and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
    return seen == remaining


def max_configuration(size, n_a=None, n_b=None):
    """Most segregated arrangement reachable at this size and population.

    DI hits its ceiling of 1 exactly when no tract holds both groups, so the
    question is combinatorial: can the 9 tracts be split into two disjoint sets
    with capacity >= n_a and >= n_b? Vacant cells are the slack that makes this
    easy — at 80% density every size from 10 to 50 can park one whole group in
    five tracts and the other in four.

    Contiguous solutions are preferred, and at these sizes one always exists
    (A = tracts 0-4, B = tracts 5-8), so the ceiling is NOT an artefact of
    letting a group occupy scattered tracts: two ordinary blobs reach 1.0 as
    long as the boundary between them runs along tract lines. What costs a real
    run its last 0.2-0.3 is boundary MISALIGNMENT, not shape — see
    two_cluster_di().

    Returns (max_di, grid); the DI is MEASURED off the returned grid rather
    than asserted, so the number and the witness cannot drift apart.
    """
    n_a, n_b = _resolve_population(size, n_a, n_b)
    caps = tract_sizes(size)
    label = fallback = None
    for cand in itertools.product((0, 1, 2), repeat=N_TRACTS):
        set_a = [i for i in range(N_TRACTS) if cand[i] == 0]
        if sum(caps[i] for i in set_a) < n_a:
            continue
        set_b = [i for i in range(N_TRACTS) if cand[i] == 1]
        if sum(caps[i] for i in set_b) < n_b:
            continue
        if _tracts_connected(set_a) and _tracts_connected(set_b):
            label = cand
            break
        if fallback is None:
            fallback = cand
    label = label if label is not None else fallback
    if label is None:
        raise ValueError(
            f"no tract-pure arrangement fits {n_a}+{n_b} agents on {size}x{size}; "
            "DI cannot reach 1.0 for this population and the ceiling would have "
            "to be solved for numerically")

    tracts = tract_map(size)
    grid = np.full((size, size), -1, dtype=np.int8)
    flat, tflat = grid.reshape(-1), tracts.reshape(-1)
    for group, target in ((0, n_a), (1, n_b)):
        remaining = target
        for tract in range(N_TRACTS):
            if remaining <= 0:
                break
            if label[tract] != group:
                continue
            cells = np.flatnonzero(tflat == tract)[:remaining]
            flat[cells] = group
            remaining -= len(cells)
    return compute_dissimilarity_from_int_grid(grid), grid


def theoretical_max(size, n_a=None, n_b=None):
    """Highest DI attainable at this grid size and population (see above)."""
    return max_configuration(size, n_a, n_b)[0]


def half_split_di(size, n_a=None, n_b=None, draws=2000, seed=DEFAULT_SEED):
    """DI of a clean left-half / right-half spatial split — an interpretive anchor.

    Not a bound; it is the configuration a reader pictures when they hear
    "totally segregated city", and it is worth knowing that it scores 0.60 on a
    10x10 and 0.70 on a 20x20 rather than 1.0. It falls short because the
    middle column band straddles the cut, so its three tracts are internally
    balanced and contribute nothing. Vacancies are placed at random within each
    half and the value averaged over `draws`.
    """
    n_a, n_b = _resolve_population(size, n_a, n_b)
    n_cells = size * size
    cut = size // 2
    left = np.flatnonzero((np.arange(n_cells) % size) < cut)
    right = np.flatnonzero((np.arange(n_cells) % size) >= cut)
    if n_a > len(left) or n_b > len(right):
        raise ValueError("population does not fit in half the grid")
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for i in range(draws):
        a_cells = rng.choice(left, size=n_a, replace=False)
        b_cells = rng.choice(right, size=n_b, replace=False)
        grid = np.full(n_cells, -1, dtype=np.int8)
        grid[a_cells] = 0
        grid[b_cells] = 1
        values[i] = compute_dissimilarity_from_int_grid(grid.reshape(size, size))
    return float(values.mean())


def two_cluster_di(size, n_a=None, n_b=None, gap="random", draws=4000,
                   seed=DEFAULT_SEED):
    """DI of a two-cluster city whose boundary falls at a RANDOM angle.

    The realistic ceiling, as opposed to the definitional one. theoretical_max()
    returns 1.0 because a boundary that follows tract lines makes every tract
    pure; a segregation pattern that emerges on its own has no reason to line up
    with the tract grid, and every cell of misalignment costs DI. This samples
    that: a straight line at a uniformly random angle, the two half-planes given
    to the two groups, vacancies scattered at random inside each.

    What varies between draws is the boundary's ANGLE only. Its POSITION is
    pinned: with n_a == n_b and the two groups filling their sides, the cut has
    to bisect the grid, so it passes through the centre in every draw. On a
    20x20 the same city then scores anywhere from 0.67 to 0.91 purely on where
    that line lands relative to the tract edges — the modifiable areal unit
    problem (MAUP), the index being a property of the (data, partition) pair
    rather than of the data alone.

    `gap` controls how much empty space separates the two cities, in CELLS
    taken out of the middle of the projection order (on a 20x20 with 80
    vacancies, 20 cells is a strip one cell thick, 80 cells is four cells
    thick — the widest the density allows):
        "random"   the default. Width drawn uniformly from 0 to every vacancy
                   in the grid, so the value is a marginal over the whole range
                   of ways two cities can sit next to each other rather than a
                   single arbitrary arrangement.
        0          the touching extreme. Vacancies still scatter inside each
                   city and some land on the boundary, but never enough to
                   separate them: measured over 2000 draws at 20x20 the two
                   cities are orthogonally adjacent somewhere in 100% of draws,
                   with ~18 contact pairs on average.
        int        a fixed buffer of exactly that many cells.
    The gap matters and is not a detail — at 20x20 the mean runs 0.79 (touching)
    -> 0.83 (one-cell strip) -> 0.93 (four-cell strip), because a buffer deletes
    exactly the mixed cells nearest the cut. Separation becomes complete as soon
    as the buffer is a full one-cell strip; below that it is ragged and the
    cities still touch.

    Returns a dict with mean, sd, the 5/95 percentiles, min and max over `draws`.
    """
    n_a, n_b = _resolve_population(size, n_a, n_b)
    rng = np.random.default_rng(seed)
    n_cells = size * size
    rows, cols = np.divmod(np.arange(n_cells), size)
    y, x = rows + 0.5, cols + 0.5
    # Widest buffer that still leaves each half able to hold its own group.
    max_gap = n_cells - 2 * max(n_a, n_b)
    if gap != "random" and not 0 <= int(gap) <= max_gap:
        raise ValueError(f"gap must be between 0 and {max_gap} cells on a {size}x{size} grid")
    values = np.empty(draws)
    for i in range(draws):
        theta = rng.uniform(0, np.pi)
        # Rank cells by their projection onto the cut's normal; the lowest cells
        # are one side of the line, the highest the other, and any buffer is the
        # band in the middle.
        proj = x * np.cos(theta) + y * np.sin(theta)
        order = np.argsort(proj + rng.uniform(0, 1e-9, n_cells))
        width = int(rng.integers(0, max_gap + 1)) if gap == "random" else int(gap)
        side = (n_cells - width) // 2
        a_cells = rng.choice(order[:side], size=n_a, replace=False)
        b_cells = rng.choice(order[n_cells - side:], size=n_b, replace=False)
        grid = np.full(n_cells, -1, dtype=np.int8)
        grid[a_cells] = 0
        grid[b_cells] = 1
        values[i] = compute_dissimilarity_from_int_grid(grid.reshape(size, size))
    q5, q95 = np.percentile(values, [5, 95])
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "p5": float(q5), "p95": float(q95),
            "min": float(values.min()), "max": float(values.max())}


def di_reference(size, n_a=None, n_b=None, draws=DEFAULT_DRAWS,
                 seed=DEFAULT_SEED):
    """Every reference value for one grid size, in one dict."""
    n_a, n_b = _resolve_population(size, n_a, n_b)
    base = random_baseline(size, n_a, n_b, draws, seed)
    return {**base,
            "band_widths": band_widths(size),
            "tract_sizes": tuple(int(v) for v in tract_sizes(size)),
            "max": theoretical_max(size, n_a, n_b),
            "half_split": half_split_di(size, n_a, n_b, seed=seed),
            "two_cluster": two_cluster_di(size, n_a, n_b, seed=seed)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
# House style for the reference annotations: muted ink, never a series colour,
# always direct-labelled so the two lines are told apart by text and dash
# pattern rather than by colour alone.

REFERENCE_INK = "#52514e"
REFERENCE_BAND = "#d9d8d4"
# Categorical slots 1/2/3 of the validated palette (all-pairs CVD safe).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
# Sequential blue ramp, steps 100 -> 700, for tract-size magnitude.
BLUE_RAMP = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b")


def _tract_colormap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("tract_blues", BLUE_RAMP)


def plot_tract_maps(sizes=DEFAULT_SIZES, out_path=None, draws=DEFAULT_DRAWS,
                    dpi=300, n_a=None, n_b=None, refs=None):
    """Figure: how the 9 tracts and the DI scale change with grid size.

    Top row, one panel per size: the partition itself, each tract shaded by its
    share of the grid on a common scale so the panels are comparable, annotated
    with that size's DI floor and ceiling; pass `refs` to reuse values already
    sampled rather than sampling them twice. Bottom row: those reference values
    against grid size, which is where the actual finding shows up — the ceiling
    is flat at 1.0 while the random floor falls as 1/sqrt(n), so the usable
    range of the index widens with the grid.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if refs is None:
        refs = [di_reference(s, n_a, n_b, draws) for s in sizes]
    shares = np.concatenate([np.asarray(r["tract_sizes"]) / (s * s)
                             for s, r in zip(sizes, refs)])
    cmap = _tract_colormap()
    vmin, vmax = shares.min(), shares.max()

    fig = plt.figure(figsize=(3.1 * len(sizes), 7.6))
    gs = fig.add_gridspec(2, len(sizes), height_ratios=[1.0, 0.78],
                          hspace=0.30, wspace=0.16, top=0.86)

    for col, (size, ref) in enumerate(zip(sizes, refs)):
        ax = fig.add_subplot(gs[0, col])
        tracts = tract_map(size)
        counts = np.asarray(ref["tract_sizes"])
        # One rectangle per tract with a 2px surface gap, rather than an
        # imshow of the label array: the gap IS the tract boundary, and it
        # stays crisp at every size.
        for tract in range(N_TRACTS):
            rows = np.flatnonzero((tracts == tract).any(axis=1))
            cols = np.flatnonzero((tracts == tract).any(axis=0))
            r0, r1 = rows.min(), rows.max() + 1
            c0, c1 = cols.min(), cols.max() + 1
            share = counts[tract] / (size * size)
            pad = size * 0.004
            ax.add_patch(Rectangle((c0 + pad, r0 + pad),
                                   (c1 - c0) - 2 * pad, (r1 - r0) - 2 * pad,
                                   facecolor=cmap((share - vmin) / (vmax - vmin)),
                                   edgecolor="none"))
            ax.text((c0 + c1) / 2, (r0 + r1) / 2,
                    f"{r1 - r0}x{c1 - c0}\n{counts[tract]} ({100 * share:.1f}%)",
                    ha="center", va="center", fontsize=7.5, linespacing=1.5,
                    color="#ffffff" if share > (vmin + vmax) / 2 else "#0b0b0b")
        # Cell lattice, kept faint: it is texture showing how fine the cells
        # get, not a readable grid.
        for k in range(size + 1):
            ax.axhline(k, color="#ffffff", lw=0.3, alpha=0.35, zorder=2)
            ax.axvline(k, color="#ffffff", lw=0.3, alpha=0.35, zorder=2)
        ax.set_xlim(0, size)
        ax.set_ylim(size, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        bw = "-".join(str(w) for w in ref["band_widths"])
        ax.set_title(f"{size}x{size}   bands {bw}", fontsize=10, pad=6)
        ax.annotate(
            f"E(random)      {ref['mean']:.3f}\n"
            f"E(two-cluster) {ref['two_cluster']['mean']:.3f}\n"
            f"maximum        {ref['max']:.3f}",
            xy=(0.5, -0.045), xycoords="axes fraction", ha="center", va="top",
            fontsize=8, color=REFERENCE_INK, family="monospace")

    ax = fig.add_subplot(gs[1, :])
    x = np.asarray(sizes)
    means = np.array([r["mean"] for r in refs])
    p5 = np.array([r["p5"] for r in refs])
    p95 = np.array([r["p95"] for r in refs])
    maxima = np.array([r["max"] for r in refs])
    tc = np.array([r["two_cluster"]["mean"] for r in refs])
    tc5 = np.array([r["two_cluster"]["p5"] for r in refs])
    tc95 = np.array([r["two_cluster"]["p95"] for r in refs])

    ax.fill_between(x, p5, p95, color=SERIES_COLORS[0], alpha=0.16, lw=0)
    ax.fill_between(x, tc5, tc95, color=SERIES_COLORS[1], alpha=0.16, lw=0)
    ax.plot(x, means, color=SERIES_COLORS[0], lw=2, marker="o", ms=8,
            markeredgecolor="#ffffff", markeredgewidth=1.2,
            label="E(random allocation) — no sorting")
    ax.plot(x, tc, color=SERIES_COLORS[1], lw=2, marker="s", ms=8,
            markeredgecolor="#ffffff", markeredgewidth=1.2,
            label="E(two-cluster city) — random boundary angle and gap")
    ax.plot(x, maxima, color=SERIES_COLORS[2], lw=2, marker="^", ms=8,
            markeredgecolor="#ffffff", markeredgewidth=1.2,
            label="attainable maximum — boundary on tract lines")

    for y, text in ((means[-1], "E(random)"),
                    (tc[-1], "E(two-cluster)"),
                    (maxima[-1], "maximum")):
        ax.annotate(text, xy=(x[-1], y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#0b0b0b")
    ax.set_xticks(x)
    ax.set_xlabel("grid size (cells per side)")
    ax.set_ylabel("dissimilarity index")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(x[0] - 2, x[-1] + 13)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c4")
    ax.legend(loc="center left", frameon=False, fontsize=9)
    ax.set_title("Where the scale's floor and ceiling sit at each grid size — the ceiling "
                 "is flat, the floor falls as 1/sqrt(agents per tract)\n"
                 "E(.) are expectations over draws; shading is the 5th-95th percentile",
                 fontsize=10.5, pad=8, linespacing=1.5)

    n_a_txt, n_b_txt = refs[0]["n_a"], refs[0]["n_b"]
    fig.suptitle("Dissimilarity index: 9-tract partition and reference values by grid size",
                 fontsize=13, y=0.985)
    fig.text(0.5, 0.945,
             f"9 tracts at every size; agent counts hold config.py's density and A:B ratio "
             f"({n_a_txt}+{n_b_txt} at {sizes[0]}x{sizes[0]}); "
             f"random baseline from {draws} draws of the exact null",
             ha="center", fontsize=9, color=REFERENCE_INK)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path
    return fig


# ---------------------------------------------------------------------------
# CLI: generate the tract maps and the reference table
# ---------------------------------------------------------------------------

def default_out_dir():
    """reports/dissimilarity_index/tract_maps, honouring the reports override."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis_tools"))
        from output_paths import get_reports_dir
        base = get_reports_dir()
    except Exception:
        base = Path("reports")
    return base / "dissimilarity_index" / "tract_maps"


def write_tract_maps(sizes=DEFAULT_SIZES, out_dir=None, draws=DEFAULT_DRAWS,
                     dpi=300, n_a=None, n_b=None):
    """Write the per-size tract maps, the reference table and the figure.

    The maps are written out as plain integer CSVs so the partition can be
    checked by eye or by another tool without importing this module.
    """
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, refs = [], []
    for size in sizes:
        ref = di_reference(size, n_a, n_b, draws)
        refs.append(ref)
        np.savetxt(out_dir / f"tract_map_{size}x{size}.csv",
                   tract_map(size), fmt="%d", delimiter=",")
        rows.append({
            "size": size,
            "band_widths": "-".join(str(w) for w in ref["band_widths"]),
            "tract_sizes": " ".join(str(v) for v in ref["tract_sizes"]),
            "n_a": ref["n_a"], "n_b": ref["n_b"],
            "E_random": round(ref["mean"], 4),
            "random_sd": round(ref["sd"], 4),
            "random_p5": round(ref["p5"], 4),
            "random_p95": round(ref["p95"], 4),
            "E_half_split": round(ref["half_split"], 4),
            "E_two_cluster": round(ref["two_cluster"]["mean"], 4),
            "two_cluster_p5": round(ref["two_cluster"]["p5"], 4),
            "two_cluster_p95": round(ref["two_cluster"]["p95"], 4),
            "max": round(ref["max"], 4),
        })

    header = ",".join(rows[0].keys())
    lines = [header] + [",".join(str(v) for v in r.values()) for r in rows]
    (out_dir / "di_reference_values.csv").write_text("\n".join(lines) + "\n")

    fig_path = plot_tract_maps(sizes, out_dir / "tract_maps_by_grid_size.png",
                               draws=draws, dpi=dpi, n_a=n_a, n_b=n_b, refs=refs)
    return out_dir, fig_path, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
                    help="grid sizes to draw (default 10 20 30 40 50)")
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS,
                    help="Monte Carlo draws for the random baseline")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out-dir", default=None,
                    help="default reports/dissimilarity_index/tract_maps")
    args = ap.parse_args()

    out_dir, fig_path, rows = write_tract_maps(
        tuple(args.sizes), args.out_dir, args.draws, args.dpi)
    width = max(len(r["tract_sizes"]) for r in rows)
    print(f"{'size':>7}  {'bands':>8}  {'tract sizes':<{width}}  "
          f"{'E(rand)':>7}  {'E(2clu)':>7}  {'E(half)':>7}  {'max':>5}")
    for r in rows:
        print(f"{r['size']:>5}x{r['size']:<1}  {r['band_widths']:>8}  "
              f"{r['tract_sizes']:<{width}}  {r['E_random']:>7.3f}  "
              f"{r['E_two_cluster']:>7.3f}  {r['E_half_split']:>7.3f}  "
              f"{r['max']:>5.3f}")
    print(f"\n[DONE] tract maps + reference table -> {out_dir}")
    print(f"[DONE] figure -> {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

