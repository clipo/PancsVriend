import numpy as np


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