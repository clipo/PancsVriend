import numpy as np


def compute_dissimilarity_index(grid):
    """
    Calculate the Dissimilarity Index for the Schelling segregation model.
    
    The Dissimilarity Index measures the proportion of one group that would need
    to move to achieve an even distribution across spatial units.
    
    Args:
        grid: 2D numpy array containing Agent objects or None for empty cells
        
    Returns:
        float: Dissimilarity index value between 0 (perfect integration) and 1 (complete segregation)
        
    Raises:
        ValueError: If grid is not 10x10
    """
    # Verify grid size is 10x10
    if grid.shape != (10, 10):
        raise ValueError(f"Grid size must be 10x10, but got {grid.shape[0]}x{grid.shape[1]}")
    
    # Initialize counters for each census tract
    # tract_counts[tract_id] = [type_0_count, type_1_count]
    tract_counts = {i: [0, 0] for i in range(9)}
    
    # Count agents by type in each census tract
    total_type_0 = 0
    total_type_1 = 0
    
    for row in range(10):
        for col in range(10):
            agent = grid[row][col]
            if agent is not None:
                tract_id = get_census_tract(row, col)
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


def get_census_tract(row, col):
    """
    Map grid coordinates to census tract ID (0-8).
    
    The 10x10 grid is divided into 9 census tracts with non-uniform sizes:
    
    Axis Division:
    - X-axis (columns): [0-2] (3 cells), [3-6] (4 cells), [7-9] (3 cells)
    - Y-axis (rows): [0-2] (3 cells), [3-6] (4 cells), [7-9] (3 cells)
    
    Resulting 9 Census Tracts:
    
    Tract Layout:
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
    
    Census Tract Properties:
    - Corner tracts (0,2,6,8): 3x3 = 9 cells each
    - Edge tracts (1,3,5,7): 3x4 or 4x3 = 12 cells each
    - Center tract (4): 4x4 = 16 cells
    - Total: 4×9 + 4×12 + 1×16 = 36 + 48 + 16 = 100 cells
    
    Args:
        row: Row index (0-9)
        col: Column index (0-9)
        
    Returns:
        int: Census tract ID (0-8)
    """
    # Determine which row section (0, 1, or 2)
    if row < 3:
        row_section = 0
    elif row < 7:
        row_section = 1
    else:
        row_section = 2
        
    # Determine which column section (0, 1, or 2)
    if col < 3:
        col_section = 0
    elif col < 7:
        col_section = 1
    else:
        col_section = 2
        
    # Calculate tract ID from row and column sections
    tract_id = row_section * 3 + col_section
    
    return tract_id