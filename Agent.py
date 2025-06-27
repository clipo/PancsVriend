import config as cfg
import random

class Agent:
    def __init__(self, type_id):
        self.type_id = type_id
        # Position tracking properties (set by simulation)
        self.starting_position = None
        self.position_history = []
        self.new_position = None
        # LLM properties (used by LLM agents, None for mechanical agents)
        self.llm_call_count = 0
        self.llm_call_time = 0.0

    def utility(self, unlike_ratio):
        """
        Simplified utility function: returns 1 if at least 50% of neighbors are similar, 0 otherwise.
        
        Args:
            unlike_ratio: Proportion of unlike neighbors (between 0 and 1)
            
        Returns:
            1 if similar_ratio >= 0.5 (at least 50% neighbors are similar), 0 otherwise
        """
        similar_ratio = 1 - unlike_ratio
        return 1 if similar_ratio >= cfg.SIMILARITY_THRESHOLD else 0

    def best_response(self, r, c, grid):
        """
        Determines the best position for the agent to move to on the grid based on utility.

        Evaluates all empty positions on the grid and computes the utility of moving to each.
        Returns the position (row, column) that yields the highest utility, or None if no better position is found.

        Args:
            r (int): Current row position of the agent.
            c (int): Current column position of the agent.
            grid (list[list[object or None]]): The grid representing the environment, where each cell may be occupied or None.

        Returns:
            tuple[int, int] or None: The (row, column) of the best position to move to, or None if no better position exists.
        """
        max_u, best_pos = self.utility(self._unlike_ratio(r, c, grid)), None
        for r_new in range(cfg.GRID_SIZE):
            for c_new in range(cfg.GRID_SIZE):
                if grid[r_new][c_new] is None:
                    u = self.utility(self._unlike_ratio(r_new, c_new, grid))
                    if u > max_u:
                        max_u = u
                        best_pos = (r_new, c_new)
        return best_pos

    def random_response(self, r, c, grid):
        """
        Pick a random empty space on the grid only if agent is dissatisfied (utility <= satisfaction threshold).
        
        Args:
            r (int): Current row position
            c (int): Current column position  
            grid (list): 2D list representing the game grid
            
        Returns:
            tuple: (row, col) of a random empty space if dissatisfied, or None if satisfied/no spaces
        """
        # Check current utility - only move if agent is dissatisfied
        current_utility = self.utility(self._unlike_ratio(r, c, grid))
        
        # If utility > satisfaction threshold, agent is satisfied and shouldn't move randomly
        if current_utility > cfg.AGENT_SATISFACTION_THRESHOLD:
            return None
        
        # Find all empty spaces on the grid
        empty_spaces = []
        for row in range(cfg.GRID_SIZE):
            for col in range(cfg.GRID_SIZE):
                if grid[row][col] is None:  # None represents empty space
                    empty_spaces.append((row, col))
        
        # If there are empty spaces, return a random one
        if empty_spaces:
            return random.choice(empty_spaces)
        
        # If no empty spaces available, return None (stay in place)
        return None
        
        # If there are empty spaces, return a random one
        if empty_spaces:
            return random.choice(empty_spaces)
        
        # If no empty spaces available, return None (stay in place)
        return None
        
        # If there are empty spaces, return a random one
        if empty_spaces:
            return random.choice(empty_spaces)
        
        # If no empty spaces available, return None (stay in place)
        return None

    def _unlike_ratio(self, r, c, grid):
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r_n, c_n = r + dr, c + dc
                if 0 <= r_n < cfg.GRID_SIZE and 0 <= c_n < cfg.GRID_SIZE:
                    agent = grid[r_n][c_n]
                    if agent is not None:
                        neighbors.append(agent)
        if not neighbors:
            return 1
        return sum(1 for n in neighbors if n.type_id != self.type_id) / len(neighbors)
