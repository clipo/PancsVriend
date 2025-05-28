import numpy as np
import config as cfg

class Agent:
    def __init__(self, type_id):
        self.type_id = type_id

    def utility(self, unlike_ratio):
        x = unlike_ratio * 100
        if x > cfg.UTILITY_CUTOFF:
            return 0
        return cfg.UTILITY_BASE + cfg.UTILITY_SLOPE * (50 - abs(x - 50))

    def best_response(self, r, c, grid):
        max_u, best_pos = self.utility(self._unlike_ratio(r, c, grid)), None
        for r_new in range(cfg.GRID_SIZE):
            for c_new in range(cfg.GRID_SIZE):
                if grid[r_new][c_new] is None:
                    u = self.utility(self._unlike_ratio(r_new, c_new, grid))
                    if u > max_u:
                        max_u = u
                        best_pos = (r_new, c_new)
        return best_pos

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
