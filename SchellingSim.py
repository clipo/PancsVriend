import pygame
import pygame_gui
import numpy as np
from Agent import Agent
from Metrics import calculate_all_metrics
from LLMAgent import maybe_use_llm_agent
import config as cfg

class Simulation:
    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
        pygame.display.set_caption("Schelling Segregation Simulation")
        self.manager = pygame_gui.UIManager((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.grid = np.full((cfg.GRID_SIZE, cfg.GRID_SIZE), None)
        self.populate_grid()
        self.running = True

    def populate_grid(self):
        agents = [Agent(type_id) for type_id in ([0] * cfg.NUM_TYPE_A + [1] * cfg.NUM_TYPE_B)]
        np.random.shuffle(agents)
        flat_positions = [(r, c) for r in range(cfg.GRID_SIZE) for c in range(cfg.GRID_SIZE)]
        np.random.shuffle(flat_positions)
        for agent, pos in zip(agents, flat_positions[:len(agents)]):
            r, c = pos
            self.grid[r][c] = agent

    def run(self):
        while self.running:
            time_delta = self.clock.tick(cfg.FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                self.manager.process_events(event)

            self.manager.update(time_delta)
            self.window.fill(cfg.BG_COLOR)
            self.draw_grid()
            self.manager.draw_ui(self.window)
            pygame.display.update()
            self.update_agents()

    def draw_grid(self):
        for r in range(cfg.GRID_SIZE):
            for c in range(cfg.GRID_SIZE):
                agent = self.grid[r][c]
                rect = pygame.Rect(c * cfg.CELL_SIZE, r * cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE)
                if agent:
                    color = cfg.COLOR_A if agent.type_id == 0 else cfg.COLOR_B
                else:
                    color = cfg.VACANT_COLOR
                pygame.draw.rect(self.window, color, rect)
                pygame.draw.rect(self.window, cfg.GRID_LINE_COLOR, rect, 1)

    def update_agents(self):
        # Update one agent per frame
        all_positions = [(r, c) for r in range(cfg.GRID_SIZE) for c in range(cfg.GRID_SIZE) if self.grid[r][c]]
        if not all_positions:
            return
        r, c = all_positions[np.random.randint(len(all_positions))]
        agent = self.grid[r][c]
        move_to = maybe_use_llm_agent(agent, r, c, self.grid) if cfg.USE_LLM else agent.best_response(r, c, self.grid)
        if move_to:
            r_new, c_new = move_to
            self.grid[r_new][c_new] = agent
            self.grid[r][c] = None

if __name__ == "__main__":
    Simulation().run()
