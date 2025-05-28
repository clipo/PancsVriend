import pygame
import pygame_gui
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
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
        self.step = 0
        self.simulation_started = False
        self.setup_ui()
        self.setup_csv()
        self.setup_plots()

    def setup_ui(self):
        self.max_steps = 1000
        self.progress_bar = pygame_gui.elements.UIProgressBar(
            relative_rect=pygame.Rect((10, 50), (660, 20)),
            manager=self.manager
        )
        self.start_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((440, 10), (120, 30)),
            text="Start",
            manager=self.manager
        )
        self.stop_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((310, 10), (120, 30)),
            text="Stop & Graph",
            manager=self.manager
        )
        self.llm_model_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=['qwen2.5-coder:32B', 'llama3.2:1B', 'phi-4:14B', 'mistral:7B'],
            starting_option=cfg.OLLAMA_MODEL,
            relative_rect=pygame.Rect((10, 10), (180, 30)),
            manager=self.manager
        )
        self.llm_toggle = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((200, 10), (100, 30)),
            text='Toggle LLM',
            manager=self.manager
        )

    def setup_csv(self):
        self.csv_file = 'segregation_metrics.csv'
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Step', 'Clusters', 'Switch Rate', 'Distance', 'Mix Deviation', 'Share', 'Ghetto Rate'])

    def setup_plots(self):
        self.metrics_history = {
            'clusters': [],
            'switch_rate': [],
            'distance': [],
            'mix_deviation': [],
            'share': [],
            'ghetto_rate': []
        }

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
                if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                    if event.ui_element == self.llm_model_dropdown:
                        cfg.OLLAMA_MODEL = event.text
                elif event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.start_button:
                        self.simulation_started = True
                    elif event.ui_element == self.stop_button:
                        self.running = False
                        self.plot_final_metrics()
                    elif event.ui_element == self.llm_toggle:
                        cfg.USE_LLM = not cfg.USE_LLM
                self.manager.process_events(event)

            # GUI rendering
            self.window.fill(cfg.BG_COLOR)
            self.draw_grid()
            self.manager.update(time_delta)
            self.progress_bar.set_current_progress(min(self.step / self.max_steps * 100, 100))
            self.manager.draw_ui(self.window)
            pygame.display.update()

            # Core simulation logic (only if started)
            if self.simulation_started:
                self.update_agents()
                self.log_metrics()

    def draw_grid(self):
        visible_agents = 0
        for r in range(cfg.GRID_SIZE):
            for c in range(cfg.GRID_SIZE):
                agent = self.grid[r][c]
                rect = pygame.Rect(c * cfg.CELL_SIZE, r * cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE)
                if agent:
                    color = cfg.COLOR_A if agent.type_id == 0 else cfg.COLOR_B
                    visible_agents += 1
                else:
                    color = cfg.VACANT_COLOR
                pygame.draw.rect(self.window, color, rect)
                pygame.draw.rect(self.window, cfg.GRID_LINE_COLOR, rect, 1)

    def update_agents(self):
        all_positions = [(r, c) for r in range(cfg.GRID_SIZE) for c in range(cfg.GRID_SIZE) if self.grid[r][c]]
        np.random.shuffle(all_positions)
        for r, c in all_positions:
            agent = self.grid[r][c]
            if not agent:
                continue
            move_to = maybe_use_llm_agent(agent, r, c, self.grid) if cfg.USE_LLM else agent.best_response(r, c, self.grid)
            if move_to:
                r_new, c_new = move_to
                if self.grid[r_new][c_new] is None:
                    self.grid[r_new][c_new] = agent
                    self.grid[r][c] = None
            break  # Only one agent moves per call

    def log_metrics(self):
        metrics = calculate_all_metrics(self.grid)
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                self.step,
                metrics['clusters'],
                metrics['switch_rate'],
                metrics['distance'],
                metrics['mix_deviation'],
                metrics['share'],
                metrics['ghetto_rate']
            ])
        for key in self.metrics_history:
            self.metrics_history[key].append(metrics[key])
        self.step += 1

    def plot_final_metrics(self):
        import matplotlib.backends.backend_pdf
        plt.ioff()
        fig, axs = plt.subplots(3, 2, figsize=(12, 8))
        for idx, key in enumerate(self.metrics_history.keys()):
            ax = axs[idx // 2][idx % 2]
            ax.plot(self.metrics_history[key])
            ax.set_title(key.replace('_', ' ').title())
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
        plt.tight_layout()
        pdf = matplotlib.backends.backend_pdf.PdfPages("final_metrics_summary.pdf")
        pdf.savefig(fig)
        pdf.close()
        plt.show()

if __name__ == "__main__":
    Simulation().run()
