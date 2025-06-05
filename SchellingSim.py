import pygame
import pygame_gui
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import json
from Agent import Agent
from Metrics import calculate_all_metrics
from LLMAgent import maybe_use_llm_agent
import config as cfg
import threading
import queue
from datetime import datetime
import pandas as pd

# Define your own direction constants
DIRECTION_LTR = 0  # Left-to-right
DIRECTION_RTL = 1  # Right-to-left

def check_llm_connection():
    import requests
    try:
        test_headers = {
            "Authorization": f"Bearer {cfg.OLLAMA_API_KEY}",
            "Content-Type": "application/json"
        }
        test_payload = {
            "model": cfg.OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "Respond with the word 'success' only."}],
            "stream": False
        }
        response = requests.post(cfg.OLLAMA_URL, headers=test_headers, json=test_payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "choices" not in data or not data["choices"]:
            print("[LLM Test Failed] Response missing 'choices'. Full response:", data)
            return False
        content = data["choices"][0]["message"]["content"]
        if "success" not in content.lower():
            print("[LLM Test Failed] Unexpected response:", content)
            return False
        return True
    except Exception as e:
        print("[LLM Startup Check Failed]", e)
        return False

class Simulation:
    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
        pygame.display.set_caption("Schelling Segregation Simulation")
        self.manager = pygame_gui.UIManager((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.grid = np.full((cfg.GRID_SIZE, cfg.GRID_SIZE), None)
        self.populate_grid()
        # Initialize integer states list after population
        self.states = [self._grid_to_int()]
        # record timestamp for this simulation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # set up experiment folder for this simulation run
        self.sim_dir = f"experiments/simulation_{self.timestamp}"
        os.makedirs(self.sim_dir, exist_ok=True)
        self.running = True
        self.step = 0
        self.simulation_started = False
        self.setup_ui()
        self.setup_csv()
        self.setup_plots()
        self.converged = False
        self.start_llm_worker()
        self.convergence_step = None
        self.no_move_steps = 0
        self.no_move_threshold = 20

    def start_llm_worker(self):
        def worker():
            while True:
                task = self.query_queue.get()
                if task is None:
                    break  # Sentinel to shut down the worker
                agent, r, c = task
                try:
                    move_to = maybe_use_llm_agent(agent, r, c, self.grid)
                    self.result_queue.put((agent, r, c, move_to))
                except Exception as e:
                    print(f"[LLM Worker Error] Agent at ({r},{c}): {e}")
                finally:
                    self.query_queue.task_done()

        self.query_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.llm_thread = threading.Thread(target=worker, daemon=True)
        self.llm_thread.start()

    def setup_ui(self):
        self.max_steps = 1000

        sidebar_x = cfg.GRID_SIZE * cfg.CELL_SIZE + 20
        button_width = 160
        button_height = 30
        y = 10
        spacing = 40

        self.progress_bar = pygame_gui.elements.UIProgressBar(
            relative_rect=pygame.Rect((sidebar_x, y), (button_width, button_height)),
            manager=self.manager
        )
        y += spacing

        self.llm_model_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=['qwen2.5-coder:32B', 'llama3.2:1B', 'phi-4:14B', 'mistral:7B'],
            starting_option=cfg.OLLAMA_MODEL,
            relative_rect=pygame.Rect((sidebar_x, y), (button_width, button_height)),
            manager=self.manager
        )
        y += spacing

        self.llm_toggle = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((sidebar_x, y), (button_width, button_height)),
            text='Toggle LLM',
            manager=self.manager
        )
        y += spacing

        self.start_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((sidebar_x, y), (button_width, button_height)),
            text="Start",
            manager=self.manager
        )
        y += spacing

        self.pause_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((sidebar_x, y), (button_width, button_height)),
            text="Pause",
            manager=self.manager
        )
        y += spacing

        self.stop_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((sidebar_x, y), (button_width, button_height)),
            text="Stop & Graph",
            manager=self.manager
        )
        y += spacing

        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((sidebar_x, y), (button_width, button_height)),
            text="Reset",
            manager=self.manager
        )

    def setup_csv(self):
        # Save segregation metrics per-step into the simulation folder
        # Use experiment folder for segregation metrics
        self.csv_file = os.path.join(self.sim_dir, f"segregation_metrics_{self.timestamp}.csv")
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
                        if not cfg.USE_LLM:
                            print("Checking LLM connection...")
                            if check_llm_connection():
                                print("LLM connection confirmed.")
                                cfg.USE_LLM = True
                            else:
                                print("LLM connection failed. Remaining in manual mode.")
                        else:
                            cfg.USE_LLM = False
                    elif event.ui_element == self.pause_button:
                        self.simulation_started = not self.simulation_started
                        print("Paused" if not self.simulation_started else "Resumed")
                    elif event.ui_element == self.reset_button:
                        print("Resetting simulation...")
                        self.grid = np.full((cfg.GRID_SIZE, cfg.GRID_SIZE), None)
                        self.populate_grid()
                        self.step = 0
                        self.metrics_history = {key: [] for key in self.metrics_history}
                        self.simulation_started = False
                        self.running = True
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
                self.check_convergence()

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

        self.last_agent_moved = False

        if not cfg.USE_LLM:
            for r, c in all_positions:
                agent = self.grid[r][c]
                move_to = agent.best_response(r, c, self.grid)
                if move_to:
                    r_new, c_new = move_to
                    if 0 <= r_new < cfg.GRID_SIZE and 0 <= c_new < cfg.GRID_SIZE:
                        if self.grid[r_new][c_new] is None:
                            self.grid[r_new][c_new] = agent
                            self.grid[r][c] = None
                            self.last_agent_moved = True
                            print(f"[Main Thread] Moved agent from ({r},{c}) to ({r_new},{c_new})")
                        else:
                            print(f"[Main Thread] Target cell ({r_new},{c_new}) is occupied.")
                    else:
                        print(f"[Main Thread] Ignored invalid move: ({r_new},{c_new})")
        else:
            for r, c in all_positions[:5]:
                agent = self.grid[r][c]
                self.query_queue.put((agent, r, c))

            for _ in range(5):
                try:
                    agent, r, c, move_to = self.result_queue.get_nowait()
                    if move_to:
                        # LLM moves are 3x3 relative, so adjust them
                        rel_r, rel_c = move_to
                        r_new = r + (rel_r - 1)
                        c_new = c + (rel_c - 1)

                        if 0 <= r_new < cfg.GRID_SIZE and 0 <= c_new < cfg.GRID_SIZE:
                            if self.grid[r_new][c_new] is None:
                                self.grid[r_new][c_new] = agent
                                self.grid[r][c] = None
                                self.last_agent_moved = True
                                print(f"[Main Thread] Moved agent from ({r},{c}) to ({r_new},{c_new})")
                            else:
                                print(f"[Main Thread] Target cell ({r_new},{c_new}) is occupied.")
                        else:
                            print(f"[Main Thread] Ignored invalid move: ({r_new},{c_new})")
                    self.result_queue.task_done()
                except queue.Empty:
                    break

    def _grid_to_int(self):
        """
        Convert self.grid of Agent objects/None into int grid:
        -1 for empty, agent.type_id for occupied.
        """
        size = cfg.GRID_SIZE
        int_grid = np.full((size, size), -1, dtype=int)
        for r in range(size):
            for c in range(size):
                agent = self.grid[r][c]
                if agent is not None:
                    int_grid[r, c] = agent.type_id
        return int_grid

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
        # Store integer snapshot of current grid
        self.states.append(self._grid_to_int())
        self.step += 1

    def plot_final_metrics(self):
        print(f"Converged at step: {self.convergence_step}")
        plt.ioff()
        fig, axs = plt.subplots(3, 2, figsize=(12, 8))
        for idx, key in enumerate(self.metrics_history.keys()):
            ax = axs[idx // 2][idx % 2]
            ax.plot(self.metrics_history[key])
            ax.set_title(key.replace('_', ' ').title())
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
        plt.tight_layout()
        from matplotlib.backends.backend_pdf import PdfPages
        # Save final metrics summary PDF into experiment folder
        sim_dir = f"experiments/simulation_{self.timestamp}"
        os.makedirs(sim_dir, exist_ok=True)
        pdf_path = os.path.join(sim_dir, f"final_metrics_summary_{self.timestamp}.pdf")
        pdf = PdfPages(pdf_path)
        pdf.savefig(fig)
        pdf.close()
        print(f"Saved final metrics summary to {pdf_path}")
        # Save all integer grid states as a numpy array
        states_array = np.stack(self.states)
        # save states
        np.savez_compressed(os.path.join(sim_dir, f"states_{self.timestamp}.npz"), states_array)
        print(f"Saved integer grid states to {sim_dir}/states_{self.timestamp}.npz (shape={states_array.shape})")
        # save metrics history
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(os.path.join(sim_dir, f"metrics_history_{self.timestamp}.csv"), index=False)
        print(f"Saved metrics history to {self.sim_dir}/metrics_history_{self.timestamp}.csv")
        # save config for reproducibility
        config = {
            'timestamp': self.timestamp,
            'grid_size': cfg.GRID_SIZE,
            'max_steps': self.max_steps,
            'use_llm': cfg.USE_LLM,
            'llm_model': cfg.OLLAMA_MODEL if hasattr(cfg, 'OLLAMA_MODEL') else None
        }
        with open(os.path.join(sim_dir, f"config_{self.timestamp}.json"), 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {self.sim_dir}/config_{self.timestamp}.json")
        # write convergence summary inside simulation folder
        cs_file = os.path.join(sim_dir, f"convergence_summary_{self.timestamp}.csv")
        if not os.path.exists(cs_file):
            with open(cs_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Step', 'USE_LLM', 'Model', 'Total Agents'])

        with open(cs_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.step,
                cfg.USE_LLM,
                cfg.OLLAMA_MODEL,
                cfg.NUM_TYPE_A + cfg.NUM_TYPE_B
            ])
        plt.show()

    def check_convergence(self):
        if not hasattr(self, 'last_agent_moved') or not self.result_queue.empty() or not self.query_queue.empty():
            return
        if not hasattr(self, 'last_agent_moved'):
            return
        if self.last_agent_moved:
            self.no_move_steps = 0
        else:
            self.no_move_steps += 1

        if self.no_move_steps >= self.no_move_threshold:
            print(f"Convergence detected at step {self.step}")
            self.converged = True
            self.convergence_step = self.step
            self.plot_final_metrics()
            self.running = False

            # write convergence summary inside simulation folder
            cs_file = os.path.join(self.sim_dir, f"convergence_summary_{self.timestamp}.csv")
            if not os.path.exists(cs_file):
                with open(cs_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Step', 'USE_LLM', 'Model', 'Total Agents'])

            with open(cs_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.step,
                    cfg.USE_LLM,
                    cfg.OLLAMA_MODEL,
                    cfg.NUM_TYPE_A + cfg.NUM_TYPE_B
                ])

if __name__ == "__main__":
    Simulation().run()

