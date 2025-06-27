# Base simulation class for Schelling model variants
import numpy as np
import config as cfg
from Metrics import calculate_all_metrics
import os
import gzip
import json
import pandas as pd
from tqdm import tqdm

class Simulation:
    def __init__(self, run_id, agent_factory, decision_func, scenario='baseline', random_seed=None):
        self.run_id = run_id
        self.scenario = scenario
        self.grid = np.full((cfg.GRID_SIZE, cfg.GRID_SIZE), None)
        self.step = 0
        self.converged = False
        self.convergence_step = None
        self.no_move_steps = 0
        self.no_move_threshold = cfg.NO_MOVE_THRESHOLD
        self.metrics_history = []
        self.states = []
        self.agent_factory = agent_factory
        self.decision_func = decision_func
        self.random_seed = random_seed
        # Track all agent moves during simulation
        self.agent_move_log = []

        if self.random_seed is None:
            np.random.seed(None)

        self.populate_grid()
        # Save the initial grid state after population
        self.log_state_per_move()
        # Log a dummy "move" to record their initial state
        self.log_agent_move(None, None, None, None, False, None, 'initial_state', verbose_move_log=False)

    def populate_grid(self):
        agents = [self.agent_factory(type_id) for type_id in ([0] * cfg.NUM_TYPE_A + [1] * cfg.NUM_TYPE_B)]
        np.random.shuffle(agents)
        flat_positions = [(r, c) for r in range(cfg.GRID_SIZE) for c in range(cfg.GRID_SIZE)]
        np.random.shuffle(flat_positions)
        for agent, pos in zip(agents, flat_positions[:len(agents)]):
            r, c = pos
            self.grid[r][c] = agent
            # Assign starting position and initialize position tracking
            agent.starting_position = (r, c)
            agent.position_history = [(r, c)]  # Track all positions throughout the run
            agent.new_position = None  # Initialize new_position attribute

    def update_agents(self, verbose_move_log=False):
        all_positions = [(r, c) for r in range(cfg.GRID_SIZE) for c in range(cfg.GRID_SIZE) if self.grid[r][c]]
        np.random.shuffle(all_positions)
        moved = False
            
        if verbose_move_log:
            print(f"[Step {self.step}] Processing {len(all_positions)} agents for movement decisions")
        
        for i, (r, c) in enumerate(all_positions):
            agent = self.grid[r][c]
            if agent is not None:
                agent.new_position = None  # Reset new position for this agent
            move_to = self.decision_func(agent, r, c, self.grid)
                        
            # Log each agent's move decision
            if move_to and move_to != (r, c):
                r_new, c_new = move_to
                if 0 <= r_new < cfg.GRID_SIZE and 0 <= c_new < cfg.GRID_SIZE:
                    if self.grid[r_new][c_new] is None: # Target position is empty
                        # Track the new position
                        agent.new_position = (r_new, c_new)
                        agent.position_history.append((r_new, c_new))
                        
                        # Move the agent
                        self.grid[r_new][c_new] = agent
                        self.grid[r][c] = None
                        moved = True
                                            
                        # Log the move
                        self.log_agent_move(agent, r, c, move_to, True, (r_new, c_new), 'successful_move', verbose_move_log)
                        self.log_state_per_move()

                    else:
                        # Target occupied
                        agent.new_position = (r, c)
                        self.log_agent_move(agent, r, c, move_to, False, (r, c), 'target_occupied', verbose_move_log)
                        self.log_state_per_move()

                else:
                    # Invalid move (out of bounds)
                    agent.new_position = (r, c)
                    self.log_agent_move(agent, r, c, move_to, False, (r, c), 'invalid_target', verbose_move_log)                        
                    self.log_state_per_move()

            else:
                # Agent stays in current position
                agent.new_position = (r, c)
                if move_to is None:
                    self.log_agent_move(agent, r, c, move_to, False, (r, c), 'chose_to_stay', verbose_move_log)
                    self.log_state_per_move()

                else:
                    self.log_agent_move(agent, r, c, move_to, False, (r, c), 'same_position', verbose_move_log)
                    self.log_state_per_move()
    
        if verbose_move_log:
            print(f"[Step {self.step}] Movement phase complete - {'Some' if moved else 'No'} agents moved this step")
        return moved

    def run_step(self, verbose_move_log=False):
        moved = self.update_agents(verbose_move_log=verbose_move_log)
        metrics = calculate_all_metrics(self.grid)
        metrics['step'] = self.step
        metrics['run_id'] = self.run_id
        self.metrics_history.append(metrics)
        # States are now saved after each individual move in update_agents()
        if not moved:
            self.no_move_steps += 1
        else:
            self.no_move_steps = 0
        if self.no_move_steps >= self.no_move_threshold:
            self.converged = True
            self.convergence_step = self.step
        if not self.converged:
            self.step += 1 # Increment step only if not converged 
        return self.converged

    def _grid_to_int(self):
        size = cfg.GRID_SIZE
        int_grid = np.full((size, size), -1, dtype=int)
        for r in range(size):
            for c in range(size):
                agent = self.grid[r][c]
                if agent is not None:
                    int_grid[r, c] = agent.type_id
        return int_grid

    def run_single_simulation(self, output_dir=None, max_steps=1000, show_progress=False):
        """Run a single simulation and optionally save agent moves."""
        progress_bar = None
        if show_progress:
            progress_bar = tqdm(total=max_steps, desc=f"Run {self.run_id} ({self.scenario})", 
                               unit="step", leave=False)
        
        while not self.converged and self.step < max_steps:
            self.run_step()
            
            if progress_bar:
                progress_bar.update(1)
                
                # Update progress bar with current status
                if self.step % 10 == 0:  # Update postfix every 10 steps to avoid spam
                    progress_bar.set_postfix({
                        'converged': self.converged,
                        'no_move_steps': self.no_move_steps,
                        'moves_logged': len(self.agent_move_log)
                    })
        
        if progress_bar:
            progress_bar.close()
        self.save_states(output_dir)
        self.save_agent_move_log(output_dir)  # Save the detailed move log
        return {
            'run_id': self.run_id,
            'converged': self.converged,
            'convergence_step': self.convergence_step,
            'final_step': self.step,
            'metrics_history': self.metrics_history,
            'states_per_move': self.states,
            'initial_grid': self.states[0] if self.states else None,
            'total_agent_moves': len(self.agent_move_log)
        }

    def save_states(self, output_dir):
        """Save grid states after every move (no run_logs)."""
        if output_dir is not None:
            states_dir = os.path.join(output_dir, "states")
            os.makedirs(states_dir, exist_ok=True)

            # Save grid states as numpy arrays (includes state after every individual move)
            states_array = np.array(self.states)
            np.savez_compressed(os.path.join(states_dir, f"states_run_{self.run_id}.npz"), 
                              states=states_array)
            
            # print(f"[Run {self.run_id}] Saved {len(self.states)} grid states (including after each move)")

    def save_agent_move_log(self, output_dir):
        """Save detailed agent move log to CSV and JSON files."""
        if output_dir is not None and self.agent_move_log:
            move_logs_dir = os.path.join(output_dir, "move_logs")
            os.makedirs(move_logs_dir, exist_ok=True)
            
            # Convert to DataFrame for easy CSV export
            df = pd.DataFrame(self.agent_move_log)
            # Save as CSV for easy analysis
            csv_path = os.path.join(move_logs_dir, f"agent_moves_run_{self.run_id}.csv")
            df.to_csv(csv_path, index=False)
            
            # Also save as compressed JSON for complete data
            json_path = os.path.join(move_logs_dir, f"agent_moves_run_{self.run_id}.json.gz")
            with gzip.open(json_path, 'wt', encoding='utf-8') as f:
                json.dump(self.agent_move_log, f, separators=(',', ':'), default=str)
            
            # print(f"[Run {self.run_id}] Saved {len(self.agent_move_log)} agent move entries to {move_logs_dir}")
            
            # Print summary statistics
            moves = sum(1 for entry in self.agent_move_log if entry['moved'])
            stays = len(self.agent_move_log) - moves
            print(f"[Run {self.run_id}] Move summary: {moves} moves, {stays} stays, {self.step} steps")

    def log_agent_move(self, agent, r, c, move_to, moved, new_position, reason, verbose_move_log=False):
        """Log an individual agent move with all relevant details."""
        # Create move entry for logging
        move_entry = {
            'step': self.step,
            'agent_id': id(agent) if agent else None,
            'type_id': agent.type_id if agent else None,
            'current_position': (r, c),
            'decision': move_to,
            'moved': moved,
            'new_position': new_position,
            'reason': reason,
            'llm_call_count': getattr(agent, 'llm_call_count', 0),
            'llm_call_time': getattr(agent, 'llm_call_time', 0.0),
            'timestamp': pd.Timestamp.now().isoformat(),
            'grid': self._grid_to_int().tolist()  # Save the current grid state after the move
        }
        
        # Add move entry to log
        self.agent_move_log.append(move_entry)
        
        # Print verbose output if requested
        if verbose_move_log:
            agent_id = f"Agent-{id(agent)}"
            if moved:
                print(f"[Step {self.step}] {agent_id} (Type {agent.type_id}) moved from ({r},{c}) to {new_position}")
            elif reason == 'target_occupied':
                print(f"[Step {self.step}] {agent_id} (Type {agent.type_id}) wanted to move from ({r},{c}) to {move_to} but target was occupied - stayed")
            elif reason == 'invalid_target':
                print(f"[Step {self.step}] {agent_id} (Type {agent.type_id}) wanted to move from ({r},{c}) to {move_to} but target was out of bounds - stayed")
            elif reason == 'chose_to_stay':
                print(f"[Step {self.step}] {agent_id} (Type {agent.type_id}) at ({r},{c}) chose to stay (decision: None)")
            else:
                print(f"[Step {self.step}] {agent_id} (Type {agent.type_id}) at ({r},{c}) chose to stay (decision: {move_to})")

    def log_state_per_move(self):
        """Save current grid state after a move."""
        self.states.append(self._grid_to_int())

    def analyze_results(results, output_dir, n_runs):   
        """Analyze simulation results and save metrics, convergence data, and step statistics."""
        all_metrics = []
        convergence_data = []

        for result in results:
            convergence_data.append({
                'run_id': result['run_id'],
                'converged': result['converged'],
                'convergence_step': result['convergence_step'],
                'final_step': result['final_step']
            })
            for metric in result['metrics_history']:
                all_metrics.append(metric)

        # Save metrics history and convergence summary
        pd.DataFrame(all_metrics).to_csv(f"{output_dir}/metrics_history.csv", index=False)
        pd.DataFrame(convergence_data).to_csv(f"{output_dir}/convergence_summary.csv", index=False)

        # Analyze step statistics
        df = pd.DataFrame(all_metrics)
        step_stats = df.groupby('step').agg({
            'clusters': ['mean', 'std', 'min', 'max'],
            'switch_rate': ['mean', 'std', 'min', 'max'],
            'distance': ['mean', 'std', 'min', 'max'],
            'mix_deviation': ['mean', 'std', 'min', 'max'],
            'share': ['mean', 'std', 'min', 'max'],
            'ghetto_rate': ['mean', 'std', 'min', 'max']
        }).reset_index()
        step_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in step_stats.columns.values]
        step_stats.to_csv(f"{output_dir}/step_statistics.csv", index=False)

        return output_dir, results, convergence_data
