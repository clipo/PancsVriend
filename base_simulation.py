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
    def __init__(self, run_id, agent_factory, decision_func, scenario='baseline', random_seed=None,
                 initial_int_grid=None, initial_step=None, initial_no_move_steps=None):
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

        # Initialize grid either randomly or from a provided int grid (resume)
        if initial_int_grid is not None:
            self.populate_from_int_grid(initial_int_grid)
            # Allow resuming at a specified step (next step index)
            if initial_step is not None:
                try:
                    self.step = int(initial_step)
                except Exception:
                    pass
        else:
            self.populate_grid()
        # Save the initial grid state after population
        self.log_state_per_move()
        # Log a dummy "move" to record their initial state
        self.log_agent_move(None, None, None, None, False, None, 'initial_state', verbose_move_log=False)

        if initial_no_move_steps is not None:
            try:
                parsed_streak = int(initial_no_move_steps)
                if parsed_streak >= 0:
                    self.no_move_steps = parsed_streak
            except Exception:
                pass

    def populate_from_int_grid(self, int_grid):
        """Populate grid from a 2D numpy/list of ints (-1 empty, 0/1 type ids)."""
        arr = np.array(int_grid)
        assert arr.shape == (cfg.GRID_SIZE, cfg.GRID_SIZE), "initial_int_grid shape mismatch"
        for r in range(cfg.GRID_SIZE):
            for c in range(cfg.GRID_SIZE):
                t = int(arr[r, c])
                if t >= 0:
                    agent = self.agent_factory(t)
                    self.grid[r][c] = agent
                    agent.starting_position = (r, c)
                    agent.position_history = [(r, c)]
                    agent.new_position = None

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
                               unit="step", leave=True, ncols=80)
        
        while not self.converged and self.step < max_steps:
            self.run_step()
            self.save_states(output_dir)
            self.save_agent_move_log(output_dir)  # Save the detailed move log
            
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

        # Print summary statistics
        moves = sum(1 for entry in self.agent_move_log if entry['moved'])
        stays = len(self.agent_move_log) - moves
        print(f"[Run {self.run_id}] Move summary: {moves} moves, {stays} stays, {self.step} steps")
        
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

        store_llm_responses = (
            getattr(cfg, 'STORE_LLM_RESPONSES', False) or
            os.environ.get('STORE_LLM_RESPONSES', '').lower() in ('true', '1', 'yes')
        )
        if store_llm_responses:
            move_entry['llm_raw_response'] = getattr(agent, 'last_llm_response_raw', None)
            move_entry['llm_parsed_decision'] = getattr(agent, 'last_llm_parsed_decision', None)
            move_entry['llm_parse_status'] = getattr(agent, 'last_llm_parse_status', None)
        
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

    # --- Resume helpers ---
    def set_state_from_int_grid(self, int_grid, step=None):
        """Set the current simulation grid from a 2D array/list of ints and optionally the next step.

        int_grid: shape (GRID_SIZE, GRID_SIZE); -1 empty, otherwise type_id
        step: if provided, sets self.step to this (the next step index)
        """
        if int_grid is None:
            return
        arr = np.array(int_grid)
        if arr.shape != (cfg.GRID_SIZE, cfg.GRID_SIZE):
            raise ValueError("int_grid shape mismatch with GRID_SIZE")
        # Clear grid
        self.grid = np.full((cfg.GRID_SIZE, cfg.GRID_SIZE), None)
        for r in range(cfg.GRID_SIZE):
            for c in range(cfg.GRID_SIZE):
                t = int(arr[r, c])
                if t >= 0:
                    agent = self.agent_factory(t)
                    self.grid[r][c] = agent
                    agent.starting_position = (r, c)
                    agent.position_history = [(r, c)]
                    agent.new_position = None
        if step is not None:
            try:
                self.step = int(step)
            except Exception:
                pass

    @staticmethod
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

    @staticmethod
    def load_results_from_output(output_dir, force_recompute: bool = False):
        """
        Load simulation results from stored output files to feed into analyze_results function.
        
        Args:
            output_dir (str): Directory containing saved simulation outputs
            force_recompute (bool): If True, ignore existing analysis files and rebuild
                from raw logs/states when possible.
            
        Returns:
            tuple: (results, n_runs) where results is a list of result dictionaries
                  compatible with analyze_results function
        """
        results = []
        
        # Check if metrics_history.csv already exists (from previous analysis)
        metrics_file = os.path.join(output_dir, "metrics_history.csv")
        convergence_file = os.path.join(output_dir, "convergence_summary.csv")
        
        if (not force_recompute) and os.path.exists(metrics_file) and os.path.exists(convergence_file):
            print(f"Loading existing analysis files from {output_dir}")
            
            # Load pre-computed metrics and convergence data
            metrics_df = pd.read_csv(metrics_file)
            convergence_df = pd.read_csv(convergence_file)
            
            # Group metrics by run_id to reconstruct results structure
            for run_id in convergence_df['run_id'].unique():
                convergence_row = convergence_df[convergence_df['run_id'] == run_id].iloc[0]
                run_metrics = metrics_df[metrics_df['run_id'] == run_id].to_dict('records')
                
                result = {
                    'run_id': run_id,
                    'converged': convergence_row['converged'],
                    'convergence_step': convergence_row['convergence_step'] if pd.notna(convergence_row['convergence_step']) else None,
                    'final_step': convergence_row['final_step'],
                    'metrics_history': run_metrics
                }
                results.append(result)
                
            n_runs = len(convergence_df)
            print(f"Loaded {n_runs} simulation runs from existing analysis files")
            
        else:
            print(f"Loading raw simulation data from {output_dir}")
            
            # Load from individual move log files 
            move_logs_dir = os.path.join(output_dir, "move_logs")
            
            if not os.path.exists(move_logs_dir):
                raise FileNotFoundError(f"Move logs directory not found: {move_logs_dir}")
            
            # Find all run files (JSON logs are the canonical source)
            move_files = [
                f for f in os.listdir(move_logs_dir)
                if f.startswith("agent_moves_run_") and f.endswith(".json.gz")
            ]

            run_ids = sorted({int(f.split("_")[-1].split(".")[0]) for f in move_files})
            
            print(f"Found {len(run_ids)} simulation runs: {run_ids}")
            
            for run_id in run_ids:
                print(f"Loading run {run_id}...")
                
                # Load agent move log to reconstruct metrics history
                json_path = os.path.join(move_logs_dir, f"agent_moves_run_{run_id}.json.gz")

                with gzip.open(json_path, 'rt', encoding='utf-8') as f:
                    move_records = json.load(f)
                move_df = pd.DataFrame(move_records)
                
                # Extract convergence information from move log
                max_step = move_df['step'].max() if not move_df.empty else 0
                
                # Check if simulation converged (look for patterns in the data)
                # Use the configured NO_MOVE_THRESHOLD for the trailing window.
                threshold = getattr(cfg, 'NO_MOVE_THRESHOLD', 5)
                step_moves = move_df.groupby('step')['moved'].sum() if not move_df.empty else pd.Series(dtype=int)
                # Consider exactly the last `threshold` steps (or all if fewer)
                if not step_moves.empty:
                    window = step_moves.tail(threshold)
                    # Converged if all moves are zero across the trailing window AND we had at least one step
                    if len(window) == threshold and (window == 0).all():
                        converged = True
                        # Convergence step is the first step in the zero window
                        convergence_step = window.index[0]
                    else:
                        converged = False
                        convergence_step = None
                else:
                    converged = False
                    convergence_step = None
                
                # Try to reconstruct metrics history from the move log
                # Group by step and calculate basic metrics if available
                metrics_history = []
                
                # If we have detailed grid states saved in move log, we can reconstruct metrics
                if 'grid' in move_df.columns and not move_df.empty:
                    step_groups = move_df.groupby('step')
                    
                    for step, group in step_groups:
                        # Take the last grid state for this step (after all moves completed)
                        last_entry = group.iloc[-1]
                        
                        # Try to parse the grid if it's stored as string
                        try:
                            if isinstance(last_entry['grid'], str):
                                grid_data = eval(last_entry['grid'])  # Parse string representation
                            else:
                                grid_data = last_entry['grid']
                            
                            # Convert to numpy array and calculate metrics
                            grid_array = np.array(grid_data)
                            
                            # Create a mock grid for metrics calculation
                            mock_grid = np.full((cfg.GRID_SIZE, cfg.GRID_SIZE), None)
                            for r in range(grid_array.shape[0]):
                                for c in range(grid_array.shape[1]):
                                    if grid_array[r, c] >= 0:
                                        # Create a simple agent-like object with type_id
                                        class MockAgent:
                                            def __init__(self, type_id):
                                                self.type_id = type_id
                                        mock_grid[r, c] = MockAgent(grid_array[r, c])
                            
                            # Calculate metrics for this step
                            step_metrics = calculate_all_metrics(mock_grid)
                            step_metrics['step'] = step
                            step_metrics['run_id'] = run_id
                            metrics_history.append(step_metrics)
                            
                        except Exception as e:
                            print(f"Warning: Could not reconstruct metrics for step {step}, run {run_id}: {e}")
                            # Create minimal metrics entry
                            metrics_history.append({
                                'step': step,
                                'run_id': run_id,
                                'clusters': 0,
                                'switch_rate': 0,
                                'distance': 0,
                                'mix_deviation': 0,
                                'share': 0.5,
                                'ghetto_rate': 0
                            })
                
                # If no metrics could be reconstructed, create minimal result
                if not metrics_history:
                    metrics_history = [{
                        'step': 0,
                        'run_id': run_id,
                        'clusters': 0,
                        'switch_rate': 0,
                        'distance': 0,
                        'mix_deviation': 0,
                        'share': 0.5,
                        'ghetto_rate': 0
                    }]
                
                result = {
                    'run_id': run_id,
                    'converged': converged,
                    'convergence_step': convergence_step,
                    'final_step': max_step,
                    'metrics_history': metrics_history
                }
                results.append(result)
            
            n_runs = len(results)
            print(f"Loaded {n_runs} simulation runs from raw data")
        
        return results, n_runs

    @staticmethod
    def load_and_analyze_results(output_dir, force_recompute: bool = False):
        """
        Convenience function that loads stored simulation outputs and runs analysis.
        
        Args:
            output_dir (str): Directory containing saved simulation outputs
            force_recompute (bool): If True, ignore existing analysis files and rebuild
                from raw logs/states when possible.
            
        Returns:
            tuple: (output_dir, results, convergence_data) from analyze_results
        """
        print(f"Loading and analyzing results from: {output_dir}")
        
        # Load the results from stored output
        # Be robust to monkeypatched or older signatures that don't accept force_recompute
        try:
            results, n_runs = Simulation.load_results_from_output(output_dir, force_recompute=force_recompute)
        except TypeError:
            try:
                # Try positional in case only positional args are supported
                results, n_runs = Simulation.load_results_from_output(output_dir, force_recompute)
            except TypeError:
                # Fall back to legacy call with only output_dir
                results, n_runs = Simulation.load_results_from_output(output_dir)
        
        if not results:
            raise ValueError(f"No simulation results found in {output_dir}")
        
        print(f"Analyzing {n_runs} simulation runs...")
        
        # Run the analysis
        return Simulation.analyze_results(results, output_dir, n_runs)
