import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
import config as cfg
from Agent import Agent
from Metrics import calculate_all_metrics
from tqdm import tqdm
import argparse

class BaselineSimulation:
    def __init__(self, run_id, config_override=None):
        self.run_id = run_id
        self.grid = np.full((cfg.GRID_SIZE, cfg.GRID_SIZE), None)
        self.step = 0
        self.converged = False
        self.convergence_step = None
        self.no_move_steps = 0
        self.no_move_threshold = 20
        self.metrics_history = []
        
        # Initialize list to store grid snapshot per step
    
        # Apply any config overrides
        if config_override:
            for key, value in config_override.items():
                setattr(cfg, key, value)
        
        self.populate_grid()
        # Initialize list to store grid snapshot per step (after population)
        # Initialize integer states list after population
        self.states = [self._grid_to_int()]
    
    def populate_grid(self):
        agents = [Agent(type_id) for type_id in ([0] * cfg.NUM_TYPE_A + [1] * cfg.NUM_TYPE_B)]
        np.random.shuffle(agents)
        flat_positions = [(r, c) for r in range(cfg.GRID_SIZE) for c in range(cfg.GRID_SIZE)]
        np.random.shuffle(flat_positions)
        for agent, pos in zip(agents, flat_positions[:len(agents)]):
            r, c = pos
            self.grid[r][c] = agent
    
    def update_agents(self):
        all_positions = [(r, c) for r in range(cfg.GRID_SIZE) for c in range(cfg.GRID_SIZE) if self.grid[r][c]]
        np.random.shuffle(all_positions)
        
        moved = False
        for r, c in all_positions:
            agent = self.grid[r][c]
            move_to = agent.best_response(r, c, self.grid)
            if move_to:
                r_new, c_new = move_to
                if 0 <= r_new < cfg.GRID_SIZE and 0 <= c_new < cfg.GRID_SIZE:
                    if self.grid[r_new][c_new] is None:
                        self.grid[r_new][c_new] = agent
                        self.grid[r][c] = None
                        moved = True
        
        return moved
    
    def run_step(self):
        moved = self.update_agents()
        metrics = calculate_all_metrics(self.grid)
        metrics['step'] = self.step
        metrics['run_id'] = self.run_id
        self.metrics_history.append(metrics)
        # Record integer snapshot of current grid
        self.states.append(self._grid_to_int())
        
        # Check convergence
        if not moved:
            self.no_move_steps += 1
        else:
            self.no_move_steps = 0
        
        if self.no_move_steps >= self.no_move_threshold:
            self.converged = True
            self.convergence_step = self.step
        
        self.step += 1
        return self.converged
    
    def run(self, max_steps=1000):
        while self.step < max_steps and not self.converged:
            self.run_step()
        
        return {
            'run_id': self.run_id,
            'converged': self.converged,
            'convergence_step': self.convergence_step,
            'final_step': self.step,
            'metrics_history': self.metrics_history,
            'states': self.states
        }
    
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

def run_single_simulation(args):
    run_id, config_override = args
    sim = BaselineSimulation(run_id, config_override)
    return sim.run()

def run_baseline_experiment(n_runs=100, max_steps=1000, config_override=None, parallel=True):
    """Run baseline experiments with mechanical agents"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"baseline_{timestamp}"
    output_dir = f"experiments/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save experiment configuration
    config_dict = {
        'n_runs': n_runs,
        'max_steps': max_steps,
        'grid_size': cfg.GRID_SIZE,
        'num_type_a': cfg.NUM_TYPE_A,
        'num_type_b': cfg.NUM_TYPE_B,
        'similarity_threshold': cfg.SIMILARITY_THRESHOLD,
        'agent_satisfaction_threshold': cfg.AGENT_SATISFACTION_THRESHOLD,
        'no_move_threshold': cfg.NO_MOVE_THRESHOLD,
        'timestamp': timestamp
    }
    if config_override:
        config_dict['overrides'] = config_override
    
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Run simulations
    args_list = [(i, config_override) for i in range(n_runs)]
    
    if parallel:
        n_processes = min(cpu_count(), n_runs)
        with Pool(n_processes) as pool:
            results = list(tqdm(
                pool.imap(run_single_simulation, args_list),
                total=n_runs,
                desc="Running baseline simulations"
            ))
    else:
        results = []
        for args in tqdm(args_list, desc="Running baseline simulations"):
            results.append(run_single_simulation(args))
    
    # Process and save results
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
    
    # Save raw data
    pd.DataFrame(all_metrics).to_csv(f"{output_dir}/metrics_history.csv", index=False)
    pd.DataFrame(convergence_data).to_csv(f"{output_dir}/convergence_summary.csv", index=False)
    
    # Calculate summary statistics
    df = pd.DataFrame(all_metrics)
    
    # Group by step and calculate statistics
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
    # Save grid timeseries for each run
    # create a 'states' subfolder in the experiment directory
    states_dir = os.path.join(output_dir, "states")
    os.makedirs(states_dir, exist_ok=True)
    for result in results:
        arr = np.stack(result['states'])
        # Save time-series grid snapshot per run inside 'states'
        np.savez_compressed(os.path.join(states_dir, f"states_run_{result['run_id']}.npz"), arr)

    print(f"\nExperiment completed. Results saved to: {output_dir}")
    print(f"Total runs: {n_runs}")
    print(f"Converged runs: {sum(1 for r in convergence_data if r['converged'])}")
    print(f"Average convergence step: {np.mean([r['convergence_step'] for r in convergence_data if r['convergence_step'] is not None]):.2f}")
    
    return output_dir, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline Schelling segregation simulations")
    parser.add_argument('--runs', type=int, default=100, help='Number of simulation runs')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps per simulation')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    
    args = parser.parse_args()
    
    run_baseline_experiment(
        n_runs=args.runs,
        max_steps=args.max_steps,
        parallel=not args.no_parallel
    )