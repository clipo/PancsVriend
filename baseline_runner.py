import numpy as np
import json
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
import config as cfg
from Agent import Agent
from tqdm import tqdm
import argparse
from base_simulation import Simulation

def mechanical_decision(agent, r, c, grid):
    return agent.random_response(r, c, grid) 

class BaselineSimulation(Simulation):
    def __init__(self, run_id, config_override=None):
        if config_override:
            for key, value in config_override.items():
                setattr(cfg, key, value)
        super().__init__(run_id, agent_factory=Agent, decision_func=mechanical_decision)

    # Optionally, add any baseline-specific logging or hooks here

def run_single_simulation(args):
    run_id, config_override, output_dir = args
    sim = BaselineSimulation(run_id, config_override)
    return sim.run_single_simulation(output_dir=output_dir, max_steps=1000)

def run_baseline_experiment(n_runs=100, max_steps=1000, config_override=None, parallel=True):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"baseline_{timestamp}"
    output_dir = f"experiments/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
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
        
    args_list = [(i, config_override, output_dir) for i in range(n_runs)]
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

    # Analyze results
    output_dir, results, convergence_data = Simulation.analyze_results(results, output_dir, n_runs)
    
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