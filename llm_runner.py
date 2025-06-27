import numpy as np
import json
import os
import random
from datetime import datetime
import config as cfg
from Agent import Agent
import requests
from tqdm import tqdm
import time
from context_scenarios import CONTEXT_SCENARIOS
import argparse
from base_simulation import Simulation
from multiprocessing import Pool, cpu_count

def check_llm_connection(llm_model=None, llm_url=None, llm_api_key=None, timeout=10):
    """
    Check if LLM connection is active and working
    
    Parameters:
    - llm_model: Model to use (overrides config.py)
    - llm_url: API URL (overrides config.py)
    - llm_api_key: API key (overrides config.py)
    - timeout: Connection timeout in seconds
    
    Returns:
    - True if connection successful
    - False if connection failed
    """
    model = llm_model or cfg.OLLAMA_MODEL
    url = llm_url or cfg.OLLAMA_URL
    api_key = llm_api_key or cfg.OLLAMA_API_KEY
    
    print("\nChecking LLM connection...")
    print(f"URL: {url}")
    print(f"Model: {model}")
    
    try:
        test_payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Respond with only the word 'OK' and nothing else."}],
            "stream": False,
            "temperature": 0
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        response = requests.post(url, headers=headers, json=test_payload, timeout=timeout)
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            print(f"❌ LLM connection failed - HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        data = response.json()
        
        # Check for proper response structure
        if "choices" not in data or not data["choices"]:
            print("❌ LLM connection failed - Invalid response structure")
            print(f"Response: {data}")
            return False
        
        content = data["choices"][0]["message"]["content"].strip()
        print(f"✅ LLM connection successful (response time: {elapsed:.2f}s)")
        print(f"Test response: '{content}'")
        
        return True
        
    except requests.exceptions.Timeout:
        print(f"❌ LLM connection failed - Timeout after {timeout}s")
        print("The LLM server is not responding. Please check:")
        print("1. Is the Ollama server running?")
        print("2. Is the URL correct?")
        print("3. Is the model loaded?")
        return False
        
    except requests.exceptions.ConnectionError as e:
        print("❌ LLM connection failed - Connection error")
        print(f"Error: {e}")
        print("\nPlease check:")
        print("1. Is the Ollama server running?")
        print("2. Is the URL correct?")
        print("3. Is your network connection working?")
        return False
        
    except Exception as e:
        print("❌ LLM connection failed - Unexpected error")
        print(f"Error: {type(e).__name__}: {e}")
        return False

class LLMAgent(Agent):
    def __init__(self, type_id, scenario='baseline', llm_model=None, llm_url=None, llm_api_key=None):
        super().__init__(type_id)
        self.scenario = scenario
        self.context_info = CONTEXT_SCENARIOS[scenario]
        self.agent_type = self.context_info['type_a'] if type_id == 0 else self.context_info['type_b']
        self.opposite_type = self.context_info['type_b'] if type_id == 0 else self.context_info['type_a']
        self.llm_model = llm_model or cfg.OLLAMA_MODEL
        self.llm_url = llm_url or cfg.OLLAMA_URL
        self.llm_api_key = llm_api_key or cfg.OLLAMA_API_KEY
    
    def get_llm_decision(self, r, c, grid, max_retries=10):
        """Get movement decision from LLM with retry logic (max_retries attempts)"""
        # Debug flag - set via environment variable
        debug = os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes')
        
        if debug:
            print(f"\n[DEBUG] LLM Decision Request for agent at ({r},{c})")
            print(f"[DEBUG] Agent type: {self.agent_type} | Scenario: {self.scenario}")
        
        # Construct 3x3 neighborhood context
        context = []
        for dr in [-1, 0, 1]:
            row = []
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < cfg.GRID_SIZE and 0 <= nc < cfg.GRID_SIZE:
                    neighbor = grid[nr][nc]
                    if neighbor is None:
                        row.append("E")  # Empty
                    elif neighbor.type_id == self.type_id:
                        row.append("S")  # Same
                    else:
                        row.append("O")  # Opposite
                else:
                    row.append("#")  # Out of bounds
            context.append(row)
        
        # Format context for prompt - mark current position
        context_with_position = []
        for i, row in enumerate(context):
            new_row = []
            for j, cell in enumerate(row):
                if i == 1 and j == 1:  # Center position (current location)
                    new_row.append("X")
                else:
                    new_row.append(cell)
            context_with_position.append(new_row)
        
        context_str = "\n".join([" ".join(row) for row in context_with_position])
        
        # Create prompt using context scenario template
        prompt = self.context_info['prompt_template'].format(
            agent_type=self.agent_type,
            opposite_type=self.opposite_type,
            context=context_str
        )
        
        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": 0.3,  # Lower temperature for more consistent responses
                    "max_tokens": 50,    # Limit response length
                    "timeout": 20000     # 20 second timeout in milliseconds
                }
                
                headers = {
                    "Authorization": f"Bearer {self.llm_api_key}",
                    "Content-Type": "application/json"
                }
                
                if debug:
                    print(f"[DEBUG] Sending LLM request to: {self.llm_url}")
                    print(f"[DEBUG] Model: {self.llm_model}")
                    print(f"[DEBUG] Context grid:\n{context_str}")
                
                # Track timing and calls
                start_time = time.time()
                response = requests.post(self.llm_url, headers=headers, json=payload, timeout=8)
                response_time = time.time() - start_time
                
                # Update agent's LLM metrics
                self.llm_call_count += 1
                self.llm_call_time += response_time
                
                if debug:
                    print(f"[DEBUG] LLM Response received in {response_time:.2f}s")
                    print(f"[DEBUG] Status code: {response.status_code}")

                
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                
                if debug:
                    print(f"[DEBUG] LLM Response text: '{text}'")
                    print("[DEBUG] Attempting to parse decision...")
                
                # Parse MOVE/STAY response
                text_upper = text.strip().upper()
                
                if "MOVE" in text_upper:
                    if debug:
                        print("[DEBUG] Decision: MOVE - finding random empty space")
                    
                    # Find all empty spaces on the grid
                    empty_spaces = []
                    for row in range(cfg.GRID_SIZE):
                        for col in range(cfg.GRID_SIZE):
                            if grid[row][col] is None:  # Empty space
                                empty_spaces.append((row, col))
                    
                    # Return random empty space if available
                    if empty_spaces:
                        chosen_pos = random.choice(empty_spaces)
                        if debug:
                            print(f"[DEBUG] Moving to random empty position: {chosen_pos}")
                        return chosen_pos
                    else:
                        if debug:
                            print("[DEBUG] No empty spaces available, staying put")
                        return None
                
                elif "STAY" in text_upper:
                    if debug:
                        print("[DEBUG] Decision: STAY")
                    return None
                
                else:
                    if debug:
                        print(f"[DEBUG] Could not parse MOVE/STAY from: '{text}'")
                        print("[DEBUG] Decision: STAY (parse failure)")
                    return None
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    print(f"[LLM Timeout] Retry {attempt + 1}/{max_retries} for agent at ({r},{c})")
                    time.sleep(10)  # Wait 10 seconds before retry
                    continue
                else:
                    print(f"[LLM Error] Max retries exceeded ({max_retries}) for agent at ({r},{c})")
                    raise Exception(f"LLM timeout after {max_retries} retries")
            except Exception as e:
                if attempt < max_retries:
                    print(f"[LLM Error] Exception - Retry {attempt + 1}/{max_retries}: {e}")
                    time.sleep(60)  # Wait 1 minute before retry
                    continue
                else:
                    print(f"[LLM Error] Exception: Unhandled error after {max_retries} retries: {e}")
                    raise Exception(f"LLM error after {max_retries} retries: {e}")

def llm_decision_function(agent, r, c, grid):
    """Decision function for LLM agents with retry logic (no fallback to mechanical decision)"""
    return agent.get_llm_decision(r, c, grid)

class LLMSimulation(Simulation):
    def __init__(self, run_id, scenario='baseline', llm_model=None, llm_url=None, llm_api_key=None, random_seed=None):
        # Store LLM parameters for agent creation
        self.scenario = scenario
        self.llm_model = llm_model or cfg.OLLAMA_MODEL
        self.llm_url = llm_url or cfg.OLLAMA_URL
        self.llm_api_key = llm_api_key or cfg.OLLAMA_API_KEY
        
        super().__init__(
            run_id=run_id, 
            agent_factory=self._create_llm_agent, 
            decision_func=llm_decision_function, 
            scenario=scenario,
            random_seed=random_seed
        )
        
        # Track LLM metrics across all agents
        self.total_llm_calls = 0
        self.total_llm_time = 0.0
    
    def _create_llm_agent(self, type_id):
        """Create LLM agent with simulation parameters"""
        return LLMAgent(type_id, self.scenario, self.llm_model, self.llm_url, self.llm_api_key)

    def run_step(self):
        """Override run_step to track LLM metrics"""
        # Update total LLM metrics from all agents
        for r in range(cfg.GRID_SIZE):
            for c in range(cfg.GRID_SIZE):
                agent = self.grid[r][c]
                if agent is not None and hasattr(agent, 'llm_call_count'):
                    self.total_llm_calls += agent.llm_call_count
                    self.total_llm_time += agent.llm_call_time
                    # Reset agent counters to avoid double counting
                    agent.llm_call_count = 0
                    agent.llm_call_time = 0.0
        
        # Call parent run_step
        return super().run_step()

    def run_single_simulation(self, output_dir=None, max_steps=1000):
        """Override to show progress bar for LLM simulations"""
        return super().run_single_simulation(output_dir=output_dir, max_steps=max_steps, show_progress=True)

    # def run_step(self):
    #     moved = self.update_agents()
    #     metrics = calculate_all_metrics(self.grid)
    #     metrics['step'] = self.step
    #     metrics['run_id'] = self.run_id
    #     self.metrics_history.append(metrics)
    #     self.states.append(self._grid_to_int())
    #     if not moved:
    #         self.no_move_steps += 1
    #     else:
    #         self.no_move_steps = 0
    #     if self.no_move_steps >= self.no_move_threshold:
    #         self.converged = True
    #         self.convergence_step = self.step
    #     self.step += 1
    #     if self.progress_file and self.step % 10 == 0:
    #         try:
    #             with open(self.progress_file, 'r') as f:
    #                 progress_data = json.load(f)
    #             progress_data.update({
    #                 "current_step": self.step,
    #                 "step_progress_percent": (self.step / progress_data["max_steps"]) * 100,
    #                 "converged": self.converged,
    #                 "timestamp": datetime.now().isoformat()
    #             })
    #             with open(self.progress_file, 'w') as f:
    #                 json.dump(progress_data, f, indent=2)
    #         except Exception:
    #             pass
    #     return self.converged

    # def run(self, max_steps=1000):
    #     progress_bar = tqdm(total=max_steps, desc=f"Run {self.run_id} ({self.scenario})")
    #     while self.step < max_steps and not self.converged:
    #         self.run_step()
    #         progress_bar.update(1)
    #         if self.step % 10 == 0:
    #             avg_llm_time = np.mean(self.llm_call_times[-100:]) if self.llm_call_times else 0
    #             progress_bar.set_postfix({'converged': self.converged, 'llm_calls': self.llm_call_count, 'avg_llm_time': f"{avg_llm_time:.2f}s"})
    #     progress_bar.close()
    #     self._shutdown_requested = True
    #     try:
    #         self.query_queue.put(None, timeout=1.0)
    #         self.llm_thread.join(timeout=5.0)
    #         if self.llm_thread.is_alive():
    #             print("[Warning] LLM worker thread did not shut down cleanly")
    #     except queue.Full:
    #         print("[Warning] Could not send shutdown signal to LLM worker")
    #     except Exception as e:
    #         print(f"[Warning] Error during LLM worker cleanup: {e}")
    #     return {
    #         'run_id': self.run_id,
    #         'scenario': self.scenario,
    #         'converged': self.converged,
    #         'convergence_step': self.convergence_step,
    #         'final_step': self.step,
    #         'metrics_history': self.metrics_history,
    #         'llm_call_count': self.llm_call_count,
    #         'avg_llm_call_time': np.mean(self.llm_call_times) if self.llm_call_times else 0,
    #         'states': self.states
    #     }

# def run_llm_experiment(scenario='baseline', n_runs=10, max_steps=1000, llm_model=None, llm_url=None, llm_api_key=None):
#     """Run LLM experiments with specified scenario"""
    
#     # Check LLM connection first with potentially custom parameters
#     if not check_llm_connection(llm_model, llm_url, llm_api_key):
#         print("\n⚠️  Cannot proceed with LLM experiments - connection check failed!")
#         print("Please ensure the LLM server is running and accessible.")
#         return None, []
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     experiment_name = f"llm_{scenario}_{timestamp}"
#     output_dir = f"experiments/{experiment_name}"
#     os.makedirs(output_dir, exist_ok=True)
    
#     config_dict = {
#         'n_runs': n_runs,
#         'max_steps': max_steps,
#         'grid_size': cfg.GRID_SIZE,
#         'num_type_a': cfg.NUM_TYPE_A,
#         'num_type_b': cfg.NUM_TYPE_B,
#         'llm_model': llm_model or cfg.OLLAMA_MODEL,
#         'llm_url': llm_url or cfg.OLLAMA_URL,
#         'llm_api_key_last4': (llm_api_key or cfg.OLLAMA_API_KEY)[-4:] if (llm_api_key or cfg.OLLAMA_API_KEY) else None,
#         'timestamp': timestamp,
#         'context_info': CONTEXT_SCENARIOS[scenario]
#     }
    
#     with open(f"{output_dir}/config.json", 'w') as f:
#         json.dump(config_dict, f, indent=2)
    
#     args_list = [(i, scenario, llm_model, llm_url, llm_api_key, output_dir) for i in range(n_runs)]
#     results = []
#     for args in tqdm(args_list, desc="Running LLM simulations"):
#         sim = LLMSimulation(*args[:-1])
#         results.append(sim.run(max_steps))

#     # Analyze results
#     output_dir, results, convergence_data = Simulation.analyze_results(results, output_dir, n_runs)
    
#     print(f"\nExperiment completed. Results saved to: {output_dir}")
#     print(f"Total runs: {n_runs}")
#     print(f"Converged runs: {sum(1 for r in convergence_data if r['converged'])}")
#     print(f"Average convergence step: {np.mean([r['convergence_step'] for r in convergence_data if r['convergence_step'] is not None]):.2f}")
#     return output_dir, results

def run_single_simulation(args):
    """Run a single LLM simulation - compatible with baseline_runner structure"""
    run_id, scenario, llm_model, llm_url, llm_api_key, output_dir = args
    sim = LLMSimulation(run_id, scenario, llm_model, llm_url, llm_api_key)
    result = sim.run_single_simulation(output_dir=output_dir, max_steps=1000)
    
    # Add LLM-specific metrics to the result
    result.update({
        'scenario': scenario,
        'llm_call_count': sim.total_llm_calls,
        'avg_llm_call_time': sim.total_llm_time / max(sim.total_llm_calls, 1),
    })
    return result
    
def run_llm_experiment(scenario='baseline', n_runs=10, max_steps=1000, llm_model=None, llm_url=None, llm_api_key=None, parallel=True, n_processes=None):
    """
    Run LLM experiments with specified scenario - compatible with baseline_runner structure
    
    Parameters:
    -----------
    scenario : str
        Scenario context to use for the experiment
    n_runs : int
        Number of simulation runs to perform
    max_steps : int
        Maximum steps per simulation
    llm_model : str, optional
        LLM model to use (overrides config.py)
    llm_url : str, optional
        LLM API URL (overrides config.py)
    llm_api_key : str, optional
        LLM API key (overrides config.py)
    parallel : bool
        Whether to use parallel processing
    n_processes : int, optional
        Number of CPU processes to use for parallel execution.
        If None, uses min(cpu_count(), n_runs). If 1, forces sequential execution.
        
    Returns:
    --------
    tuple
        (output_dir, results) where results contains simulation outcomes
    """
    
    # Check LLM connection first with potentially custom parameters
    if not check_llm_connection(llm_model, llm_url, llm_api_key):
        print("\n⚠️  Cannot proceed with LLM experiments - connection check failed!")
        print("Please ensure the LLM server is running and accessible.")
        return None, []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"llm_{scenario}_{timestamp}"
    output_dir = f"experiments/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    config_dict = {
        'n_runs': n_runs,
        'max_steps': max_steps,
        'grid_size': cfg.GRID_SIZE,
        'num_type_a': cfg.NUM_TYPE_A,
        'num_type_b': cfg.NUM_TYPE_B,
        'scenario': scenario,
        'llm_model': llm_model or cfg.OLLAMA_MODEL,
        'llm_url': llm_url or cfg.OLLAMA_URL,
        'llm_api_key_last4': (llm_api_key or cfg.OLLAMA_API_KEY)[-4:] if (llm_api_key or cfg.OLLAMA_API_KEY) else None,
        'no_move_threshold': cfg.NO_MOVE_THRESHOLD,
        'timestamp': timestamp,
        'context_info': CONTEXT_SCENARIOS[scenario],
        'parallel_execution': parallel,
        'n_processes': n_processes if parallel else 1,
        'cpu_count': cpu_count()
    }
    
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    args_list = [(i, scenario, llm_model, llm_url, llm_api_key, output_dir) for i in range(n_runs)]
    
    # Determine number of processes to use
    if n_processes is None:
        n_processes = min(cpu_count(), n_runs)
    elif n_processes == 1:
        parallel = False  # Force sequential execution if only 1 process requested
    elif n_processes > cpu_count():
        print(f"⚠️  Warning: Requested {n_processes} processes but only {cpu_count()} CPU cores available.")
        print(f"   Using {cpu_count()} processes instead.")
        n_processes = cpu_count()
    elif n_processes > n_runs:
        print(f"⚠️  Warning: Requested {n_processes} processes but only {n_runs} runs to execute.")
        print(f"   Using {n_runs} processes instead.")
        n_processes = n_runs
    elif n_processes < 1:
        print(f"⚠️  Warning: Invalid number of processes ({n_processes}). Using 1 process (sequential).")
        n_processes = 1
        parallel = False
    
    if parallel and n_processes > 1:
        print(f"Running {n_runs} simulations using {n_processes} parallel processes...")
        with Pool(n_processes) as pool:
            results = list(tqdm(
                pool.imap(run_single_simulation, args_list),
                total=n_runs,
                desc="Running LLM simulations"
            ))
    else:
        print(f"Running {n_runs} simulations sequentially...")
        results = []
        for args in tqdm(args_list, desc="Running LLM simulations"):
            results.append(run_single_simulation(args))

    # Analyze results using Simulation's analyze_results method
    output_dir, results, convergence_data = Simulation.analyze_results(results, output_dir, n_runs)
    
    print(f"\nExperiment completed. Results saved to: {output_dir}")
    print(f"Total runs: {n_runs}")
    print(f"Converged runs: {sum(1 for r in convergence_data if r['converged'])}")
    converged_steps = [r['convergence_step'] for r in convergence_data if r['convergence_step'] is not None]
    if converged_steps:
        print(f"Average convergence step: {np.mean(converged_steps):.2f}")
    
    # Calculate LLM-specific statistics
    llm_calls = [r.get('llm_call_count', 0) for r in results if 'llm_call_count' in r]
    llm_times = [r.get('avg_llm_call_time', 0) for r in results if 'avg_llm_call_time' in r]
    if llm_calls:
        print(f"Average LLM calls per run: {np.mean(llm_calls):.1f}")
        print(f"Average LLM response time: {np.mean(llm_times):.3f}s")
    
    return output_dir, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-based Schelling segregation simulations")
    parser.add_argument('--runs', type=int, default=10, help='Number of simulation runs')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps per simulation')
    parser.add_argument('--scenario', type=str, default='baseline', choices=list(CONTEXT_SCENARIOS.keys()), help='Scenario to simulate')
    parser.add_argument('--llm-model', type=str, help='LLM model to use (overrides config.py)')
    parser.add_argument('--llm-url', type=str, help='LLM API URL (overrides config.py)')
    parser.add_argument('--llm-api-key', type=str, help='LLM API key (overrides config.py)')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--processes', type=int, default=None, 
                       help=f'Number of CPU processes to use (default: min(cpu_count={cpu_count()}, n_runs)). Use 1 for sequential execution.')
    args = parser.parse_args()

    run_llm_experiment(
        scenario=args.scenario,
        n_runs=args.runs,
        max_steps=args.max_steps,
        llm_model=args.llm_model,
        llm_url=args.llm_url,
        llm_api_key=args.llm_api_key,
        parallel=not args.no_parallel,
        n_processes=args.processes
    )