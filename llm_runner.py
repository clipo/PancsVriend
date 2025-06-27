import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import config as cfg
from Agent import Agent
from Metrics import calculate_all_metrics
import requests
import re
from tqdm import tqdm
import argparse
import time
import threading
import queue
from context_scenarios import CONTEXT_SCENARIOS
import argparse

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
    
    def get_llm_decision(self, r, c, grid, max_retries=2):
        """Get movement decision from LLM with retry logic (max_retries attempts)"""
        # Debug flag - set via environment variable
        debug = os.environ.get('DEBUG_LLM', '').lower() in ('true', '1', 'yes')
        
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
                    row.append("X")  # Out of bounds
            context.append(row)
        
        # Format context for prompt
        context_str = "\n".join([" ".join(row) for row in context])
        
        # Create prompt
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
                    "timeout": 10000     # 10 second timeout in milliseconds
                }
                
                headers = {
                    "Authorization": f"Bearer {self.llm_api_key}",
                    "Content-Type": "application/json"
                }
                
                if debug:
                    print(f"[DEBUG] Sending LLM request to: {self.llm_url}")
                    print(f"[DEBUG] Model: {self.llm_model}")
                    print(f"[DEBUG] Context grid:\n{context_str}")
                
                # Shorter timeout for individual requests
                start_time = time.time()
                response = requests.post(self.llm_url, headers=headers, json=payload, timeout=8)
                response_time = time.time() - start_time
                
                if debug:
                    print(f"[DEBUG] LLM Response received in {response_time:.2f}s")
                    print(f"[DEBUG] Status code: {response.status_code}")
                
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                
                if debug:
                    print(f"[DEBUG] LLM Response text: '{text}'")
                    print(f"[DEBUG] Attempting to parse decision...")
                
                # Parse response
                match = re.search(r"\((\d+),\s*(\d+)\)", text)
                if match:
                    move_to = (int(match.group(1)), int(match.group(2)))
                    # Convert from relative to absolute coordinates
                    r_new = r + (move_to[0] - 1)
                    c_new = c + (move_to[1] - 1)
                    if debug:
                        print(f"[DEBUG] Parsed move: ({move_to[0]},{move_to[1]}) relative -> ({r_new},{c_new}) absolute")
                        print(f"[DEBUG] Decision: MOVE to ({r_new},{c_new})")
                    return (r_new, c_new)
                
                if "none" in text.strip().lower():
                    if debug:
                        print(f"[DEBUG] Decision: STAY (agent chose not to move)")
                    return None
                
                if debug:
                    print(f"[DEBUG] Could not parse decision from: '{text}'")
                    print(f"[DEBUG] Decision: STAY (parse failure)")
                return None
                
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    print(f"[LLM Timeout] Retry {attempt + 1}/{max_retries} for agent at ({r},{c})")
                    time.sleep(1)  # Brief pause before retry
                    continue
                else:
                    print(f"[LLM Error] Max retries exceeded ({max_retries}) for agent at ({r},{c})")
                    return self.best_response(r, c, grid)
                    
            except Exception as e:
                if attempt < max_retries:
                    print(f"[LLM Error] {type(e).__name__} - Retry {attempt + 1}/{max_retries}")
                    time.sleep(1)
                    continue
                else:
                    print(f"[LLM Error] {type(e).__name__}: {e}")
                    return self.best_response(r, c, grid)

class LLMSimulation:
    def __init__(self, run_id, scenario='baseline', use_llm_probability=1.0, 
                 llm_model=None, llm_url=None, llm_api_key=None, progress_file=None):
        self.run_id = run_id
        self.scenario = scenario
        self.use_llm_probability = use_llm_probability
        self.llm_model = llm_model
        self.llm_url = llm_url
        self.llm_api_key = llm_api_key
        self.progress_file = progress_file
        self.grid = np.full((cfg.GRID_SIZE, cfg.GRID_SIZE), None)
        self.step = 0
        self.converged = False
        self.convergence_step = None
        self.no_move_steps = 0
        self.no_move_threshold = 20
        self.metrics_history = []
        self.llm_call_count = 0
        self.llm_call_times = []
        
        # Circuit breaker for LLM failures
        self.llm_failure_count = 0
        self.max_llm_failures = 20  # After 20 failures, switch to mechanical only
        self.llm_circuit_open = False
        
        self.populate_grid()
        # Initialize integer states list after population
        self.states = [self._grid_to_int()]
        self.setup_llm_worker()
    
    def populate_grid(self):
        agents = [LLMAgent(type_id, self.scenario, self.llm_model, self.llm_url, self.llm_api_key) 
                 for type_id in ([0] * cfg.NUM_TYPE_A + [1] * cfg.NUM_TYPE_B)]
        np.random.shuffle(agents)
        flat_positions = [(r, c) for r in range(cfg.GRID_SIZE) for c in range(cfg.GRID_SIZE)]
        np.random.shuffle(flat_positions)
        for agent, pos in zip(agents, flat_positions[:len(agents)]):
            r, c = pos
            self.grid[r][c] = agent
    
    def restart_worker_if_needed(self):
        """Restart worker thread if it died"""
        if not self.llm_thread.is_alive():
            print("[Warning] Worker thread died, restarting...")
            self.setup_llm_worker()
            return True
        return False
    
    def setup_llm_worker(self):
        def worker():
            while True:
                try:
                    # Use timeout to prevent hanging
                    task = self.query_queue.get(timeout=1.0)
                    if task is None:  # Shutdown signal
                        break
                    
                    agent, r, c, task_id = task
                    start_time = time.time()
                    
                    try:
                        # Check circuit breaker
                        use_llm = (not self.llm_circuit_open and 
                                 np.random.random() < self.use_llm_probability)
                        
                        if use_llm:
                            # Use threading timeout for LLM calls
                            move_to = None
                            call_start = time.time()
                            
                            try:
                                move_to = agent.get_llm_decision(r, c, self.grid)
                                call_elapsed = time.time() - call_start
                                
                                # Check if call took too long
                                if call_elapsed > 20.0:  # 20 second warning
                                    print(f"[LLM Warning] Slow response ({call_elapsed:.1f}s) for agent at ({r},{c})")
                                
                                self.llm_call_count += 1
                            except Exception as e:
                                call_elapsed = time.time() - call_start
                                print(f"[LLM Error] Failed after {call_elapsed:.1f}s for agent at ({r},{c}): {e}")
                                move_to = agent.best_response(r, c, self.grid)
                                self.llm_failure_count += 1
                        else:
                            move_to = agent.best_response(r, c, self.grid)
                        
                        elapsed = time.time() - start_time
                        self.llm_call_times.append(elapsed)
                        
                        # Add task_id to track which request this is
                        self.result_queue.put((agent, r, c, move_to, task_id))
                        
                    except Exception as e:
                        print(f"[LLM Worker Error] Agent ({r},{c}): {e}")
                        self.llm_failure_count += 1
                        
                        # Open circuit breaker if too many failures
                        if self.llm_failure_count >= self.max_llm_failures and not self.llm_circuit_open:
                            self.llm_circuit_open = True
                            print(f"[Circuit Breaker] Too many LLM failures ({self.llm_failure_count}), switching to mechanical agents only")
                        
                        # Fallback to mechanical behavior
                        move_to = agent.best_response(r, c, self.grid)
                        elapsed = time.time() - start_time
                        self.llm_call_times.append(elapsed)
                        self.result_queue.put((agent, r, c, move_to, task_id))
                    
                    finally:
                        self.query_queue.task_done()
                        
                except queue.Empty:
                    # No tasks available, check if we should shutdown
                    if hasattr(self, '_shutdown_requested') and self._shutdown_requested:
                        break
                    continue
                    
                except Exception as e:
                    print(f"[LLM Worker Critical Error] {e}")
                    break
        
        self.query_queue = queue.Queue(maxsize=100)  # Limit queue size
        self.result_queue = queue.Queue(maxsize=100)
        self._shutdown_requested = False
        self._task_counter = 0
        self.llm_thread = threading.Thread(target=worker, daemon=True)
        self.llm_thread.start()
    
    def _grid_to_int(self):
        """
        Convert self.grid of Agent objects/None into int grid:
        -1 for empty, agent.type_id for occupied.
        """
        size = cfg.GRID_SIZE
        int_grid = np.full((size, size), -1, dtype=int)
        for r in range(size):
            for c in range(size):
                cell = self.grid[r][c]
                if cell is not None:
                    int_grid[r, c] = cell.type_id
        return int_grid

    def update_agents(self):
        all_positions = [(r, c) for r in range(cfg.GRID_SIZE) for c in range(cfg.GRID_SIZE) if self.grid[r][c]]
        np.random.shuffle(all_positions)
        
        # Limit batch size to prevent overwhelming the LLM
        batch_size = min(5, len(all_positions))  # Reduced from 10 to 5
        moved = False
        
        pending_agents = []
        for i in range(0, len(all_positions) + len(pending_agents), batch_size):
            # Combine pending agents from previous batch with new batch
            if pending_agents:
                batch = pending_agents + all_positions[i:i+max(0, batch_size-len(pending_agents))]
                pending_agents = []
            else:
                batch = all_positions[i:i+batch_size]
            task_ids = []
            batch_retry = []
            # Queue batch with task IDs for tracking
            for r, c in batch:
                agent = self.grid[r][c]
                if agent is not None:  # Double-check agent still exists
                    task_id = self._task_counter
                    self._task_counter += 1
                    task_ids.append(task_id)
                    try:
                        # Check if worker thread is alive before queuing
                        if not self.llm_thread.is_alive():
                            print("[Error] Worker thread died, attempting restart...")
                            if self.restart_worker_if_needed():
                                try:
                                    self.query_queue.put((agent, r, c, task_id), timeout=2.0)
                                except queue.Full:
                                    batch_retry.append((r, c))
                                    continue
                        else:
                            self.query_queue.put((agent, r, c, task_id), timeout=2.0)
                    except queue.Full:
                        batch_retry.append((r, c))
                        continue
            # Add any agents that couldn't be queued to pending_agents for next batch
            pending_agents.extend(batch_retry)
            
            # Wait for results with proper timeout handling
            results = []
            expected_results = len(task_ids)
            timeout_per_batch = 30.0  # 30 seconds total for entire batch
            
            start_wait = time.time()
            while len(results) < expected_results:
                time_elapsed = time.time() - start_wait
                
                # Check if we've exceeded total batch timeout
                if time_elapsed > timeout_per_batch:
                    print(f"[Warning] Batch timeout exceeded. Got {len(results)}/{expected_results} results in {time_elapsed:.1f}s")
                    break
                
                # Calculate remaining time for this get() call
                remaining_time = timeout_per_batch - time_elapsed
                individual_timeout = min(5.0, remaining_time)  # Max 5s per individual result
                
                if individual_timeout <= 0:
                    print(f"[Warning] No time remaining for more results. Got {len(results)}/{expected_results}")
                    break
                
                try:
                    result = self.result_queue.get(timeout=individual_timeout)
                    results.append(result)
                except queue.Empty:
                    # Check if worker thread is still alive
                    if not self.llm_thread.is_alive():
                        print(f"[Error] Worker thread died. Got {len(results)}/{expected_results} results")
                        break
                    
                    # Continue waiting if thread is alive and we have time
                    if time_elapsed < timeout_per_batch * 0.8:  # Give up at 80% of total timeout
                        continue
                    else:
                        print(f"[Warning] Timeout waiting for remaining results. Got {len(results)}/{expected_results}")
                        break
            
            # Process moves
            for result in results:
                if len(result) >= 4:  # New format with task_id
                    agent, r, c, move_to, task_id = result
                else:  # Old format fallback
                    agent, r, c, move_to = result[:4]
                
                if self._process_move(agent, r, c, move_to):
                    moved = True
        
        return moved
    
    def _process_move(self, agent, r, c, move_to):
        """Helper method to process a single agent move"""
        if move_to and move_to != (r, c):
            r_new, c_new = move_to
            if 0 <= r_new < cfg.GRID_SIZE and 0 <= c_new < cfg.GRID_SIZE:
                if self.grid[r_new][c_new] is None and self.grid[r][c] == agent:
                    self.grid[r_new][c_new] = agent
                    self.grid[r][c] = None
                    return True
        return False
    
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
        
        # Update progress file if available (every 10 steps to avoid too frequent writes)
        if self.progress_file and self.step % 10 == 0:
            try:
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                progress_data["current_step"] = self.step
                progress_data["step_progress_percent"] = (self.step / progress_data["max_steps"]) * 100
                progress_data["converged"] = self.converged
                progress_data["timestamp"] = datetime.now().isoformat()
                with open(self.progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
            except Exception as e:
                pass  # Don't let progress file errors break the simulation
        
        return self.converged
    
    def run(self, max_steps=1000):
        progress_bar = tqdm(total=max_steps, desc=f"Run {self.run_id} ({self.scenario})")
        
        while self.step < max_steps and not self.converged:
            self.run_step()
            progress_bar.update(1)
            
            if self.step % 10 == 0:
                avg_llm_time = np.mean(self.llm_call_times[-100:]) if self.llm_call_times else 0
                progress_bar.set_postfix({'converged': self.converged, 'llm_calls': self.llm_call_count, 'avg_llm_time': f"{avg_llm_time:.2f}s"})
        
        progress_bar.close()
        
        # Clean up worker thread safely
        self._shutdown_requested = True
        try:
            # Send shutdown signal
            self.query_queue.put(None, timeout=1.0)
            # Wait for thread to finish, but not forever
            self.llm_thread.join(timeout=5.0)
            if self.llm_thread.is_alive():
                print("[Warning] LLM worker thread did not shut down cleanly")
        except queue.Full:
            print("[Warning] Could not send shutdown signal to LLM worker")
        except Exception as e:
            print(f"[Warning] Error during LLM worker cleanup: {e}")
        
        return {
            'run_id': self.run_id,
            'scenario': self.scenario,
            'converged': self.converged,
            'convergence_step': self.convergence_step,
            'final_step': self.step,
            'metrics_history': self.metrics_history,
            'llm_call_count': self.llm_call_count,
            'avg_llm_call_time': np.mean(self.llm_call_times) if self.llm_call_times else 0
           , 'states': self.states
        }

def run_llm_experiment(scenario='baseline', n_runs=10, max_steps=1000, use_llm_probability=1.0, 
                      llm_model=None, llm_url=None, llm_api_key=None):
    """Run LLM experiments with specified scenario"""
    
    # Check LLM connection first with potentially custom parameters
    if not check_llm_connection(llm_model, llm_url, llm_api_key):
        print("\n⚠️  Cannot proceed with LLM experiments - connection check failed!")
        print("Please ensure the LLM server is running and accessible.")
        print("\nTo start Ollama locally:")
        print("  ollama serve")
        print(f"  ollama pull {llm_model or cfg.OLLAMA_MODEL}")
        return None, []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"llm_{scenario}_{timestamp}"
    output_dir = f"experiments/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save experiment configuration
    config_dict = {
        'experiment_type': 'llm',
        'scenario': scenario,
        'n_runs': n_runs,
        'max_steps': max_steps,
        'use_llm_probability': use_llm_probability,
        'grid_size': cfg.GRID_SIZE,
        'num_type_a': cfg.NUM_TYPE_A,
        'num_type_b': cfg.NUM_TYPE_B,
        'llm_model': llm_model or cfg.OLLAMA_MODEL,
        'llm_url': llm_url or cfg.OLLAMA_URL,
        'llm_api_key_last4': (llm_api_key or cfg.OLLAMA_API_KEY)[-4:] if (llm_api_key or cfg.OLLAMA_API_KEY) else None,
        'no_move_threshold': cfg.NO_MOVE_THRESHOLD,
        'timestamp': timestamp,
        'context_info': CONTEXT_SCENARIOS[scenario]
    }
    
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Create progress file for real-time monitoring
    progress_file = f"{output_dir}/progress_realtime.json"
    
    # Run simulations (sequential due to LLM rate limits)
    results = []
    for i in range(n_runs):
        print(f"\nRunning simulation {i+1}/{n_runs} for scenario: {scenario}")
        
        # Update progress file before starting run
        progress_data = {
            "scenario": scenario,
            "current_run": i + 1,
            "total_runs": n_runs,
            "current_step": 0,
            "max_steps": max_steps,
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "run_progress_percent": (i / n_runs) * 100,
            "step_progress_percent": 0
        }
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        sim = LLMSimulation(i, scenario, use_llm_probability, llm_model, llm_url, llm_api_key, progress_file)
        result = sim.run(max_steps)
        results.append(result)
        
        # Update progress file after completing run
        progress_data["run_progress_percent"] = ((i + 1) / n_runs) * 100
        progress_data["step_progress_percent"] = 100 if result['converged'] else (result['final_step'] / max_steps) * 100
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        # Small delay between runs to avoid overwhelming the LLM
        time.sleep(2)
    
    # Process and save results
    all_metrics = []
    convergence_data = []
    
    for result in results:
        convergence_data.append({
            'run_id': result['run_id'],
            'scenario': result['scenario'],
            'converged': result['converged'],
            'convergence_step': result['convergence_step'],
            'final_step': result['final_step'],
            'llm_call_count': result['llm_call_count'],
            'avg_llm_call_time': result['avg_llm_call_time']
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
    print(f"Scenario: {scenario}")
    print(f"Total runs: {n_runs}")
    print(f"Converged runs: {sum(1 for r in convergence_data if r['converged'])}")
    if any(r['convergence_step'] for r in convergence_data):
        print(f"Average convergence step: {np.mean([r['convergence_step'] for r in convergence_data if r['convergence_step'] is not None]):.2f}")
    print(f"Total LLM calls: {sum(r['llm_call_count'] for r in convergence_data)}")
    print(f"Average LLM call time: {np.mean([r['avg_llm_call_time'] for r in convergence_data]):.2f}s")
    
    return output_dir, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-based Schelling segregation simulations")
    parser.add_argument('--scenario', type=str, default='baseline', 
                        choices=list(CONTEXT_SCENARIOS.keys()),
                        help='Scenario to run')
    parser.add_argument('--runs', type=int, default=10, help='Number of simulation runs')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps per simulation')
    parser.add_argument('--llm-probability', type=float, default=1.0, 
                        help='Probability of using LLM for each agent (vs mechanical)')
    parser.add_argument('--llm-model', type=str, help='LLM model to use (overrides config.py)')
    parser.add_argument('--llm-url', type=str, help='LLM API URL (overrides config.py)')
    parser.add_argument('--llm-api-key', type=str, help='LLM API key (overrides config.py)')
    
    args = parser.parse_args()
    
    run_llm_experiment(
        scenario=args.scenario,
        n_runs=args.runs,
        max_steps=args.max_steps,
        use_llm_probability=args.llm_probability,
        llm_model=args.llm_model,
        llm_url=args.llm_url,
        llm_api_key=args.llm_api_key
    )