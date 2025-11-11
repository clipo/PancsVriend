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
import pandas as pd
import ast
import glob
import re
import gzip

def check_llm_connection(llm_model=None, llm_url=None, llm_api_key=None, timeout=10, max_retries=5):
    """
    Check if LLM connection is active and working
    
    Parameters:
    - llm_model: Model to use (overrides config.py)
    - llm_url: API URL (overrides config.py)
    - llm_api_key: API key (overrides config.py)
    - timeout: Connection timeout in seconds
    - max_retries: Max retry attempts on transient failures (timeouts/connection errors)
    
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

    for attempt in range(1, max_retries + 1):
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=test_payload, timeout=timeout)
            elapsed = time.time() - start_time

            if response.status_code != 200:
                print(f"❌ LLM connection failed - HTTP {response.status_code}")
                print(f"Response: {response.text}")
                # Retry only for server errors (5xx) if attempts remain
                if attempt < max_retries and 500 <= response.status_code < 600:
                    print(f"↻ Retrying... ({attempt}/{max_retries})")
                    time.sleep(1)
                    continue
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
            print(f"❌ Timeout after {timeout}s (attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(5)
                continue
            print("The LLM server is not responding. Please check:")
            print("1. Is the Ollama server running?")
            print("2. Is the URL correct?")
            print("3. Is the model loaded?")
            return False

        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection error (attempt {attempt}/{max_retries})")
            print(f"Error: {e}")
            if attempt < max_retries:
                time.sleep(1)
                continue
            print("\nPlease check:")
            print("1. Is the Ollama server running?")
            print("2. Is the URL correct?")
            print("3. Is your network connection working?")
            return False

        except Exception as e:
            print("❌ LLM connection failed - Unexpected error")
            print(f"Error: {type(e).__name__}: {e}")
            return False

    return False

class LLMAgent(Agent):
    def __init__(self, type_id, scenario='baseline', llm_model=None, llm_url=None, llm_api_key=None, run_id=None, step=None):
        super().__init__(type_id)
        self.scenario = scenario
        self.context_info = CONTEXT_SCENARIOS[scenario]
        self.agent_type = self.context_info['type_a'] if type_id == 0 else self.context_info['type_b']
        self.opposite_type = self.context_info['type_b'] if type_id == 0 else self.context_info['type_a']
        self.llm_model = llm_model or cfg.OLLAMA_MODEL
        self.llm_url = llm_url or cfg.OLLAMA_URL
        self.llm_api_key = llm_api_key or cfg.OLLAMA_API_KEY
        self.run_id = run_id
        # Initialize LLM tracking metrics
        self.llm_call_count = 0
        self.llm_call_time = 0.0
        self.step = step
    
    def get_context_grid(self, r, c, grid):
        """
        Create a 3x3 neighborhood context string for the LLM prompt.
        
        Parameters:
        -----------
        r : int
            Row position of the agent
        c : int  
            Column position of the agent
        grid : list
            2D grid representing the simulation state
            
        Returns:
        --------
        str
            Formatted context string showing the 3x3 neighborhood with:
            - X: Current agent position (center)
            - S: Same type agent
            - O: Opposite type agent  
            - E: Empty space
            - #: Out of bounds
        """
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
        
        return "\n".join([" ".join(row) for row in context_with_position])
    
    def get_llm_decision(self, r, c, grid, max_retries=300):
        """Get movement decision from LLM with retry logic (max_retries attempts)"""
        # Debug flag - set via environment variable
        debug = os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes')
        
        if debug:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] [DEBUG] LLM Decision Request for agent at ({r},{c})")
            print(f"[DEBUG] Agent type: {self.agent_type} | Scenario: {self.scenario}")
        
        # Get context string using the new method
        context_str = self.get_context_grid(r, c, grid)
        
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
                    "timeout": 10000     # 10 second timeout in milliseconds
                }
                
                headers = {
                    "Authorization": f"Bearer {self.llm_api_key}",
                    "Content-Type": "application/json"
                }
                
                if debug:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] [DEBUG] Sending LLM request to: {self.llm_url}")
                    print(f"[DEBUG] Model: {self.llm_model}")
                    print(f"[DEBUG] Context grid:\n{context_str}")
                
                # Track timing and calls
                start_time = time.time()
                response = requests.post(self.llm_url, headers=headers, json=payload, timeout=20)
                response_time = time.time() - start_time
                
                # Update agent's LLM metrics
                self.llm_call_count += 1
                self.llm_call_time += response_time
                
                if debug:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] [DEBUG] LLM Response received in {response_time:.2f}s")
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
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if attempt < max_retries:
                    print(f"[{timestamp}] [LLM Timeout] Retry {attempt + 1}/{max_retries} for agent at ({r},{c}) [run {self.run_id}, step {self.step}]")
                    time.sleep(5)  # Wait 5 seconds before retry
                    continue
                else:
                    print(f"[{timestamp}] [LLM Error] Max retries exceeded ({max_retries}) for agent at ({r},{c}) [run {self.run_id}, step {self.step}]")
                    raise Exception(f"LLM timeout after {max_retries} retries")
            except Exception as e:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if attempt < max_retries:
                    print(f"[{timestamp}] [LLM Error] Exception - Retry {attempt + 1}/{max_retries}: {e} [run {self.run_id}, step {self.step}]")
                    time.sleep(5)  # Wait 5 seconds before retry
                    continue
                else:
                    print(f"[{timestamp}] [LLM Error] Exception: Unhandled error after {max_retries} retries: {e} [run {self.run_id}, step {self.step}]")
                    raise Exception(f"LLM error after {max_retries} retries: {e}")
                    raise Exception(f"LLM error after {max_retries} retries: {e}")

def llm_decision_function(agent, r, c, grid):
    """Decision function for LLM agents with retry logic (no fallback to mechanical decision)"""
    return agent.get_llm_decision(r, c, grid)

class LLMSimulation(Simulation):
    def __init__(self, run_id, scenario='baseline', llm_model=None, llm_url=None, llm_api_key=None, random_seed=None,
                 initial_int_grid=None, initial_step=None, initial_no_move_steps=None):
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
            random_seed=random_seed,
            initial_int_grid=initial_int_grid,
            initial_step=initial_step,
            initial_no_move_steps=initial_no_move_steps
        )
        
        # Track LLM metrics across all agents
        self.total_llm_calls = 0
        self.total_llm_time = 0.0
    
    def _create_llm_agent(self, type_id):
        """Create LLM agent with simulation parameters"""
        return LLMAgent(type_id, self.scenario, self.llm_model, self.llm_url, self.llm_api_key, self.run_id, self.step)

    def run_step(self, verbose_move_log: bool = False):
        """Override run_step to track LLM metrics and add timestamps"""
        step_start_time = datetime.now()
        
        # Sync step/run_id to agents (for accurate logging) and update total LLM metrics
        for r in range(cfg.GRID_SIZE):
            for c in range(cfg.GRID_SIZE):
                agent = self.grid[r][c]
                if agent is not None:
                    # Ensure agents know the current simulation step and run id
                    try:
                        agent.step = self.step
                        agent.run_id = self.run_id
                    except Exception:
                        pass

                    # Aggregate and reset LLM metrics if present
                    if hasattr(agent, 'llm_call_count'):
                        self.total_llm_calls += agent.llm_call_count
                        self.total_llm_time += agent.llm_call_time
                        # Reset agent counters to avoid double counting
                        agent.llm_call_count = 0
                        agent.llm_call_time = 0.0

        # Call parent run_step
        result = super().run_step(verbose_move_log=verbose_move_log)

        # Add timestamp for step completion
        step_end_time = datetime.now()
        step_duration = (step_end_time - step_start_time).total_seconds()

        # Only print timestamp for longer steps or periodically
        if step_duration > 10: # or (hasattr(self, 'step') and self.step % 10 == 0)
            print(f"[{step_end_time.strftime('%Y-%m-%d %H:%M:%S')}] Step {getattr(self, 'step', '?')} completed in {step_duration:.1f}s [run {self.run_id}]")

        return result

    def run_single_simulation(self, output_dir=None, max_steps=1000, show_progress=False):
        """Override to show progress bar for LLM simulations and add timestamps"""
        start_time = datetime.now()
        print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting LLM simulation run {self.run_id}")
        
        result = super().run_single_simulation(output_dir=output_dir, max_steps=max_steps, show_progress=show_progress)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] Completed LLM simulation run {self.run_id} in {duration:.1f}s")
        
        return result

def run_single_simulation(args):
    """Run a single LLM simulation - compatible with baseline_runner structure.
    Supports optional resume seeds and max_steps.
    Args tuples supported:
      (run_id, scenario, llm_model, llm_url, llm_api_key, output_dir)
      (run_id, scenario, llm_model, llm_url, llm_api_key, output_dir, max_steps)
      (run_id, scenario, llm_model, llm_url, llm_api_key, output_dir, max_steps, initial_int_grid, initial_step)
    """
    run_id, scenario, llm_model, llm_url, llm_api_key, output_dir = args[:6]
    max_steps = args[6] if len(args) >= 7 and args[6] is not None else 1000
    initial_int_grid = args[7] if len(args) >= 8 else None
    initial_step = args[8] if len(args) >= 9 else None
    initial_no_move_steps = args[9] if len(args) >= 10 else None

    sim = LLMSimulation(run_id, scenario, llm_model, llm_url, llm_api_key,
                        initial_int_grid=initial_int_grid, initial_step=initial_step,
                        initial_no_move_steps=initial_no_move_steps)
    result = sim.run_single_simulation(output_dir=output_dir, max_steps=max_steps)
    
    # Add LLM-specific metrics to the result
    result.update({
        'scenario': scenario,
        'llm_call_count': sim.total_llm_calls,
        'avg_llm_call_time': sim.total_llm_time / max(sim.total_llm_calls, 1),
    })
    return result

def _analyze_run_status(output_dir, run_id, max_steps):
    """Classify a run by inspecting logs and extract resume seeds if needed.
    Returns dict with keys: status in {'converged','reached_max','aborted','missing'},
    last_step (int or None), next_step (int or None), seed_grid (2D list or None).
    """
    move_csv = os.path.join(output_dir, "move_logs", f"agent_moves_run_{run_id}.csv")
    states_npz = os.path.join(output_dir, "states", f"states_run_{run_id}.npz")

    if os.path.exists(move_csv):
        try:
            df = pd.read_csv(move_csv)
        except Exception:
            df = pd.DataFrame()
        last_step = int(df['step'].max()) if (not df.empty and 'step' in df.columns) else -1
        step_moves = df.groupby('step')['moved'].sum() if (not df.empty and 'moved' in df.columns) else pd.Series(dtype=int)
        threshold = getattr(cfg, 'NO_MOVE_THRESHOLD', 5)
        converged = False
        convergence_step = None
        no_move_streak = 0
        if not step_moves.empty:
            window = step_moves.tail(threshold)
            if len(window) == threshold and (window == 0).all():
                converged = True
                convergence_step = int(window.index[0])
            # Count trailing zero-move steps for resuming aborted runs
            for moves in reversed(step_moves.tolist()):
                if pd.isna(moves):
                    break
                if int(moves) == 0:
                    no_move_streak += 1
                else:
                    break
        reached_max = (last_step + 1) >= max_steps if last_step >= 0 else False

        status = 'converged' if converged else ('reached_max' if reached_max else 'aborted')

        # Extract last grid snapshot for resuming aborted runs
        seed_grid = None
        next_step = None
        if status == 'aborted':
            if 'grid' in df.columns and not df.empty:
                try:
                    raw = df.iloc[-1]['grid']
                    if isinstance(raw, str):
                        seed_grid = ast.literal_eval(raw)
                    else:
                        seed_grid = raw
                except Exception:
                    seed_grid = None
            if seed_grid is None and os.path.exists(states_npz):
                try:
                    data = np.load(states_npz)
                    arr = data['states']
                    if len(arr) > 0:
                        seed_grid = arr[-1].tolist()
                except Exception:
                    seed_grid = None
            next_step = (last_step + 1) if last_step is not None and last_step >= 0 else None

        return {
            'status': status,
            'last_step': None if last_step < 0 else last_step,
            'next_step': next_step,
            'seed_grid': seed_grid,
            'convergence_step': convergence_step if converged else None,
            'no_move_streak': int(no_move_streak)
        }

    # No move log; try states as existence indicator
    if os.path.exists(states_npz):
        # Without move log, we can't infer steps reliably; treat as aborted with last grid
        seed_grid = None
        try:
            data = np.load(states_npz)
            arr = data['states']
            if len(arr) > 0:
                seed_grid = arr[-1].tolist()
        except Exception:
            seed_grid = None
        return {'status': 'aborted', 'last_step': None, 'next_step': None, 'seed_grid': seed_grid, 'convergence_step': None, 'no_move_streak': 0}
    return {'status': 'missing', 'last_step': None, 'next_step': None, 'seed_grid': None, 'convergence_step': None, 'no_move_streak': 0}
    
def list_available_experiments():
    """List all available experiments that can be resumed"""
    exp_dir = "experiments"
    if not os.path.exists(exp_dir):
        print("No experiments directory found.")
        return []
    
    experiments = []
    for exp_name in os.listdir(exp_dir):
        exp_path = os.path.join(exp_dir, exp_name)
        if os.path.isdir(exp_path):
            # Use the updated check_existing_experiment function
            exists, completed_runs, _, existing_run_ids = check_existing_experiment(exp_name)
            
            # Load config to get total runs
            config_file = os.path.join(exp_path, "config.json")
            total_runs = "unknown"
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        total_runs = config.get('n_runs', 'unknown')
                except Exception:
                    pass
            
            experiments.append({
                'name': exp_name,
                'completed': completed_runs,
                'total': total_runs,
                'path': exp_path
            })
    
    return experiments

def check_existing_experiment(experiment_name):
    """
    Check if an experiment already exists and find completed run IDs
    
    Parameters:
    -----------
    experiment_name : str
        Name of the experiment directory to check
        
    Returns:
    --------
    tuple
        (exists, completed_runs, output_dir, existing_run_ids) where:
        - exists: bool indicating if experiment directory exists
        - completed_runs: int number of completed simulation runs found
        - output_dir: str path to the experiment directory
        - existing_run_ids: set of run IDs that already exist
    """
    output_dir = f"experiments/{experiment_name}"
    
    if not os.path.exists(output_dir):
        return False, 0, output_dir, set()
    
    # Track existing run IDs
    existing_run_ids = set()
    
    # Look for different possible result file patterns
    import glob
    import re
    patterns_to_check = [
        os.path.join(output_dir, "run_*.json.gz"),  # Original pattern
        os.path.join(output_dir, "states", "states_run_*.npz"),  # Actual pattern used
        os.path.join(output_dir, "states_run_*.npz"),  # Alternative pattern
    ]
    
    for pattern in patterns_to_check:
        existing_files = glob.glob(pattern)
        if existing_files:
            # Extract run IDs from filenames
            for file_path in existing_files:
                filename = os.path.basename(file_path)
                match = re.search(r'run_(\d+)', filename)
                if match:
                    run_id = int(match.group(1))
                    existing_run_ids.add(run_id)
            break  # Use the first pattern that finds files
    
    completed_runs = len(existing_run_ids)
    return True, completed_runs, output_dir, existing_run_ids

def run_llm_experiment(scenario=None, n_runs=None, max_steps=None, llm_model=None, llm_url=None, llm_api_key=None, parallel=True, n_processes=None, resume_experiment=None):
    """
    Run LLM experiments with specified scenario - compatible with baseline_runner structure
    
    Parameters:
    -----------
    scenario : str or None
        Scenario context to use. When resuming and omitted, falls back to the stored experiment config.
    n_runs : int or None
        Number of simulation runs to perform. When resuming and omitted, uses the stored experiment config.
    max_steps : int or None
        Maximum steps per simulation. When resuming and omitted, uses the stored experiment config.
    llm_model : str or None
        LLM model to use (overrides config.py if provided; otherwise uses config/defaults).
    llm_url : str or None
        LLM API URL (overrides config.py if provided; otherwise uses config/defaults).
    llm_api_key : str or None
        LLM API key to use (overrides config.py if provided; otherwise uses config/defaults).
    parallel : bool
        Whether to use parallel processing
    n_processes : int, optional
        Number of CPU processes to use for parallel execution.
        If None, uses min(cpu_count(), n_runs). If 1, forces sequential execution.
    resume_experiment : str, optional
        Name of an existing experiment to resume (skip runs that are already completed)
        
    Returns:
    --------
    tuple
        (output_dir, results) where results contains simulation outcomes
    """
    
    # Handle experiment resumption
    if resume_experiment:
        print(f"Checking for existing experiment: {resume_experiment}")
        exists, completed_runs, output_dir, existing_run_ids = check_existing_experiment(resume_experiment)
        
        if not exists:
            print(f"❌ Experiment '{resume_experiment}' not found in experiments/ directory")
            print("Available experiments:")
            exp_dir = "experiments"
            if os.path.exists(exp_dir):
                for exp in os.listdir(exp_dir):
                    if os.path.isdir(os.path.join(exp_dir, exp)):
                        print(f"  - {exp}")
            return None, []
        
        print(f"✅ Found existing experiment with {completed_runs} completed runs")
        if existing_run_ids:
            print(f"   Existing run IDs: {sorted(existing_run_ids)}")
        
        # Load existing config to match original parameters when CLI does not override them
        config_file = os.path.join(output_dir, "config.json")
        config_defaults_used = []
        config_values_from_file = {}
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                existing_config = json.load(f)
                if scenario is None:
                    scenario_from_config = existing_config.get('scenario')
                    if scenario_from_config:
                        scenario = scenario_from_config
                        config_defaults_used.append('scenario')
                        config_values_from_file['scenario'] = scenario_from_config
                if max_steps is None:
                    max_steps_from_config = existing_config.get('max_steps')
                    if max_steps_from_config is not None:
                        max_steps = max_steps_from_config
                        config_defaults_used.append('max_steps')
                        config_values_from_file['max_steps'] = max_steps_from_config
                if llm_model is None:
                    llm_model_from_config = existing_config.get('llm_model')
                    if llm_model_from_config:
                        llm_model = llm_model_from_config
                        config_defaults_used.append('llm_model')
                        config_values_from_file['llm_model'] = llm_model_from_config
                if llm_url is None:
                    llm_url_from_config = existing_config.get('llm_url')
                    if llm_url_from_config:
                        llm_url = llm_url_from_config
                        config_defaults_used.append('llm_url')
                        config_values_from_file['llm_url'] = llm_url_from_config
                if llm_api_key is None and existing_config.get('llm_api_key_last4'):
                    # API key cannot be reconstructed from last4; fall back to cfg if not provided
                    pass
                if n_runs is None:
                    n_runs_from_config = existing_config.get('n_runs')
                    if n_runs_from_config is not None:
                        n_runs = n_runs_from_config
                        config_defaults_used.append('n_runs')
                        config_values_from_file['n_runs'] = n_runs_from_config
                if n_processes is None:
                    n_proc_from_config = existing_config.get('n_processes')
                    if isinstance(n_proc_from_config, int) and n_proc_from_config > 0:
                        n_processes = n_proc_from_config
                        config_defaults_used.append('n_processes')
                        config_values_from_file['n_processes'] = n_proc_from_config

                # Persist new n_runs value if CLI provided different target count
                orig_runs = existing_config.get('n_runs')
                if orig_runs is not None and n_runs is not None and orig_runs != n_runs:
                    print(f"📝 Updating config n_runs from {orig_runs} to {n_runs} to match CLI")
                    existing_config['n_runs'] = n_runs
                    try:
                        with open(config_file, 'w') as fw:
                            json.dump(existing_config, fw, indent=2)
                    except Exception as e:
                        print(f"Warning: failed to update config.json: {e}")
        else:
            existing_config = {}
            print("⚠️ config.json not found for this experiment; falling back to default settings where necessary.")

        # Apply fallback defaults if config did not supply missing values
        scenario = scenario or 'baseline'
        max_steps = max_steps if (isinstance(max_steps, int) and max_steps > 0) else 1000
        n_runs = n_runs if (isinstance(n_runs, int) and n_runs > 0) else (completed_runs if completed_runs > 0 else 10)
        llm_model = llm_model or cfg.OLLAMA_MODEL
        llm_url = llm_url or cfg.OLLAMA_URL
        llm_api_key = llm_api_key or cfg.OLLAMA_API_KEY
        if not isinstance(n_processes, int) or n_processes < 1:
            n_processes = None

        value_map = {
            'scenario': scenario,
            'max_steps': max_steps,
            'llm_model': llm_model,
            'llm_url': llm_url,
            'n_runs': n_runs,
            'n_processes': n_processes,
        }
        config_defaults_used = [key for key in config_defaults_used if value_map.get(key) == config_values_from_file.get(key)]

        print("📋 Resuming with experiment parameters:")
        print(f"   Scenario: {scenario}{' (from config)' if 'scenario' in config_defaults_used else ''}")
        print(f"   Max steps: {max_steps}{' (from config)' if 'max_steps' in config_defaults_used else ''}")
        if 'n_runs' in config_defaults_used:
            print(f"   Total planned runs: {n_runs} (from config)")
        else:
            print(f"   Total planned runs: {n_runs}")
        if 'llm_model' in config_defaults_used:
            print(f"   LLM model: {llm_model} (from config)")
        if 'llm_url' in config_defaults_used:
            print(f"   LLM URL: {llm_url} (from config)")
        if n_processes is not None:
            suffix = " (from config)" if 'n_processes' in config_defaults_used else ""
            print(f"   Parallel processes: {n_processes}{suffix}")
        
        # Plan and classify all target run IDs
        planned_run_ids = list(range(n_runs))
        statuses = {}
        for rid in planned_run_ids:
            statuses[rid] = _analyze_run_status(output_dir, rid, max_steps)

        completed_ids = [rid for rid, s in statuses.items() if s['status'] in ('converged', 'reached_max')]
        aborted_items = [(rid, s) for rid, s in statuses.items() if s['status'] == 'aborted']
        missing_run_ids = [rid for rid, s in statuses.items() if s['status'] == 'missing']

        print(f"   Classified runs → completed: {len(completed_ids)}, aborted: {len(aborted_items)}, missing: {len(missing_run_ids)}")
        print(f"     Aborted IDs: {[rid for rid,_ in aborted_items]}, Missing IDs: {missing_run_ids}")

        if len(completed_ids) >= n_runs and not aborted_items and not missing_run_ids:
            print(f"⚠️  Experiment already complete! ({len(completed_ids)}/{n_runs} runs)")
            print("Loading existing results...")
            # Load analysis from stored outputs for accurate results
            out_dir, results, convergence_data = Simulation.load_and_analyze_results(output_dir, force_recompute=False)
            return out_dir, results

        # Prepare args: aborted runs (resume) first, then missing runs (new)
        args_list = []
        for rid, s in aborted_items:
            seed_grid = s.get('seed_grid')
            next_step = s.get('next_step')
            no_move_streak = s.get('no_move_streak')
            args_list.append((rid, scenario, llm_model, llm_url, llm_api_key, output_dir, max_steps, seed_grid, next_step, no_move_streak))
        if aborted_items:
            print(f"   Will resume aborted run IDs first: {[rid for rid,_ in aborted_items]}")
            for rid, s in aborted_items:
                if s.get('no_move_streak'):
                    print(f"     Run {rid}: trailing no-move steps = {s['no_move_streak']}")
        if missing_run_ids:
            print(f"   Will execute missing run IDs: {missing_run_ids}")
        for rid in missing_run_ids:
            args_list.append((rid, scenario, llm_model, llm_url, llm_api_key, output_dir, max_steps))

        runs_to_execute = len(args_list)
        remaining_runs = runs_to_execute
    else:
        completed_runs = 0
        existing_run_ids = set()
        scenario = scenario or 'baseline'
        max_steps = max_steps or 1000
        n_runs = n_runs or 10
        llm_model = llm_model or cfg.OLLAMA_MODEL
        llm_url = llm_url or cfg.OLLAMA_URL
        llm_api_key = llm_api_key or cfg.OLLAMA_API_KEY
        # Create output directory for new experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"llm_{scenario}_{timestamp}"
        output_dir = f"experiments/{experiment_name}"
        os.makedirs(output_dir, exist_ok=True)
        remaining_runs = n_runs
        # Generate args list for new experiment
        args_list = [(i, scenario, llm_model, llm_url, llm_api_key, output_dir, max_steps) for i in range(n_runs)]
    
    # Check LLM connection first with potentially custom parameters
    if not check_llm_connection(llm_model, llm_url, llm_api_key):
        print("\n⚠️  Cannot proceed with LLM experiments - connection check failed!")
        print("Please ensure the LLM server is running and accessible.")
        return None, []
    
    # Create or update config (for new experiments only)
    if not resume_experiment:
        config_dict = {
            'n_runs': n_runs,
            'max_steps': max_steps,
            'grid_size': cfg.GRID_SIZE,
            'num_type_a': cfg.NUM_TYPE_A,
            'num_type_b': cfg.NUM_TYPE_B,
            'scenario': scenario,
            'llm_model': llm_model,
            'llm_url': llm_url,
            'llm_api_key_last4': (llm_api_key)[-4:] if llm_api_key else None,
            'no_move_threshold': cfg.NO_MOVE_THRESHOLD,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'context_info': CONTEXT_SCENARIOS[scenario],
            'parallel_execution': parallel,
            'n_processes': n_processes if parallel else 1,
            'cpu_count': cpu_count()
        }
        
        with open(f"{output_dir}/config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    # Determine number of processes to use
    runs_to_execute = remaining_runs if resume_experiment else n_runs
    if n_processes is None:
        n_processes = min(cpu_count(), runs_to_execute)
    elif n_processes == 1:
        parallel = False  # Force sequential execution if only 1 process requested
    elif n_processes > cpu_count():
        print(f"⚠️  Warning: Requested {n_processes} processes but only {cpu_count()} CPU cores available.")
        print(f"   Using {cpu_count()} processes instead.")
        n_processes = cpu_count()
    elif n_processes > runs_to_execute:
        print(f"⚠️  Warning: Requested {n_processes} processes but only {runs_to_execute} runs to execute.")
        print(f"   Using {runs_to_execute} processes instead.")
        n_processes = runs_to_execute
    elif n_processes < 1:
        print(f"⚠️  Warning: Invalid number of processes ({n_processes}). Using 1 process (sequential).")
        n_processes = 1
        parallel = False
    
    if parallel and n_processes > 1:
        print(f"Running {runs_to_execute} simulations using {n_processes} parallel processes...")
        with Pool(n_processes) as pool:
            results = list(tqdm(
                pool.imap(run_single_simulation, args_list),
                total=runs_to_execute,
                desc="Running LLM simulations",
                ncols=80,
            ))
    else:
        print(f"Running {runs_to_execute} simulations sequentially...")
        results = []
        for args in tqdm(args_list, desc="Running LLM simulations", ncols=80):
            results.append(run_single_simulation(args))

    # Load existing results if resuming
    if resume_experiment and completed_runs > 0:
        print(f"Loading {completed_runs} existing results...")
        # Try different result file patterns
        patterns_to_check = [
            os.path.join(output_dir, "run_*.json.gz"),
            os.path.join(output_dir, "states", "states_run_*.npz"),
            os.path.join(output_dir, "states_run_*.npz"),
        ]
        
        existing_result_files = []
        for pattern in patterns_to_check:
            files = glob.glob(pattern)
            if files:
                existing_result_files = files
                break
        
        # Load existing results and filter out run_ids that were just re-executed
        newly_executed_run_ids = {r['run_id'] for r in results}
        
        existing_results = []
        for result_file in sorted(existing_result_files):
            if result_file.endswith('.json.gz'):
                with gzip.open(result_file, 'rt') as f:
                    result = json.load(f)
                    # Skip if this run was just re-executed (avoids duplicates)
                    if result.get('run_id') not in newly_executed_run_ids:
                        existing_results.append(result)
            elif result_file.endswith('.npz'):
                # For .npz files, create a minimal result structure
                # Extract run number from filename
                match = re.search(r'run_(\d+)', result_file)
                run_id = int(match.group(1)) if match else 0
                
                # Skip if this run was just re-executed (avoids duplicates)
                if run_id not in newly_executed_run_ids:
                    existing_results.append({
                        'run_id': run_id,
                        'converged': True,  # Assume converged if file exists
                        'convergence_step': None,  # Unknown from .npz file alone
                        'final_step': 'unknown',
                        'metrics_history': [],  # Empty metrics for .npz files
                        'file_path': result_file
                    })
        
        # Combine existing and new results (no duplicates now)
        all_results = existing_results + results
        total_runs = len(all_results)
        print(f"Combined {len(existing_results)} existing + {len(results)} new = {total_runs} total results")
    else:
        all_results = results
        total_runs = n_runs

    # Analyze results using Simulation's analyze_results method
    output_dir, final_results, convergence_data = Simulation.analyze_results(all_results, output_dir, total_runs)
    
    print(f"\nExperiment completed. Results saved to: {output_dir}")
    if resume_experiment:
        print(f"Resumed experiment: {completed_runs} existing + {len(results)} new = {total_runs} total runs")
    else:
        print(f"Total runs: {total_runs}")
    print(f"Converged runs: {sum(1 for r in convergence_data if r['converged'])}")
    converged_steps = [r['convergence_step'] for r in convergence_data if r['convergence_step'] is not None]
    if converged_steps:
        print(f"Average convergence step: {np.mean(converged_steps):.2f}")
    
    # Calculate LLM-specific statistics
    llm_calls = [r.get('llm_call_count', 0) for r in final_results if 'llm_call_count' in r]
    llm_times = [r.get('avg_llm_call_time', 0) for r in final_results if 'avg_llm_call_time' in r]
    if llm_calls:
        print(f"Average LLM calls per run: {np.mean(llm_calls):.1f}")
        print(f"Average LLM response time: {np.mean(llm_times):.3f}s")
    
    return output_dir, final_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-based Schelling segregation simulations")
    parser.add_argument('--runs', type=int, default=None, help='Number of simulation runs')
    parser.add_argument('--max-steps', type=int, default=None, help='Maximum steps per simulation')
    parser.add_argument('--scenario', type=str, default=None, choices=list(CONTEXT_SCENARIOS.keys()), help='Scenario to simulate')
    parser.add_argument('--llm-model', type=str, help='LLM model to use (overrides config.py)')
    parser.add_argument('--llm-url', type=str, help='LLM API URL (overrides config.py)')
    parser.add_argument('--llm-api-key', type=str, help='LLM API key (overrides config.py)')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--processes', type=int, default=None, 
                       help=f'Number of CPU processes to use (default: min(cpu_count={cpu_count()}, n_runs)). Use 1 for sequential execution.')
    parser.add_argument('--resume', type=str, help='Resume existing experiment by name (e.g., "llm_baseline_20250706_143022")')
    parser.add_argument('--list-experiments', action='store_true', help='List all available experiments that can be resumed')
    args = parser.parse_args()

    # Handle listing experiments
    if args.list_experiments:
        experiments = list_available_experiments()
        if not experiments:
            print("No experiments found.")
        else:
            print("\nAvailable experiments:")
            print("-" * 80)
            for exp in experiments:
                status = f"{exp['completed']}/{exp['total']}"
                if exp['completed'] == exp['total'] and exp['total'] != 'unknown':
                    status += " (complete)"
                elif exp['total'] != 'unknown' and exp['completed'] < exp['total']:
                    status += " (incomplete - can resume)"
                print(f"{exp['name']:<50} {status}")
            print("-" * 80)
            print("\nTo resume an experiment, use: --resume <experiment_name>")
        exit(0)

    run_llm_experiment(
        scenario=args.scenario,
        n_runs=args.runs,
        max_steps=args.max_steps,
        llm_model=args.llm_model,
        llm_url=args.llm_url,
        llm_api_key=args.llm_api_key,
        parallel=not args.no_parallel,
        n_processes=args.processes,
        resume_experiment=args.resume
    )