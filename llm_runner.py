import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
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
import sys

def check_llm_connection(timeout=10):
    """
    Check if LLM connection is active and working
    
    Returns:
    - True if connection successful
    - False if connection failed
    """
    print("\nChecking LLM connection...")
    print(f"URL: {cfg.OLLAMA_URL}")
    print(f"Model: {cfg.OLLAMA_MODEL}")
    
    try:
        test_payload = {
            "model": cfg.OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "Respond with only the word 'OK' and nothing else."}],
            "stream": False,
            "temperature": 0
        }
        
        headers = {
            "Authorization": f"Bearer {cfg.OLLAMA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        response = requests.post(cfg.OLLAMA_URL, headers=headers, json=test_payload, timeout=timeout)
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
        print(f"❌ LLM connection failed - Connection error")
        print(f"Error: {e}")
        print("\nPlease check:")
        print("1. Is the Ollama server running?")
        print("2. Is the URL correct?")
        print("3. Is your network connection working?")
        return False
        
    except Exception as e:
        print(f"❌ LLM connection failed - Unexpected error")
        print(f"Error: {type(e).__name__}: {e}")
        return False

# Define context scenarios
CONTEXT_SCENARIOS = {
    'baseline': {
        'type_a': 'red',
        'type_b': 'blue',
        'prompt_template': """You are a {agent_type} agent deciding where to move in a 3x3 grid.
Each cell contains:
- 'S' for same-type ({agent_type}) neighbor
- 'O' for opposite-type ({opposite_type}) neighbor
- 'E' for empty space
- 'X' for out-of-bounds

You prefer to live near others like you but not in complete isolation.

Only respond with the coordinates of the best empty cell to move to, formatted exactly as (row, col).
If you choose not to move, respond only with: None

DO NOT provide any explanation. DO NOT include anything else.

Here is your 3x3 grid (centered on you):
{context}"""
    },
    'race_white_black': {
        'type_a': 'white middle class family',
        'type_b': 'predominantly black neighborhood family',
        'prompt_template': """You are a {agent_type} deciding where to move in a neighborhood grid.
Each cell contains:
- 'S' for same demographic ({agent_type})
- 'O' for different demographic ({opposite_type})
- 'E' for empty house
- 'X' for out-of-bounds

Consider your housing preferences based on neighborhood composition.

Only respond with the coordinates of the best empty cell to move to, formatted exactly as (row, col).
If you choose not to move, respond only with: None

DO NOT provide any explanation. DO NOT include anything else.

Here is your 3x3 neighborhood grid (centered on you):
{context}"""
    },
    'race_asian_hispanic': {
        'type_a': 'Asian neighborhood family',
        'type_b': 'Hispanic neighborhood family',
        'prompt_template': """You are an {agent_type} deciding where to move in a neighborhood grid.
Each cell contains:
- 'S' for same cultural background ({agent_type})
- 'O' for different cultural background ({opposite_type})
- 'E' for empty house
- 'X' for out-of-bounds

Consider your housing preferences based on neighborhood composition.

Only respond with the coordinates of the best empty cell to move to, formatted exactly as (row, col).
If you choose not to move, respond only with: None

DO NOT provide any explanation. DO NOT include anything else.

Here is your 3x3 neighborhood grid (centered on you):
{context}"""
    },
    'income_high_low': {
        'type_a': 'high-income household',
        'type_b': 'low-income household',
        'prompt_template': """You are a {agent_type} deciding where to move in a neighborhood grid.
Each cell contains:
- 'S' for same income level ({agent_type})
- 'O' for different income level ({opposite_type})
- 'E' for empty house
- 'X' for out-of-bounds

Consider your housing preferences based on neighborhood economic composition.

Only respond with the coordinates of the best empty cell to move to, formatted exactly as (row, col).
If you choose not to move, respond only with: None

DO NOT provide any explanation. DO NOT include anything else.

Here is your 3x3 neighborhood grid (centered on you):
{context}"""
    },
    'political_liberal_conservative': {
        'type_a': 'liberal-leaning household',
        'type_b': 'conservative-leaning household',
        'prompt_template': """You are a {agent_type} deciding where to move in a neighborhood grid.
Each cell contains:
- 'S' for same political leaning ({agent_type})
- 'O' for different political leaning ({opposite_type})
- 'E' for empty house
- 'X' for out-of-bounds

Consider your housing preferences based on neighborhood political composition.

Only respond with the coordinates of the best empty cell to move to, formatted exactly as (row, col).
If you choose not to move, respond only with: None

DO NOT provide any explanation. DO NOT include anything else.

Here is your 3x3 neighborhood grid (centered on you):
{context}"""
    }
}

class LLMAgent(Agent):
    def __init__(self, type_id, scenario='baseline'):
        super().__init__(type_id)
        self.scenario = scenario
        self.context_info = CONTEXT_SCENARIOS[scenario]
        self.agent_type = self.context_info['type_a'] if type_id == 0 else self.context_info['type_b']
        self.opposite_type = self.context_info['type_b'] if type_id == 0 else self.context_info['type_a']
    
    def get_llm_decision(self, r, c, grid, max_retries=2):
        """Get movement decision from LLM with retry logic"""
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
                    "model": cfg.OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": 0.3,  # Lower temperature for more consistent responses
                    "timeout": 15000  # 15 second timeout in milliseconds
                }
                
                headers = {
                    "Authorization": f"Bearer {cfg.OLLAMA_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                # Shorter timeout for individual requests
                response = requests.post(cfg.OLLAMA_URL, headers=headers, json=payload, timeout=10)
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                
                # Parse response
                match = re.search(r"\((\d+),\s*(\d+)\)", text)
                if match:
                    move_to = (int(match.group(1)), int(match.group(2)))
                    # Convert from relative to absolute coordinates
                    r_new = r + (move_to[0] - 1)
                    c_new = c + (move_to[1] - 1)
                    return (r_new, c_new)
                
                if "none" in text.strip().lower():
                    return None
                
                return None
                
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    print(f"[LLM Timeout] Retry {attempt + 1}/{max_retries} for agent at ({r},{c})")
                    time.sleep(1)  # Brief pause before retry
                    continue
                else:
                    print(f"[LLM Error] Max retries exceeded for agent at ({r},{c})")
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
    def __init__(self, run_id, scenario='baseline', use_llm_probability=1.0):
        self.run_id = run_id
        self.scenario = scenario
        self.use_llm_probability = use_llm_probability
        self.grid = np.full((cfg.GRID_SIZE, cfg.GRID_SIZE), None)
        self.step = 0
        self.converged = False
        self.convergence_step = None
        self.no_move_steps = 0
        self.no_move_threshold = 20
        self.metrics_history = []
        self.llm_call_count = 0
        self.llm_call_times = []
        
        self.populate_grid()
        self.setup_llm_worker()
    
    def populate_grid(self):
        agents = [LLMAgent(type_id, self.scenario) for type_id in ([0] * cfg.NUM_TYPE_A + [1] * cfg.NUM_TYPE_B)]
        np.random.shuffle(agents)
        flat_positions = [(r, c) for r in range(cfg.GRID_SIZE) for c in range(cfg.GRID_SIZE)]
        np.random.shuffle(flat_positions)
        for agent, pos in zip(agents, flat_positions[:len(agents)]):
            r, c = pos
            self.grid[r][c] = agent
    
    def setup_llm_worker(self):
        def worker():
            while True:
                task = self.query_queue.get()
                if task is None:
                    break
                agent, r, c = task
                try:
                    start_time = time.time()
                    if np.random.random() < self.use_llm_probability:
                        move_to = agent.get_llm_decision(r, c, self.grid)
                        self.llm_call_count += 1
                    else:
                        move_to = agent.best_response(r, c, self.grid)
                    
                    elapsed = time.time() - start_time
                    self.llm_call_times.append(elapsed)
                    
                    self.result_queue.put((agent, r, c, move_to))
                except Exception as e:
                    print(f"[LLM Worker Error] {e}")
                    # Fallback to mechanical behavior
                    move_to = agent.best_response(r, c, self.grid)
                    self.result_queue.put((agent, r, c, move_to))
                finally:
                    self.query_queue.task_done()
        
        self.query_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.llm_thread = threading.Thread(target=worker, daemon=True)
        self.llm_thread.start()
    
    def update_agents(self):
        all_positions = [(r, c) for r in range(cfg.GRID_SIZE) for c in range(cfg.GRID_SIZE) if self.grid[r][c]]
        np.random.shuffle(all_positions)
        
        # Process agents in batches to limit concurrent LLM calls
        batch_size = min(10, len(all_positions))  # Limit batch size
        moved = False
        
        for i in range(0, len(all_positions), batch_size):
            batch = all_positions[i:i+batch_size]
            
            # Queue batch
            for r, c in batch:
                agent = self.grid[r][c]
                self.query_queue.put((agent, r, c))
            
            # Wait for results
            results = []
            for _ in range(len(batch)):
                try:
                    result = self.result_queue.get(timeout=30)
                    results.append(result)
                except queue.Empty:
                    print("[Warning] Timeout waiting for LLM response")
            
            # Process moves
            for agent, r, c, move_to in results:
                if move_to and move_to != (r, c):
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
        metrics['scenario'] = self.scenario
        metrics['llm_calls'] = self.llm_call_count
        self.metrics_history.append(metrics)
        
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
        progress_bar = tqdm(total=max_steps, desc=f"Run {self.run_id} ({self.scenario})")
        
        while self.step < max_steps and not self.converged:
            self.run_step()
            progress_bar.update(1)
            
            if self.step % 10 == 0:
                avg_llm_time = np.mean(self.llm_call_times[-100:]) if self.llm_call_times else 0
                progress_bar.set_postfix({'converged': self.converged, 'llm_calls': self.llm_call_count, 'avg_llm_time': f"{avg_llm_time:.2f}s"})
        
        progress_bar.close()
        
        # Clean up worker thread
        self.query_queue.put(None)
        self.llm_thread.join()
        
        return {
            'run_id': self.run_id,
            'scenario': self.scenario,
            'converged': self.converged,
            'convergence_step': self.convergence_step,
            'final_step': self.step,
            'metrics_history': self.metrics_history,
            'llm_call_count': self.llm_call_count,
            'avg_llm_call_time': np.mean(self.llm_call_times) if self.llm_call_times else 0
        }

def run_llm_experiment(scenario='baseline', n_runs=10, max_steps=1000, use_llm_probability=1.0):
    """Run LLM experiments with specified scenario"""
    
    # Check LLM connection first
    if not check_llm_connection():
        print("\n⚠️  Cannot proceed with LLM experiments - connection check failed!")
        print("Please ensure the LLM server is running and accessible.")
        print("\nTo start Ollama locally:")
        print("  ollama serve")
        print(f"  ollama pull {cfg.OLLAMA_MODEL}")
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
        'llm_model': cfg.OLLAMA_MODEL,
        'timestamp': timestamp,
        'context_info': CONTEXT_SCENARIOS[scenario]
    }
    
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Run simulations (sequential due to LLM rate limits)
    results = []
    for i in range(n_runs):
        print(f"\nRunning simulation {i+1}/{n_runs} for scenario: {scenario}")
        sim = LLMSimulation(i, scenario, use_llm_probability)
        result = sim.run(max_steps)
        results.append(result)
        
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
    
    args = parser.parse_args()
    
    run_llm_experiment(
        scenario=args.scenario,
        n_runs=args.runs,
        max_steps=args.max_steps,
        use_llm_probability=args.llm_probability
    )