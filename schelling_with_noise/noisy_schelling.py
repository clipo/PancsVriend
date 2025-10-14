"""
Noisy Schelling Segregation Model

This module implements a variant of the Schelling segregation model where agents
can misperceive the types of their neighbors with a given probability. This
introduces uncertainty and noise into the decision-making process.

Key Features:
- Agents may mistake neighbors' types with configurable probability
- Different noise levels can be tested (e.g., 0.1 = 10% chance of misperception)
- Compatible with both mechanical and LLM-based agent decision making
- Maintains all original simulation functionality with added perception noise
"""

import numpy as np
import random
import config as cfg
from base_simulation import Simulation
from Agent import Agent


class NoisyAgent(Agent):
    """
    Agent with noisy perception - may mistake neighbor types with given probability
    """
    
    def __init__(self, type_id, noise_probability=0.1):
        """
        Initialize noisy agent
        
        Args:
            type_id: Agent type (0 or 1)
            noise_probability: Probability of misperceiving each neighbor's type (0.0-1.0)
        """
        super().__init__(type_id)
        self.noise_probability = noise_probability
    
    def _unlike_ratio_with_noise(self, r, c, grid):
        """
        Calculate unlike ratio with perception noise
        
        For each neighbor, there's a noise_probability chance that the agent
        will perceive them as the opposite type from what they actually are.
        
        Args:
            r: Row position
            c: Column position
            grid: Simulation grid
            
        Returns:
            float: Ratio of perceived unlike neighbors (with noise)
        """
        neighbors = []
        perceived_unlike_count = 0
        
        # Get all actual neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r_n, c_n = r + dr, c + dc
                if 0 <= r_n < cfg.GRID_SIZE and 0 <= c_n < cfg.GRID_SIZE:
                    agent = grid[r_n][c_n]
                    if agent is not None:
                        neighbors.append(agent)
        
        if not neighbors:
            return 0  # No neighbors means no need to move
        
        # Apply noise to perception of each neighbor
        for neighbor in neighbors:
            actual_type = neighbor.type_id
            
            # With noise_probability, perceive as opposite type
            if random.random() < self.noise_probability:
                perceived_type = 1 - actual_type  # Flip the type
            else:
                perceived_type = actual_type  # Perceive correctly
            
            # Count as unlike if perceived type differs from self
            if perceived_type != self.type_id:
                perceived_unlike_count += 1
        
        return perceived_unlike_count / len(neighbors)
    
    def _unlike_ratio(self, r, c, grid):
        """Override parent method to use noisy perception"""
        return self._unlike_ratio_with_noise(r, c, grid)


class NoisySimulation(Simulation):
    """
    Schelling simulation with noisy agent perception
    """
    
    def __init__(self, run_id, decision_func, noise_probability=0.1, scenario='noisy_baseline', random_seed=None):
        """
        Initialize noisy simulation
        
        Args:
            run_id: Unique identifier for this simulation run
            decision_func: Function that determines agent movement decisions
            noise_probability: Probability of misperceiving neighbor types (0.0-1.0)
            scenario: Scenario name for tracking
            random_seed: Random seed for reproducibility
        """
        self.noise_probability = noise_probability
        
        # Create agent factory that produces noisy agents
        def noisy_agent_factory(type_id):
            return NoisyAgent(type_id, noise_probability)
        
        super().__init__(
            run_id=run_id,
            agent_factory=noisy_agent_factory,
            decision_func=decision_func,
            scenario=scenario,
            random_seed=random_seed
        )
    
    def run_single_simulation(self, output_dir=None, max_steps=1000, show_progress=False):
        """Override to include final metrics calculation"""
        # Call parent method to run the simulation
        result = super().run_single_simulation(output_dir=output_dir, max_steps=max_steps, show_progress=show_progress)
        
        # Calculate final metrics including segregation
        from Metrics import calculate_all_metrics
        final_metrics = calculate_all_metrics(self.grid)
        
        # Add final segregation measure (using mix_deviation as segregation metric)
        result['final_segregation'] = final_metrics.get('mix_deviation', 0.0)
        result['final_metrics'] = final_metrics
        result['noise_probability'] = self.noise_probability
        
        return result


def mechanical_decision_noisy(agent, r, c, grid):
    """
    Mechanical decision function for noisy agents
    Uses the same logic as original but with noisy perception
    """
    return agent.best_response(r, c, grid)


def random_decision_noisy(agent, r, c, grid):
    """
    Random decision function for noisy agents
    Uses the same logic as original but with noisy perception
    """
    return agent.random_response(r, c, grid)


# Example usage and testing functions
def run_noisy_experiment(n_runs=10, noise_levels=None, max_steps=1000, decision_func=mechanical_decision_noisy):
    """
    Run experiments with different noise levels
    
    Args:
        n_runs: Number of runs per noise level
        noise_levels: List of noise probabilities to test (default: [0.0, 0.1, 0.2, 0.3])
        max_steps: Maximum steps per simulation
        decision_func: Decision function to use
        
    Returns:
        dict: Results organized by noise level
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3]
    
    results = {}
    
    for noise_prob in noise_levels:
        print(f"\n🔊 Testing noise level: {noise_prob:.1%}")
        noise_results = []
        
        for run_id in range(n_runs):
            sim = NoisySimulation(
                run_id=run_id,
                decision_func=decision_func,
                noise_probability=noise_prob,
                scenario=f'noisy_{noise_prob:.1f}',
                random_seed=run_id  # For reproducibility
            )
            
            # Run simulation
            result = sim.run_single_simulation(max_steps=max_steps, show_progress=False)
            result['noise_probability'] = noise_prob
            noise_results.append(result)
            
            print(f"  Run {run_id:2d}: Converged={result['converged']:5} "
                  f"Steps={result['final_step']:3d} "
                  f"Final_segregation={result['final_segregation']:.3f}")
        
        results[noise_prob] = noise_results
    
    return results


def analyze_noise_effects(results):
    """
    Analyze the effects of different noise levels
    
    Args:
        results: Results dictionary from run_noisy_experiment
        
    Returns:
        dict: Summary statistics by noise level
    """
    summary = {}
    
    print("\n📊 NOISE EFFECTS ANALYSIS")
    print("=" * 60)
    print(f"{'Noise Level':<12} {'Convergence Rate':<16} {'Avg Steps':<12} {'Avg Segregation':<16}")
    print("-" * 60)
    
    for noise_prob in sorted(results.keys()):
        runs = results[noise_prob]
        
        # Calculate statistics
        convergence_rate = sum(1 for r in runs if r['converged']) / len(runs)
        converged_runs = [r for r in runs if r['converged']]
        avg_steps = np.mean([r['final_step'] for r in converged_runs]) if converged_runs else float('inf')
        avg_segregation = np.mean([r['final_segregation'] for r in runs])
        
        summary[noise_prob] = {
            'convergence_rate': convergence_rate,
            'avg_steps': avg_steps,
            'avg_segregation': avg_segregation,
            'n_runs': len(runs)
        }
        
        print(f"{noise_prob:<12.1%} {convergence_rate:<16.1%} {avg_steps:<12.1f} {avg_segregation:<16.3f}")
    
    return summary


class NoisyLLMAgent(NoisyAgent):
    """
    LLM-based agent with noisy perception
    Combines LLM decision making with perception noise
    """
    
    def __init__(self, type_id, scenario='noisy_baseline', noise_probability=0.1, 
                 llm_model=None, llm_url=None, llm_api_key=None):
        """
        Initialize noisy LLM agent
        
        Args:
            type_id: Agent type (0 or 1)
            scenario: Context scenario for LLM prompts
            noise_probability: Probability of misperceiving neighbor types
            llm_model: LLM model to use
            llm_url: LLM API URL
            llm_api_key: LLM API key
        """
        super().__init__(type_id, noise_probability)
        self.scenario = scenario
        self.llm_model = llm_model or cfg.OLLAMA_MODEL
        self.llm_url = llm_url or cfg.OLLAMA_URL
        self.llm_api_key = llm_api_key or cfg.OLLAMA_API_KEY
        
        # Import context scenarios
        from context_scenarios import CONTEXT_SCENARIOS
        self.context_info = CONTEXT_SCENARIOS.get(scenario, CONTEXT_SCENARIOS['baseline'])
        self.agent_type = self.context_info['type_a'] if type_id == 0 else self.context_info['type_b']
        self.opposite_type = self.context_info['type_b'] if type_id == 0 else self.context_info['type_a']
    
    def get_noisy_context_grid(self, r, c, grid):
        """
        Generate context grid with noisy perception for LLM
        
        Args:
            r: Current row position
            c: Current column position
            grid: Simulation grid
            
        Returns:
            str: Context grid string with noise applied to perception
        """
        context = []
        for dr in [-1, 0, 1]:
            row = []
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < cfg.GRID_SIZE and 0 <= nc < cfg.GRID_SIZE:
                    neighbor = grid[nr][nc]
                    if neighbor is None:
                        row.append("E")  # Empty
                    elif dr == 0 and dc == 0:
                        row.append("X")  # Self (center position)
                    else:
                        # Apply noise to neighbor perception
                        actual_type = neighbor.type_id
                        
                        # With noise_probability, perceive as opposite type
                        if random.random() < self.noise_probability:
                            perceived_type = 1 - actual_type  # Flip the type
                        else:
                            perceived_type = actual_type  # Perceive correctly
                        
                        if perceived_type == self.type_id:
                            row.append("S")  # Same (perceived)
                        else:
                            row.append("O")  # Opposite (perceived)
                else:
                    row.append("#")  # Out of bounds
            context.append(row)
        
        return "\n".join([" ".join(row) for row in context])
    
    def get_llm_decision_with_noise(self, r, c, grid, max_retries=10):
        """
        Get movement decision from LLM using noisy context perception
        
        Args:
            r: Current row position
            c: Current column position
            grid: Simulation grid
            max_retries: Maximum number of retry attempts
            
        Returns:
            tuple or None: New position (r, c) or None to stay
        """
        import requests
        import time
        import os
        
        # Debug flag
        debug = os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes')
        
        if debug:
            print(f"\n[DEBUG] Noisy LLM Decision Request for agent at ({r},{c})")
            print(f"[DEBUG] Agent type: {self.agent_type} | Noise: {self.noise_probability:.1%}")
        
        # Get noisy context grid
        context_str = self.get_noisy_context_grid(r, c, grid)
        
        # Create prompt using context scenario template
        prompt = self.context_info['prompt_template'].format(
            agent_type=self.agent_type,
            opposite_type=self.opposite_type,
            context=context_str
        )
        
        # Add noise information to prompt if significant
        if self.noise_probability > 0.05:  # Only mention if noise > 5%
            noise_note = f"\n\nNote: You may occasionally misperceive neighbors' types due to uncertainty (noise level: {self.noise_probability:.1%})."
            prompt += noise_note
        
        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": 0.3,
                    "max_tokens": 50,
                    "timeout": 20000
                }
                
                headers = {
                    "Authorization": f"Bearer {self.llm_api_key}",
                    "Content-Type": "application/json"
                }
                
                if debug:
                    print(f"[DEBUG] Noisy context grid:\n{context_str}")
                
                start_time = time.time()
                response = requests.post(self.llm_url, headers=headers, json=payload, timeout=8)
                response_time = time.time() - start_time
                
                # Update agent's LLM metrics
                self.llm_call_count += 1
                self.llm_call_time += response_time
                
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                
                if debug:
                    print(f"[DEBUG] LLM Response: '{text}'")
                
                # Parse MOVE/STAY response
                text_upper = text.strip().upper()
                
                if "MOVE" in text_upper:
                    if debug:
                        print("[DEBUG] Decision: MOVE")
                    
                    # Find random empty space
                    empty_spaces = []
                    for row in range(cfg.GRID_SIZE):
                        for col in range(cfg.GRID_SIZE):
                            if grid[row][col] is None:
                                empty_spaces.append((row, col))
                    
                    if empty_spaces:
                        chosen_pos = random.choice(empty_spaces)
                        if debug:
                            print(f"[DEBUG] Moving to: {chosen_pos}")
                        return chosen_pos
                    else:
                        if debug:
                            print("[DEBUG] No empty spaces, staying")
                        return None
                
                elif "STAY" in text_upper:
                    if debug:
                        print("[DEBUG] Decision: STAY")
                    return None
                
                else:
                    if debug:
                        print("[DEBUG] Could not parse, defaulting to STAY")
                    return None
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    print(f"[LLM Timeout] Retry {attempt + 1}/{max_retries} for noisy agent at ({r},{c})")
                    time.sleep(10)
                    continue
                else:
                    print(f"[LLM Error] Max retries exceeded for noisy agent at ({r},{c})")
                    raise Exception(f"LLM timeout after {max_retries} retries")
            except Exception as e:
                if attempt < max_retries:
                    print(f"[LLM Error] Exception - Retry {attempt + 1}/{max_retries}: {e}")
                    time.sleep(60)
                    continue
                else:
                    print(f"[LLM Error] Unhandled error after {max_retries} retries: {e}")
                    raise Exception(f"LLM error after {max_retries} retries: {e}")


def noisy_llm_decision_function(agent, r, c, grid):
    """Decision function for noisy LLM agents"""
    if hasattr(agent, 'get_llm_decision_with_noise'):
        return agent.get_llm_decision_with_noise(r, c, grid)


class NoisyLLMSimulation(NoisySimulation):
    """
    Noisy simulation with LLM agents
    """
    
    def __init__(self, run_id, scenario='noisy_baseline', noise_probability=0.1,
                 llm_model=None, llm_url=None, llm_api_key=None, random_seed=None):
        """
        Initialize noisy LLM simulation
        
        Args:
            run_id: Unique identifier for this simulation run
            scenario: Context scenario for LLM prompts
            noise_probability: Probability of misperceiving neighbor types
            llm_model: LLM model to use
            llm_url: LLM API URL
            llm_api_key: LLM API key
            random_seed: Random seed for reproducibility
        """
        self.scenario = scenario
        self.llm_model = llm_model or cfg.OLLAMA_MODEL
        self.llm_url = llm_url or cfg.OLLAMA_URL
        self.llm_api_key = llm_api_key or cfg.OLLAMA_API_KEY
        
        # Create noisy LLM agent factory
        def noisy_llm_agent_factory(type_id):
            return NoisyLLMAgent(
                type_id=type_id,
                scenario=scenario,
                noise_probability=noise_probability,
                llm_model=self.llm_model,
                llm_url=self.llm_url,
                llm_api_key=self.llm_api_key
            )
        
        # Initialize with noisy LLM agents and decision function
        Simulation.__init__(
            self,
            run_id=run_id,
            agent_factory=noisy_llm_agent_factory,
            decision_func=noisy_llm_decision_function,
            scenario=scenario,
            random_seed=random_seed
        )
        
        # Store noise probability
        self.noise_probability = noise_probability
        
        # Track LLM metrics
        self.total_llm_calls = 0
        self.total_llm_time = 0.0
        self.total_moves = 0
    
    def run_step(self):
        """Override to track LLM metrics with noise"""
        # Track moves and LLM calls like in original LLM runner
        moves_count = 0
        
        # Update total LLM metrics from all agents
        for r in range(cfg.GRID_SIZE):
            for c in range(cfg.GRID_SIZE):
                agent = self.grid[r][c]
                if agent is not None and hasattr(agent, 'llm_call_count'):
                    self.total_llm_calls += agent.llm_call_count
                    self.total_llm_time += agent.llm_call_time
                    # Reset agent counters
                    agent.llm_call_count = 0
                    agent.llm_call_time = 0.0
        
        # Track positions for move counting
        positions_before = {}
        for r in range(cfg.GRID_SIZE):
            for c in range(cfg.GRID_SIZE):
                if self.grid[r][c] is not None:
                    agent_id = id(self.grid[r][c])
                    positions_before[agent_id] = (r, c)
        
        # Call parent run_step
        result = super().run_step()
        
        # Count moves
        positions_after = {}
        for r in range(cfg.GRID_SIZE):
            for c in range(cfg.GRID_SIZE):
                if self.grid[r][c] is not None:
                    agent_id = id(self.grid[r][c])
                    positions_after[agent_id] = (r, c)
        
        for agent_id in positions_before:
            if agent_id in positions_after:
                if positions_before[agent_id] != positions_after[agent_id]:
                    moves_count += 1
        
        self.total_moves += moves_count
        
        return result
    
    def run_single_simulation(self, output_dir=None, max_steps=1000, show_progress=False):
        """Override to include LLM metrics and final segregation calculation"""
        # Call parent method (NoisySimulation) to run the simulation
        result = super().run_single_simulation(output_dir=output_dir, max_steps=max_steps, show_progress=show_progress)
        
        # Add LLM-specific metrics to the result
        result.update({
            'llm_call_count': self.total_llm_calls,
            'avg_llm_call_time': self.total_llm_time / max(self.total_llm_calls, 1),
            'total_moves': self.total_moves
        })
        
        return result


def run_noise_comparison_study(n_runs=10, noise_levels=None, max_steps=1000, 
                              use_llm=False, scenario='baseline'):
    """
    Run a comprehensive comparison study between regular and noisy simulations
    
    Args:
        n_runs: Number of runs per condition
        noise_levels: List of noise probabilities to test
        max_steps: Maximum steps per simulation
        use_llm: Whether to use LLM agents (requires LLM server)
        scenario: Scenario context for simulations
        
    Returns:
        dict: Comprehensive results including statistical comparisons
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    
    print("🧪 NOISE COMPARISON STUDY")
    print("=" * 50)
    print(f"Agent Type: {'LLM' if use_llm else 'Mechanical'}")
    print(f"Runs per condition: {n_runs}")
    print(f"Max steps: {max_steps}")
    print(f"Noise levels: {[f'{n:.1%}' for n in noise_levels]}")
    print("=" * 50)
    
    all_results = {}
    
    for noise_prob in noise_levels:
        print(f"\n🔊 Testing noise level: {noise_prob:.1%}")
        condition_results = []
        
        for run_id in range(n_runs):
            if use_llm:
                # Use noisy LLM simulation
                sim = NoisyLLMSimulation(
                    run_id=run_id,
                    scenario=scenario,
                    noise_probability=noise_prob,
                    random_seed=run_id + 1000  # Offset for reproducibility
                )
            else:
                # Use mechanical noisy simulation
                sim = NoisySimulation(
                    run_id=run_id,
                    decision_func=mechanical_decision_noisy,
                    noise_probability=noise_prob,
                    scenario=f'noisy_{scenario}',
                    random_seed=run_id + 1000
                )
            
            try:
                result = sim.run_single_simulation(max_steps=max_steps, show_progress=False)
                result['noise_probability'] = noise_prob
                result['agent_type'] = 'LLM' if use_llm else 'Mechanical'
                
                if use_llm and hasattr(sim, 'total_llm_calls'):
                    result['llm_call_count'] = sim.total_llm_calls
                    result['avg_llm_call_time'] = sim.total_llm_time / max(sim.total_llm_calls, 1)
                    result['total_moves'] = sim.total_moves
                
                condition_results.append(result)
                
                print(f"  Run {run_id:2d}: Converged={str(result['converged']):5} "
                      f"Steps={result['final_step']:3d} "
                      f"Segregation={result['final_segregation']:.3f}")
                      
            except Exception as e:
                print(f"  Run {run_id:2d}: ERROR - {e}")
                continue
        
        all_results[noise_prob] = condition_results
    
    # Generate analysis
    analysis = analyze_noise_comparison(all_results)
    
    return {
        'results': all_results,
        'analysis': analysis,
        'metadata': {
            'n_runs': n_runs,
            'max_steps': max_steps,
            'agent_type': 'LLM' if use_llm else 'Mechanical',
            'scenario': scenario,
            'noise_levels': noise_levels
        }
    }


def analyze_noise_comparison(results):
    """
    Analyze and compare results across different noise levels
    
    Args:
        results: Results dictionary from run_noise_comparison_study
        
    Returns:
        dict: Statistical analysis and comparisons
    """
    import scipy.stats as stats
    
    analysis = {}
    noise_levels = sorted(results.keys())
    
    print("\n📊 DETAILED ANALYSIS")
    print("=" * 80)
    print(f"{'Noise':<8} {'Conv%':<8} {'Avg Steps':<12} {'Segregation':<12} {'Std Dev':<10} {'Stat Sig':<10}")
    print("-" * 80)
    
    baseline_segregation = None
    
    for noise_prob in noise_levels:
        runs = results[noise_prob]
        if not runs:
            continue
            
        # Basic statistics
        convergence_rate = sum(1 for r in runs if r['converged']) / len(runs)
        converged_runs = [r for r in runs if r['converged']]
        avg_steps = np.mean([r['final_step'] for r in converged_runs]) if converged_runs else float('inf')
        
        segregation_values = [r['final_segregation'] for r in runs]
        avg_segregation = np.mean(segregation_values)
        std_segregation = np.std(segregation_values)
        
        # Statistical significance test against baseline (noise = 0.0)
        stat_sig = "N/A"
        if noise_prob == 0.0:
            baseline_segregation = segregation_values
            stat_sig = "Baseline"
        elif baseline_segregation is not None:
            try:
                t_stat, p_value = stats.ttest_ind(baseline_segregation, segregation_values)
                stat_sig = f"p={p_value:.3f}" if p_value < 0.05 else "n.s."
            except Exception:
                stat_sig = "Error"
        
        analysis[noise_prob] = {
            'convergence_rate': convergence_rate,
            'avg_steps': avg_steps,
            'avg_segregation': avg_segregation,
            'std_segregation': std_segregation,
            'n_runs': len(runs),
            'segregation_values': segregation_values
        }
        
        print(f"{noise_prob:<8.1%} {convergence_rate:<8.1%} {avg_steps:<12.1f} "
              f"{avg_segregation:<12.3f} {std_segregation:<10.3f} {stat_sig:<10}")
    
    # Effect size analysis
    print("\n📈 EFFECT SIZES (compared to no noise)")
    if 0.0 in analysis and baseline_segregation:
        baseline_mean = analysis[0.0]['avg_segregation']
        baseline_std = analysis[0.0]['std_segregation']
        
        for noise_prob in noise_levels:
            if noise_prob == 0.0:
                continue
            if noise_prob in analysis:
                current_mean = analysis[noise_prob]['avg_segregation']
                # Cohen's d effect size
                pooled_std = np.sqrt((baseline_std**2 + analysis[noise_prob]['std_segregation']**2) / 2)
                cohens_d = (baseline_mean - current_mean) / pooled_std if pooled_std > 0 else 0
                
                effect_size_desc = "Small" if abs(cohens_d) < 0.5 else "Medium" if abs(cohens_d) < 0.8 else "Large"
                print(f"  Noise {noise_prob:.1%}: Cohen's d = {cohens_d:.3f} ({effect_size_desc})")
    
    return analysis


def save_noise_study_results(study_results, filename=None):
    """
    Save noise study results to files
    
    Args:
        study_results: Results from run_noise_comparison_study
        filename: Base filename (will add extensions)
    """
    import pandas as pd
    import json
    from datetime import datetime
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_type = study_results['metadata']['agent_type'].lower()
        filename = f"noise_study_{agent_type}_{timestamp}"
    
    # Convert to DataFrame
    all_data = []
    for noise_level, runs in study_results['results'].items():
        for run in runs:
            run_data = run.copy()
            run_data['noise_level'] = noise_level
            all_data.append(run_data)
    
    df = pd.DataFrame(all_data)
    
    # Save CSV
    csv_file = f"{filename}.csv"
    df.to_csv(csv_file, index=False)
    print(f"💾 Detailed results saved to: {csv_file}")
    
    # Save summary
    summary_file = f"{filename}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(study_results['analysis'], f, indent=2, default=str)
    print(f"💾 Summary analysis saved to: {summary_file}")
    
    return csv_file, summary_file


# Example usage functions
def quick_noise_test():
    """Quick test of noisy vs regular simulation"""
    print("🚀 Quick Noise Test (Mechanical Agents)")
    
    results = run_noise_comparison_study(
        n_runs=100,
        noise_levels=[0.0, 0.05],
        max_steps=200,
        use_llm=False
    )
    
    return results


def comprehensive_noise_study():
    """Comprehensive noise study with multiple conditions"""
    print("🔬 Comprehensive Noise Study")
    
    results = run_noise_comparison_study(
        n_runs=100,
        noise_levels=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        max_steps=1000,
        use_llm=False
    )
    
    # Save results
    save_noise_study_results(results)
    
    return results

if __name__ == "__main__":
    # Run quick test
    quick_results = quick_noise_test()
    
    # Run comprehensive study
    comprehensive_results = comprehensive_noise_study()
    
    # Analyze effects
    analyze_noise_effects(comprehensive_results['results'])
    
    print("\n✅ Noise study completed successfully!")