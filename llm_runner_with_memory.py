"""
Enhanced LLM Runner with Agent Memory Support
This version maintains agent histories and relationships
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import config as cfg
from llm_agent_with_memory import LLMAgentWithMemory
from Metrics import calculate_all_metrics
import requests
import re
from tqdm import tqdm
import time
import threading
import queue
from context_scenarios import CONTEXT_SCENARIOS

class LLMSimulationWithMemory:
    """Enhanced simulation where agents remember their experiences"""
    
    def __init__(self, run_id, scenario='baseline', use_llm_probability=1.0, 
                 llm_model=None, llm_url=None, llm_api_key=None, enable_memory=True):
        self.run_id = run_id
        self.scenario = scenario
        self.use_llm_probability = use_llm_probability
        self.llm_model = llm_model
        self.llm_url = llm_url
        self.llm_api_key = llm_api_key
        self.enable_memory = enable_memory
        
        # Grid and simulation state
        self.grid = np.full((cfg.GRID_SIZE, cfg.GRID_SIZE), None)
        self.step = 0
        self.converged = False
        self.convergence_step = None
        self.no_move_steps = 0
        self.no_move_threshold = 20
        self.metrics_history = []
        
        # Agent tracking for memory
        self.agent_positions = {}  # agent_id -> (r, c)
        self.position_history = []  # Track all positions over time
        
        # LLM tracking
        self.llm_call_count = 0
        self.llm_call_times = []
        self.llm_failure_count = 0
        self.max_llm_failures = 20
        self.llm_circuit_open = False
        
        # Initialize simulation
        self.populate_grid()
        self.states = [self._grid_to_int()]
        
        # Setup LLM worker thread
        self.setup_llm_worker()
        
    def populate_grid(self):
        """Create agents with memory and place them on grid"""
        if self.enable_memory:
            agents = [LLMAgentWithMemory(type_id, self.scenario, self.llm_model, self.llm_url, self.llm_api_key) 
                     for type_id in ([0] * cfg.NUM_TYPE_A + [1] * cfg.NUM_TYPE_B)]
        else:
            # Fallback to regular agents
            from llm_runner import LLMAgent
            agents = [LLMAgent(type_id, self.scenario, self.llm_model, self.llm_url, self.llm_api_key) 
                     for type_id in ([0] * cfg.NUM_TYPE_A + [1] * cfg.NUM_TYPE_B)]
            
        np.random.shuffle(agents)
        flat_positions = [(r, c) for r in range(cfg.GRID_SIZE) for c in range(cfg.GRID_SIZE)]
        np.random.shuffle(flat_positions)
        
        for agent, pos in zip(agents, flat_positions[:len(agents)]):
            r, c = pos
            self.grid[r][c] = agent
            if hasattr(agent, 'personal_id'):
                self.agent_positions[agent.personal_id] = (r, c)
                
    def get_neighborhood_context(self, r, c):
        """Get 3x3 neighborhood as both grid and interpreted context"""
        context = []
        for dr in [-1, 0, 1]:
            row = []
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < cfg.GRID_SIZE and 0 <= nc < cfg.GRID_SIZE:
                    neighbor = self.grid[nr][nc]
                    if neighbor is None:
                        row.append("E")  # Empty
                    elif neighbor.type_id == self.grid[r][c].type_id:
                        row.append("S")  # Same
                    else:
                        row.append("O")  # Opposite
                else:
                    row.append("X")  # Out of bounds
            context.append(row)
        return context
    
    def update_agent_memories(self, agent, old_pos, new_pos, context_before, context_after):
        """Update agent's memory based on move decision"""
        if not hasattr(agent, 'remember_move'):
            return
            
        r_old, c_old = old_pos
        
        if new_pos and new_pos != old_pos:
            # Agent moved
            reason = self.generate_move_reason(agent, context_before, context_after)
            agent.remember_move(old_pos, new_pos, reason, context_before)
            
            # Check for significant neighborhood changes
            changes = agent.analyze_neighborhood_change(context_before, context_after)
            for change in changes:
                agent.add_neighborhood_experience(change)
                
        else:
            # Agent stayed
            satisfaction = self.calculate_satisfaction(agent, context_before)
            reason = self.generate_stay_reason(agent, context_before, satisfaction)
            agent.remember_staying(old_pos, reason, satisfaction)
            
        # Update neighbor relationships
        self.update_neighbor_relationships(agent, old_pos, context_before)
        
    def generate_move_reason(self, agent, context_before, context_after):
        """Generate a human-like reason for moving"""
        same_before = sum(1 for row in context_before for cell in row if cell == 'S')
        same_after = sum(1 for row in context_after for cell in row if cell == 'S')
        
        if same_after > same_before:
            return "Found a neighborhood with more people like me"
        elif hasattr(agent.identity, 'children') and agent.identity.get('children', 0) > 0:
            return "Better environment for my children"
        elif hasattr(agent.identity, 'primary_concerns') and 'affordability' in agent.identity.get('primary_concerns', []):
            return "Found more affordable housing"
        else:
            return "Seeking a better community fit"
            
    def generate_stay_reason(self, agent, context, satisfaction):
        """Generate a human-like reason for staying"""
        if satisfaction >= 8:
            return "Very happy with current neighbors and location"
        elif agent.time_in_current_location > 5:
            return "Established roots here, don't want to disrupt"
        elif hasattr(agent.identity, 'children') and agent.identity.get('children', 0) > 0:
            return "Children are settled in schools and have friends"
        else:
            return "Current situation is acceptable"
            
    def calculate_satisfaction(self, agent, context):
        """Calculate agent's satisfaction level (0-10)"""
        same_neighbors = sum(1 for row in context for cell in row if cell == 'S')
        total_neighbors = sum(1 for row in context for cell in row if cell in ['S', 'O'])
        
        if total_neighbors == 0:
            return 5  # Neutral if no neighbors
            
        ratio = same_neighbors / total_neighbors
        
        # Adjust based on agent personality
        if hasattr(agent, 'identity'):
            if agent.identity.get('tolerance_level') == 'high':
                # High tolerance agents are satisfied with fewer same-type neighbors
                satisfaction = 5 + (ratio * 5)
            elif agent.identity.get('tolerance_level') == 'low':
                # Low tolerance agents need more same-type neighbors
                satisfaction = ratio * 10
            else:
                # Moderate tolerance
                satisfaction = 2 + (ratio * 8)
        else:
            satisfaction = ratio * 10
            
        return min(10, max(0, satisfaction))
        
    def update_neighbor_relationships(self, agent, pos, context):
        """Track relationships with immediate neighbors"""
        if not hasattr(agent, 'remember_neighbor_interaction'):
            return
            
        r, c = pos
        dr_dc_pairs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Direct neighbors only
        
        for dr, dc in dr_dc_pairs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < cfg.GRID_SIZE and 0 <= nc < cfg.GRID_SIZE:
                neighbor = self.grid[nr][nc]
                if neighbor:
                    neighbor_type = "same_type" if neighbor.type_id == agent.type_id else "different_type"
                    
                    # Determine interaction quality based on context
                    if neighbor.type_id == agent.type_id:
                        quality = "positive"
                    elif hasattr(agent, 'identity') and agent.identity.get('tolerance_level') == 'high':
                        quality = "neutral"
                    else:
                        quality = "negative" if np.random.random() < 0.3 else "neutral"
                        
                    agent.remember_neighbor_interaction(neighbor_type, quality)
                    
    def get_llm_decision_with_memory(self, agent, r, c, max_retries=2):
        """Get LLM decision using agent's memory and identity"""
        # Debug flag - set via environment variable
        debug = os.environ.get('DEBUG_LLM', '').lower() in ('true', '1', 'yes')
        
        if debug:
            print(f"\n[DEBUG-MEMORY] LLM Decision Request for agent at ({r},{c})")
            agent_type = agent.identity.get('type', f'Type {agent.type_id}') if hasattr(agent, 'identity') else f'Type {agent.type_id}'
            print(f"[DEBUG-MEMORY] Agent type: {agent_type} | Scenario: {self.scenario}")
            decision_count = len(agent.move_history) if hasattr(agent, 'move_history') else 0
            print(f"[DEBUG-MEMORY] Agent has {decision_count} previous decisions")
        
        context = self.get_neighborhood_context(r, c)
        context_str = "\n".join([" ".join(row) for row in context])
        
        if hasattr(agent, 'get_enhanced_prompt'):
            # Use memory-enhanced prompt
            prompt = agent.get_enhanced_prompt(context_str, r, c, self.grid)
        else:
            # Fallback to standard prompt
            # For memory agents, get type from identity dict
            if hasattr(agent, 'identity'):
                agent_type = agent.identity.get('type', f'Type {agent.type_id}')
                # Determine opposite type based on scenario
                context_info = CONTEXT_SCENARIOS[self.scenario]
                if agent.type_id == 0:
                    opposite_type = context_info['type_b']
                else:
                    opposite_type = context_info['type_a']
            else:
                # Ultimate fallback
                agent_type = f'Type {agent.type_id}'
                opposite_type = f'Type {1 - agent.type_id}'
            
            prompt = CONTEXT_SCENARIOS[self.scenario]['prompt_template'].format(
                agent_type=agent_type,
                opposite_type=opposite_type,
                context=context_str
            )
            
        # Make LLM call (similar to original implementation)
        for attempt in range(max_retries + 1):
            try:
                payload = {
                    "model": agent.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "temperature": 0.3,
                    "max_tokens": 50,
                    "timeout": 10000
                }
                
                headers = {
                    "Authorization": f"Bearer {agent.llm_api_key}",
                    "Content-Type": "application/json"
                }
                
                if debug:
                    print(f"[DEBUG-MEMORY] Sending LLM request to: {agent.llm_url}")
                    print(f"[DEBUG-MEMORY] Model: {agent.llm_model}")
                    print(f"[DEBUG-MEMORY] Context grid:\n{context_str}")
                    if hasattr(agent, 'decision_history') and agent.decision_history:
                        print(f"[DEBUG-MEMORY] Recent history: {agent.decision_history[-3:]}")
                
                start_time = time.time()
                response = requests.post(agent.llm_url, headers=headers, json=payload, timeout=8)
                response_time = time.time() - start_time
                
                if debug:
                    print(f"[DEBUG-MEMORY] LLM Response received in {response_time:.2f}s")
                    print(f"[DEBUG-MEMORY] Status code: {response.status_code}")
                
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                
                if debug:
                    print(f"[DEBUG-MEMORY] LLM Response text: '{text}'")
                    print(f"[DEBUG-MEMORY] Attempting to parse decision...")
                
                # Parse response
                match = re.search(r"\((\d+),\s*(\d+)\)", text)
                if match:
                    move_to = (int(match.group(1)), int(match.group(2)))
                    # Convert from relative to absolute coordinates
                    r_new = r + (move_to[0] - 1)
                    c_new = c + (move_to[1] - 1)
                    
                    if debug:
                        print(f"[DEBUG-MEMORY] Parsed move: ({move_to[0]},{move_to[1]}) relative -> ({r_new},{c_new}) absolute")
                    
                    # Validate the move
                    if 0 <= r_new < cfg.GRID_SIZE and 0 <= c_new < cfg.GRID_SIZE and self.grid[r_new][c_new] is None:
                        if debug:
                            print(f"[DEBUG-MEMORY] Decision: MOVE to ({r_new},{c_new})")
                        return (r_new, c_new)
                    elif debug:
                        print(f"[DEBUG-MEMORY] Move invalid (out of bounds or occupied), will STAY")
                
                if "none" in text.strip().lower():
                    if debug:
                        print(f"[DEBUG-MEMORY] Decision: STAY (agent chose not to move)")
                    return None
                
                if debug:
                    print(f"[DEBUG-MEMORY] Could not parse decision from: '{text}'")
                    print(f"[DEBUG-MEMORY] Decision: STAY (parse failure)")
                return None
                
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                else:
                    # Fall back to mechanical decision
                    return agent.best_response(r, c, self.grid)
                    
    def setup_llm_worker(self):
        """Setup worker thread for LLM queries"""
        self.query_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=100)
        self._shutdown_requested = False
        
        def worker():
            while True:
                item = self.query_queue.get()
                if item is None or self._shutdown_requested:
                    break
                    
                agent, r, c, task_id = item
                
                if self.llm_circuit_open:
                    # Circuit is open, use mechanical agent
                    result = agent.best_response(r, c, self.grid)
                else:
                    # Try LLM decision
                    result = self.get_llm_decision_with_memory(agent, r, c)
                    self.llm_call_count += 1
                    
                self.result_queue.put((task_id, agent, r, c, result))
                
        self.llm_thread = threading.Thread(target=worker, daemon=True)
        self.llm_thread.start()
        
    def run_step(self):
        """Run one simulation step with memory updates"""
        # Get all agent positions
        all_positions = [(r, c) for r in range(cfg.GRID_SIZE) 
                        for c in range(cfg.GRID_SIZE) if self.grid[r][c]]
        np.random.shuffle(all_positions)
        
        moved = False
        batch_size = 5
        
        # Process agents in batches
        for i in range(0, len(all_positions), batch_size):
            batch = all_positions[i:i+batch_size]
            results = []
            
            # Queue batch
            task_ids = []
            for r, c in batch:
                agent = self.grid[r][c]
                if agent:
                    task_id = len(task_ids)
                    task_ids.append(task_id)
                    
                    # Save context before move
                    context_before = self.get_neighborhood_context(r, c)
                    agent._context_before = context_before
                    
                    self.query_queue.put((agent, r, c, task_id))
                    
            # Collect results
            for _ in range(len(task_ids)):
                try:
                    result = self.result_queue.get(timeout=30)
                    results.append(result)
                except queue.Empty:
                    print("Warning: LLM timeout")
                    
            # Process moves
            for task_id, agent, r, c, move_to in results:
                if move_to and move_to != (r, c):
                    r_new, c_new = move_to
                    if self.grid[r_new][c_new] is None:  # Double-check empty
                        # Execute move
                        self.grid[r_new][c_new] = agent
                        self.grid[r][c] = None
                        
                        # Update memory
                        context_after = self.get_neighborhood_context(r_new, c_new)
                        self.update_agent_memories(agent, (r, c), (r_new, c_new), 
                                                 agent._context_before, context_after)
                        
                        moved = True
                else:
                    # Agent stayed - update memory
                    self.update_agent_memories(agent, (r, c), None, 
                                             agent._context_before, agent._context_before)
                    
        # Update simulation state
        if not moved:
            self.no_move_steps += 1
            if self.no_move_steps >= self.no_move_threshold:
                self.converged = True
                self.convergence_step = self.step
        else:
            self.no_move_steps = 0
            
        # Calculate metrics
        metrics = calculate_all_metrics(self.grid)
        metrics['step'] = self.step
        metrics['llm_calls'] = self.llm_call_count
        self.metrics_history.append(metrics)
        
        # Save grid state
        self.states.append(self._grid_to_int())
        self.step += 1
        
        return moved
        
    def _grid_to_int(self):
        """Convert grid to integer representation"""
        int_grid = np.full((cfg.GRID_SIZE, cfg.GRID_SIZE), -1, dtype=int)
        for r in range(cfg.GRID_SIZE):
            for c in range(cfg.GRID_SIZE):
                if self.grid[r][c]:
                    int_grid[r, c] = self.grid[r][c].type_id
        return int_grid
        
    def run(self, max_steps=1000):
        """Run the simulation"""
        pbar = tqdm(range(max_steps), desc=f"Run {self.run_id} ({self.scenario})")
        
        for _ in pbar:
            if self.converged:
                pbar.set_postfix(converged=True, step=self.convergence_step)
                break
                
            moved = self.run_step()
            
            # Update progress bar
            pbar.set_postfix(
                converged=self.converged,
                llm_calls=self.llm_call_count,
                no_move_steps=self.no_move_steps
            )
            
        # Cleanup
        self._shutdown_requested = True
        self.query_queue.put(None)
        self.llm_thread.join(timeout=5)
        
        # Collect agent memories for analysis
        agent_memories = []
        for r in range(cfg.GRID_SIZE):
            for c in range(cfg.GRID_SIZE):
                agent = self.grid[r][c]
                if agent and hasattr(agent, 'move_history'):
                    agent_memories.append({
                        'agent_id': agent.personal_id,
                        'type': agent.type_id,
                        'identity': agent.identity,
                        'total_moves': agent.total_moves,
                        'final_satisfaction': agent.satisfaction_history[-1]['satisfaction'] if agent.satisfaction_history else None,
                        'move_history': agent.move_history,
                        'time_in_final_location': agent.time_in_current_location
                    })
                    
        return {
            'run_id': self.run_id,
            'scenario': self.scenario,
            'converged': self.converged,
            'convergence_step': self.convergence_step,
            'steps': self.step,
            'metrics_history': self.metrics_history,
            'llm_call_count': self.llm_call_count,
            'llm_failure_count': self.llm_failure_count,
            'agent_memories': agent_memories,
            'states': self.states
        }