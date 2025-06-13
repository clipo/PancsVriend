#!/usr/bin/env python3
"""
Experimental Design Space Explorer
Systematically explores different agent configurations and LLM combinations
"""

import os
import json
import itertools
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import argparse
import subprocess
import shutil

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    experiment_id: str
    name: str
    description: str
    
    # Agent configuration
    agent_type: str  # 'standard', 'memory'
    scenario: str
    grid_size: int
    num_type_a: int
    num_type_b: int
    
    # LLM configuration
    llm_model: str
    llm_url: str
    llm_api_key: str
    
    # Experiment parameters
    n_runs: int
    max_steps: int
    use_llm_probability: float
    
    # Output tracking
    timestamp: str = ""
    status: str = "pending"  # pending, running, completed, failed
    output_dir: str = ""
    runtime_seconds: float = 0.0
    error_message: str = ""

class ExperimentDesignExplorer:
    """Main class for exploring experimental design space"""
    
    def __init__(self, base_output_dir="design_space_exploration"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.base_output_dir / "experiments").mkdir(exist_ok=True)
        (self.base_output_dir / "analysis").mkdir(exist_ok=True)
        (self.base_output_dir / "comparisons").mkdir(exist_ok=True)
        (self.base_output_dir / "logs").mkdir(exist_ok=True)
        
        self.experiments: List[ExperimentConfig] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def define_llm_configurations(self) -> List[Dict[str, str]]:
        """Define different LLM configurations to test"""
        return [
            {
                "name": "qwen_local",
                "model": "qwen2.5-coder:32B",
                "url": "https://chat.binghamton.edu/api/chat/completions",
                "api_key": "sk-571df6eec7f5495faef553ab5cb2c67a"
            },
            # Add more LLM configurations here
            # {
            #     "name": "gpt4",
            #     "model": "gpt-4",
            #     "url": "https://api.openai.com/v1/chat/completions",
            #     "api_key": "your-openai-key"
            # },
            # {
            #     "name": "claude_sonnet",
            #     "model": "claude-3-sonnet-20240229",
            #     "url": "https://api.anthropic.com/v1/messages",
            #     "api_key": "your-anthropic-key"
            # }
        ]
    
    def define_agent_configurations(self) -> List[Dict[str, Any]]:
        """Define different agent configurations to test"""
        return [
            {
                "type": "standard",
                "description": "Standard LLM agents (current snapshot only)"
            },
            {
                "type": "memory",
                "description": "Memory-enhanced LLM agents (with personal history)"
            }
        ]
    
    def define_scenario_configurations(self) -> List[str]:
        """Define social scenarios to test"""
        return [
            "baseline",
            "race_white_black",
            "ethnic_asian_hispanic", 
            "economic_high_working",
            "political_liberal_conservative"
        ]
    
    def define_grid_configurations(self) -> List[Dict[str, int]]:
        """Define different grid sizes and populations"""
        return [
            {"name": "small", "grid_size": 10, "type_a": 25, "type_b": 25},
            {"name": "medium", "grid_size": 15, "type_a": 75, "type_b": 75},
            {"name": "large", "grid_size": 20, "type_a": 150, "type_b": 150},
            {"name": "xlarge", "grid_size": 25, "type_a": 200, "type_b": 200}
        ]
    
    def generate_experiment_matrix(self, 
                                 llms: Optional[List[str]] = None,
                                 agents: Optional[List[str]] = None,
                                 scenarios: Optional[List[str]] = None,
                                 grids: Optional[List[str]] = None,
                                 runs_per_config: int = 20) -> List[ExperimentConfig]:
        """Generate full experimental design matrix"""
        
        llm_configs = self.define_llm_configurations()
        agent_configs = self.define_agent_configurations()
        scenario_configs = self.define_scenario_configurations()
        grid_configs = self.define_grid_configurations()
        
        # Filter configurations if specified
        if llms:
            llm_configs = [c for c in llm_configs if c["name"] in llms]
        if agents:
            agent_configs = [c for c in agent_configs if c["type"] in agents]
        if scenarios:
            scenario_configs = [s for s in scenario_configs if s in scenarios]
        if grids:
            grid_configs = [g for g in grid_configs if g["name"] in grids]
            
        experiments = []
        exp_id = 1
        
        # Generate all combinations
        for llm_config in llm_configs:
            for agent_config in agent_configs:
                for scenario in scenario_configs:
                    for grid_config in grid_configs:
                        
                        experiment_id = f"exp_{exp_id:04d}"
                        exp_name = f"{llm_config['name']}_{agent_config['type']}_{scenario}_{grid_config['name']}"
                        
                        exp = ExperimentConfig(
                            experiment_id=experiment_id,
                            name=exp_name,
                            description=f"LLM: {llm_config['name']}, Agents: {agent_config['type']}, Scenario: {scenario}, Grid: {grid_config['name']}",
                            
                            # Agent configuration
                            agent_type=agent_config['type'],
                            scenario=scenario,
                            grid_size=grid_config['grid_size'],
                            num_type_a=grid_config['type_a'],
                            num_type_b=grid_config['type_b'],
                            
                            # LLM configuration
                            llm_model=llm_config['model'],
                            llm_url=llm_config['url'],
                            llm_api_key=llm_config['api_key'],
                            
                            # Experiment parameters
                            n_runs=runs_per_config,
                            max_steps=1000,
                            use_llm_probability=1.0
                        )
                        
                        experiments.append(exp)
                        exp_id += 1
                        
        return experiments
    
    def save_experiment_plan(self, experiments: List[ExperimentConfig]):
        """Save the experimental design plan"""
        plan_file = self.base_output_dir / f"experiment_plan_{self.session_id}.json"
        
        plan_data = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "total_experiments": len(experiments),
            "experiments": [asdict(exp) for exp in experiments]
        }
        
        with open(plan_file, 'w') as f:
            json.dump(plan_data, f, indent=2)
            
        print(f"📋 Experiment plan saved: {plan_file}")
        
        # Also save a human-readable summary
        summary_file = self.base_output_dir / f"experiment_summary_{self.session_id}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"EXPERIMENTAL DESIGN SPACE EXPLORATION\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Created: {datetime.now().isoformat()}\n")
            f.write(f"Total Experiments: {len(experiments)}\n\n")
            
            # Summary by category
            llm_counts = {}
            agent_counts = {}
            scenario_counts = {}
            grid_counts = {}
            
            for exp in experiments:
                llm_counts[exp.llm_model] = llm_counts.get(exp.llm_model, 0) + 1
                agent_counts[exp.agent_type] = agent_counts.get(exp.agent_type, 0) + 1
                scenario_counts[exp.scenario] = scenario_counts.get(exp.scenario, 0) + 1
                grid_name = f"{exp.grid_size}x{exp.grid_size}"
                grid_counts[grid_name] = grid_counts.get(grid_name, 0) + 1
                
            f.write("CONFIGURATION BREAKDOWN:\n")
            f.write(f"LLM Models: {dict(llm_counts)}\n")
            f.write(f"Agent Types: {dict(agent_counts)}\n")
            f.write(f"Scenarios: {dict(scenario_counts)}\n")
            f.write(f"Grid Sizes: {dict(grid_counts)}\n\n")
            
            # Detailed experiment list
            f.write("DETAILED EXPERIMENT LIST:\n")
            f.write("-" * 80 + "\n")
            for exp in experiments:
                f.write(f"{exp.experiment_id}: {exp.name}\n")
                f.write(f"  Description: {exp.description}\n")
                f.write(f"  Runs: {exp.n_runs}, Max Steps: {exp.max_steps}\n\n")
                
        print(f"📄 Experiment summary saved: {summary_file}")
        
    def estimate_runtime_and_cost(self, experiments: List[ExperimentConfig]):
        """Estimate total runtime and API costs"""
        
        # Base estimates (adjust based on your experience)
        time_estimates = {
            'small': {'standard': 45, 'memory': 60},      # seconds per run
            'medium': {'standard': 120, 'memory': 180},
            'large': {'standard': 300, 'memory': 450},
            'xlarge': {'standard': 600, 'memory': 900}
        }
        
        # Cost estimates (tokens per run)
        token_estimates = {
            'small': {'standard': 50000, 'memory': 65000},
            'medium': {'standard': 200000, 'memory': 260000},
            'large': {'standard': 500000, 'memory': 650000},
            'xlarge': {'standard': 800000, 'memory': 1040000}
        }
        
        # LLM costs per 1K tokens
        llm_costs = {
            "qwen2.5-coder:32B": 0.0,  # Local model
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.001,
            "claude-3-sonnet-20240229": 0.003,
            "claude-3-haiku-20240307": 0.00025
        }
        
        total_time = 0
        total_cost = 0
        
        for exp in experiments:
            grid_name = f"{exp.grid_size}x{exp.grid_size}"
            if exp.grid_size <= 10:
                size_key = 'small'
            elif exp.grid_size <= 15:
                size_key = 'medium'
            elif exp.grid_size <= 20:
                size_key = 'large'
            else:
                size_key = 'xlarge'
                
            # Time estimate
            time_per_run = time_estimates[size_key][exp.agent_type]
            exp_time = time_per_run * exp.n_runs
            total_time += exp_time
            
            # Cost estimate
            tokens_per_run = token_estimates[size_key][exp.agent_type]
            cost_per_1k = llm_costs.get(exp.llm_model, 0.002)  # Default estimate
            exp_cost = (tokens_per_run / 1000) * cost_per_1k * exp.n_runs
            total_cost += exp_cost
            
        # Save estimates
        estimates_file = self.base_output_dir / f"estimates_{self.session_id}.json"
        estimates = {
            "total_experiments": len(experiments),
            "estimated_runtime_hours": total_time / 3600,
            "estimated_cost_usd": total_cost,
            "breakdown_by_experiment": []
        }
        
        for exp in experiments:
            # Calculate individual estimates
            if exp.grid_size <= 10:
                size_key = 'small'
            elif exp.grid_size <= 15:
                size_key = 'medium' 
            elif exp.grid_size <= 20:
                size_key = 'large'
            else:
                size_key = 'xlarge'
                
            time_per_run = time_estimates[size_key][exp.agent_type]
            tokens_per_run = token_estimates[size_key][exp.agent_type]
            cost_per_1k = llm_costs.get(exp.llm_model, 0.002)
            
            estimates["breakdown_by_experiment"].append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "estimated_time_minutes": (time_per_run * exp.n_runs) / 60,
                "estimated_cost_usd": (tokens_per_run / 1000) * cost_per_1k * exp.n_runs
            })
            
        with open(estimates_file, 'w') as f:
            json.dump(estimates, f, indent=2)
            
        print(f"\n💰 COST & TIME ESTIMATES")
        print(f"========================")
        print(f"Total Experiments: {len(experiments)}")
        print(f"Estimated Runtime: {total_time/3600:.1f} hours")
        print(f"Estimated Cost: ${total_cost:.2f}")
        print(f"Details saved: {estimates_file}")
        
        return total_time, total_cost
    
    def run_single_experiment(self, exp: ExperimentConfig) -> bool:
        """Run a single experiment"""
        
        print(f"\n🚀 Running experiment: {exp.experiment_id}")
        print(f"   Description: {exp.description}")
        
        exp.status = "running"
        exp.timestamp = datetime.now().isoformat()
        start_time = time.time()
        
        try:
            # Prepare experiment directory
            exp_output_dir = self.base_output_dir / "experiments" / exp.experiment_id
            exp_output_dir.mkdir(exist_ok=True)
            exp.output_dir = str(exp_output_dir)
            
            # Save experiment config
            config_file = exp_output_dir / "experiment_config.json"
            with open(config_file, 'w') as f:
                json.dump(asdict(exp), f, indent=2)
                
            # Prepare command based on agent type
            if exp.agent_type == "memory":
                # Use memory-enhanced runner
                cmd = [
                    "python", "-c", f'''
import sys
sys.path.append(".")
import config as cfg
cfg.GRID_SIZE = {exp.grid_size}
cfg.NUM_TYPE_A = {exp.num_type_a}
cfg.NUM_TYPE_B = {exp.num_type_b}

from llm_runner_with_memory import LLMSimulationWithMemory
import json

results = []
for run_id in range({exp.n_runs}):
    print(f"Run {{run_id+1}}/{exp.n_runs}")
    sim = LLMSimulationWithMemory(
        run_id=run_id,
        scenario="{exp.scenario}",
        use_llm_probability={exp.use_llm_probability},
        llm_model="{exp.llm_model}",
        llm_url="{exp.llm_url}",
        llm_api_key="{exp.llm_api_key}",
        enable_memory=True
    )
    result = sim.run(max_steps={exp.max_steps})
    results.append(result)

# Save results
with open("{exp_output_dir}/results.json", "w") as f:
    json.dump(results, f, indent=2)
'''
                ]
            else:
                # Use standard runner
                cmd = [
                    "python", "llm_runner.py",
                    "--scenario", exp.scenario,
                    "--runs", str(exp.n_runs),
                    "--llm-model", exp.llm_model,
                    "--llm-url", exp.llm_url,
                    "--llm-api-key", exp.llm_api_key
                ]
                
            # Run experiment
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if result.returncode == 0:
                exp.status = "completed"
                print(f"   ✅ Completed successfully")
            else:
                exp.status = "failed"
                exp.error_message = result.stderr
                print(f"   ❌ Failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            exp.status = "failed"
            exp.error_message = "Timeout after 2 hours"
            print(f"   ⏰ Timeout after 2 hours")
            
        except Exception as e:
            exp.status = "failed"
            exp.error_message = str(e)
            print(f"   ❌ Error: {e}")
            
        exp.runtime_seconds = time.time() - start_time
        
        # Save updated experiment status
        status_file = self.base_output_dir / "logs" / f"{exp.experiment_id}_status.json"
        with open(status_file, 'w') as f:
            json.dump(asdict(exp), f, indent=2)
            
        return exp.status == "completed"
    
    def run_experiment_batch(self, experiments: List[ExperimentConfig], 
                           start_idx: int = 0, max_experiments: Optional[int] = None):
        """Run a batch of experiments"""
        
        if max_experiments:
            experiments = experiments[start_idx:start_idx + max_experiments]
        else:
            experiments = experiments[start_idx:]
            
        print(f"\n🔬 STARTING EXPERIMENT BATCH")
        print(f"=====================================")
        print(f"Running {len(experiments)} experiments")
        print(f"Session ID: {self.session_id}")
        
        successful = 0
        failed = 0
        
        for i, exp in enumerate(experiments):
            print(f"\n--- Experiment {i+1}/{len(experiments)} ---")
            
            success = self.run_single_experiment(exp)
            if success:
                successful += 1
            else:
                failed += 1
                
            # Save progress
            progress_file = self.base_output_dir / f"progress_{self.session_id}.json"
            progress = {
                "session_id": self.session_id,
                "total_planned": len(experiments),
                "completed": i + 1,
                "successful": successful,
                "failed": failed,
                "progress_percent": ((i + 1) / len(experiments)) * 100
            }
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
                
        print(f"\n🏁 BATCH COMPLETE")
        print(f"=================")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total: {len(experiments)}")
        
    def generate_comparative_analysis(self):
        """Generate comprehensive comparative analysis"""
        
        print(f"\n📊 GENERATING COMPARATIVE ANALYSIS")
        print(f"===================================")
        
        # Implementation for analysis would go here
        # This would include:
        # - Collecting all experiment results
        # - Statistical comparisons across configurations
        # - Visualization generation
        # - Report creation
        
        analysis_dir = self.base_output_dir / "analysis"
        print(f"Analysis will be saved to: {analysis_dir}")


def main():
    parser = argparse.ArgumentParser(description="Experimental Design Space Explorer")
    parser.add_argument("--mode", choices=["plan", "run", "analyze"], required=True,
                       help="Mode: plan experiments, run them, or analyze results")
    parser.add_argument("--llms", nargs="+", help="LLM configurations to include")
    parser.add_argument("--agents", nargs="+", choices=["standard", "memory"], 
                       help="Agent types to include")
    parser.add_argument("--scenarios", nargs="+", help="Scenarios to include")
    parser.add_argument("--grids", nargs="+", choices=["small", "medium", "large", "xlarge"],
                       help="Grid sizes to include")
    parser.add_argument("--runs", type=int, default=20, help="Runs per configuration")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index for batch runs")
    parser.add_argument("--max-experiments", type=int, help="Maximum experiments to run")
    parser.add_argument("--output-dir", default="design_space_exploration", 
                       help="Output directory")
    
    args = parser.parse_args()
    
    explorer = ExperimentDesignExplorer(args.output_dir)
    
    if args.mode == "plan":
        print("🔬 EXPERIMENTAL DESIGN SPACE PLANNER")
        print("====================================")
        
        experiments = explorer.generate_experiment_matrix(
            llms=args.llms,
            agents=args.agents,
            scenarios=args.scenarios,
            grids=args.grids,
            runs_per_config=args.runs
        )
        
        explorer.save_experiment_plan(experiments)
        explorer.estimate_runtime_and_cost(experiments)
        
        print(f"\n✅ Experiment plan generated!")
        print(f"Total experiments: {len(experiments)}")
        
    elif args.mode == "run":
        # Load experiment plan
        plan_files = list(Path(args.output_dir).glob("experiment_plan_*.json"))
        if not plan_files:
            print("❌ No experiment plan found. Run with --mode plan first.")
            return
            
        latest_plan = max(plan_files, key=lambda p: p.stat().st_mtime)
        print(f"📋 Loading experiment plan: {latest_plan}")
        
        with open(latest_plan) as f:
            plan_data = json.load(f)
            
        experiments = [ExperimentConfig(**exp) for exp in plan_data["experiments"]]
        explorer.session_id = plan_data["session_id"]
        
        explorer.run_experiment_batch(experiments, args.start_idx, args.max_experiments)
        
    elif args.mode == "analyze":
        explorer.generate_comparative_analysis()

if __name__ == "__main__":
    main()