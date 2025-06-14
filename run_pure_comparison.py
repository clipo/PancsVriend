#!/usr/bin/env python3
"""
Pure Comparison Study Runner
Executes clean comparison between Mechanical, Standard LLM, and Memory LLM agents
"""

import subprocess
import sys
from datetime import datetime
import os

def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n🚀 {description}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            print(f"Output directory: {result.stdout.strip().split()[-1] if result.stdout.strip() else 'N/A'}")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def main():
    """Run the pure comparison study"""
    print("🔬 PURE COMPARISON STUDY")
    print("=" * 60)
    print("Comparing three pure agent types:")
    print("1. 📐 Mechanical Baseline (traditional utility maximization)")
    print("2. 🤖 Standard LLM Agents (100% LLM, current context)")
    print("3. 🧠 Memory LLM Agents (100% LLM with persistent memory)")
    print("4. 🎯 Two Key Scenarios (baseline control + racial context)")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configuration - optimized for fast execution
    runs = 2
    max_steps = 25
    
    scenarios = ["baseline"]  # Start with just baseline for testing
    
    results = []
    
    # 1. Run Mechanical Baseline
    for scenario in scenarios:
        cmd = f"python baseline_runner.py --runs {runs} --max-steps {max_steps}"
        if run_command(cmd, f"Mechanical Baseline - {scenario}"):
            results.append(f"mechanical_{scenario}")
    
    # 2. Run Standard LLM Agents (100% LLM)
    for scenario in scenarios:
        cmd = f"python llm_runner.py --scenario {scenario} --runs {runs} --max-steps {max_steps} --llm-probability 1.0"
        if run_command(cmd, f"Standard LLM Agents - {scenario}"):
            results.append(f"standard_llm_{scenario}")
    
    # 3. Run Memory LLM Agents (with memory enabled)
    # Note: Memory is controlled by ENABLE_AGENT_MEMORY in config.py
    print(f"\n💡 For Memory LLM agents, ensure ENABLE_AGENT_MEMORY = True in config.py")
    
    for scenario in scenarios:
        cmd = f"python llm_runner.py --scenario {scenario} --runs {runs} --max-steps {max_steps} --llm-probability 1.0"
        if run_command(cmd, f"Memory LLM Agents - {scenario}"):
            results.append(f"memory_llm_{scenario}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 PURE COMPARISON STUDY COMPLETE")
    print("=" * 60)
    print(f"Completed experiments: {len(results)}")
    for result in results:
        print(f"  ✅ {result}")
    
    print(f"\n📊 Results saved in experiments/ directory with timestamp {timestamp}")
    print("📈 Use visualization.py or statistical_analysis.py to analyze results")
    print("\n💡 Next steps:")
    print("  1. Run: python statistical_analysis.py")
    print("  2. Run: python visualization.py --baseline-dir experiments/baseline_* --llm-dirs experiments/llm_*")

if __name__ == "__main__":
    main()