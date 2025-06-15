#!/usr/bin/env python3
"""
Restart with Memory LLM Experiments Only
Quick script to run just memory experiments to complement existing work
"""

import subprocess
import sys
from pathlib import Path

def run_memory_only_experiments():
    """Run memory-only LLM experiments"""
    
    print("🧠 MEMORY LLM EXPERIMENTS ONLY")
    print("=" * 50)
    print("This will run only memory-enhanced LLM agents to complement")
    print("your existing mechanical baseline and standard LLM work.")
    print()
    
    # Option 1: Use design space exploration with memory filter
    print("🚀 Option 1: Direct Design Space Exploration")
    print("-" * 45)
    
    cmd = [
        "python", "run_design_space_exploration.py",
        "--all",                           # Plan, run, and analyze
        "--agents", "memory",              # Only memory agents
        "--output-dir", "memory_llm_experiments"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nThis will:")
    print("- Plan experiments for memory agents only")
    print("- Run all planned memory experiments")
    print("- Generate analysis results")
    print("- Save to: memory_llm_experiments/")
    
    if input("\n🤔 Run Option 1? (y/n): ").lower().startswith('y'):
        print("\n🚀 Starting memory-only experiments...")
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print("✅ Memory experiments completed!")
            print("📁 Results saved in: memory_llm_experiments/")
        else:
            print("❌ Memory experiments failed")
        return result.returncode == 0
    
    # Option 2: Use comprehensive study with memory-only config
    print("\n🚀 Option 2: Comprehensive Study (Memory Only)")
    print("-" * 48)
    
    cmd2 = [
        "python", "comprehensive_comparison_study.py",
        "--config", "memory_only_study.yaml",
        "--preset", "mixtral"
    ]
    
    print(f"Command: {' '.join(cmd2)}")
    print("\nThis will:")
    print("- Reuse existing mechanical baseline data")
    print("- Run only memory LLM experiments")
    print("- Create unified analysis combining all three agent types")
    print("- Save to: comprehensive_study_[timestamp]/")
    
    if input("\n🤔 Run Option 2? (y/n): ").lower().startswith('y'):
        print("\n🚀 Starting comprehensive study (memory only)...")
        result = subprocess.run(cmd2)
        if result.returncode == 0:
            print("✅ Comprehensive study completed!")
        else:
            print("❌ Comprehensive study failed")
        return result.returncode == 0
    
    print("\n💡 You can also run these commands manually:")
    print(f"   {' '.join(cmd)}")
    print(f"   {' '.join(cmd2)}")
    
    return False

def main():
    print("🔄 RESTART: MEMORY LLM EXPERIMENTS")
    print("=" * 60)
    
    # Check current status
    print("📊 Current Status Check:")
    print("-" * 25)
    
    # Check for existing mechanical baselines
    mechanical_count = len(list(Path("experiments").glob("baseline_*"))) if Path("experiments").exists() else 0
    print(f"✅ Mechanical baselines available: {mechanical_count} experiments")
    
    # Check for existing comprehensive studies
    comp_studies = list(Path(".").glob("comprehensive_study_*"))
    if comp_studies:
        latest_study = max(comp_studies, key=lambda x: x.name)
        print(f"📁 Latest comprehensive study: {latest_study.name}")
        
        # Check what's in it
        mech_dir = latest_study / "mechanical_baseline"
        llm_dir = latest_study / "llm_results"
        
        if mech_dir.exists():
            mech_configs = len(list(mech_dir.glob("baseline_*")))
            print(f"   🔧 Mechanical configs: {mech_configs}")
        
        if llm_dir.exists():
            exp_dir = llm_dir / "experiments"
            if exp_dir.exists():
                llm_experiments = len(list(exp_dir.glob("exp_*")))
                print(f"   🤖 LLM experiments: {llm_experiments}")
    
    print(f"\n💡 Recommendation:")
    print("Since you have existing mechanical baseline data, you can:")
    print("1. Run ONLY memory experiments to complete your three-way comparison")
    print("2. Reuse existing mechanical baselines (no redundant computation)")
    print("3. Combine everything for comprehensive analysis")
    
    success = run_memory_only_experiments()
    
    if success:
        print("\n🎉 SUCCESS!")
        print("You now have data for all three agent types:")
        print("✅ Mechanical (existing)")
        print("✅ Standard LLM (existing)")
        print("✅ Memory LLM (just completed)")
        print("\nUse analyze_comprehensive_study.py to combine and analyze all data!")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())