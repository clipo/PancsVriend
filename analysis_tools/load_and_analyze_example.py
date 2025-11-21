#!/usr/bin/env python3
"""
Example script demonstrating how to load stored simulation outputs and analyze them.

Usage:
    python load_and_analyze_example.py <output_directory>

Example:
    python load_and_analyze_example.py ./experiments/baseline_20250710_120000
"""

import sys
import os
from analysis_tools.output_paths import get_reports_dir
from base_simulation import Simulation

def main():
    if len(sys.argv) != 2:
        print("Usage: python load_and_analyze_example.py <output_directory>")
        print("\nExample:")
        print("    python load_and_analyze_example.py ./experiments/baseline_20250710_120000")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    if not os.path.exists(output_dir):
        print(f"Error: Output directory does not exist: {output_dir}")
        sys.exit(1)
    
    try:
        print("=" * 60)
        print("LOADING AND ANALYZING SIMULATION RESULTS")
        print("=" * 60)
        
        # Option 1: Load and analyze in one step (recommended)
        print("\n1. Using load_and_analyze_results() - One-step approach:")
        output_dir_result, results, convergence_data = Simulation.load_and_analyze_results(output_dir)
        
        print("\nAnalysis complete!")
        print(f"- Results saved to: {output_dir_result}")
        print(f"- Number of runs analyzed: {len(results)}")
        print(f"- Convergence rate: {sum(1 for cd in convergence_data if cd['converged'])} / {len(convergence_data)}")
        
        # Show some basic statistics
        if convergence_data:
            converged_runs = [cd for cd in convergence_data if cd['converged']]
            if converged_runs:
                avg_convergence_step = sum(cd['convergence_step'] for cd in converged_runs) / len(converged_runs)
                print(f"- Average convergence step: {avg_convergence_step:.1f}")
        
        print("\n" + "=" * 60)
        print("ALTERNATIVE APPROACH (TWO-STEP)")
        print("=" * 60)
        
        # Option 2: Load and analyze separately (for more control)
        print("\n2. Using load_results_from_output() + analyze_results() - Two-step approach:")
        
        # Step 1: Load the data
        results_loaded, n_runs = Simulation.load_results_from_output(output_dir)
        print(f"Loaded {n_runs} simulation runs")
        
        # Step 2: Analyze the data (this will overwrite the previous analysis files)
        # Re-analysis outputs in reports directory
        base_name = os.path.basename(os.path.normpath(output_dir))
        analysis_output_dir = get_reports_dir() / f"{base_name}_reanalyzed"
        analysis_output_dir.mkdir(parents=True, exist_ok=True)

        output_dir_result2, results2, convergence_data2 = Simulation.analyze_results(
            results_loaded, str(analysis_output_dir), n_runs
        )
        
        print("Re-analysis complete!")
        print(f"- Results saved to: {output_dir_result2}")
        
        print("\n" + "=" * 60)
        print("FILES GENERATED")
        print("=" * 60)
        
        # List the analysis files that were created
        analysis_files = [
            "metrics_history.csv",
            "convergence_summary.csv", 
            "step_statistics.csv"
        ]
        
        for filename in analysis_files:
            filepath = analysis_output_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                print(f"✓ {filename} ({size:,} bytes)")
            else:
                print(f"✗ {filename} (not found)")
        
        print("\nYou can now use these CSV files for further analysis, visualization, or reporting.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
