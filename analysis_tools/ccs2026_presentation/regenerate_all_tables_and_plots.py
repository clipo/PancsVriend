#!/usr/bin/env python3
"""
Regenerate all tables and plots for the CCS 2026 presentation.

This script runs all the individual generation scripts in sequence,
deploying outputs to pres-overleaf.link/pres/pics/.

Usage:
    python regenerate_all_tables_and_plots.py           # Interactive mode
    python regenerate_all_tables_and_plots.py --yes     # Auto-deploy all
    python regenerate_all_tables_and_plots.py --no      # Generate only, no deploy

Scripts run (in order):
    1. generate_choice_function_plots.py    - Choice function plots (6-panel grids)
    2. generate_convergence_plots.py        - Convergence trajectory plots
    3. generate_sorted_segregation_plots.py - Segregation bar charts
    4. generate_and_deploy_tables.py        - LaTeX tables (key_results_ordering.tex)
    5. generate_and_deploy_heatmaps.py      - Regenerate P(MOVE) heatmaps from JSONs
    6. deploy_model_images.py               - Copy model images to pics/

Requires:
    - Access to experiments_with_llama_cpp/ data
    - Python environment with pandas, numpy, matplotlib
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# Scripts to run in order
SCRIPTS = [
    "generate_choice_function_plots.py",
    "generate_convergence_plots.py",
    "generate_sorted_segregation_plots.py",
    "generate_and_deploy_tables.py",
    "generate_and_deploy_heatmaps.py",
    "deploy_model_images.py",
]


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate all tables and plots for CCS 2026 presentation"
    )
    parser.add_argument("--yes", action="store_true", help="Auto-deploy without asking")
    parser.add_argument("--no", action="store_true", help="Generate only, no deployment")
    args = parser.parse_args()

    # Determine flags to pass
    if args.yes:
        flags = ["--yes"]
    elif args.no:
        flags = ["--no"]
    else:
        flags = []

    print("=" * 70)
    print("Regenerating All Tables and Plots for CCS 2026 Presentation")
    print("=" * 70)

    failed = []
    succeeded = []

    for script in SCRIPTS:
        script_path = SCRIPT_DIR / script
        if not script_path.exists():
            print(f"\n[SKIP] {script} - file not found")
            failed.append(script)
            continue

        print(f"\n{'=' * 70}")
        print(f"Running: {script}")
        print("=" * 70)

        cmd = [sys.executable, str(script_path)] + flags
        result = subprocess.run(cmd, cwd=SCRIPT_DIR)

        if result.returncode != 0:
            print(f"\n[FAILED] {script} exited with code {result.returncode}")
            failed.append(script)
        else:
            print(f"\n[OK] {script}")
            succeeded.append(script)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Succeeded: {len(succeeded)}/{len(SCRIPTS)}")
    print(f"  Failed:    {len(failed)}/{len(SCRIPTS)}")

    if failed:
        print(f"\n  Failed scripts: {', '.join(failed)}")
        return 1

    print("\nAll scripts completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
