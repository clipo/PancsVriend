#!/usr/bin/env python3
"""
Generate presentation tables and optionally deploy to the pres folder.

This script:
1. Runs compute_scenario_orderings.py to generate the LaTeX tables
2. If on hostname ECON-0FM96LD-L, asks if you want to copy tables to pres folder
3. Copies tables so the presentation can use \input{} to include them

Usage:
    python generate_and_deploy_tables.py          # Interactive mode
    python generate_and_deploy_tables.py --yes    # Auto-deploy without asking
    python generate_and_deploy_tables.py --no     # Generate only, no deploy
"""

import socket
import shutil
import argparse
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PRES_DIR = PROJECT_ROOT / "pres-overleaf.link" / "pres"

# Files to deploy
LATEX_FILES = [
    "key_results_table.tex",
]

TARGET_HOSTNAME = "ECON-0FM96LD-L"


def get_hostname():
    """Get the current machine's hostname."""
    return socket.gethostname()


def run_analysis():
    """Run the compute_scenario_orderings.py script."""
    print("=" * 60)
    print("Running compute_scenario_orderings.py...")
    print("=" * 60)

    # Import and run the main function
    from compute_scenario_orderings import main
    main()


def deploy_tables():
    """Copy LaTeX tables to the presentation folder."""
    print("\n" + "=" * 60)
    print("Deploying tables to presentation folder...")
    print("=" * 60)

    if not PRES_DIR.exists():
        print(f"ERROR: Presentation directory not found: {PRES_DIR}")
        return False

    for filename in LATEX_FILES:
        src = SCRIPT_DIR / filename
        dst = PRES_DIR / filename

        if not src.exists():
            print(f"WARNING: Source file not found: {src}")
            continue

        shutil.copy2(src, dst)
        print(f"  Copied: {filename} -> {dst}")

    print("\nDone! Tables deployed to presentation folder.")
    print(f"  Location: {PRES_DIR}")
    print("\nTo use in your presentation, add to the preamble or where needed:")
    print("  \\input{key_results_table.tex}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate and deploy presentation tables"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Auto-deploy without asking"
    )
    parser.add_argument(
        "--no", "-n",
        action="store_true",
        help="Generate only, skip deployment"
    )
    args = parser.parse_args()

    # Run the analysis
    run_analysis()

    # Check hostname
    hostname = get_hostname()
    print(f"\nHostname: {hostname}")

    if args.no:
        print("Skipping deployment (--no flag)")
        return

    if hostname == TARGET_HOSTNAME or args.yes:
        if args.yes:
            do_deploy = True
        else:
            # Ask user
            print(f"\nYou are on {TARGET_HOSTNAME}.")
            response = input("Copy LaTeX tables to pres folder? [y/N]: ").strip().lower()
            do_deploy = response in ('y', 'yes')

        if do_deploy:
            deploy_tables()
        else:
            print("Skipping deployment.")
    else:
        print(f"Not on {TARGET_HOSTNAME}, skipping deployment prompt.")
        print(f"Use --yes to force deployment on any machine.")


if __name__ == "__main__":
    main()
