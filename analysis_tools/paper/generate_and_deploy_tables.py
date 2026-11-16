#!/usr/bin/env python3
"""
Generate LaTeX tables and optionally deploy to the presentation or paper folder.

This script:
1. Runs compute_scenario_orderings.py to generate the LaTeX tables
2. If on hostname ECON-0FM96LD-L, asks if you want to copy tables to target folder
3. Copies tables so the presentation/paper can use \input{} to include them

Usage:
    python generate_and_deploy_tables.py                    # Interactive, deploy to pres (default)
    python generate_and_deploy_tables.py --target paper     # Deploy to paper folder
    python generate_and_deploy_tables.py --yes              # Auto-deploy without asking
    python generate_and_deploy_tables.py --no               # Generate only, no deploy
"""

import socket
import shutil
import argparse
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Base path for Overleaf sync
OVERLEAF_BASE = Path.home() / "Documents/d/PM/sakoka-schelling-llm-abm/paper-overleaf"

# Target configurations (tables go to root of pres/paper, not pics/figs)
TARGETS = {
    "pres": {
        "output_dir": OVERLEAF_BASE / "pres",
        "description": "CCS 2026 presentation",
    },
    "paper": {
        "output_dir": OVERLEAF_BASE,
        "description": "paper",
    },
}

# Will be set in main() based on --target argument
deploy_dir = None
target_desc = None

# Files to deploy
LATEX_FILES = [
    "key_results_ordering.tex",
    "key_results_ordering_info.txt",
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
    """Copy LaTeX tables to the target folder."""
    print("\n" + "=" * 60)
    print(f"Deploying tables to {target_desc} folder...")
    print("=" * 60)

    if not deploy_dir.exists():
        print(f"ERROR: Target directory not found: {deploy_dir}")
        return False

    for filename in LATEX_FILES:
        src = SCRIPT_DIR / filename
        dst = deploy_dir / filename

        if not src.exists():
            print(f"WARNING: Source file not found: {src}")
            continue

        shutil.copy2(src, dst)
        print(f"  Copied: {filename} -> {dst}")

    print(f"\nDone! Tables deployed to {target_desc} folder.")
    print(f"  Location: {deploy_dir}")
    print("\nTo use in your document, add:")
    print("  \\input{key_results_ordering.tex}")

    return True


def main():
    global deploy_dir, target_desc

    parser = argparse.ArgumentParser(
        description="Generate and deploy LaTeX tables"
    )
    parser.add_argument(
        "--target",
        choices=["pres", "paper"],
        default="pres",
        help="Target: 'pres' (presentation) or 'paper' (default: pres)"
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

    target_config = TARGETS[args.target]
    deploy_dir = target_config["output_dir"]
    target_desc = target_config["description"]

    print(f"Target: {target_desc} ({deploy_dir})")

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
