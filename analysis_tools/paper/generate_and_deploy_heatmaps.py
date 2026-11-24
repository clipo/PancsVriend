#!/usr/bin/env python3
"""
Regenerate and deploy choice function heatmaps for presentation or paper.

Reads value function JSON files from each model's experiment folder and regenerates
the heatmap images using the current plotting code (including political color swap).

Usage:
    python generate_and_deploy_heatmaps.py                    # Interactive, deploy to pres (default)
    python generate_and_deploy_heatmaps.py --target paper     # Deploy to paper/figs
    python generate_and_deploy_heatmaps.py --yes              # Auto-deploy
    python generate_and_deploy_heatmaps.py --no               # Generate only, no deploy
"""

import argparse
import json
import shutil
import socket
import sys
from pathlib import Path

import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments_with_llama_cpp"
CROSS_MODEL_SOURCES = EXPERIMENTS_DIR / "cross_model" / "cross_model_vf-lp_sources.csv"

# Base path for Overleaf sync
OVERLEAF_BASE = Path.home() / "Documents/d/PM/sakoka-schelling-llm-abm/paper-overleaf"

# Target configurations
TARGETS = {
    "pres": {
        "output_dir": OVERLEAF_BASE / "pres" / "pics",
        "description": "CCS 2026 presentation",
    },
    "paper": {
        "output_dir": OVERLEAF_BASE / "figs",
        "description": "paper",
    },
}

# Will be set in main() based on --target argument
deploy_dir = None
target_desc = None

# Add plot_value_functions to path
sys.path.insert(0, str(PROJECT_ROOT / "value_functions" / "sampling"))

# Target hostname for deployment
TARGET_HOSTNAME = "ECON-0FM96LD-L"

# Scenarios to generate individual heatmaps for
SCENARIOS = [
    "baseline",
    "race_white_black",
    "ethnic_asian_hispanic",
    "income_high_low",
    "political_liberal_conservative",
    "green_yellow",
]


def load_value_functions(vf_dir):
    """Load all value function JSONs from a directory."""
    found = {}
    for p in sorted(vf_dir.glob("vf_*.json")):
        vf = json.loads(p.read_text())
        scenario = vf["meta"]["scenario"]
        found[scenario] = vf
    return found


def main():
    global deploy_dir, target_desc, EXPERIMENTS_DIR, CROSS_MODEL_SOURCES

    parser = argparse.ArgumentParser(description="Regenerate and deploy choice function heatmaps")
    parser.add_argument("--target", choices=["pres", "paper"], default="pres",
                        help="Target: 'pres' (presentation) or 'paper' (default: pres)")
    parser.add_argument("--yes", action="store_true", help="Auto-deploy without asking")
    parser.add_argument("--no", action="store_true", help="Generate only, no deployment")
    parser.add_argument("--experiments-dir", type=Path, default=None,
                        help="Run root holding the model run folders and cross_model/ "
                             "(default: experiments_with_llama_cpp)")
    args = parser.parse_args()
    if args.experiments_dir is not None:
        EXPERIMENTS_DIR = args.experiments_dir.resolve()
        CROSS_MODEL_SOURCES = EXPERIMENTS_DIR / "cross_model" / "cross_model_vf-lp_sources.csv"

    target_config = TARGETS[args.target]
    deploy_dir = target_config["output_dir"]
    target_desc = target_config["description"]

    print("=" * 60)
    print(f"Regenerating Choice Function Heatmaps for {target_desc}")
    print(f"Target: {deploy_dir}")
    print("=" * 60)

    # Import plotting functions (after path setup)
    try:
        from plot_value_functions import (
            fig_heatmaps, fig_single_scenario_heatmap, SCENARIO_ORDER
        )
    except ImportError as e:
        print(f"Error importing plot_value_functions: {e}")
        print("Make sure you're running from the correct environment.")
        return 1

    # Load model sources
    print("\n1. Loading model sources...")
    if not CROSS_MODEL_SOURCES.exists():
        print(f"   ERROR: Sources file not found: {CROSS_MODEL_SOURCES}")
        return 1
    sources = pd.read_csv(CROSS_MODEL_SOURCES)
    print(f"   Found {len(sources)} models")

    # Process each model
    print("\n2. Regenerating heatmaps...")
    generated = []
    for _, row in sources.iterrows():
        model = row['model']
        run_dir = row['run_dir']
        short_name = model.split('-')[0]
        vf_dir = EXPERIMENTS_DIR / run_dir / "value_functions"

        if not vf_dir.exists():
            print(f"   WARNING: Missing value_functions dir for {model}")
            continue

        # Load value functions
        found = load_value_functions(vf_dir)
        if not found:
            print(f"   WARNING: No value function JSONs for {model}")
            continue

        # Sort scenarios
        scenarios = sorted(found, key=lambda x: (SCENARIO_ORDER.index(x)
                                                  if x in SCENARIO_ORDER else 99, x))

        # Generate combined heatmap grid
        combined_path = vf_dir / "value_function_heatmaps.png"
        fig_heatmaps(found, scenarios, str(combined_path))
        generated.append((short_name, "combined", combined_path))
        print(f"   {short_name}: combined heatmap grid")

        # Generate individual per-scenario heatmaps
        for scenario in scenarios:
            if scenario in found:
                scenario_path = vf_dir / f"heatmap_{scenario}.png"
                fig_single_scenario_heatmap(found[scenario], scenario, str(scenario_path))
                generated.append((short_name, scenario, scenario_path))
        print(f"   {short_name}: {len(scenarios)} individual scenario heatmaps")

    print(f"\n   Generated {len(generated)} heatmap files")

    if args.no:
        print("\n--no flag: skipping deployment")
        return 0

    # Check hostname for deployment
    hostname = socket.gethostname()
    print(f"\nHostname: {hostname}")

    if hostname != TARGET_HOSTNAME and not args.yes:
        print(f"\nNot on target machine ({TARGET_HOSTNAME}).")
        print(f"Use --yes to force deployment on any machine.")
        return 0

    # Deploy
    if args.yes:
        do_deploy = True
    else:
        response = input(f"\nDeploy {len(generated)} heatmaps to {deploy_dir}? [y/N] ")
        do_deploy = response.lower() in ('y', 'yes')

    if do_deploy:
        print(f"\n3. Deploying to {target_desc} folder...")
        deploy_dir.mkdir(parents=True, exist_ok=True)
        deployed = 0

        for short_name, scenario, src_path in generated:
            if scenario == "combined":
                dst_path = deploy_dir / f"{short_name}_heatmaps.png"
            else:
                dst_path = deploy_dir / f"{short_name}_heatmap_{scenario}.png"

            shutil.copy2(src_path, dst_path)
            deployed += 1

            # Special case: also create short name for "Full Decision Surface" slide
            if short_name == "deepseek" and scenario == "political_liberal_conservative":
                short_dst = deploy_dir / f"{short_name}_heatmap_political.png"
                shutil.copy2(src_path, short_dst)
                print(f"   Copied: {short_dst.name} (alias)")
                deployed += 1

        print(f"\nDone! Deployed {deployed} heatmaps to {deploy_dir}")
    else:
        print("\nDeployment skipped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
