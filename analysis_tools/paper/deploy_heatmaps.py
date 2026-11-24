#!/usr/bin/env python3
"""
Deploy value function heatmaps to the presentation or paper folder.

Copies existing heatmaps from each model's run folder to the target folder.
Includes both combined grids and individual per-scenario heatmaps.

Usage:
    python deploy_heatmaps.py                    # Interactive, deploy to pres (default)
    python deploy_heatmaps.py --target paper     # Deploy to paper/figs
    python deploy_heatmaps.py --yes              # Auto-deploy
    python deploy_heatmaps.py --no               # List only, no deploy
"""

import argparse
import shutil
import socket
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

# Target hostname for deployment
TARGET_HOSTNAME = "ECON-0FM96LD-L"

# Individual scenario heatmaps to deploy
SCENARIOS = [
    "baseline",
    "race_white_black",
    "ethnic_asian_hispanic",
    "income_high_low",
    "political_liberal_conservative",
    "green_yellow",
]


def main():
    global EXPERIMENTS_DIR, CROSS_MODEL_SOURCES
    parser = argparse.ArgumentParser(description="Deploy value function heatmaps")
    parser.add_argument("--target", choices=["pres", "paper"], default="pres",
                        help="Target: 'pres' (presentation) or 'paper' (default: pres)")
    parser.add_argument("--yes", action="store_true", help="Auto-deploy without asking")
    parser.add_argument("--no", action="store_true", help="List only, no deployment")
    parser.add_argument("--experiments-dir", type=Path, default=None,
                        help="Run root holding the model run folders and cross_model/ "
                             "(default: experiments_with_llama_cpp)")
    args = parser.parse_args()
    if args.experiments_dir is not None:
        EXPERIMENTS_DIR = args.experiments_dir.resolve()
        CROSS_MODEL_SOURCES = EXPERIMENTS_DIR / "cross_model" / "cross_model_vf-lp_sources.csv"

    target_config = TARGETS[args.target]
    output_dir = target_config["output_dir"]
    target_desc = target_config["description"]

    print("=" * 60)
    print(f"Deploying Choice Function Heatmaps for {target_desc}")
    print(f"Target: {output_dir}")
    print("=" * 60)

    # Load model sources
    print("\n1. Loading model sources...")
    sources = pd.read_csv(CROSS_MODEL_SOURCES)
    print(f"   Found {len(sources)} models")

    # Find heatmaps (combined grids)
    print("\n2. Locating combined heatmap grids...")
    combined_heatmaps = []
    for _, row in sources.iterrows():
        model = row['model']
        run_dir = row['run_dir']
        short_name = model.split('-')[0]

        heatmap_path = EXPERIMENTS_DIR / run_dir / "value_functions" / "value_function_heatmaps.png"
        if heatmap_path.exists():
            combined_heatmaps.append((model, short_name, heatmap_path))
            print(f"   Found: {short_name} -> {heatmap_path.name}")
        else:
            print(f"   WARNING: Missing combined heatmap for {model}")

    print(f"\n   Found {len(combined_heatmaps)} combined heatmaps")

    # Find individual per-scenario heatmaps
    print("\n3. Locating individual scenario heatmaps...")
    individual_heatmaps = []
    for _, row in sources.iterrows():
        model = row['model']
        run_dir = row['run_dir']
        short_name = model.split('-')[0]
        vf_dir = EXPERIMENTS_DIR / run_dir / "value_functions"

        for scenario in SCENARIOS:
            heatmap_path = vf_dir / f"heatmap_{scenario}.png"
            if heatmap_path.exists():
                individual_heatmaps.append((short_name, scenario, heatmap_path))

    print(f"   Found {len(individual_heatmaps)} individual scenario heatmaps")

    if args.no:
        print("\n--no flag: skipping deployment")
        return

    # Check hostname for deployment
    hostname = socket.gethostname()
    print(f"\nHostname: {hostname}")

    if hostname != TARGET_HOSTNAME and not args.yes:
        print(f"\nNot on target machine ({TARGET_HOSTNAME}).")
        print(f"Use --yes to force deployment on any machine.")
        return

    # Deploy
    total = len(combined_heatmaps) + len(individual_heatmaps)
    if args.yes:
        do_deploy = True
    else:
        response = input(f"\nDeploy {total} heatmaps to {output_dir}? [y/N] ")
        do_deploy = response.lower() in ('y', 'yes')

    if do_deploy:
        print(f"\n4. Deploying to {target_desc} folder...")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Deploy combined grids
        for model, short_name, src_path in combined_heatmaps:
            dst_path = output_dir / f"{short_name}_heatmaps.png"
            shutil.copy2(src_path, dst_path)
            print(f"   Copied: {dst_path.name}")

        # Deploy individual scenario heatmaps
        for short_name, scenario, src_path in individual_heatmaps:
            dst_path = output_dir / f"{short_name}_heatmap_{scenario}.png"
            shutil.copy2(src_path, dst_path)
            print(f"   Copied: {dst_path.name}")

            # Special case: also create short name for "Full Decision Surface" slide
            if short_name == "deepseek" and scenario == "political_liberal_conservative":
                short_dst = output_dir / f"{short_name}_heatmap_political.png"
                shutil.copy2(src_path, short_dst)
                print(f"   Copied: {short_dst.name} (alias for Full Decision Surface slide)")

        print(f"\nDone! Deployed {total} heatmaps to {output_dir}")
    else:
        print("\nDeployment skipped.")


if __name__ == "__main__":
    main()
