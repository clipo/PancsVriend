#!/usr/bin/env python3
"""
Deploy value function heatmaps to the presentation folder.

Copies existing heatmaps from each model's run folder to pres-overleaf.link/pres/pics/
Includes both combined grids and individual per-scenario heatmaps.

Usage:
    python deploy_heatmaps.py           # Interactive
    python deploy_heatmaps.py --yes     # Auto-deploy
    python deploy_heatmaps.py --no      # List only, no deploy
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
PRES_DIR = PROJECT_ROOT / "pres-overleaf.link" / "pres"
PICS_DIR = PRES_DIR / "pics"

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
    parser = argparse.ArgumentParser(description="Deploy value function heatmaps")
    parser.add_argument("--yes", action="store_true", help="Auto-deploy without asking")
    parser.add_argument("--no", action="store_true", help="List only, no deployment")
    args = parser.parse_args()

    print("=" * 60)
    print("Deploying Choice Function Heatmaps for CCS 2026")
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
        response = input(f"\nDeploy {total} heatmaps to {PICS_DIR}? [y/N] ")
        do_deploy = response.lower() in ('y', 'yes')

    if do_deploy:
        print("\n4. Deploying to presentation folder...")
        PICS_DIR.mkdir(parents=True, exist_ok=True)

        # Deploy combined grids
        for model, short_name, src_path in combined_heatmaps:
            dst_path = PICS_DIR / f"{short_name}_heatmaps.png"
            shutil.copy2(src_path, dst_path)
            print(f"   Copied: {dst_path.name}")

        # Deploy individual scenario heatmaps
        for short_name, scenario, src_path in individual_heatmaps:
            dst_path = PICS_DIR / f"{short_name}_heatmap_{scenario}.png"
            shutil.copy2(src_path, dst_path)
            print(f"   Copied: {dst_path.name}")

            # Special case: also create short name for "Full Decision Surface" slide
            if short_name == "deepseek" and scenario == "political_liberal_conservative":
                short_dst = PICS_DIR / f"{short_name}_heatmap_political.png"
                shutil.copy2(src_path, short_dst)
                print(f"   Copied: {short_dst.name} (alias for Full Decision Surface slide)")

        print(f"\nDone! Deployed {total} heatmaps to {PICS_DIR}")
    else:
        print("\nDeployment skipped.")


if __name__ == "__main__":
    main()
