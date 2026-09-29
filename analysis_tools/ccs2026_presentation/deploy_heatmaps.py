#!/usr/bin/env python3
"""
Deploy value function heatmaps to the presentation folder.

Copies existing heatmaps from each model's run folder to pres-overleaf.link/pres/pics/

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


def main():
    parser = argparse.ArgumentParser(description="Deploy value function heatmaps")
    parser.add_argument("--yes", action="store_true", help="Auto-deploy without asking")
    parser.add_argument("--no", action="store_true", help="List only, no deployment")
    args = parser.parse_args()

    print("=" * 60)
    print("Deploying Value Function Heatmaps for CCS 2026")
    print("=" * 60)

    # Load model sources
    print("\n1. Loading model sources...")
    sources = pd.read_csv(CROSS_MODEL_SOURCES)
    print(f"   Found {len(sources)} models")

    # Find heatmaps
    print("\n2. Locating heatmaps...")
    heatmaps = []
    for _, row in sources.iterrows():
        model = row['model']
        run_dir = row['run_dir']
        short_name = model.split('-')[0]

        heatmap_path = EXPERIMENTS_DIR / run_dir / "value_functions" / "value_function_heatmaps.png"
        if heatmap_path.exists():
            heatmaps.append((model, short_name, heatmap_path))
            print(f"   Found: {short_name} -> {heatmap_path.name}")
        else:
            print(f"   WARNING: Missing heatmap for {model}")

    print(f"\n   Found {len(heatmaps)} heatmaps")

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
    if args.yes:
        do_deploy = True
    else:
        response = input(f"\nDeploy {len(heatmaps)} heatmaps to {PICS_DIR}? [y/N] ")
        do_deploy = response.lower() in ('y', 'yes')

    if do_deploy:
        print("\n3. Deploying to presentation folder...")
        PICS_DIR.mkdir(parents=True, exist_ok=True)
        for model, short_name, src_path in heatmaps:
            dst_path = PICS_DIR / f"{short_name}_heatmaps.png"
            shutil.copy2(src_path, dst_path)
            print(f"   Copied: {dst_path.name}")
        print(f"\nDone! Deployed to {PICS_DIR}")
    else:
        print("\nDeployment skipped.")


if __name__ == "__main__":
    main()
