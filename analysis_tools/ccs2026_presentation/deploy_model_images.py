#!/usr/bin/env python3
"""
Deploy model result images to the presentation pics folder.

Copies segregation and convergence plots from each model's run directory
to pres-overleaf.link/pres/pics/ with consistent naming for use in slides.

Source images (from <run>/plots/):
  - segregation_metrics_comparison_dissimilarity_index.png
  - convergence_patterns_dissimilarity_index.png

Destination naming:
  - <model_short>_segregation_by_context.png
  - <model_short>_convergence_patterns.png

Usage:
    python deploy_model_images.py           # Interactive: ask before deploying
    python deploy_model_images.py --yes     # Auto-deploy without asking
    python deploy_model_images.py --no      # List what would be copied, no action
"""

import argparse
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
PRES_DIR = PROJECT_ROOT / "pres-overleaf.link" / "pres"
PICS_DIR = PRES_DIR / "pics"

# Target hostname for deployment
TARGET_HOSTNAME = "ECON-0FM96LD-L"

# Image mappings: (source_filename, dest_suffix)
IMAGE_MAPPINGS = [
    ("segregation_metrics_comparison_dissimilarity_index.png", "segregation_by_context.png"),
    ("convergence_patterns_dissimilarity_index.png", "convergence_patterns.png"),
]


def get_model_short_name(model: str) -> str:
    """Extract short name from model identifier (e.g., 'gemma-4-31b' -> 'gemma')."""
    return model.split('-')[0]


def load_model_sources() -> pd.DataFrame:
    """Load the cross-model sources file."""
    return pd.read_csv(CROSS_MODEL_SOURCES)


def list_images_to_copy(sources: pd.DataFrame) -> list:
    """Generate list of (src, dst) tuples for all images to copy."""
    copies = []

    for _, row in sources.iterrows():
        model = row['model']
        run_dir = row['run_dir']
        short_name = get_model_short_name(model)

        plots_dir = EXPERIMENTS_DIR / run_dir / "plots"

        for src_filename, dst_suffix in IMAGE_MAPPINGS:
            src_path = plots_dir / src_filename
            dst_path = PICS_DIR / f"{short_name}_{dst_suffix}"

            if src_path.exists():
                copies.append((src_path, dst_path, model))
            else:
                print(f"WARNING: Missing {src_path}")

    return copies


def deploy_images(copies: list) -> int:
    """Copy images to destination. Returns count of files copied."""
    PICS_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for src, dst, model in copies:
        shutil.copy2(src, dst)
        print(f"  Copied: {model} -> {dst.name}")
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Deploy model images to presentation folder")
    parser.add_argument("--yes", action="store_true", help="Auto-deploy without asking")
    parser.add_argument("--no", action="store_true", help="List only, no deployment")
    args = parser.parse_args()

    print("=" * 60)
    print("Model Image Deployment for CCS 2026 Presentation")
    print("=" * 60)

    # Load sources
    print("\n1. Loading model sources...")
    sources = load_model_sources()
    print(f"   Found {len(sources)} models")

    # List images to copy
    print("\n2. Checking source images...")
    copies = list_images_to_copy(sources)
    print(f"   Found {len(copies)} images to copy")

    if args.no:
        print("\n3. Would copy (--no flag, no action taken):")
        for src, dst, model in copies:
            print(f"   {model}: {src.name} -> {dst.name}")
        return

    # Check hostname
    hostname = socket.gethostname()
    print(f"\nHostname: {hostname}")

    if hostname != TARGET_HOSTNAME:
        print(f"\nNot on target machine ({TARGET_HOSTNAME}).")
        print("Images generated but not deployed.")
        print(f"\nTo deploy manually, copy from experiment plots/ folders to:")
        print(f"  {PICS_DIR}")
        return

    # Deploy
    if args.yes:
        do_deploy = True
    else:
        response = input(f"\nDeploy {len(copies)} images to {PICS_DIR}? [y/N] ")
        do_deploy = response.lower() in ('y', 'yes')

    if do_deploy:
        print("\n3. Deploying images...")
        count = deploy_images(copies)
        print(f"\nDone! Deployed {count} images to {PICS_DIR}")
    else:
        print("\nDeployment skipped.")


if __name__ == "__main__":
    main()
