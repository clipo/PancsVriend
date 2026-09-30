#!/usr/bin/env python3
"""
Generate 2D value function plots for CCS 2026 presentation.

Creates plots showing P(MOVE) vs fraction of similar neighbors for each model.
These are cleaner than the full heatmaps and easier to interpret on slides.

Outputs to pres-overleaf.link/pres/pics/<model>_value_functions.png

X-AXIS TRANSFORMATION NOTE (2026-09-29):
    The value function JSON stores `ratio_float` as the OPPOSITE fraction:
        ratio_float = (n_occupied - n_similar) / n_occupied

    However, we want the x-axis to show "Fraction Similar Neighbors" to match
    the heatmaps which show P(MOVE) indexed by (n_similar, n_occupied).

    Therefore, we transform x-data to similar fraction:
        similar_fraction = 1.0 - ratio_float

    This ensures consistency:
    - Heatmaps: LOW n_similar → HIGH P(MOVE) (top-left is hot)
    - 1D plots: LOW similar fraction → HIGH P(MOVE) (left side is high)

    The mechanical baseline step function [0,0.5,0.5,1] → [1,1,0,0] is correct
    for similar fraction: agents move when similar < 50%, stay when similar ≥ 50%.

Usage:
    python generate_value_function_plots.py           # Interactive
    python generate_value_function_plots.py --yes     # Auto-deploy
    python generate_value_function_plots.py --no      # Generate only, no deploy
"""

import argparse
import json
import shutil
import socket
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

# Short labels - bold for the 3 empirical comparison scenarios
SCENARIO_SHORT_LABELS = {
    'baseline': 'Color(RvB)',
    'green_yellow': 'Color(GvY)',
    'race_white_black': r'$\bf{Racial}$',
    'ethnic_asian_hispanic': 'Ethnic',
    'income_high_low': r'$\bf{Economic}$',
    'political_liberal_conservative': r'$\bf{Political}$',
}

# Colors for scenarios
SCENARIO_COLORS = {
    'baseline': '#7f7f7f',           # gray
    'green_yellow': '#2ca02c',       # green
    'race_white_black': '#d62728',   # red
    'ethnic_asian_hispanic': '#ff7f0e',  # orange
    'income_high_low': '#1f77b4',    # blue
    'political_liberal_conservative': '#9467bd',  # purple
}

# Scenario order for plotting
SCENARIO_ORDER = [
    'baseline', 'green_yellow', 'race_white_black',
    'ethnic_asian_hispanic', 'income_high_low', 'political_liberal_conservative'
]


def load_value_functions(run_dir: str) -> dict:
    """Load all value function JSON files for a model."""
    vf_dir = EXPERIMENTS_DIR / run_dir / "value_functions"
    vfs = {}

    for json_path in vf_dir.glob("vf_*__*__R3_dual_count.json"):
        with open(json_path) as f:
            vf = json.load(f)
        scenario = vf["meta"]["scenario"]
        vfs[scenario] = vf

    return vfs


def generate_value_function_plot(model: str, run_dir: str, output_path: Path) -> bool:
    """Generate a 2D value function plot for one model."""
    vfs = load_value_functions(run_dir)
    if not vfs:
        print(f"  WARNING: No value function data for {model}")
        return False

    # Set up figure - 2x3 grid of scenarios
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, scenario in enumerate(SCENARIO_ORDER):
        ax = axes[idx]

        if scenario not in vfs:
            ax.set_visible(False)
            continue

        vf = vfs[scenario]
        color = SCENARIO_COLORS.get(scenario, '#999999')
        label = SCENARIO_SHORT_LABELS.get(scenario, scenario)

        # Plot mechanical agent step function
        ax.step([0, 0.5, 0.5, 1.0], [1, 1, 0, 0], where="post",
                color="black", ls="--", lw=1.5, alpha=0.5, label="Mechanical")

        # Plot both roles
        for role, role_color, role_label in [("red", "#d62728", "Type A"),
                                              ("blue", "#1f77b4", "Type B")]:
            if role not in vf["ratios"]:
                continue

            # Extract ratio data points
            # Transform ratio_float (opposite fraction) to similar fraction
            points = []
            for r in vf["ratios"][role]:
                if r["ratio_float"] is not None and r["p_move_effective"] is not None:
                    similar_frac = 1.0 - r["ratio_float"]
                    points.append((similar_frac, r["p_move_effective"]))

            if not points:
                continue

            points.sort()
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            ax.plot(xs, ys, color=role_color, lw=2.0, marker='o', ms=3,
                    alpha=0.85, label=role_label)

        # Title and formatting
        ax.set_title(label, fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

        if idx >= 3:  # Bottom row
            ax.set_xlabel("Fraction Similar Neighbors")
        if idx % 3 == 0:  # Left column
            ax.set_ylabel("P(MOVE)")

    # Add legend to first subplot
    axes[0].legend(loc='upper right', fontsize=7)

    # Get model display name
    model_display = model.split('-')[0].capitalize()
    fig.suptitle(f"Value Functions: {model_display}", fontsize=12, y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return True


def generate_single_scenario_plot(model: str, run_dir: str, scenario: str, output_path: Path) -> bool:
    """Generate a single-scenario value function plot for use in slides."""
    vfs = load_value_functions(run_dir)
    if scenario not in vfs:
        print(f"  WARNING: No {scenario} data for {model}")
        return False

    vf = vfs[scenario]

    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(5, 4))

    # Mechanical baseline
    ax.step([0, 0.5, 0.5, 1.0], [1, 1, 0, 0], where="post",
            color="black", ls="--", lw=1.5, alpha=0.5, label="Mechanical")

    # Plot both roles
    for role, role_color, role_label in [("red", "#d62728", "Type A"),
                                          ("blue", "#1f77b4", "Type B")]:
        if role not in vf["ratios"]:
            continue

        # Extract ratio data points - transform to similar fraction
        points = []
        for r in vf["ratios"][role]:
            if r["ratio_float"] is not None and r["p_move_effective"] is not None:
                similar_frac = 1.0 - r["ratio_float"]
                points.append((similar_frac, r["p_move_effective"]))

        if not points:
            continue

        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        ax.plot(xs, ys, color=role_color, lw=2.5, marker='o', ms=5,
                alpha=0.85, label=role_label)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Fraction Similar Neighbors")
    ax.set_ylabel("P(MOVE)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    label = SCENARIO_SHORT_LABELS.get(scenario, scenario)
    model_display = model.split('-')[0].capitalize()
    ax.set_title(f"{model_display}: {label}", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return True


def generate_combined_example_plot(output_path: Path) -> bool:
    """Generate a single plot comparing two contrasting models/scenarios."""
    # Load DeepSeek (context-sensitive) and Phi (context-insensitive)
    # Shows contrast between models that respond to context vs those that don't
    sources = pd.read_csv(CROSS_MODEL_SOURCES)

    deepseek_row = sources[sources['model'].str.startswith('deepseek')].iloc[0]
    phi_row = sources[sources['model'].str.startswith('phi')].iloc[0]

    deepseek_vfs = load_value_functions(deepseek_row['run_dir'])
    phi_vfs = load_value_functions(phi_row['run_dir'])

    if not deepseek_vfs or not phi_vfs:
        print("  WARNING: Could not load DeepSeek or Phi value functions")
        return False

    # Create 1x2 comparison: DeepSeek (context-sensitive) vs Phi (context-insensitive)
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    scenarios_to_show = ['baseline', 'political_liberal_conservative']
    models_data = [
        ("DeepSeek V4 Flash", deepseek_vfs),
        ("Phi 4 14B", phi_vfs),
    ]

    for ax_idx, (model_name, vfs) in enumerate(models_data):
        ax = axes[ax_idx]

        # Mechanical step
        ax.step([0, 0.5, 0.5, 1.0], [1, 1, 0, 0], where="post",
                color="black", ls="--", lw=1.5, alpha=0.4, label="Mechanical")

        for scenario in scenarios_to_show:
            if scenario not in vfs:
                continue

            vf = vfs[scenario]
            color = SCENARIO_COLORS.get(scenario, '#999999')
            label = SCENARIO_SHORT_LABELS.get(scenario, scenario)

            # Average both roles for simplicity
            # Transform ratio_float (opposite fraction) to similar fraction
            all_points = {}
            for role in ["red", "blue"]:
                if role not in vf["ratios"]:
                    continue
                for r in vf["ratios"][role]:
                    if r["ratio_float"] is not None and r["p_move_effective"] is not None:
                        similar_frac = 1.0 - r["ratio_float"]
                        x = round(similar_frac, 3)
                        if x not in all_points:
                            all_points[x] = []
                        all_points[x].append(r["p_move_effective"])

            if not all_points:
                continue

            xs = sorted(all_points.keys())
            ys = [np.mean(all_points[x]) for x in xs]

            ax.plot(xs, ys, color=color, lw=2.5, marker='o', ms=4,
                    alpha=0.9, label=label)

        ax.set_title(model_name, fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("Fraction Similar Neighbors", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

        if ax_idx == 0:
            ax.set_ylabel("P(MOVE)", fontsize=10)

    fig.suptitle("Value Function Comparison: Context Sensitivity", fontsize=13, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate value function plots")
    parser.add_argument("--yes", action="store_true", help="Auto-deploy without asking")
    parser.add_argument("--no", action="store_true", help="Generate only, no deployment")
    args = parser.parse_args()

    print("=" * 60)
    print("Generating Value Function Plots for CCS 2026")
    print("=" * 60)

    # Load model sources
    print("\n1. Loading model sources...")
    sources = pd.read_csv(CROSS_MODEL_SOURCES)
    print(f"   Found {len(sources)} models")

    # Create output directory
    output_dir = SCRIPT_DIR / "value_function_plots"
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    print("\n2. Generating value function plots...")
    generated = []

    # Generate per-model plots
    for _, row in sources.iterrows():
        model = row['model']
        run_dir = row['run_dir']
        short_name = model.split('-')[0]

        output_path = output_dir / f"{short_name}_value_functions.png"

        if generate_value_function_plot(model, run_dir, output_path):
            generated.append((model, short_name, output_path))
            print(f"   Generated: {short_name}_value_functions.png")

    # Generate comparison example
    example_path = output_dir / "value_function_comparison.png"
    if generate_combined_example_plot(example_path):
        generated.append(("comparison", "comparison", example_path))
        print(f"   Generated: value_function_comparison.png")

    # Generate single-scenario plots used in main presentation slides
    single_scenario_plots = [
        ("deepseek", "political_liberal_conservative", "deepseek_vf_political.png"),
    ]
    for model_prefix, scenario, filename in single_scenario_plots:
        row = sources[sources['model'].str.startswith(model_prefix)].iloc[0]
        output_path = output_dir / filename
        if generate_single_scenario_plot(row['model'], row['run_dir'], scenario, output_path):
            generated.append((row['model'], filename.replace('.png', ''), output_path))
            print(f"   Generated: {filename}")

    print(f"\n   Generated {len(generated)} plots in {output_dir}")

    if args.no:
        print("\n--no flag: skipping deployment")
        return

    # Check hostname for deployment
    hostname = socket.gethostname()
    print(f"\nHostname: {hostname}")

    if hostname != TARGET_HOSTNAME:
        print(f"\nNot on target machine ({TARGET_HOSTNAME}).")
        print(f"Plots saved to: {output_dir}")
        return

    # Deploy
    if args.yes:
        do_deploy = True
    else:
        response = input(f"\nDeploy {len(generated)} plots to {PICS_DIR}? [y/N] ")
        do_deploy = response.lower() in ('y', 'yes')

    if do_deploy:
        print("\n3. Deploying to presentation folder...")
        PICS_DIR.mkdir(parents=True, exist_ok=True)
        for model, short_name, src_path in generated:
            if short_name == "comparison":
                dst_path = PICS_DIR / "value_function_comparison.png"
            else:
                dst_path = PICS_DIR / f"{short_name}_value_functions.png"
            shutil.copy2(src_path, dst_path)
            print(f"   Copied: {dst_path.name}")
        print(f"\nDone! Deployed to {PICS_DIR}")
    else:
        print("\nDeployment skipped.")


if __name__ == "__main__":
    main()
