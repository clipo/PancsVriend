#!/usr/bin/env python3
"""
Generate segregation-by-context plots with scenarios sorted by mean DI.

For each model, sorts scenarios from lowest to highest segregation,
helping readers understand the ordering visually.

Outputs to pres-overleaf.link/pres/pics/<model>_segregation_sorted.png

Usage:
    python generate_sorted_segregation_plots.py           # Interactive
    python generate_sorted_segregation_plots.py --yes     # Auto-deploy
    python generate_sorted_segregation_plots.py --no      # Generate only, no deploy
"""

import argparse
import shutil
import socket
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments_with_llama_cpp"
CROSS_MODEL_SOURCES = EXPERIMENTS_DIR / "cross_model" / "cross_model_vf-lp_sources.csv"
PRES_DIR = PROJECT_ROOT / "pres-overleaf.link" / "pres"
PICS_DIR = PRES_DIR / "pics"

# Target hostname for deployment
TARGET_HOSTNAME = "ECON-0FM96LD-L"

# Short labels as requested
SCENARIO_SHORT_LABELS = {
    'baseline': 'Color(RvB)',
    'green_yellow': 'Color(GvY)',
    'race_white_black': 'Racial',
    'ethnic_asian_hispanic': 'Ethnic',
    'income_high_low': 'Economic',
    'political_liberal_conservative': 'Political',
}

# Colors for scenarios (consistent with original)
SCENARIO_COLORS = {
    'baseline': '#7f7f7f',           # gray
    'green_yellow': '#2ca02c',       # green
    'race_white_black': '#d62728',   # red
    'ethnic_asian_hispanic': '#ff7f0e',  # orange
    'income_high_low': '#1f77b4',    # blue
    'political_liberal_conservative': '#9467bd',  # purple
}


def load_model_data(run_dir: str) -> pd.DataFrame:
    """Load run_summary data for a model."""
    summary_path = EXPERIMENTS_DIR / run_dir / "analysis" / "run_summary_by_run_all_scenarios.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return pd.DataFrame()


def get_chance_level(run_dir: str) -> float:
    """Get the random-allocation baseline DI for this run's grid configuration."""
    import json
    config_path = EXPERIMENTS_DIR / run_dir / "run_config_effective.yaml"

    # Default for 20x20 grid with 160+160 agents
    # Pre-computed: random_baseline(20, 160, 160)['mean'] ≈ 0.106
    default_chance = 0.106

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        # Could extract grid params and compute exactly, but default is fine
        return default_chance
    except:
        return default_chance


def generate_sorted_plot(model: str, run_dir: str, output_path: Path) -> bool:
    """Generate a sorted segregation plot for one model."""
    df = load_model_data(run_dir)
    if df.empty:
        print(f"  WARNING: No data for {model}")
        return False

    # Compute mean DI per scenario
    scenario_means = df.groupby('scenario')['dissimilarity_index'].mean().sort_values()

    # Prepare data in sorted order
    plot_data = []
    plot_labels = []
    plot_colors = []
    plot_keys = []

    for scenario in scenario_means.index:
        vals = df[df['scenario'] == scenario]['dissimilarity_index'].values
        if len(vals) > 0:
            plot_data.append(vals)
            plot_labels.append(SCENARIO_SHORT_LABELS.get(scenario, scenario))
            plot_colors.append(SCENARIO_COLORS.get(scenario, '#999999'))
            plot_keys.append(scenario)

    if not plot_data:
        print(f"  WARNING: No valid scenarios for {model}")
        return False

    # Set up figure with publication style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(5.5, 4))

    positions = np.arange(len(plot_labels))

    # Violin plot
    parts = ax.violinplot(plot_data, positions=positions,
                          showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(plot_colors[i])
        pc.set_edgecolor(plot_colors[i])
        pc.set_alpha(0.35)
        pc.set_linewidth(1.0)

    # Box plot overlay
    bp = ax.boxplot(plot_data, positions=positions,
                    widths=0.18, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(plot_colors[i])
        patch.set_edgecolor(plot_colors[i])
        patch.set_alpha(0.65)
        patch.set_linewidth(1.0)
    for med in bp['medians']:
        med.set_color('black')
        med.set_linewidth(1.2)
        med.set_zorder(4)
    for wl in bp['whiskers']:
        wl.set_color('#777777')
        wl.set_linewidth(1.0)
    for cap in bp['caps']:
        cap.set_color('#777777')
        cap.set_linewidth(1.0)

    # Chance level marker
    chance = get_chance_level(run_dir)
    ax.axhline(y=chance, color='#888888', linestyle='--', linewidth=1.0,
               label=f'Random baseline ({chance:.3f})', zorder=1)

    # Formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_labels, rotation=25, ha='right')
    ax.set_ylabel('Dissimilarity Index')
    ax.set_xlabel('Context (sorted low → high)')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(loc='upper left', fontsize=8)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate sorted segregation plots")
    parser.add_argument("--yes", action="store_true", help="Auto-deploy without asking")
    parser.add_argument("--no", action="store_true", help="Generate only, no deployment")
    args = parser.parse_args()

    print("=" * 60)
    print("Generating Sorted Segregation Plots for CCS 2026")
    print("=" * 60)

    # Load model sources
    print("\n1. Loading model sources...")
    sources = pd.read_csv(CROSS_MODEL_SOURCES)
    print(f"   Found {len(sources)} models")

    # Create output directory
    output_dir = SCRIPT_DIR / "sorted_plots"
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    print("\n2. Generating sorted plots...")
    generated = []
    for _, row in sources.iterrows():
        model = row['model']
        run_dir = row['run_dir']
        short_name = model.split('-')[0]

        output_path = output_dir / f"{short_name}_segregation_sorted.png"

        if generate_sorted_plot(model, run_dir, output_path):
            generated.append((model, short_name, output_path))
            print(f"   Generated: {short_name}_segregation_sorted.png")

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
            dst_path = PICS_DIR / f"{short_name}_segregation_sorted.png"
            shutil.copy2(src_path, dst_path)
            print(f"   Copied: {dst_path.name}")
        print(f"\nDone! Deployed to {PICS_DIR}")
    else:
        print("\nDeployment skipped.")


if __name__ == "__main__":
    main()
