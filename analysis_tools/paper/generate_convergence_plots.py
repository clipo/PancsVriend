#!/usr/bin/env python3
"""
Generate convergence pattern plots for presentation or paper.

Modifications from original:
1. Shorter labels (Color(GvY), Economic, etc.)
2. No "active runs" strip at bottom (cleaner for slides)
3. Highest DI scenarios drawn last (visible on top)
4. Economic, Political, Racial labels in boldface

Outputs to target folder (pres: pics/, paper: figs/)

Usage:
    python generate_convergence_plots.py                    # Interactive, deploy to pres (default)
    python generate_convergence_plots.py --target paper     # Deploy to paper/figs
    python generate_convergence_plots.py --yes              # Auto-deploy
    python generate_convergence_plots.py --no               # Generate only, no deploy
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


# Steps drawn (the x-axis ends here). --max-step changes it; 1000 was the
# simulation cap until 2026-11-23.
MAX_STEP = 1000


def load_dissimilarity_by_step(run_dir: str) -> pd.DataFrame:
    """Load dissimilarity_by_step data for a model (only the four columns the
    plot uses; scenario as a category, so a 10,000-step model's ~600M rows fit
    in memory — same parser, same values)."""
    # Try the analysis folder first
    dissim_path = EXPERIMENTS_DIR / run_dir / "analysis" / "dissimilarity_index" / "dissimilarity_by_step_all.csv.gz"
    if dissim_path.exists():
        return pd.read_csv(dissim_path,
                           usecols=['scenario', 'run_id', 'step', 'dissimilarity_index'],
                           dtype={'scenario': 'category'})
    return pd.DataFrame()


def step_stats_forward_filled(df: pd.DataFrame, metric: str):
    """Vectorised _step_stats_forward_filled_loop: the same numbers.

    The loop filtered the whole frame once per step and walked every run in
    Python — 10,000 steps x 10,000 runs per scenario at a 10,000-step cap,
    which does not finish. Here the runs x steps table is built once, each run
    carries its final value forward past its last step (exactly the loop's
    rule), and every step's mean and population SD (np.std, ddof=0) are taken
    over a contiguous vector of runs in run_id order, as the loop's 1-D
    np.mean / np.std were (checked bit for bit, 2026-11-23).
    """
    if df.empty or metric not in df.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    steps = sorted(df['step'].unique())
    table = df.pivot(index='run_id', columns='step', values=metric)
    table = table.reindex(columns=steps).sort_index()
    last = df.groupby('run_id')['step'].max().reindex(table.index).to_numpy()
    values = table.to_numpy()
    step_arr = np.asarray(steps)
    # past a run's last step: its final value (the loop's "run ended" case)
    final = values[np.arange(len(values)), np.searchsorted(step_arr, last)]
    ended = step_arr[None, :] > last[:, None]
    values = np.where(ended, final[:, None], values)
    rows = np.ascontiguousarray(values.T)          # one contiguous row per step
    means, cis = [], []
    for row in rows:
        row = row[~np.isnan(row)]                  # a run absent mid-way: skipped, as before
        if row.size:
            means.append(row.mean())
            cis.append(1.96 * row.std() / np.sqrt(len(row)))
        else:
            means.append(np.nan)
            cis.append(np.nan)
    return pd.Series(means, index=steps), pd.Series(cis, index=steps)


def _step_stats_forward_filled_loop(df: pd.DataFrame, metric: str):
    """
    Compute mean and CI at each step, forward-filling for runs that ended early.

    Returns: (mean_series, ci_series)
    """
    if df.empty or metric not in df.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Get max step per run
    max_steps = df.groupby('run_id')['step'].max()
    global_max = int(df['step'].max())

    # For each step, compute stats using forward-filled values
    steps = sorted(df['step'].unique())
    means = []
    cis = []

    # Get final value for each run (for forward filling)
    final_vals = df.loc[df.groupby('run_id')['step'].idxmax(), ['run_id', metric]].set_index('run_id')[metric]

    for step in steps:
        # Get values at this step
        step_data = df[df['step'] == step].set_index('run_id')[metric]

        # For runs that ended before this step, use their final value
        all_runs = final_vals.index
        values = []
        for run_id in all_runs:
            if run_id in step_data.index:
                values.append(step_data[run_id])
            elif max_steps[run_id] < step:
                # Run ended, use final value
                values.append(final_vals[run_id])

        if values:
            arr = np.array(values)
            means.append(arr.mean())
            # 95% CI
            cis.append(1.96 * arr.std() / np.sqrt(len(arr)))
        else:
            means.append(np.nan)
            cis.append(np.nan)

    return pd.Series(means, index=steps), pd.Series(cis, index=steps)


def generate_convergence_plot(model: str, run_dir: str, output_path: Path) -> bool:
    """Generate a clean convergence plot for one model."""
    df = load_dissimilarity_by_step(run_dir)
    if df.empty:
        print(f"  WARNING: No dissimilarity_by_step data for {model}")
        return False

    # Compute final mean DI per scenario for sorting (highest drawn last = on top)
    scenario_final_di = {}
    for scenario in df['scenario'].unique():
        scenario_df = df[df['scenario'] == scenario]
        final_step = scenario_df['step'].max()
        final_vals = scenario_df[scenario_df['step'] == final_step]['dissimilarity_index']
        scenario_final_di[scenario] = final_vals.mean()

    # Sort scenarios by final DI (ascending, so highest is drawn last)
    sorted_scenarios = sorted(scenario_final_di.keys(), key=lambda s: scenario_final_di[s])

    # Set up figure
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
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Plot each scenario (sorted so highest DI is drawn last = on top)
    for scenario in sorted_scenarios:
        scenario_df = df[df['scenario'] == scenario]
        if scenario_df.empty:
            continue

        mean_values, ci = step_stats_forward_filled(scenario_df, 'dissimilarity_index')
        if mean_values.empty:
            continue

        # Limit to MAX_STEP steps
        max_step = min(MAX_STEP, mean_values.index.max())
        steps = mean_values.index[mean_values.index <= max_step]

        color = SCENARIO_COLORS.get(scenario, '#999999')
        label = SCENARIO_SHORT_LABELS.get(scenario, scenario)

        # Plot mean line
        ax.plot(steps, mean_values[steps],
                label=label,
                linewidth=2.0, alpha=0.95, color=color)

        # Add confidence interval
        ax.fill_between(steps,
                        mean_values[steps] - ci[steps],
                        mean_values[steps] + ci[steps],
                        alpha=0.15, color=color)

    # Formatting
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Dissimilarity Index')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(loc='lower right', fontsize=8)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return True


def main():
    global deploy_dir, EXPERIMENTS_DIR, CROSS_MODEL_SOURCES, MAX_STEP

    parser = argparse.ArgumentParser(description="Generate clean convergence plots")
    parser.add_argument("--target", choices=["pres", "paper"], default="pres",
                        help="Target: 'pres' (presentation) or 'paper' (default: pres)")
    parser.add_argument("--yes", action="store_true", help="Auto-deploy without asking")
    parser.add_argument("--no", action="store_true", help="Generate only, no deployment")
    parser.add_argument("--experiments-dir", type=Path, default=None,
                        help="Run root holding the model run folders and cross_model/ "
                             "(default: experiments_with_llama_cpp)")
    parser.add_argument("--max-step", type=int, default=None,
                        help=f"Last step drawn (default {MAX_STEP})")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where the PNGs go (default: analysis_tools/paper/convergence_plots)")
    args = parser.parse_args()

    if args.experiments_dir is not None:
        EXPERIMENTS_DIR = args.experiments_dir.resolve()
        CROSS_MODEL_SOURCES = EXPERIMENTS_DIR / "cross_model" / "cross_model_vf-lp_sources.csv"
    if args.max_step is not None:
        MAX_STEP = args.max_step

    target_config = TARGETS[args.target]
    deploy_dir = target_config["output_dir"]
    target_desc = target_config["description"]

    print("=" * 60)
    print(f"Generating Clean Convergence Plots for {target_desc}")
    print(f"Target: {deploy_dir}")
    print("=" * 60)

    # Load model sources
    print("\n1. Loading model sources...")
    sources = pd.read_csv(CROSS_MODEL_SOURCES)
    print(f"   Found {len(sources)} models")

    # Create output directory
    output_dir = args.output_dir or (SCRIPT_DIR / "convergence_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\n2. Generating convergence plots...")
    generated = []
    for _, row in sources.iterrows():
        model = row['model']
        run_dir = row['run_dir']
        short_name = model.split('-')[0]

        output_path = output_dir / f"{short_name}_convergence_clean.png"

        if generate_convergence_plot(model, run_dir, output_path):
            generated.append((model, short_name, output_path))
            print(f"   Generated: {short_name}_convergence_clean.png")

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
        response = input(f"\nDeploy {len(generated)} plots to {deploy_dir}? [y/N] ")
        do_deploy = response.lower() in ('y', 'yes')

    if do_deploy:
        print(f"\n3. Deploying to {target_desc} folder...")
        deploy_dir.mkdir(parents=True, exist_ok=True)
        for model, short_name, src_path in generated:
            dst_path = deploy_dir / f"{short_name}_convergence_clean.png"
            shutil.copy2(src_path, dst_path)
            print(f"   Copied: {dst_path.name}")
        print(f"\nDone! Deployed to {deploy_dir}")
    else:
        print("\nDeployment skipped.")


if __name__ == "__main__":
    main()
