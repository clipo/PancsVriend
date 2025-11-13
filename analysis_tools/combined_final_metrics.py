import pandas as pd
from pathlib import Path
from scipy import stats
import argparse
from experiment_list_for_analysis import SCENARIOS as scenarios
from analysis_tools.output_paths import get_reports_dir

# Optional import for fallback reconstruction if metrics files are missing
try:
    from base_simulation import Simulation  # Provides load_and_analyze_results
except Exception:
    Simulation = None  # Fallback if not available; we'll warn instead

# Scenarios are now imported from centralized module

# Extract final metrics for each run
def get_final_metrics(df):
    """Extract the final metrics for each run (last step)"""
    return df.loc[df.groupby('run_id')['step'].idxmax()]

def process_scenarios(recompute: bool = True):
    """Process scenarios, optionally forcing recomputation of metrics.

    Args:
        recompute (bool): If True, attempt to (re)compute metrics & convergence data
                          even when existing files are present.
    Returns:
        pd.DataFrame: Combined final metrics dataframe with 'scenario' column.
    """
    results = {}
    for scenario_name, folder in scenarios.items():
        exp_dir = Path('experiments') / folder
        metrics_path = exp_dir / 'metrics_history.csv'
        step_stats_path = exp_dir / 'step_statistics.csv'
        convergence_path = exp_dir / 'convergence_summary.csv'

        all_exist = metrics_path.exists() and step_stats_path.exists() and convergence_path.exists()
        need_recompute = recompute or not all_exist
        if need_recompute:
            print(
                f"[INFO] (Re)computing metrics for '{scenario_name}' "
                f"(force={recompute}, metrics={metrics_path.exists()}, step_stats={step_stats_path.exists()}, convergence={convergence_path.exists()})"
            )
            if Simulation:
                move_logs_dir = exp_dir / 'move_logs'
                if move_logs_dir.exists():
                    try:
                        Simulation.load_and_analyze_results(str(exp_dir), force_recompute=True)
                    except Exception as e:
                        print(f"[WARN] Failed to (re)compute metrics for {scenario_name}: {e}")
                else:
                    if not metrics_path.exists():
                        print(f"[WARN] No move_logs directory for {scenario_name}; cannot construct metrics.")
            else:
                if not metrics_path.exists():
                    print("[WARN] Simulation class unavailable; cannot reconstruct missing metrics.")

        # Load metrics if present after (re)compute attempt
        if metrics_path.exists():
            try:
                df = pd.read_csv(metrics_path)
                if 'run_id' not in df.columns or 'step' not in df.columns:
                    raise ValueError("metrics_history.csv missing required columns 'run_id' or 'step'")
                final_metrics = get_final_metrics(df)
                results[scenario_name] = final_metrics
                print(f"\n{scenario_name.upper()}:")
                print(f"Number of runs: {len(final_metrics)}")
                print("Final metrics summary:")
                print(final_metrics[['clusters', 'switch_rate', 'distance', 'mix_deviation', 'share', 'ghetto_rate']].describe())
            except Exception as e:
                print(f"[ERROR] Could not load metrics for {scenario_name}: {e}")
        else:
            print(f"[INFO] Skipping scenario '{scenario_name}' – metrics unavailable after recompute attempt.")

    # Compile combined
    all_results = []
    for scenario, df in results.items():
        df_copy = df.copy()
        df_copy['scenario'] = scenario
        all_results.append(df_copy)

    if not all_results:
        raise SystemExit("No scenarios produced final metrics; aborting.")

    combined_df = pd.concat(all_results, ignore_index=True)

    # Statistical comparison across scenarios
    print("\n\nSTATISTICAL COMPARISON ACROSS SCENARIOS")
    print("=" * 60)

    metrics_list = ['clusters', 'switch_rate', 'distance', 'mix_deviation', 'share', 'ghetto_rate']
    for metric in metrics_list:
        print(f"\n{metric.upper()}:")
        groups = [results[s][metric].values for s in scenarios.keys() if s in results]
        if len(groups) >= 2:
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                print(f"  ANOVA: F={f_stat:.4f}, p={p_value:.4f}")
                if p_value < 0.05:
                    print("  Significant differences found between scenarios!")
            except Exception as e:
                print(f"  [WARN] ANOVA failed for {metric}: {e}")
        for scenario in scenarios.keys():
            if scenario in results:
                mean_val = results[scenario][metric].mean()
                std_val = results[scenario][metric].std()
                print(f"  {scenario}: {mean_val:.4f} ± {std_val:.4f}")

    # Persist combined
    out_path = get_reports_dir() / 'combined_final_metrics.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(out_path, index=False)
    print(f"\n\nSaved combined results to {out_path}")
    return combined_df


def parse_args():
    parser = argparse.ArgumentParser(description="Combine final metrics across scenarios.")
    parser.add_argument('--no-recompute', action='store_true', help='Skip forced recomputation; only use existing metrics files.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    recompute = not args.no_recompute  # default True
    print(f"[INFO] Starting processing with recompute={recompute}")
    process_scenarios(recompute=recompute)