import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats

# Define scenarios and their paths
scenarios = {
    'baseline': 'llm_baseline_20250703_101243',
    'green_yellow': 'llm_green_yellow_20250912_072712',
    'ethnic_asian_hispanic': 'llm_ethnic_asian_hispanic_20250713_221759',
    'income_high_low': 'llm_income_high_low_20250724_154316',
    'economic_high_working' : "llm_economic_high_working_20250728_220134",
    'political_liberal_conservative': 'llm_political_liberal_conservative_20250724_154733',
    'race_white_black': 'llm_race_white_black_20250718_195455'
}

# Extract final metrics for each run
def get_final_metrics(df):
    """Extract the final metrics for each run (last step)"""
    return df.loc[df.groupby('run_id')['step'].idxmax()]

# Load and process all data
results = {}
for scenario_name, folder in scenarios.items():
    filepath = Path(f'experiments/{folder}/metrics_history.csv')
    if filepath.exists():
        df = pd.read_csv(filepath)
        final_metrics = get_final_metrics(df)
        results[scenario_name] = final_metrics
        print(f"\n{scenario_name.upper()}:")
        print(f"Number of runs: {len(final_metrics)}")
        print(f"Final metrics summary:")
        print(final_metrics[['clusters', 'switch_rate', 'distance', 'mix_deviation', 'share', 'ghetto_rate']].describe())

# Compile all results into a single dataframe for comparison
all_results = []
for scenario, df in results.items():
    df_copy = df.copy()
    df_copy['scenario'] = scenario
    all_results.append(df_copy)

combined_df = pd.concat(all_results, ignore_index=True)

# Statistical comparison across scenarios
print("\n\nSTATISTICAL COMPARISON ACROSS SCENARIOS")
print("=" * 60)

metrics = ['clusters', 'switch_rate', 'distance', 'mix_deviation', 'share', 'ghetto_rate']

for metric in metrics:
    print(f"\n{metric.upper()}:")
    
    # Prepare data for ANOVA
    groups = []
    for scenario in scenarios.keys():
        if scenario in results:
            groups.append(results[scenario][metric].values)
    
    # One-way ANOVA
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"  ANOVA: F={f_stat:.4f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            print("  Significant differences found between scenarios!")
            
    # Mean and std for each scenario
    for scenario in scenarios.keys():
        if scenario in results:
            mean_val = results[scenario][metric].mean()
            std_val = results[scenario][metric].std()
            print(f"  {scenario}: {mean_val:.4f} ± {std_val:.4f}")

# Save combined results
combined_df.to_csv('experiments/combined_final_metrics.csv', index=False)
print("\n\nSaved combined results to experiments/combined_final_metrics.csv")