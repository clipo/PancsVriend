import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import savgol_filter

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

BASE_DIR = Path(__file__).resolve().parent
EXP_DIR = BASE_DIR / "experiments"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Scenarios and labels (kept consistent with other analysis scripts)
scenarios = {
    'baseline': 'llm_baseline_20250703_101243',
    'ethnic_asian_hispanic': 'llm_ethnic_asian_hispanic_20250713_221759',
    'income_high_low': 'llm_income_high_low_20250724_154316',
    'economic_high_working' : "llm_economic_high_working_20250728_220134",
    'political_liberal_conservative': 'llm_political_liberal_conservative_20250724_154733',
    'race_white_black': 'llm_race_white_black_20250718_195455'
}

scenario_labels = {
    'baseline': 'Baseline (Control)',
    'ethnic_asian_hispanic': 'Ethnic (Asian/Hispanic)',
    'income_high_low': 'Economic (High/Low Income)',
    'economic_high_working': 'Economic (High/Working)',
    'political_liberal_conservative': 'Political (Liberal/Conservative)',
    'race_white_black': 'Racial (White/Black)'
}

scenario_colors = {
    'baseline': '#666666',  # Gray for neutral
    'race_white_black': '#e74c3c',  # Red for racial
    'ethnic_asian_hispanic': '#f39c12',  # Orange for ethnic
    'income_high_low': '#27ae60',  # Green for economic
    'economic_high_working': '#2ecc71',  # Light Green for economic
    'political_liberal_conservative': '#8e44ad'  # Purple for political
}

metrics = ['clusters', 'switch_rate', 'distance', 'mix_deviation', 'share', 'ghetto_rate']
metric_labels = {
    'clusters': 'Number of Clusters',
    'switch_rate': 'Switch Rate',
    'distance': 'Average Distance',
    'mix_deviation': 'Mix Deviation',
    'share': 'Segregation Share',
    'ghetto_rate': 'Ghetto Formation Rate'
}


def calculate_rate_of_change(series: np.ndarray, window: int = 7) -> np.ndarray:
    if series is None or len(series) == 0:
        return np.array([])
    if len(series) < window * 2 + 1:
        # Fall back to simple gradient if too short for Savitzky-Golay
        return np.gradient(series)
    smoothed = savgol_filter(series, window_length=window * 2 + 1, polyorder=3)
    return np.gradient(smoothed)


def load_data():
    data = {}
    for scenario_name, folder in scenarios.items():
        filepath = EXP_DIR / folder / 'metrics_history.csv'
        if filepath.exists():
            df = pd.read_csv(filepath)
            data[scenario_name] = df
    return data


def compute_convergence_steps(df: pd.DataFrame, metric: str) -> list:
    steps_list = []
    for run_id in df['run_id'].unique():
        run = df[df['run_id'] == run_id].sort_values('step')
        series = run[metric].values
        if len(series) < 2:
            continue
        initial_val = series[0]
        final_val = series[-1]
        if final_val == initial_val:
            continue
        target = initial_val + 0.9 * (final_val - initial_val)
        if final_val > initial_val:
            idx = np.argmax(series >= target)
            if series[idx] >= target:
                steps_list.append(int(run['step'].values[idx]))
        else:
            idx = np.argmax(series <= target)
            if series[idx] <= target:
                steps_list.append(int(run['step'].values[idx]))
    return steps_list


def make_metric_panel(metric: str, data_by_scenario: dict):
    # Prepare figure: 1x3 layout (all three panels in a single row)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    ax_ts, ax_box, ax_conv = axes

    # Panel A: Convergence time series with 95% CI
    for scenario in scenarios.keys():
        if scenario not in data_by_scenario:
            continue
        df = data_by_scenario[scenario]
        grouped = df.groupby('step')[metric]
        mean_values = grouped.mean()
        std_values = grouped.std()
        count_values = grouped.count().replace(0, np.nan)
        ci = 1.96 * std_values / np.sqrt(count_values)

        steps = mean_values.index
        ax_ts.plot(steps, mean_values.values, label=scenario_labels[scenario],
                   color=scenario_colors.get(scenario, None), linewidth=2.5, alpha=0.9)
        if ci.notna().any():
            ax_ts.fill_between(steps,
                               (mean_values - ci).values,
                               (mean_values + ci).values,
                               color=scenario_colors.get(scenario, None), alpha=0.15)
    ax_ts.set_title(f"A. {metric_labels[metric]} — Convergence Over Time", fontsize=13, fontweight='bold')
    ax_ts.set_xlabel('Simulation Step')
    ax_ts.set_ylabel(metric_labels[metric])
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend(loc='best', fontsize=9)

    # Panel B: Final value distribution (boxplot) across scenarios
    final_rows = []
    for scenario in scenarios.keys():
        if scenario not in data_by_scenario:
            continue
        df = data_by_scenario[scenario]
        # final step per run
        final_per_run = df.loc[df.groupby('run_id')['step'].idxmax(), ['run_id', metric]].copy()
        final_per_run['scenario'] = scenario_labels[scenario]
        final_rows.append(final_per_run)
    if final_rows:
        final_df = pd.concat(final_rows, ignore_index=True)
        # Map label -> color for consistent palette and suppress seaborn deprecation
        label_to_color = {scenario_labels[k]: scenario_colors.get(k, '#999999') for k in scenarios.keys() if k in data_by_scenario}
        sns.boxplot(data=final_df, x='scenario', y=metric, hue='scenario', ax=ax_box,
                    palette=label_to_color, legend=False)
        ax_box.set_title(f"B. {metric_labels[metric]} — Final Values Across Scenarios", fontsize=13, fontweight='bold')
        ax_box.set_xlabel('Scenario')
        ax_box.set_ylabel(metric_labels[metric])
        ax_box.tick_params(axis='x', rotation=30)
        ax_box.grid(True, axis='y', alpha=0.3)
    else:
        ax_box.text(0.5, 0.5, 'No data for boxplot', ha='center')
        ax_box.axis('off')

    # Panel C: Convergence speed (steps to reach 90% of final)
    conv_values = []
    conv_labels = []
    bar_colors = []
    for scenario in scenarios.keys():
        if scenario not in data_by_scenario:
            continue
        df = data_by_scenario[scenario]
        steps_list = compute_convergence_steps(df, metric)
        if len(steps_list) > 0:
            conv_values.append(float(np.mean(steps_list)))
            conv_labels.append(scenario_labels[scenario])
            bar_colors.append(scenario_colors.get(scenario, '#999999'))
    if conv_values:
        x = np.arange(len(conv_values))
        ax_conv.bar(x, conv_values, color=bar_colors)
        ax_conv.set_xticks(x)
        ax_conv.set_xticklabels(conv_labels, rotation=30, ha='right')
        ax_conv.set_ylabel('Steps to 90% Convergence')
        ax_conv.set_title(f"C. {metric_labels[metric]} — Convergence Speed", fontsize=13, fontweight='bold')
        ax_conv.grid(True, axis='y', alpha=0.3)
        # annotate bars
        for i, v in enumerate(conv_values):
            ax_conv.text(i, v + max(conv_values) * 0.02, f"{v:.0f}", ha='center', va='bottom', fontsize=9)
    else:
        ax_conv.text(0.5, 0.5, 'Insufficient data to compute convergence speed', ha='center')
        ax_conv.axis('off')

    plt.suptitle(f"Comprehensive Panel — {metric_labels[metric]}", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # Save
    out_png = EXP_DIR / f"metric_panel_{metric}.png"
    out_pdf = EXP_DIR / f"metric_panel_{metric}.pdf"
    fig.savefig(str(out_png), dpi=300, bbox_inches='tight')
    fig.savefig(str(out_pdf), bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def main():
    data = load_data()
    if not data:
        print("No experiment data found. Ensure metrics_history.csv exist under experiments/<scenario>/.")
        return
    for metric in metrics:
        make_metric_panel(metric, data)


if __name__ == "__main__":
    main()
