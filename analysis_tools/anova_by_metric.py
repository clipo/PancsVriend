"""One-way ANOVA across scenarios for every metric, from run_summary_by_run_all_scenarios.csv.

    python analysis_tools/anova_by_metric.py            # after the run_summary step

Reads <reports>/run_summary_by_run_all_scenarios.csv — the per-model roll-up of every
experiment's run_summary.csv (one row per run, final-step metrics) — groups
by `scenario_key` and writes anova_results_by_metric.csv / .md plus a
per-scenario describe() to stdout.

This is what is left of combined_final_metrics.py (removed 2026-09-05). That
module also wrote combined_final_metrics.csv, the same rows as
run_summary_by_run_all_scenarios.csv under a different column name, and forced a full
rebuild of every experiment's metrics_history on every pass; the rebuild is
now the repair_stored_metrics step and the roll-up is run_summary's.
"""

from pathlib import Path

import pandas as pd
from scipy import stats

from analysis_tools.output_paths import get_reports_dir

METRICS = ['clusters', 'switch_rate', 'distance', 'mix_deviation', 'share', 'ghetto_rate',
           'dissimilarity_index']
SUMMARY_FILENAME = 'run_summary_by_run_all_scenarios.csv'


def load_final_metrics(reports_dir=None):
    """The roll-up with the analysis scenario key as `scenario`."""
    reports_dir = Path(reports_dir or get_reports_dir())
    path = reports_dir / SUMMARY_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (run the run_summary step first)")
    df = pd.read_csv(path, float_precision='round_trip')
    if 'scenario_key' in df.columns:
        df = df.drop(columns=['scenario']).rename(columns={'scenario_key': 'scenario'})
    return df


def anova_by_metric(scenario_order=None, reports_dir=None):
    """Write the ANOVA table; returns the roll-up DataFrame it was computed from."""
    reports_dir = Path(reports_dir or get_reports_dir())
    df = load_final_metrics(reports_dir)
    present = set(df['scenario'])
    scenarios = [s for s in (scenario_order or list(dict.fromkeys(df['scenario']))) if s in present]
    groups_by_scenario = {s: df[df['scenario'] == s] for s in scenarios}
    metrics = [m for m in METRICS if m in df.columns]

    for scenario in scenarios:
        print(f"\n{scenario.upper()}:")
        print(f"Number of runs: {len(groups_by_scenario[scenario])}")
        print("Final metrics summary:")
        print(groups_by_scenario[scenario][metrics].describe())

    print("\n\nSTATISTICAL COMPARISON ACROSS SCENARIOS")
    print("=" * 60)
    anova_rows = []
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        groups = [groups_by_scenario[s][metric].values for s in scenarios]
        if len(groups) >= 2:
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                print(f"  ANOVA: F={f_stat:.4f}, p={p_value:.4f}")
                anova_rows.append({'metric': metric, 'n_groups': len(groups),
                                   'f_statistic': float(f_stat), 'p_value': float(p_value),
                                   'significant_p_lt_0_05': bool(p_value < 0.05),
                                   'status': 'ok', 'error': ''})
                if p_value < 0.05:
                    print("  Significant differences found between scenarios!")
            except Exception as e:
                print(f"  [WARN] ANOVA failed for {metric}: {e}")
                anova_rows.append({'metric': metric, 'n_groups': len(groups),
                                   'f_statistic': None, 'p_value': None,
                                   'significant_p_lt_0_05': None, 'status': 'failed',
                                   'error': str(e)})
        else:
            anova_rows.append({'metric': metric, 'n_groups': len(groups),
                               'f_statistic': None, 'p_value': None,
                               'significant_p_lt_0_05': None, 'status': 'skipped',
                               'error': 'Need at least 2 scenario groups for ANOVA'})
        for scenario in scenarios:
            values = groups_by_scenario[scenario][metric]
            print(f"  {scenario}: {values.mean():.4f} ± {values.std():.4f}")

    reports_dir.mkdir(parents=True, exist_ok=True)
    anova_out_path = reports_dir / 'anova_results_by_metric.csv'
    pd.DataFrame(anova_rows).to_csv(anova_out_path, index=False)

    md_lines = ['# ANOVA Results by Metric', '',
                '| Metric | Groups | F-statistic | p-value | Significant (p<0.05) | Status |',
                '|---|---:|---:|---:|:---:|---|']
    for row in anova_rows:
        f_stat = '' if row['f_statistic'] is None else f"{row['f_statistic']:.6f}"
        p_val = '' if row['p_value'] is None else f"{row['p_value']:.6f}"
        sig = '' if row['significant_p_lt_0_05'] is None else ('*' if row['significant_p_lt_0_05'] else '')
        md_lines.append(f"| {row['metric']} | {row['n_groups']} | {f_stat} | {p_val} | {sig} | {row['status']} |")
    (reports_dir / 'anova_results_by_metric.md').write_text('\n'.join(md_lines), encoding='utf-8')
    print(f"\n\nSaved ANOVA results to {anova_out_path}")
    return df


if __name__ == '__main__':
    anova_by_metric()
