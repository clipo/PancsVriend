#!/usr/bin/env python3
"""
Compute statistically rigorous scenario orderings for each LLM model.

This script generates the "Key Results" table for the CCS 2026 presentation,
comparing LLM segregation patterns to empirical data.

Methodology:
- Paired t-tests on per-run dissimilarity index values between scenario pairs
- Holm-Bonferroni correction for multiple comparisons within each model
- Classification: < (p<0.01), ≤ (p<0.05), ≈ (p≥0.05)

Outputs:
- pairwise_tests_all.csv: Full pairwise test results
- ordering_3way.csv: Income vs Political vs Racial orderings
- ordering_6way.csv: All scenario orderings
- key_results_table.tex: LaTeX code for presentation
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from itertools import combinations
import warnings

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments_with_llama_cpp"
CROSS_MODEL_SOURCES = EXPERIMENTS_DIR / "cross_model" / "cross_model_vf-lp_sources.csv"

# Significance thresholds
P_SIGNIFICANT = 0.01  # < for p < 0.01
P_MARGINAL = 0.05     # ≤ for 0.01 ≤ p < 0.05, ≈ for p ≥ 0.05

# Scenario mappings for 3-way comparison
SCENARIO_3WAY = {
    'income_high_low': 'Economic',
    'political_liberal_conservative': 'Political',
    'race_white_black': 'Racial'
}

# Display names for all scenarios
SCENARIO_DISPLAY = {
    'baseline': 'Baseline',
    'race_white_black': 'Racial',
    'ethnic_asian_hispanic': 'Ethnic',
    'income_high_low': 'Economic',
    'political_liberal_conservative': 'Political',
    'green_yellow': 'Green/Yellow'
}

# Short names for 6-way table
SCENARIO_SHORT = {
    'baseline': 'Base',
    'race_white_black': 'Race',
    'ethnic_asian_hispanic': 'Ethn',
    'income_high_low': 'Econ',
    'political_liberal_conservative': 'Polit',
    'green_yellow': 'G/Y'
}


def load_model_data():
    """Load run_summary data for all models from cross_model_vf-lp_sources.csv."""
    sources = pd.read_csv(CROSS_MODEL_SOURCES)
    model_data = {}

    for _, row in sources.iterrows():
        model = row['model']
        run_dir = row['run_dir']
        summary_path = EXPERIMENTS_DIR / run_dir / "analysis" / "run_summary_by_run_all_scenarios.csv"

        if summary_path.exists():
            df = pd.read_csv(summary_path)
            model_data[model] = df
            print(f"Loaded {model}: {len(df)} rows")
        else:
            print(f"WARNING: Missing {summary_path}")

    return model_data


def paired_ttest(df, scenario_a, scenario_b, metric='dissimilarity_index'):
    """
    Compute paired t-test between two scenarios.

    Pairs runs by run_id, computes differences, tests if mean diff != 0.
    Returns t-statistic, p-value, mean values, and mean difference.
    """
    df_a = df[df['scenario'] == scenario_a][['run_id', metric]].set_index('run_id')
    df_b = df[df['scenario'] == scenario_b][['run_id', metric]].set_index('run_id')

    # Inner join to get paired observations
    paired = df_a.join(df_b, lsuffix='_a', rsuffix='_b', how='inner')

    if len(paired) == 0:
        return None

    values_a = paired[f'{metric}_a'].values
    values_b = paired[f'{metric}_b'].values

    # Paired t-test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_stat, p_value = stats.ttest_rel(values_a, values_b)

    return {
        'n_pairs': len(paired),
        'mean_a': values_a.mean(),
        'mean_b': values_b.mean(),
        'mean_diff': (values_a - values_b).mean(),
        'std_diff': (values_a - values_b).std(),
        't_stat': t_stat,
        'p_raw': p_value
    }


def holm_correction(p_values):
    """Apply Holm-Bonferroni correction to a list of p-values."""
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Holm correction: p_adj[i] = max(p_adj[i-1], p[i] * (n - i))
    adjusted = np.zeros(n)
    for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        corrected = p * (n - i)
        if i == 0:
            adjusted[idx] = min(corrected, 1.0)
        else:
            # Must be at least as large as previous
            adjusted[idx] = min(max(corrected, adjusted[sorted_indices[i-1]]), 1.0)

    return adjusted.tolist()


def classify_comparison(p_holm, mean_a, mean_b):
    """
    Classify a comparison based on Holm-corrected p-value.

    Returns (symbol, direction) where:
    - symbol: '<', '≤', or '≈'
    - direction: 1 if a < b, -1 if a > b, 0 if equal
    """
    if mean_a < mean_b:
        direction = 1  # a < b
        if p_holm < P_SIGNIFICANT:
            return '<', direction
        elif p_holm < P_MARGINAL:
            return '≤', direction
        else:
            return '≈', 0
    else:
        direction = -1  # a > b
        if p_holm < P_SIGNIFICANT:
            return '>', direction
        elif p_holm < P_MARGINAL:
            return '≥', direction
        else:
            return '≈', 0


def compute_all_pairwise_tests(model_data, scenarios=None):
    """
    Compute pairwise t-tests for all scenario pairs for each model.

    Returns DataFrame with all test results.
    """
    results = []

    for model, df in model_data.items():
        available_scenarios = df['scenario'].unique()
        if scenarios is None:
            test_scenarios = [s for s in available_scenarios if s in SCENARIO_DISPLAY]
        else:
            test_scenarios = [s for s in scenarios if s in available_scenarios]

        # Compute all pairwise tests
        pair_results = []
        for s_a, s_b in combinations(test_scenarios, 2):
            result = paired_ttest(df, s_a, s_b)
            if result:
                pair_results.append({
                    'model': model,
                    'scenario_a': s_a,
                    'scenario_b': s_b,
                    **result
                })

        # Apply Holm correction within model
        if pair_results:
            p_raw = [r['p_raw'] for r in pair_results]
            p_holm = holm_correction(p_raw)

            for i, r in enumerate(pair_results):
                r['p_holm'] = p_holm[i]
                symbol, direction = classify_comparison(p_holm[i], r['mean_a'], r['mean_b'])
                r['symbol'] = symbol
                r['direction'] = direction
                results.append(r)

    return pd.DataFrame(results)


def build_ordering_string(pairwise_df, model, scenarios, display_names):
    """
    Build an ordering string like "Economic < Political ≈ Racial" for a model.

    Uses mean values to determine order, then pairwise tests for symbols.
    Since we sort by mean (lowest first), adjacent pairs always have
    mean[i] < mean[i+1], so we just need to determine significance level.
    """
    model_df = pairwise_df[pairwise_df['model'] == model]

    # Get mean DI for each scenario
    scenario_means = {}
    for s in scenarios:
        rows = model_df[(model_df['scenario_a'] == s) | (model_df['scenario_b'] == s)]
        if len(rows) > 0:
            # Extract mean from any row containing this scenario
            for _, row in rows.iterrows():
                if row['scenario_a'] == s:
                    scenario_means[s] = row['mean_a']
                    break
                elif row['scenario_b'] == s:
                    scenario_means[s] = row['mean_b']
                    break

    if len(scenario_means) < len(scenarios):
        return "Incomplete data"

    # Sort scenarios by mean DI (lowest first)
    sorted_scenarios = sorted(scenario_means.keys(), key=lambda s: scenario_means[s])

    # Build ordering string
    parts = [display_names[sorted_scenarios[0]]]

    for i in range(len(sorted_scenarios) - 1):
        s_low = sorted_scenarios[i]      # lower mean
        s_high = sorted_scenarios[i + 1]  # higher mean

        # Find the comparison (might be stored in either order)
        row = model_df[
            ((model_df['scenario_a'] == s_low) & (model_df['scenario_b'] == s_high)) |
            ((model_df['scenario_a'] == s_high) & (model_df['scenario_b'] == s_low))
        ]

        if len(row) == 0:
            symbol = '?'
        else:
            row = row.iloc[0]
            p_holm = row['p_holm']

            # We know s_low has lower mean than s_high (from sorting)
            # So we just need to check significance level
            if p_holm < P_SIGNIFICANT:
                symbol = '<'
            elif p_holm < P_MARGINAL:
                symbol = '≤'
            else:
                symbol = '≈'

        parts.append(f" {symbol} ")
        parts.append(display_names[sorted_scenarios[i + 1]])

    return ''.join(parts)


def check_empirical_match(ordering_str):
    """
    Check if each comparison in the ordering matches empirical (including transitive).

    Empirical: Economic < Political < Racial (ranks: Economic=1, Political=2, Racial=3)
    A comparison A < B matches empirical if empirical_rank(A) < empirical_rank(B).

    Returns list of positions where adjacent comparisons match empirical.
    """
    # Empirical ranks (lower = less segregation)
    EMPIRICAL_RANK = {
        'Economic': 1,
        'Political': 2,
        'Racial': 3
    }

    import re
    # Split on comparison symbols, keeping them
    parts = re.split(r'\s*([<≤≈>≥])\s*', ordering_str)

    scenarios = [p.strip() for p in parts[::2] if p.strip()]
    symbols = [p.strip() for p in parts[1::2] if p.strip()]

    matches = []
    for i, (s1, s2, sym) in enumerate(zip(scenarios[:-1], scenarios[1:], symbols)):
        # s1 < s2 means s1 has lower segregation than s2
        # Check if this is true in empirical
        if s1 in EMPIRICAL_RANK and s2 in EMPIRICAL_RANK:
            empirical_match = EMPIRICAL_RANK[s1] < EMPIRICAL_RANK[s2]
            if empirical_match and sym in ['<', '≤']:
                matches.append(i)  # Position of this comparison

    return matches


def count_all_pairwise_matches(ordering_str):
    """
    Count ALL pairwise matches to empirical, including transitive comparisons.

    For 3-way ordering [A, B, C], checks all 3 pairs: A<B, B<C, A<C.
    Empirical: Economic < Political < Racial

    Returns count of matching pairs (0-3 for 3-way comparison).
    """
    EMPIRICAL_RANK = {
        'Economic': 1,
        'Political': 2,
        'Racial': 3
    }

    import re
    parts = re.split(r'\s*([<≤≈>≥])\s*', ordering_str)
    scenarios = [p.strip() for p in parts[::2] if p.strip()]

    # Check if all scenarios are in empirical ranking
    if not all(s in EMPIRICAL_RANK for s in scenarios):
        return 0

    # Count all pairwise matches
    # The ordering [A, B, C] implies A < B < C (by mean values)
    # Check each pair against empirical
    count = 0
    n = len(scenarios)
    for i in range(n):
        for j in range(i + 1, n):
            s_low, s_high = scenarios[i], scenarios[j]
            # s_low < s_high in LLM ordering
            # Check if empirical agrees
            if EMPIRICAL_RANK[s_low] < EMPIRICAL_RANK[s_high]:
                count += 1

    return count


def check_grouping_pattern(ordering_str):
    """
    Check if ordering should use parentheses grouping.

    For 3-way ordering [A, B, C]:
    - Case 1: A matches all (A<B and A<C) but B<C doesn't → "A <* (B < C)"
    - Case 2: C matches all (A<C and B<C) but A<B doesn't → "(A < B) <* C"

    Returns: (pattern_type, grouped_scenarios, ungrouped_scenario, symbol, internal_symbol)
             where pattern_type is 'first_matches', 'last_matches', or None
    """
    EMPIRICAL_RANK = {
        'Economic': 1,
        'Political': 2,
        'Racial': 3
    }

    import re
    parts = re.split(r'\s*([<≤≈>≥])\s*', ordering_str)
    scenarios = [p.strip() for p in parts[::2] if p.strip()]
    symbols = [p.strip() for p in parts[1::2] if p.strip()]

    # Only handle 3-way case for now
    if len(scenarios) != 3:
        return (None, None, None, None, None)

    A, B, C = scenarios
    sym_AB, sym_BC = symbols[0], symbols[1]

    # Check if all scenarios are in empirical ranking
    if not all(s in EMPIRICAL_RANK for s in scenarios):
        return (None, None, None, None, None)

    # Check each pairwise comparison against empirical
    A_lt_B_matches = EMPIRICAL_RANK[A] < EMPIRICAL_RANK[B] and sym_AB in ['<', '≤']
    A_lt_C_matches = EMPIRICAL_RANK[A] < EMPIRICAL_RANK[C]  # Always significant if A is lowest
    B_lt_C_matches = EMPIRICAL_RANK[B] < EMPIRICAL_RANK[C] and sym_BC in ['<', '≤']

    # Case 1: A matches all but B<C doesn't → "A <* (B < C)"
    if A_lt_B_matches and A_lt_C_matches and not B_lt_C_matches:
        return ('first_matches', A, (B, C), sym_AB, sym_BC)

    # Case 2: C matches all but A<B doesn't → "(A < B) <* C"
    if B_lt_C_matches and A_lt_C_matches and not A_lt_B_matches:
        return ('last_matches', (A, B), C, sym_BC, sym_AB)

    return (None, None, None, None, None)


def generate_latex_table(ordering_3way_df):
    """Generate LaTeX code for the Key Results table.

    Uses grouping notation for cases where the first element matches empirical
    against all others, but internal ordering doesn't match.
    E.g., Olmo: "Economic <* (Racial < Political)" shows Economic lowest matches
    empirical, but Racial < Political doesn't match (should be Political < Racial).
    """
    from datetime import datetime
    import re

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_models = len(ordering_3way_df)
    models_list = ", ".join(ordering_3way_df['model'].tolist())

    header = f"""% =============================================================================
% Key Results Table - Auto-generated
% =============================================================================
% Source: analysis_tools/ccs2026_presentation/compute_scenario_orderings.py
% Generated: {timestamp}
% Models: {n_models} ({models_list})
% Data: experiments_with_llama_cpp/cross_model/cross_model_vf-lp_sources.csv
% Methodology: Paired t-tests, Holm-corrected, p<0.01 for '<', p<0.05 for '≤'
% Regenerate: python analysis_tools/ccs2026_presentation/generate_and_deploy_tables.py --yes
% =============================================================================

"""

    latex = header + r"""\begin{frame}[t]{Key Result. Segregation Ordering: Empirical vs. LLMs}


\begin{center}
\small
\begin{tabular}{ll}
\toprule
\textbf{Source} & \textbf{Ordering (lowest $\rightarrow$ highest segregation)} \\
\midrule
\textbf{Empirical} & Economic $<^{\star}$ Political $<^{\star}$ Racial \\
\midrule
"""

    for _, row in ordering_3way_df.iterrows():
        model = row['model']
        ordering = row['ordering']
        match_positions = row.get('empirical_matches', [])

        # Check if we should use grouping notation
        pattern_type, grouped, ungrouped, outer_sym, inner_sym = check_grouping_pattern(ordering)

        if pattern_type == 'first_matches':
            # Format as "A <* (B < C)"
            A = grouped
            B, C = ungrouped

            # Convert symbols to LaTeX
            if outer_sym == '<':
                latex_outer = '$<^{\\star}$'
            elif outer_sym == '≤':
                latex_outer = '$\\leq^{\\star}$'
            else:
                latex_outer = f'${outer_sym}$'

            if inner_sym == '<':
                latex_inner = '$<$'
            elif inner_sym == '≤':
                latex_inner = '$\\leq$'
            elif inner_sym == '≈':
                latex_inner = '$\\approx$'
            else:
                latex_inner = f'${inner_sym}$'

            latex_ordering = f"{A} {latex_outer} ({B} {latex_inner} {C})"

        elif pattern_type == 'last_matches':
            # Format as "(A < B) <* C"
            A, B = grouped
            C = ungrouped

            # Convert symbols to LaTeX
            if outer_sym == '<':
                latex_outer = '$<^{\\star}$'
            elif outer_sym == '≤':
                latex_outer = '$\\leq^{\\star}$'
            else:
                latex_outer = f'${outer_sym}$'

            if inner_sym == '<':
                latex_inner = '$<$'
            elif inner_sym == '≤':
                latex_inner = '$\\leq$'
            elif inner_sym == '≈':
                latex_inner = '$\\approx$'
            else:
                latex_inner = f'${inner_sym}$'

            latex_ordering = f"({A} {latex_inner} {B}) {latex_outer} {C}"

        else:
            # Standard format with stars at matched positions
            parts = re.split(r'\s*([<≤≈>≥])\s*', ordering)
            scenarios = [p.strip() for p in parts[::2] if p.strip()]
            symbols = [p.strip() for p in parts[1::2] if p.strip()]

            latex_parts = [scenarios[0]]
            for i, (sym, next_scenario) in enumerate(zip(symbols, scenarios[1:])):
                # Convert symbol to LaTeX
                if sym == '<':
                    latex_sym = '$<$'
                elif sym == '≤':
                    latex_sym = '$\\leq$'
                elif sym == '≈':
                    latex_sym = '$\\approx$'
                elif sym == '>':
                    latex_sym = '$>$'
                elif sym == '≥':
                    latex_sym = '$\\geq$'
                else:
                    latex_sym = sym

                # Add star if this position matches empirical
                if i in match_positions:
                    if sym == '<':
                        latex_sym = '$<^{\\star}$'
                    elif sym == '≤':
                        latex_sym = '$\\leq^{\\star}$'

                latex_parts.append(f' {latex_sym} ')
                latex_parts.append(next_scenario)

            latex_ordering = ''.join(latex_parts)

        # Format model name
        model_display = model.replace('-', ' ').replace('_', ' ').title()

        latex += f"{model_display} & {latex_ordering} \\\\\n"

    latex += r"""\bottomrule
\end{tabular} \\
{$<^{\star}$ indicates match to the empirical order.}
\end{center}

\smallskip

\textbf{Pattern:} Most LLMs show Racial segregation as \emph{lowest} (opposite of empirical). Most correctly order Economic $<$ Political, but none show Political $<$ Racial.

\end{frame}
"""

    return latex


def main():
    print("=" * 60)
    print("Computing scenario orderings for CCS 2026 presentation")
    print("=" * 60)

    # Load data
    print("\n1. Loading model data...")
    model_data = load_model_data()

    if not model_data:
        print("ERROR: No model data loaded")
        return

    # Compute all pairwise tests
    print("\n2. Computing pairwise t-tests...")
    all_scenarios = list(SCENARIO_DISPLAY.keys())
    pairwise_all = compute_all_pairwise_tests(model_data, scenarios=all_scenarios)
    print(f"   Computed {len(pairwise_all)} pairwise comparisons")

    # Save full pairwise results
    pairwise_path = SCRIPT_DIR / "pairwise_tests_all.csv"
    pairwise_all.to_csv(pairwise_path, index=False)
    print(f"   Saved: {pairwise_path}")

    # Build 3-way orderings
    print("\n3. Building 3-way orderings (Economic, Political, Racial)...")
    scenarios_3way = list(SCENARIO_3WAY.keys())
    pairwise_3way = compute_all_pairwise_tests(model_data, scenarios=scenarios_3way)

    ordering_3way_rows = []
    for model in model_data.keys():
        ordering = build_ordering_string(pairwise_3way, model, scenarios_3way, SCENARIO_3WAY)
        adjacent_matches = check_empirical_match(ordering)
        all_matches = count_all_pairwise_matches(ordering)
        ordering_3way_rows.append({
            'model': model,
            'ordering': ordering,
            'empirical_matches': adjacent_matches,  # List of positions that match (for star placement)
            'n_adjacent_matches': len(adjacent_matches),
            'n_all_matches': all_matches  # Total pairwise matches including transitive
        })
        match_str = f" ({all_matches}/3 pairwise matches)" if all_matches else " (0/3 pairwise matches)"
        print(f"   {model}: {ordering}{match_str}")

    ordering_3way_df = pd.DataFrame(ordering_3way_rows)

    # Sort by goodness (n_all_matches descending) then alphabetically by model
    ordering_3way_df = ordering_3way_df.sort_values(
        by=['n_all_matches', 'model'],
        ascending=[False, True]
    ).reset_index(drop=True)

    print(f"\n   Sorted order (best first, then alphabetical):")
    for _, row in ordering_3way_df.iterrows():
        print(f"     {row['model']}: {row['n_all_matches']}/3 matches")

    # Save to CSV (convert list to string for CSV compatibility)
    ordering_3way_csv = ordering_3way_df.copy()
    ordering_3way_csv['empirical_matches'] = ordering_3way_csv['empirical_matches'].apply(str)
    ordering_3way_path = SCRIPT_DIR / "ordering_3way.csv"
    ordering_3way_csv.to_csv(ordering_3way_path, index=False)
    print(f"   Saved: {ordering_3way_path}")

    # Build 6-way orderings
    print("\n4. Building 6-way orderings (all scenarios)...")
    ordering_6way_rows = []
    for model in model_data.keys():
        ordering = build_ordering_string(pairwise_all, model, all_scenarios, SCENARIO_SHORT)
        ordering_6way_rows.append({
            'model': model,
            'ordering': ordering
        })
        print(f"   {model}: {ordering}")

    ordering_6way_df = pd.DataFrame(ordering_6way_rows)
    ordering_6way_path = SCRIPT_DIR / "ordering_6way.csv"
    ordering_6way_df.to_csv(ordering_6way_path, index=False)
    print(f"   Saved: {ordering_6way_path}")

    # Generate LaTeX
    print("\n5. Generating LaTeX table...")
    latex_code = generate_latex_table(ordering_3way_df)
    latex_path = SCRIPT_DIR / "key_results_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_code)
    print(f"   Saved: {latex_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
