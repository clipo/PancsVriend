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
- key_results_ordering.tex: LaTeX model rows only (for \input in presentation)
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

# Effect size threshold: minimum Cohen's d to be considered practically significant
# Cohen's d thresholds: 0.2 = small, 0.5 = medium, 0.8 = large
# We require at least a small effect (|d| >= 0.2) to be considered meaningful
MIN_COHENS_D = 0.2

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


def independent_ttest(df, scenario_a, scenario_b, metric='dissimilarity_index'):
    """
    Compute independent samples t-test between two scenarios.

    These are independent distributions (run_id is just a seed, not a pairing).
    Uses Welch's t-test which doesn't assume equal variances.
    """
    values_a = df[df['scenario'] == scenario_a][metric].values
    values_b = df[df['scenario'] == scenario_b][metric].values

    if len(values_a) == 0 or len(values_b) == 0:
        return None

    # Independent samples t-test (Welch's, doesn't assume equal variance)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)

    # Compute Cohen's d using pooled standard deviation
    std_a, std_b = values_a.std(), values_b.std()
    n_a, n_b = len(values_a), len(values_b)
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    cohens_d = (values_a.mean() - values_b.mean()) / pooled_std if pooled_std > 0 else 0

    return {
        'n_a': n_a,
        'n_b': n_b,
        'mean_a': values_a.mean(),
        'mean_b': values_b.mean(),
        'std_a': std_a,
        'std_b': std_b,
        'mean_diff': values_a.mean() - values_b.mean(),
        'cohens_d': cohens_d,
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


def classify_comparison(p_holm, mean_a, mean_b, cohens_d):
    """
    Classify a comparison based on Holm-corrected p-value AND effect size.

    Returns (symbol, direction) where:
    - symbol: '<', '≤', or '≈'
    - direction: 1 if a < b, -1 if a > b, 0 if equal

    A comparison must meet BOTH criteria to be considered significant:
    1. Statistical significance: p < threshold
    2. Practical significance: |Cohen's d| >= MIN_COHENS_D (0.2 = small effect)

    This prevents declaring trivial differences as "significant" just because n is large.
    """
    # If effect size is below threshold, treat as equivalent
    if abs(cohens_d) < MIN_COHENS_D:
        return '≈', 0

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
    Compute independent samples t-tests for all scenario pairs for each model.

    Uses Welch's t-test (doesn't assume equal variances).
    Returns DataFrame with all test results including Cohen's d effect size.
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
            result = independent_ttest(df, s_a, s_b)
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
                symbol, direction = classify_comparison(p_holm[i], r['mean_a'], r['mean_b'], r['cohens_d'])
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
            cohens_d = abs(row['cohens_d'])

            # Check both statistical AND practical significance
            # Require |Cohen's d| >= 0.2 (small effect) to be meaningful
            if cohens_d < MIN_COHENS_D:
                symbol = '≈'
            elif p_holm < P_SIGNIFICANT:
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

    Only counts a match if the comparison is SIGNIFICANT (< or ≤), not ≈.
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
    symbols = [p.strip() for p in parts[1::2] if p.strip()]

    # Check if all scenarios are in empirical ranking
    if not all(s in EMPIRICAL_RANK for s in scenarios):
        return 0

    # Count pairwise matches - only for SIGNIFICANT comparisons
    # For adjacent pairs, use the symbol directly
    # For transitive pairs (A vs C), both A<B and B<C must be significant
    count = 0
    n = len(scenarios)

    # Check adjacent pairs
    for i in range(n - 1):
        s_low, s_high = scenarios[i], scenarios[i + 1]
        sym = symbols[i] if i < len(symbols) else '≈'

        # Only count if significant (< or ≤) AND matches empirical
        if sym in ['<', '≤'] and EMPIRICAL_RANK[s_low] < EMPIRICAL_RANK[s_high]:
            count += 1

    # Check transitive pair (first vs last) - only if BOTH adjacent pairs are significant
    if n >= 3 and len(symbols) >= 2:
        all_adjacent_significant = all(s in ['<', '≤'] for s in symbols[:n-1])
        if all_adjacent_significant:
            s_first, s_last = scenarios[0], scenarios[-1]
            if EMPIRICAL_RANK[s_first] < EMPIRICAL_RANK[s_last]:
                count += 1

    return count


def count_significant_differences(ordering_str):
    """
    Count the number of significant differences (< or ≤) in an ordering string.

    For "Racial < Economic < Political" returns 2.
    For "Racial ≈ Economic ≈ Political" returns 0.
    For "Racial < Political ≈ Economic" returns 1.
    """
    import re
    symbols = re.findall(r'[<≤≈>≥]', ordering_str)
    return sum(1 for s in symbols if s in ['<', '≤', '>', '≥'])


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
    """Generate LaTeX code for the Key Results ordering rows only.

    Uses grouping notation for cases where the first element matches empirical
    against all others, but internal ordering doesn't match.
    E.g., Olmo: "Economic <* (Racial < Political)" shows Economic lowest matches
    empirical, but Racial < Political doesn't match (should be Political < Racial).

    Only generates the model rows (\midrule to \bottomrule) - the rest of the
    frame is in the main presentation .tex file.
    """
    from datetime import datetime
    import re

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # No comments at start - they break \noalign in tabular when using \input
    latex = ""

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

    # Remove trailing \\\n from last row - main .tex will add \\ before \bottomrule
    latex = latex.rstrip('\n').rstrip('\\').rstrip('\\')
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
        n_sig_diffs = count_significant_differences(ordering)
        ordering_3way_rows.append({
            'model': model,
            'ordering': ordering,
            'empirical_matches': adjacent_matches,  # List of positions that match (for star placement)
            'n_adjacent_matches': len(adjacent_matches),
            'n_all_matches': all_matches,  # Total pairwise matches including transitive
            'n_significant_diffs': n_sig_diffs  # Number of < or ≤ symbols
        })
        match_str = f" ({all_matches}/3 pairwise matches, {n_sig_diffs} sig diffs)"
        print(f"   {model}: {ordering}{match_str}")

    ordering_3way_df = pd.DataFrame(ordering_3way_rows)

    # Sort by: 1) n_all_matches desc, 2) n_significant_diffs desc, 3) model alpha
    # This groups models with no significant differences (granite, mistral, phi) at the bottom
    ordering_3way_df = ordering_3way_df.sort_values(
        by=['n_all_matches', 'n_significant_diffs', 'model'],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    print(f"\n   Sorted order (by matches, then sig diffs, then alphabetical):")
    for _, row in ordering_3way_df.iterrows():
        print(f"     {row['model']}: {row['n_all_matches']}/3 matches, {row['n_significant_diffs']} sig diffs")

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
    latex_path = SCRIPT_DIR / "key_results_ordering.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_code.rstrip('\n'))  # No trailing newline - needed for \bottomrule
    print(f"   Saved: {latex_path}")

    # Write metadata file
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info_path = SCRIPT_DIR / "key_results_ordering_info.txt"
    with open(info_path, 'w') as f:
        f.write(f"Source: analysis_tools/ccs2026_presentation/compute_scenario_orderings.py\n")
        f.write(f"Generated: {timestamp}\n")
    print(f"   Saved: {info_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
