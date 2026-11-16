# Statistical Methodology

This document describes the statistical methods used to compare segregation outcomes across social context scenarios in the PancsVriend framework.

## Overview

The analysis compares dissimilarity index (DI) distributions across six scenarios (baseline, race_white_black, ethnic_asian_hispanic, income_high_low, political_liberal_conservative, green_yellow) for each LLM model. With 10,000 runs per scenario, statistical power is high, so we require both statistical significance AND practical significance (effect size) to declare meaningful differences.

## Primary Pairwise Comparison Method

**Code:** `analysis_tools/ccs2026_presentation/compute_scenario_orderings.py`

### Test: Welch's Independent Samples t-test

Since runs are seeded independently (run_id is a seed, not a pairing), scenarios are compared using **independent samples t-tests** with Welch's correction (does not assume equal variances).

```python
t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
```

### Multiple Comparison Correction: Holm-Bonferroni

All pairwise comparisons within a model are corrected using the **Holm-Bonferroni method**, which controls the family-wise error rate while being less conservative than Bonferroni.

**Code:** `compute_scenario_orderings.py:128-148` (`holm_correction()` function)

### Effect Size: Cohen's d

Cohen's d is computed using pooled standard deviation:

```
d = (mean_a - mean_b) / pooled_std
pooled_std = sqrt(((n_a - 1) * std_a^2 + (n_b - 1) * std_b^2) / (n_a + n_b - 2))
```

**Code:** `compute_scenario_orderings.py:108-112`

### Significance Classification

A comparison is classified as significant only if **both** criteria are met:

| Criterion | Threshold |
|-----------|-----------|
| Statistical significance | p < 0.01 (Holm-corrected) |
| Practical significance | \|Cohen's d\| >= 0.2 (small effect) |

This dual requirement prevents trivial differences from being declared "significant" just because n is large.

**Symbols used:**
- `<` or `>`: Significant (p < 0.01 AND \|d\| >= 0.2)
- `≤` or `≥`: Marginal (0.01 <= p < 0.05 AND \|d\| >= 0.2)
- `≈`: Not significant (p >= 0.05 OR \|d\| < 0.2)

**Code:** `compute_scenario_orderings.py:151-184` (`classify_comparison()` function)

## ANOVA Across Scenarios

**Code:** `analysis_tools/anova_by_metric.py`

For omnibus tests across all scenarios, one-way ANOVA is used via `scipy.stats.f_oneway()`. This tests whether any scenario differs from any other, without specifying which pairs differ.

```python
f_stat, p_value = stats.f_oneway(*groups)
```

Results are written to `anova_results_by_metric.csv` and `.md` in the analysis directory.

## Comprehensive Statistical Analysis

**Code:** `statistical_analysis.py`

For detailed experiment comparisons, a full statistical pipeline is available:

1. **Normality testing:** Shapiro-Wilk test (`stats.shapiro()`)
2. **Homogeneity of variance:** Levene's test (`stats.levene()`)
3. **Main test selection:**
   - If assumptions met: One-way ANOVA with Tukey HSD post-hoc
   - If assumptions violated: Kruskal-Wallis H-test with Dunn's post-hoc (Bonferroni-corrected)
4. **Two-group comparisons:** Independent t-test or Mann-Whitney U depending on normality
5. **Effect size:** Cohen's d with interpretation thresholds:
   - \|d\| < 0.2: negligible
   - 0.2 <= \|d\| < 0.5: small
   - 0.5 <= \|d\| < 0.8: medium
   - \|d\| >= 0.8: large

## Data Sources

- **Per-run data:** `<run>/analysis/run_summary_by_run_all_scenarios.csv` (60,000 rows for 10,000 runs × 6 scenarios)
- **Pairwise test results:** `analysis_tools/ccs2026_presentation/pairwise_tests_all.csv`
- **Scenario orderings:** `analysis_tools/ccs2026_presentation/ordering_3way.csv`, `ordering_6way.csv`

## Key Design Decisions

1. **Why Welch's t-test?** Does not assume equal variances between scenarios, which is appropriate given different segregation dynamics across contexts.

2. **Why Holm-Bonferroni?** Controls FWER while being less conservative than Bonferroni, appropriate for the moderate number of comparisons (15 pairs for 6 scenarios).

3. **Why require effect size?** With n=10,000 per group, even tiny differences achieve statistical significance. The \|d\| >= 0.2 threshold ensures only practically meaningful differences are reported.

4. **Why p < 0.01?** More stringent than the conventional 0.05, appropriate for a study with high statistical power and multiple models being compared.

## References

- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.)
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2), 65-70.
- Welch, B. L. (1947). The generalization of "Student's" problem when several different population variances are involved. Biometrika, 34(1-2), 28-35.
