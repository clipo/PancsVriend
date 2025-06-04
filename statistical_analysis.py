import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import scikit_posthocs as sp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def perform_statistical_tests(experiments_data, metric='distance', alpha=0.05):
    """
    Perform comprehensive statistical tests comparing experiments
    
    Parameters:
    - experiments_data: Dictionary mapping experiment names to directories
    - metric: The metric to analyze
    - alpha: Significance level
    
    Returns:
    - results: Dictionary containing test results
    """
    results = {}
    
    # Collect final values for each experiment
    experiment_values = {}
    all_data = []
    
    for exp_name, exp_dir in experiments_data.items():
        df = pd.read_csv(f"{exp_dir}/metrics_history.csv")
        final_values = df.groupby('run_id').last()[metric].values
        experiment_values[exp_name] = final_values
        
        # For ANOVA
        for value in final_values:
            all_data.append({'experiment': exp_name, metric: value})
    
    # 1. Descriptive statistics
    descriptive_stats = {}
    for exp_name, values in experiment_values.items():
        descriptive_stats[exp_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'q1': np.percentile(values, 25),
            'q3': np.percentile(values, 75),
            'min': np.min(values),
            'max': np.max(values),
            'n': len(values)
        }
    results['descriptive'] = descriptive_stats
    
    # 2. Normality tests
    normality_tests = {}
    for exp_name, values in experiment_values.items():
        if len(values) >= 3:
            stat, p_value = stats.shapiro(values)
            normality_tests[exp_name] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > alpha
            }
    results['normality'] = normality_tests
    
    # 3. Homogeneity of variance (Levene's test)
    if len(experiment_values) > 1:
        values_list = list(experiment_values.values())
        stat, p_value = stats.levene(*values_list)
        results['levene_test'] = {
            'statistic': stat,
            'p_value': p_value,
            'equal_variance': p_value > alpha
        }
    
    # 4. ANOVA or Kruskal-Wallis test
    if len(experiment_values) > 2:
        # Check if all groups are normally distributed
        all_normal = all(test['is_normal'] for test in normality_tests.values() if 'is_normal' in test)
        
        if all_normal and results.get('levene_test', {}).get('equal_variance', False):
            # Parametric ANOVA
            df_anova = pd.DataFrame(all_data)
            model = ols(f'{metric} ~ C(experiment)', data=df_anova).fit()
            anova_table = anova_lm(model, typ=2)
            
            results['anova'] = {
                'type': 'One-way ANOVA',
                'f_statistic': anova_table['F'][0],
                'p_value': anova_table['PR(>F)'][0],
                'significant': anova_table['PR(>F)'][0] < alpha
            }
            
            # Post-hoc Tukey HSD if significant
            if results['anova']['significant']:
                tukey = pairwise_tukeyhsd(df_anova[metric], df_anova['experiment'], alpha=alpha)
                results['tukey_hsd'] = {
                    'summary': str(tukey),
                    'results': []
                }
                for i in range(len(tukey.groupsunique)):
                    for j in range(i+1, len(tukey.groupsunique)):
                        results['tukey_hsd']['results'].append({
                            'group1': tukey.groupsunique[i],
                            'group2': tukey.groupsunique[j],
                            'meandiff': tukey.meandiffs[i,j],
                            'p_adj': tukey.pvalues[i,j],
                            'reject': tukey.reject[i,j]
                        })
        else:
            # Non-parametric Kruskal-Wallis test
            stat, p_value = stats.kruskal(*values_list)
            results['kruskal_wallis'] = {
                'type': 'Kruskal-Wallis H-test',
                'h_statistic': stat,
                'p_value': p_value,
                'significant': p_value < alpha
            }
            
            # Post-hoc Dunn test if significant
            if results['kruskal_wallis']['significant']:
                df_kw = pd.DataFrame(all_data)
                dunn_results = sp.posthoc_dunn(df_kw, val_col=metric, group_col='experiment', p_adjust='bonferroni')
                results['dunn_test'] = dunn_results.to_dict()
    
    # 5. Pairwise comparisons (if only 2 groups)
    elif len(experiment_values) == 2:
        exp_names = list(experiment_values.keys())
        values1 = experiment_values[exp_names[0]]
        values2 = experiment_values[exp_names[1]]
        
        # Check normality
        normal1 = normality_tests.get(exp_names[0], {}).get('is_normal', False)
        normal2 = normality_tests.get(exp_names[1], {}).get('is_normal', False)
        
        if normal1 and normal2 and results.get('levene_test', {}).get('equal_variance', False):
            # Independent t-test
            stat, p_value = stats.ttest_ind(values1, values2)
            test_type = 'Independent t-test'
        else:
            # Mann-Whitney U test
            stat, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
            test_type = 'Mann-Whitney U test'
        
        results['pairwise_test'] = {
            'type': test_type,
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'effect_size': calculate_effect_size(values1, values2)
        }
    
    return results

def calculate_effect_size(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    # Interpretation
    if abs(d) < 0.2:
        interpretation = "negligible"
    elif abs(d) < 0.5:
        interpretation = "small"
    elif abs(d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return {
        'cohens_d': d,
        'interpretation': interpretation
    }

def analyze_convergence_patterns(experiments_data):
    """
    Analyze and compare convergence patterns across experiments
    """
    convergence_analysis = {}
    
    for exp_name, exp_dir in experiments_data.items():
        conv_df = pd.read_csv(f"{exp_dir}/convergence_summary.csv")
        metrics_df = pd.read_csv(f"{exp_dir}/metrics_history.csv")
        
        # Convergence statistics
        conv_steps = conv_df['convergence_step'].dropna()
        
        convergence_analysis[exp_name] = {
            'convergence_rate': len(conv_steps) / len(conv_df),
            'mean_convergence_step': conv_steps.mean() if len(conv_steps) > 0 else np.nan,
            'std_convergence_step': conv_steps.std() if len(conv_steps) > 0 else np.nan,
            'convergence_variability': conv_steps.std() / conv_steps.mean() if len(conv_steps) > 0 and conv_steps.mean() > 0 else np.nan
        }
        
        # Analyze rate of change patterns
        for metric in ['distance', 'mix_deviation', 'ghetto_rate']:
            if metric in metrics_df.columns:
                # Calculate average rate of change in early steps
                early_steps = metrics_df[metrics_df['step'] <= 50]
                if len(early_steps) > 0:
                    grouped = early_steps.groupby('step')[metric].mean()
                    if len(grouped) > 1:
                        rates = np.diff(grouped.values)
                        convergence_analysis[exp_name][f'{metric}_early_rate'] = np.mean(np.abs(rates))
    
    return convergence_analysis

def perform_multivariate_analysis(experiments_data, metrics=['distance', 'mix_deviation', 'ghetto_rate']):
    """
    Perform multivariate analysis (PCA) on final metric values
    """
    # Collect data
    data_points = []
    labels = []
    
    for exp_name, exp_dir in experiments_data.items():
        df = pd.read_csv(f"{exp_dir}/metrics_history.csv")
        final_values = df.groupby('run_id').last()
        
        for _, row in final_values.iterrows():
            point = [row[metric] for metric in metrics if metric in row]
            if len(point) == len(metrics):
                data_points.append(point)
                labels.append(exp_name)
    
    if len(data_points) < 2:
        return None
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_points)
    
    # PCA
    pca = PCA(n_components=min(len(metrics), 2))
    data_pca = pca.fit_transform(data_scaled)
    
    results = {
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'components': pca.components_,
        'transformed_data': data_pca,
        'labels': labels,
        'feature_importance': {}
    }
    
    # Feature importance
    for i, metric in enumerate(metrics):
        importance = np.abs(pca.components_[:, i]).mean()
        results['feature_importance'][metric] = importance
    
    return results

def create_statistical_report(experiments_data, output_file='statistical_analysis.txt'):
    """
    Create a comprehensive statistical report
    """
    metrics = ['clusters', 'distance', 'mix_deviation', 'share', 'ghetto_rate']
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STATISTICAL ANALYSIS REPORT\n")
        f.write("Schelling Segregation Model - Experiment Comparison\n")
        f.write("="*80 + "\n\n")
        
        # Analyze each metric
        for metric in metrics:
            f.write(f"\n{'='*60}\n")
            f.write(f"METRIC: {metric.upper().replace('_', ' ')}\n")
            f.write(f"{'='*60}\n")
            
            try:
                results = perform_statistical_tests(experiments_data, metric)
                
                # Descriptive statistics
                f.write("\nDESCRIPTIVE STATISTICS:\n")
                f.write("-"*40 + "\n")
                for exp_name, stats in results['descriptive'].items():
                    f.write(f"\n{exp_name}:\n")
                    f.write(f"  Mean: {stats['mean']:.4f} (SD: {stats['std']:.4f})\n")
                    f.write(f"  Median: {stats['median']:.4f} [Q1: {stats['q1']:.4f}, Q3: {stats['q3']:.4f}]\n")
                    f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
                    f.write(f"  N: {stats['n']}\n")
                
                # Normality tests
                f.write("\nNORMALITY TESTS (Shapiro-Wilk):\n")
                f.write("-"*40 + "\n")
                for exp_name, test in results['normality'].items():
                    f.write(f"{exp_name}: p={test['p_value']:.4f} ")
                    f.write(f"({'Normal' if test['is_normal'] else 'Not Normal'})\n")
                
                # Main test results
                if 'anova' in results:
                    f.write(f"\n{results['anova']['type']}:\n")
                    f.write("-"*40 + "\n")
                    f.write(f"F-statistic: {results['anova']['f_statistic']:.4f}\n")
                    f.write(f"p-value: {results['anova']['p_value']:.4f}\n")
                    f.write(f"Result: {'Significant differences' if results['anova']['significant'] else 'No significant differences'}\n")
                    
                    if 'tukey_hsd' in results:
                        f.write("\nPost-hoc Tukey HSD:\n")
                        for comp in results['tukey_hsd']['results']:
                            f.write(f"  {comp['group1']} vs {comp['group2']}: ")
                            f.write(f"diff={comp['meandiff']:.4f}, p={comp['p_adj']:.4f} ")
                            f.write(f"({'*' if comp['reject'] else 'ns'})\n")
                
                elif 'kruskal_wallis' in results:
                    f.write(f"\n{results['kruskal_wallis']['type']}:\n")
                    f.write("-"*40 + "\n")
                    f.write(f"H-statistic: {results['kruskal_wallis']['h_statistic']:.4f}\n")
                    f.write(f"p-value: {results['kruskal_wallis']['p_value']:.4f}\n")
                    f.write(f"Result: {'Significant differences' if results['kruskal_wallis']['significant'] else 'No significant differences'}\n")
                
                elif 'pairwise_test' in results:
                    f.write(f"\n{results['pairwise_test']['type']}:\n")
                    f.write("-"*40 + "\n")
                    f.write(f"Statistic: {results['pairwise_test']['statistic']:.4f}\n")
                    f.write(f"p-value: {results['pairwise_test']['p_value']:.4f}\n")
                    f.write(f"Result: {'Significant difference' if results['pairwise_test']['significant'] else 'No significant difference'}\n")
                    f.write(f"Effect size (Cohen's d): {results['pairwise_test']['effect_size']['cohens_d']:.4f} ")
                    f.write(f"({results['pairwise_test']['effect_size']['interpretation']})\n")
                
            except Exception as e:
                f.write(f"\nError analyzing {metric}: {str(e)}\n")
        
        # Convergence analysis
        f.write("\n\n" + "="*80 + "\n")
        f.write("CONVERGENCE ANALYSIS\n")
        f.write("="*80 + "\n")
        
        conv_analysis = analyze_convergence_patterns(experiments_data)
        for exp_name, analysis in conv_analysis.items():
            f.write(f"\n{exp_name}:\n")
            f.write(f"  Convergence rate: {analysis['convergence_rate']:.2%}\n")
            if not np.isnan(analysis['mean_convergence_step']):
                f.write(f"  Mean convergence step: {analysis['mean_convergence_step']:.1f} (SD: {analysis['std_convergence_step']:.1f})\n")
                f.write(f"  Convergence variability (CV): {analysis['convergence_variability']:.2f}\n")
            
            for metric in ['distance', 'mix_deviation', 'ghetto_rate']:
                key = f'{metric}_early_rate'
                if key in analysis:
                    f.write(f"  Early {metric} rate of change: {analysis[key]:.4f}\n")
        
        # Multivariate analysis
        f.write("\n\n" + "="*80 + "\n")
        f.write("MULTIVARIATE ANALYSIS (PCA)\n")
        f.write("="*80 + "\n")
        
        pca_results = perform_multivariate_analysis(experiments_data)
        if pca_results:
            f.write("\nPrincipal Component Analysis:\n")
            f.write(f"  Explained variance ratio: {pca_results['explained_variance_ratio']}\n")
            f.write("\nFeature Importance:\n")
            for metric, importance in sorted(pca_results['feature_importance'].items(), 
                                           key=lambda x: x[1], reverse=True):
                f.write(f"  {metric}: {importance:.3f}\n")
    
    print(f"Statistical report saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    # Example usage
    experiments = {
        'baseline': 'experiments/baseline_20240101_120000',
        'llm_baseline': 'experiments/llm_baseline_20240101_130000',
        'llm_race': 'experiments/llm_race_white_black_20240101_140000'
    }
    
    # Run analysis
    create_statistical_report(experiments)