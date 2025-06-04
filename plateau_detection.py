import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def detect_plateau(series, window_size=10, threshold=0.01, min_plateau_length=20):
    """
    Detect plateau in a time series using rolling statistics
    
    Parameters:
    - series: The time series data
    - window_size: Window for calculating rolling statistics
    - threshold: Threshold for considering change as significant
    - min_plateau_length: Minimum length to consider as plateau
    
    Returns:
    - plateau_start: Step where plateau begins (None if no plateau)
    - plateau_info: Dictionary with plateau statistics
    """
    
    if len(series) < window_size * 2:
        return None, {}
    
    # Smooth the series
    smoothed = savgol_filter(series, window_length=min(len(series), window_size*2+1), polyorder=3)
    
    # Calculate rolling statistics
    rolling_mean = pd.Series(smoothed).rolling(window=window_size).mean()
    rolling_std = pd.Series(smoothed).rolling(window=window_size).std()
    
    # Calculate rate of change
    rate_of_change = np.diff(smoothed)
    rolling_roc = pd.Series(rate_of_change).rolling(window=window_size).mean()
    
    # Detect plateau: low variance and low rate of change
    plateau_mask = (rolling_std < threshold) & (np.abs(rolling_roc) < threshold)
    
    # Find consecutive plateau regions
    plateau_groups = []
    current_group = []
    
    for i, is_plateau in enumerate(plateau_mask):
        if is_plateau and not pd.isna(is_plateau):
            current_group.append(i)
        else:
            if len(current_group) >= min_plateau_length:
                plateau_groups.append(current_group)
            current_group = []
    
    if len(current_group) >= min_plateau_length:
        plateau_groups.append(current_group)
    
    if plateau_groups:
        # Take the first significant plateau
        first_plateau = plateau_groups[0]
        plateau_start = first_plateau[0]
        plateau_value = np.mean(series[plateau_start:plateau_start+window_size])
        
        plateau_info = {
            'start_step': plateau_start,
            'length': len(first_plateau),
            'value': plateau_value,
            'stability': 1 - rolling_std.iloc[plateau_start] if plateau_start < len(rolling_std) else 0
        }
        
        return plateau_start, plateau_info
    
    return None, {}

def calculate_convergence_rate(series, plateau_start=None):
    """
    Calculate the rate of convergence to plateau
    
    Parameters:
    - series: The time series data
    - plateau_start: Step where plateau begins
    
    Returns:
    - convergence_rate: Dictionary with convergence metrics
    """
    
    if plateau_start is None:
        plateau_start, _ = detect_plateau(series)
    
    if plateau_start is None or plateau_start < 5:
        return {}
    
    # Normalize the series
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(np.array(series).reshape(-1, 1)).flatten()
    
    # Fit exponential decay to pre-plateau data
    x = np.arange(plateau_start)
    y = normalized[:plateau_start]
    
    # Try to fit exponential: y = a * exp(-b * x) + c
    try:
        from scipy.optimize import curve_fit
        
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        popt, _ = curve_fit(exp_decay, x, y, p0=[1, 0.01, y[-1]], maxfev=5000)
        
        # Calculate half-life (time to reach halfway to plateau)
        half_life = -np.log(0.5) / popt[1] if popt[1] > 0 else np.inf
        
        # Calculate time to 90% of plateau value
        time_to_90 = -np.log(0.1) / popt[1] if popt[1] > 0 else np.inf
        
        convergence_rate = {
            'decay_constant': popt[1],
            'half_life': half_life,
            'time_to_90_percent': time_to_90,
            'initial_value': popt[0] + popt[2],
            'plateau_value': popt[2],
            'fit_quality': 1 - np.mean((y - exp_decay(x, *popt))**2)
        }
        
    except:
        # Fallback to linear approximation
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        
        convergence_rate = {
            'linear_slope': slope,
            'r_squared': r_value**2,
            'estimated_convergence_time': plateau_start
        }
    
    return convergence_rate

def analyze_all_metrics_convergence(metrics_df, run_id=None):
    """
    Analyze convergence for all metrics in a simulation run
    
    Parameters:
    - metrics_df: DataFrame with metrics history
    - run_id: Specific run to analyze (None for aggregate)
    
    Returns:
    - convergence_analysis: Dictionary with analysis for each metric
    """
    
    if run_id is not None:
        metrics_df = metrics_df[metrics_df['run_id'] == run_id]
    
    metric_columns = ['clusters', 'switch_rate', 'distance', 'mix_deviation', 'share', 'ghetto_rate']
    convergence_analysis = {}
    
    for metric in metric_columns:
        if metric in metrics_df.columns:
            series = metrics_df.groupby('step')[metric].mean().values
            
            plateau_start, plateau_info = detect_plateau(series)
            convergence_rate = calculate_convergence_rate(series, plateau_start)
            
            convergence_analysis[metric] = {
                'plateau_detected': plateau_start is not None,
                'plateau_info': plateau_info,
                'convergence_rate': convergence_rate,
                'final_value': series[-1] if len(series) > 0 else None,
                'total_change': abs(series[-1] - series[0]) if len(series) > 1 else 0
            }
    
    return convergence_analysis

def compare_convergence_across_runs(results_dir):
    """
    Compare convergence characteristics across multiple runs
    
    Parameters:
    - results_dir: Directory containing experiment results
    
    Returns:
    - comparison_df: DataFrame with convergence metrics for all runs
    """
    
    metrics_df = pd.read_csv(f"{results_dir}/metrics_history.csv")
    
    comparison_data = []
    
    for run_id in metrics_df['run_id'].unique():
        run_analysis = analyze_all_metrics_convergence(metrics_df, run_id)
        
        run_summary = {'run_id': run_id}
        
        for metric, analysis in run_analysis.items():
            if analysis['plateau_detected']:
                run_summary[f'{metric}_plateau_start'] = analysis['plateau_info']['start_step']
                run_summary[f'{metric}_plateau_value'] = analysis['plateau_info']['value']
                
                if 'half_life' in analysis['convergence_rate']:
                    run_summary[f'{metric}_half_life'] = analysis['convergence_rate']['half_life']
                elif 'estimated_convergence_time' in analysis['convergence_rate']:
                    run_summary[f'{metric}_convergence_time'] = analysis['convergence_rate']['estimated_convergence_time']
        
        comparison_data.append(run_summary)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate summary statistics
    summary_stats = {}
    for col in comparison_df.columns:
        if col != 'run_id':
            summary_stats[col] = {
                'mean': comparison_df[col].mean(),
                'std': comparison_df[col].std(),
                'min': comparison_df[col].min(),
                'max': comparison_df[col].max()
            }
    
    return comparison_df, summary_stats

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Generate sample data
    steps = np.arange(100)
    series = 10 * np.exp(-0.05 * steps) + 2 + np.random.normal(0, 0.1, 100)
    
    plateau_start, plateau_info = detect_plateau(series)
    convergence_rate = calculate_convergence_rate(series, plateau_start)
    
    print(f"Plateau detected at step: {plateau_start}")
    print(f"Plateau info: {plateau_info}")
    print(f"Convergence rate: {convergence_rate}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, series, 'b-', alpha=0.5, label='Original')
    if plateau_start:
        plt.axvline(x=plateau_start, color='r', linestyle='--', label=f'Plateau start (step {plateau_start})')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('Plateau Detection Example')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()