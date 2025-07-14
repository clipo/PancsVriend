# Resume Experiments Guide

The `llm_runner.py` now supports resuming interrupted experiments, which is very useful for long-running LLM experiments.

## Features

### List Available Experiments

```bash
python llm_runner.py --list-experiments
```

This will show all available experiments with their completion status:

**Current Actual Status (manually verified):**
```
Available experiments:
--------------------------------------------------------------------------------
baseline_20250627_092459                           0/100 (not started)
baseline_20250627_221103                           0/100 (not started)  
baseline_20250703_100955                           100/100 (complete)
llm_baseline_20250627_121212                       0/100 (not started)
llm_baseline_20250703_101243                       94/100 (incomplete - can resume)
llm_baseline_20250707_015928                       0/8 (not started)
llm_ethnic_asian_hispanic_20250707_015824          0/100 (not started)
--------------------------------------------------------------------------------


### Resume an Experiment
```bash
python llm_runner.py --resume llm_baseline_20250706_143022 --runs 100
```

This will:
1. Check the existing experiment directory
2. Count completed runs (e.g., 45/100)
3. Resume from where it left off (runs 46-100)
4. Use the original experiment parameters (scenario, max_steps, etc.)
5. Combine existing and new results for final analysis

## Example Usage

### 1. Start a Long Experiment
```bash
python llm_runner.py --runs 100 --scenario ethnic_asian_hispanic
```

### 2. If Interrupted, Check Status
```bash
python llm_runner.py --list-experiments
```

### 3. Resume from Where It Left Off
```bash
python llm_runner.py --resume llm_ethnic_asian_hispanic_20250707_015824 --runs 100
```

## How It Works

1. **Detection**: The system scans the experiment directory for existing `run_*.json.gz` files
2. **Counting**: Counts completed runs and determines remaining runs needed
3. **ID Management**: Starts new run IDs from where the previous experiment left off
4. **Result Merging**: Combines existing and new results for final analysis
5. **Config Preservation**: Uses original experiment parameters unless explicitly overridden

## Safety Features

- **Prevents Overwriting**: Won't overwrite existing results
- **Completion Detection**: Detects already-complete experiments
- **Parameter Matching**: Uses original experiment parameters for consistency
- **Error Handling**: Gracefully handles missing or corrupted experiment directories

## Command Reference

```bash
# List experiments
python llm_runner.py --list-experiments

# Resume specific experiment
python llm_runner.py --resume <experiment_name> --runs <total_runs>

# Resume with custom parameters (not recommended - may cause inconsistency)
python llm_runner.py --resume <experiment_name> --runs 100 --scenario baseline --max-steps 1000
```

## Tips

1. **Always specify --runs**: The total number of runs you want (not just remaining)
2. **Use original parameters**: Let the system use the original experiment configuration
3. **Monitor progress**: Use `tail -f` on log files to monitor resumed experiments
4. **Background execution**: Use `nohup` and `&` for long-running resumed experiments
