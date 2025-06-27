#!/usr/bin/env python3
"""
Experiment Output Format Verification Tool

This script validates the consistency and integrity of experiment outputs by checking:
1. Agent move logs coherence with state files
2. Metrics history consistency 
3. Convergence summary accuracy
4. Cross-validation between different output formats

The tool ensures that all experiment outputs are synchronized and contain
the expected data structures for downstream analysis.

Author: Generated for PancsVriend simulation validation
"""

# Standard library imports
import os
import json
import gzip

# Scientific computing imports
import numpy as np
import pandas as pd


def load_agent_moves(output_dir, run_id=None):
    """
    Load agent move logs from experiment output directory.
    
    Agent move logs contain detailed information about each agent's decision
    and movement during the simulation, including:
    - Agent ID and type
    - Current and new positions
    - Decision made (target position or None)
    - Whether the move was successful
    - Reason for the outcome (successful_move, target_occupied, etc.)
    - Grid state after the move
    
    Parameters:
    -----------
    output_dir : str
        Path to experiment output directory
    run_id : int, optional
        Specific run ID to load. If None, loads all runs
        
    Returns:
    --------
    list
        List of move logs, one per run. Each element is a list of move entries.
        
    Raises:
    -------
    FileNotFoundError
        If the output directory or move logs don't exist
    """
    move_logs = []
    
    # Determine the correct directory for move logs
    # Some experiments store logs in a subdirectory, others in the main directory
    move_logs_dir = os.path.join(output_dir, "move_logs")
    if os.path.exists(move_logs_dir):
        search_dir = move_logs_dir
    else:
        search_dir = output_dir
    
    # Collect all matching move log files and sort by run_id (ascending)
    move_files = []
    for fname in os.listdir(search_dir):
        if fname.startswith("agent_moves_run_") and fname.endswith(".json.gz"):
            # Filter by specific run_id if provided
            if run_id is not None and f"agent_moves_run_{run_id}.json.gz" != fname:
                continue
            # Extract run_id from filename for sorting
            rid = int(fname.split('_')[-1].split('.')[0])
            move_files.append((rid, fname))
    
    move_files.sort()  # Sort by run_id (ascending)
    
    # Load and decompress each move log file
    for rid, fname in move_files:
        with gzip.open(os.path.join(search_dir, fname), 'rt', encoding='utf-8') as f:
            move_logs.append(json.load(f))
    
    return move_logs

def load_states(states_dir, run_id=None):
    """
    Load grid states from the states directory.
    If run_id is None, loads all runs.
    Returns a dict: run_id -> np.ndarray (steps, grid, grid)
    """
    states = {}
    
    # Collect all matching state files and sort by run_id (ascending)
    state_files = []
    for fname in os.listdir(states_dir):
        if fname.startswith("states_run_") and fname.endswith(".npz"):
            # Extract run_id from filename
            rid = int(fname.split('_')[-1].split('.')[0])
            # Filter by specific run_id if provided
            if run_id is not None and rid != run_id:
                continue
            state_files.append((rid, fname))
    
    state_files.sort()  # Sort by run_id (ascending)
    
    # Load each state file
    for rid, fname in state_files:
        arr = np.load(os.path.join(states_dir, fname))
        # Extract the array data (npz files can contain multiple arrays)
        key = list(arr.keys())[0]
        states[rid] = arr[key]
    
    return states

def load_metrics_history(output_dir):
    """
    Load metrics history as a pandas DataFrame.
    """
    metrics_path = os.path.join(output_dir, "metrics_history.csv")
    if os.path.exists(metrics_path):
        return pd.read_csv(metrics_path)
    return None

def load_convergence_summary(output_dir):
    """
    Load convergence summary as a pandas DataFrame.
    """
    summary_path = os.path.join(output_dir, "convergence_summary.csv")
    if os.path.exists(summary_path):
        return pd.read_csv(summary_path)
    return None


# =============================================================================
# EXPERIMENT DIRECTORY DISCOVERY
# =============================================================================

# Locate the most recent experiment directory
# This section automatically finds and loads the latest experiment for validation
main_dir = "/data/siyer/PancsVriend/"
experiments_path = os.path.join(main_dir, "experiments")

# Find all experiment directories
exp_dirs = [d for d in os.listdir(experiments_path) 
           if os.path.isdir(os.path.join(experiments_path, d))]

if not exp_dirs:
    raise ValueError("No experiment directories found in experiments/")

# Use the most recently created directory
# Note: This assumes directory names sort chronologically (e.g., contain timestamps)
exp_dir = exp_dirs[-1]  # Get the last one in the list
print(f"🔍 Using latest experiment directory: {exp_dir}")

output_dir = os.path.join(experiments_path, exp_dir)

# =============================================================================
# DATA LOADING
# =============================================================================

print("📊 Loading experiment data...")

# Load agent movement logs (detailed decision and movement data)
all_run_logs = load_agent_moves(output_dir)
print(f"✓ Loaded movement logs for {len(all_run_logs)} runs")

# Load aggregated metrics history
metrics = load_metrics_history(output_dir)
if metrics is not None:
    print(f"✓ Loaded metrics history: {len(metrics)} timesteps")
else:
    print("⚠️  No metrics history file found")

# Load convergence summary
summary = load_convergence_summary(output_dir)
if summary is not None:
    print(f"✓ Loaded convergence summary: {len(summary)} runs")
else:
    print("⚠️  No convergence summary file found")

# Load grid states (spatial configurations over time)
states = load_states(os.path.join(output_dir, "states"))
print(f"✓ Loaded grid states for {len(states)} runs")


# =============================================================================
# DATA CONSISTENCY VALIDATION
# =============================================================================

print("\n🔍 Validating data consistency...")

# Gather statistics for comparison across different data sources
run_log_lengths = [len(run_log) for run_log in all_run_logs]
states_lengths = [len(states[rid]) for rid in states.keys()]

# Extract maximum step values from run logs to verify simulation duration
max_steps_per_run = []
for run_log in all_run_logs:
    if len(run_log) > 0:
        # Get all step values from the run log
        steps = [entry['step'] for entry in run_log if 'step' in entry]
        max_steps_per_run.append(max(steps) if steps else -1)
    else:
        max_steps_per_run.append(-1)

# Create comprehensive comparison table
if summary is not None and 'final_step' in summary.columns:
    comparison_df = pd.DataFrame({
        'run_log_length': run_log_lengths,      # Number of log entries per run
        'run_log_max_step_value': max_steps_per_run,  # Highest step number in logs
        'final_step': summary['final_step'],     # Final step from convergence summary
        'states_length': states_lengths,         # Number of state snapshots per run
    })
    
    print("\n📋 Data Consistency Comparison:")
    print(comparison_df)
    
    # CRITICAL VALIDATION 1: Run log lengths must match state lengths
    # Each agent decision should correspond to exactly one state snapshot
    try:
        assert comparison_df['run_log_length'].equals(comparison_df['states_length']), \
            "❌ CRITICAL ERROR: Run log lengths do not match states lengths!"
        print("✅ Run log and state lengths are consistent")
    except AssertionError as e:
        print(f"❌ {e}")
        print("This indicates a synchronization problem between movement logging and state saving.")
        raise
    
    # CRITICAL VALIDATION 2: Maximum step values must match final steps
    # The highest step number in logs should match the convergence summary
    try:
        assert comparison_df['run_log_max_step_value'].equals(comparison_df['final_step']), \
            "❌ CRITICAL ERROR: Run log max step values do not match final steps in summary!"
        print("✅ Step numbering is consistent across data sources")
    except AssertionError as e:
        print(f"❌ {e}")
        print("This indicates inconsistent step counting between logging systems.")
        raise
        
else:
    print("⚠️  Cannot perform full validation - convergence summary missing 'final_step' column")
    comparison_df = pd.DataFrame({
        'run_log_length': run_log_lengths,
        'states_length': states_lengths,
    })
    print("📋 Partial Comparison:")
    print(comparison_df)

# =============================================================================
# GRID STATE VALIDATION
# =============================================================================

print("\n🎯 Validating grid state consistency...")

# For each run, verify that grid states from movement logs match
# the grid states stored in separate state files
validation_errors = []

for run_no in range(len(all_run_logs)):
    print(f"  Validating run {run_no}...")
    
    # Extract grid states from movement logs
    # Each movement log entry contains a 'grid' field with the state after that move
    grid_from_run_log = [
        all_run_logs[run_no][i]['grid'] 
        for i in range(len(all_run_logs[run_no])) 
        if 'grid' in all_run_logs[run_no][i]
    ]
    
    # Get corresponding states from state files
    state = states[run_no]
    state = [s.tolist() for s in state]  # Convert numpy arrays to lists for comparison
    
    print(f"    Grid states in movement log: {len(grid_from_run_log)}")
    print(f"    Grid states in state file:   {len(state)}")
    
    # CRITICAL VALIDATION 3: Number of grid states must match
    if len(grid_from_run_log) != len(state):
        error_msg = f"Run {run_no}: Grid count mismatch - Log: {len(grid_from_run_log)}, State: {len(state)}"
        validation_errors.append(error_msg)
        print(f"    ❌ {error_msg}")
        continue
    
    # CRITICAL VALIDATION 4: Grid contents must be identical at each timestep
    mismatches = 0
    for i in range(len(grid_from_run_log)):
        if not np.array_equal(grid_from_run_log[i], state[i]):
            mismatches += 1
            if mismatches <= 3:  # Only report first few mismatches to avoid spam
                validation_errors.append(f"Run {run_no}, Step {i}: Grid state mismatch")
                print(f"    ❌ Step {i}: Grid states don't match")
    
    if mismatches == 0:
        print(f"    ✅ All {len(grid_from_run_log)} grid states match perfectly")
    else:
        print(f"    ❌ {mismatches} grid state mismatches found")

# Final validation report
if validation_errors:
    print(f"\n❌ VALIDATION FAILED: {len(validation_errors)} errors found:")
    for error in validation_errors[:10]:  # Show first 10 errors
        print(f"  - {error}")
    if len(validation_errors) > 10:
        print(f"  ... and {len(validation_errors) - 10} more errors")
    
    raise AssertionError(f"Grid state validation failed with {len(validation_errors)} errors. "
                        "This indicates inconsistencies between movement logs and state files.")
else:
    print("\n🎉 ALL VALIDATIONS PASSED!")
    print(f"✅ Data consistency verified across {len(all_run_logs)} runs")
    print("✅ Movement logs and state files are perfectly synchronized")
    print("✅ Step numbering is consistent across all data sources")
    print("✅ Grid states match exactly between logging systems")
    
print("\n📊 Validation Summary:")
print(f"  Runs processed: {len(all_run_logs)}")
print(f"  Total timesteps validated: {sum(len(states[rid]) for rid in states.keys())}")
print(f"  Total movement decisions checked: {sum(run_log_lengths)}")
print("  All data sources are consistent and ready for analysis! 🚀")


def main():
    """
    Main validation function that orchestrates the complete experiment output verification.
    
    Returns:
    --------
    bool
        True if all validations pass, False otherwise
    """
    try:
        # =============================================================================
        # EXPERIMENT DIRECTORY DISCOVERY
        # =============================================================================

        # Locate the most recent experiment directory
        # This section automatically finds and loads the latest experiment for validation
        main_dir = "/data/siyer/PancsVriend/"
        experiments_path = os.path.join(main_dir, "experiments")

        # Find all experiment directories
        exp_dirs = [d for d in os.listdir(experiments_path) 
                   if os.path.isdir(os.path.join(experiments_path, d))]

        if not exp_dirs:
            raise ValueError("No experiment directories found in experiments/")

        # Use the most recently created directory
        # Note: This assumes directory names sort chronologically (e.g., contain timestamps)
        exp_dir = exp_dirs[-1]  # Get the last one in the list
        print(f"🔍 Using latest experiment directory: {exp_dir}")

        output_dir = os.path.join(experiments_path, exp_dir)

        # =============================================================================
        # DATA LOADING
        # =============================================================================

        print("📊 Loading experiment data...")

        # Load agent movement logs (detailed decision and movement data)
        all_run_logs = load_agent_moves(output_dir)
        print(f"✓ Loaded movement logs for {len(all_run_logs)} runs")

        # Load aggregated metrics history
        metrics = load_metrics_history(output_dir)
        if metrics is not None:
            print(f"✓ Loaded metrics history: {len(metrics)} timesteps")
        else:
            print("⚠️  No metrics history file found")

        # Load convergence summary
        summary = load_convergence_summary(output_dir)
        if summary is not None:
            print(f"✓ Loaded convergence summary: {len(summary)} runs")
        else:
            print("⚠️  No convergence summary file found")

        # Load grid states (spatial configurations over time)
        states = load_states(os.path.join(output_dir, "states"))
        print(f"✓ Loaded grid states for {len(states)} runs")

        # Continue with validation...
        return validate_experiment_data(all_run_logs, states, summary)
        
    except Exception as e:
        print(f"❌ Validation failed with error: {str(e)}")
        return False


def validate_experiment_data(all_run_logs, states, summary):
    """
    Perform comprehensive validation of experiment data consistency.
    
    Parameters:
    -----------
    all_run_logs : list
        List of movement logs for each run
    states : dict
        Dictionary mapping run_id to grid states
    summary : pandas.DataFrame or None
        Convergence summary data
        
    Returns:
    --------
    bool
        True if all validations pass, False otherwise
    """
    validation_errors = []
    
    # =============================================================================
    # DATA CONSISTENCY VALIDATION
    # =============================================================================

    print("\n🔍 Validating data consistency...")

    # Gather statistics for comparison across different data sources
    run_log_lengths = [len(run_log) for run_log in all_run_logs]
    states_lengths = [len(states[rid]) for rid in states.keys()]

    # Extract maximum step values from run logs to verify simulation duration
    max_steps_per_run = []
    for run_log in all_run_logs:
        if len(run_log) > 0:
            # Get all step values from the run log
            steps = [entry['step'] for entry in run_log if 'step' in entry]
            max_steps_per_run.append(max(steps) if steps else -1)
        else:
            max_steps_per_run.append(-1)

    # Create comprehensive comparison table
    if summary is not None and 'final_step' in summary.columns:
        comparison_df = pd.DataFrame({
            'run_log_length': run_log_lengths,      # Number of log entries per run
            'run_log_max_step_value': max_steps_per_run,  # Highest step number in logs
            'final_step': summary['final_step'],     # Final step from convergence summary
            'states_length': states_lengths,         # Number of state snapshots per run
        })
        
        print("\n📋 Data Consistency Comparison:")
        print(comparison_df)
        
        # CRITICAL VALIDATION 1: Run log lengths must match state lengths
        # Each agent decision should correspond to exactly one state snapshot
        if not comparison_df['run_log_length'].equals(comparison_df['states_length']):
            validation_errors.append("Run log lengths do not match states lengths!")
            print("❌ CRITICAL ERROR: Run log lengths do not match states lengths!")
            print("This indicates a synchronization problem between movement logging and state saving.")
        else:
            print("✅ Run log and state lengths are consistent")
        
        # CRITICAL VALIDATION 2: Maximum step values must match final steps
        # The highest step number in logs should match the convergence summary
        if not comparison_df['run_log_max_step_value'].equals(comparison_df['final_step']):
            validation_errors.append("Run log max step values do not match final steps in summary!")
            print("❌ CRITICAL ERROR: Run log max step values do not match final steps in summary!")
            print("This indicates inconsistent step counting between logging systems.")
        else:
            print("✅ Step numbering is consistent across data sources")
            
    else:
        print("⚠️  Cannot perform full validation - convergence summary missing 'final_step' column")
        comparison_df = pd.DataFrame({
            'run_log_length': run_log_lengths,
            'states_length': states_lengths,
        })
        print("📋 Partial Comparison:")
        print(comparison_df)

    # =============================================================================
    # GRID STATE VALIDATION
    # =============================================================================

    print("\n🎯 Validating grid state consistency...")

    # For each run, verify that grid states from movement logs match
    # the grid states stored in separate state files
    for run_no in range(len(all_run_logs)):
        print(f"  Validating run {run_no}...")
        
        # Check if this run_no exists in states dictionary
        if run_no not in states:
            validation_errors.append(f"Run {run_no}: No state file found")
            print(f"    ❌ No state file found for run {run_no}")
            continue
        
        # Extract grid states from movement logs
        # Each movement log entry contains a 'grid' field with the state after that move
        grid_from_run_log = [
            all_run_logs[run_no][i]['grid'] 
            for i in range(len(all_run_logs[run_no])) 
            if 'grid' in all_run_logs[run_no][i]
        ]
        
        # Get corresponding states from state files
        state = states[run_no]
        state = [s.tolist() for s in state]  # Convert numpy arrays to lists for comparison
        
        print(f"    Grid states in movement log: {len(grid_from_run_log)}")
        print(f"    Grid states in state file:   {len(state)}")
        
        # CRITICAL VALIDATION 3: Number of grid states must match
        if len(grid_from_run_log) != len(state):
            error_msg = f"Run {run_no}: Grid count mismatch - Log: {len(grid_from_run_log)}, State: {len(state)}"
            validation_errors.append(error_msg)
            print(f"    ❌ {error_msg}")
            continue
        
        # CRITICAL VALIDATION 4: Grid contents must be identical at each timestep
        mismatches = 0
        for i in range(len(grid_from_run_log)):
            if not np.array_equal(grid_from_run_log[i], state[i]):
                mismatches += 1
                if mismatches <= 3:  # Only report first few mismatches to avoid spam
                    validation_errors.append(f"Run {run_no}, Step {i}: Grid state mismatch")
                    print(f"    ❌ Step {i}: Grid states don't match")
        
        if mismatches == 0:
            print(f"    ✅ All {len(grid_from_run_log)} grid states match perfectly")
        else:
            print(f"    ❌ {mismatches} grid state mismatches found")

    # Final validation report
    if validation_errors:
        print(f"\n❌ VALIDATION FAILED: {len(validation_errors)} errors found:")
        for error in validation_errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(validation_errors) > 10:
            print(f"  ... and {len(validation_errors) - 10} more errors")
        
        return False
    else:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print(f"✅ Data consistency verified across {len(all_run_logs)} runs")
        print("✅ Movement logs and state files are perfectly synchronized")
        print("✅ Step numbering is consistent across all data sources")
        print("✅ Grid states match exactly between logging systems")
        
        print("\n📊 Validation Summary:")
        print(f"  Runs processed: {len(all_run_logs)}")
        print(f"  Total timesteps validated: {sum(len(states[rid]) for rid in states.keys())}")
        print(f"  Total movement decisions checked: {sum(len(run_log) for run_log in all_run_logs)}")
        print("  All data sources are consistent and ready for analysis! 🚀")
        
        return True


if __name__ == "__main__":
    """
    Script entry point for command-line execution.
    """
    print("=" * 70)
    print("🔬 EXPERIMENT OUTPUT FORMAT VERIFICATION")
    print("=" * 70)
    print("This tool validates the consistency and integrity of simulation outputs")
    print("by cross-checking movement logs, grid states, and summary data.")
    print("=" * 70)
    
    success = main()
    
    if success:
        print("\n" + "=" * 70)
        print("✅ VALIDATION COMPLETED SUCCESSFULLY")
        print("All experiment outputs are consistent and ready for analysis.")
        print("=" * 70)
        exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ VALIDATION FAILED")
        print("Please check the errors above and fix data inconsistencies.")
        print("=" * 70)
        exit(1)


