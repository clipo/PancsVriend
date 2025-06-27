# Parallel Processing Parameter for LLM Runner

The `llm_runner.py` script now supports a `--processes` parameter to control the number of CPU processes used for parallel execution of simulations.

## Usage Examples

### Basic Usage
```bash
# Use default number of processes (min of CPU cores and number of runs)
python llm_runner.py --runs 20

# Use specific number of processes
python llm_runner.py --runs 20 --processes 8

# Force sequential execution (1 process)
python llm_runner.py --runs 20 --processes 1

# Use maximum available CPU cores (careful with system resources)
python llm_runner.py --runs 20 --processes $(nproc)
```

### Advanced Examples
```bash
# Run 50 simulations with 16 processes
python llm_runner.py --runs 50 --processes 16 --scenario baseline

# Run with specific LLM settings and controlled parallelism
python llm_runner.py --runs 30 --processes 4 --llm-model llama3.1:8b --scenario conflict

# Sequential execution for debugging
python llm_runner.py --runs 5 --processes 1 --max-steps 100
```

## Parameter Details

### `--processes INTEGER`
- **Default**: `min(cpu_count, n_runs)` - Uses the smaller of available CPU cores or number of runs
- **Range**: 1 to CPU core count
- **Special values**:
  - `1`: Forces sequential execution (useful for debugging)
  - `> cpu_count`: Automatically capped at CPU core count with warning
  - `> n_runs`: Automatically capped at number of runs with warning

### Automatic Validation
The system automatically validates the `--processes` parameter:
- Warns if you request more processes than CPU cores available
- Warns if you request more processes than simulation runs
- Caps the value at reasonable limits
- Forces sequential execution for invalid values

### Performance Considerations
- **Recommended**: Use 50-80% of available CPU cores for optimal performance
- **Memory**: Each process consumes additional memory - monitor system resources
- **LLM API**: Consider rate limits of your LLM service when using many processes
- **I/O**: High process counts may cause disk I/O bottlenecks during result saving

### Example Output
```
Running 20 simulations using 8 parallel processes...
Running LLM simulations: 100%|████████| 20/20 [02:15<00:00,  6.78s/it]

Experiment completed. Results saved to: experiments/llm_baseline_20250626_143022
Total runs: 20
Converged runs: 18
Average convergence step: 156.34
Average LLM calls per run: 312.5
Average LLM response time: 0.245s
```

## Configuration File Integration
The number of processes used is automatically logged in the experiment configuration:
```json
{
  "parallel_execution": true,
  "n_processes": 8,
  "cpu_count": 124,
  ...
}
```

This helps with reproducibility and performance analysis of experiments.
