# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research framework comparing traditional utility-maximizing agents with LLM-based agents in the classic Schelling Segregation Model. The project studies how different social contexts (race, income, politics) affect residential segregation patterns.

## Key Commands

### Setup and Configuration
```bash
# Install dependencies
pip install -r requirements.txt

# Test LLM connectivity (uses config.py by default)
python check_llm.py

# Test LLM connectivity with custom configuration
python check_llm.py --llm-model "gpt-4" --llm-url "https://api.openai.com/v1/chat/completions" --llm-api-key "your-key"

# Test parallel LLM processing robustness
python test_llm_parallel.py

# Test with custom LLM configuration
python test_llm_parallel.py --llm-model "claude-3-sonnet" --llm-url "https://api.anthropic.com/v1/messages"
```

### Running Experiments
```bash
# Full experiment suite (uses config.py by default)
python run_experiments.py

# Full experiment suite with custom LLM
python run_experiments.py --llm-model "gpt-4o" --llm-url "https://api.openai.com/v1/chat/completions" --llm-api-key "your-key"

# Quick test version
python run_experiments.py --quick-test

# Custom configuration with multiple parameters
python run_experiments.py --baseline-runs 50 --llm-runs 20 --scenarios baseline race_white_black --llm-model "claude-3-sonnet"

# Individual components
python baseline_runner.py --runs 100
python llm_runner.py --scenario race_white_black --runs 10

# Interactive GUI simulation
python SchellingSim.py
```

### Analysis and Visualization
```bash
# Generate statistical analysis
python statistical_analysis.py

# Create visualization reports
python visualization.py --baseline-dir experiments/baseline_xxx --llm-dirs experiments/llm_*
```

## Architecture

### Core Components
- **Agent.py**: Traditional utility-maximizing agents with best-response dynamics
- **LLMAgent.py**: LLM-powered agents that make authentic human-like housing decisions
- **Metrics.py**: Six segregation metrics (clusters, switch rate, distance, mix deviation, share, ghetto rate)
- **config.py**: Central configuration for all simulation parameters

### Experiment Framework
- **run_experiments.py**: Master orchestrator for complete experiment suites
- **baseline_runner.py**: Runs mechanical agent simulations
- **llm_runner.py**: Runs LLM agent simulations with social context scenarios
- **plateau_detection.py**: Detects convergence points and calculates segregation speeds

### Analysis Pipeline
- **statistical_analysis.py**: ANOVA, effect sizes, multivariate analysis
- **visualization.py**: Comprehensive PDF reports with time series and comparisons
- **SchellingSim.py**: Interactive GUI for real-time simulation viewing

## Social Context Scenarios

The framework supports multiple social contexts defined in `llm_runner.py`:
- **baseline**: Red vs blue teams (control)
- **race_white_black**: White middle class vs Black families
- **ethnic_asian_hispanic**: Asian American vs Hispanic/Latino families
- **economic_high_working**: High-income vs working-class households
- **political_liberal_conservative**: Liberal vs conservative households

## LLM Configuration

### Default Configuration (config.py)
LLM settings are configured in `config.py`:
- `OLLAMA_MODEL`: Model identifier (default: "qwen2.5-coder:32B")
- `OLLAMA_URL`: API endpoint
- `OLLAMA_API_KEY`: Authentication key

### Command-Line Override
All scripts support command-line LLM configuration that overrides `config.py`:
- `--llm-model`: Specify different model (e.g., "gpt-4", "claude-3-sonnet", "llama2")
- `--llm-url`: Specify different API endpoint
- `--llm-api-key`: Specify different API key

### Supported LLM Providers
The framework works with any OpenAI-compatible API, including:
- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude-3-sonnet, Claude-3-haiku (via proxy)
- **Local models**: Ollama, LM Studio, vLLM
- **Cloud providers**: Azure OpenAI, AWS Bedrock (via proxy)

The system includes robust error handling with circuit breakers and automatic fallback to mechanical agents when LLM services fail.

## Output Structure

Experiments generate structured outputs:
```
experiments/
├── baseline_[timestamp]/       # Mechanical agent results
├── llm_[scenario]_[timestamp]/ # LLM scenario results
└── ...

reports/
├── comprehensive_report_[timestamp].pdf
├── statistical_analysis_[timestamp].txt
└── experiment_summary_[timestamp].json
```

## Development Notes

- The system uses parallel processing for LLM queries with configurable batch sizes
- All simulations include plateau detection to identify convergence points
- Statistical analysis includes normality testing, ANOVA, and effect size calculations
- Visualization generates comprehensive PDF reports with confidence intervals
- The codebase follows a modular design with clear separation between mechanical and LLM agent logic