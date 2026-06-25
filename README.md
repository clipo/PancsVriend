# PancsVriend: Revealing the Bias Paradox in Large Language Models

A groundbreaking research framework that uncovers how Large Language Models (LLMs), despite having no explicit programming for discrimination, reproduce human segregation patterns with disturbing accuracy. Using the classic Schelling Segregation Model, we demonstrate the "bias paradox": LLMs' absorption of societal prejudices makes them both concerning perpetuators of bias AND superior tools for studying human social dynamics.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)](https://github.com/clipo/PancsVriend)

## Current Task

Run production simulations with both locally-available GGUF models, sequentially.

| # | Model label | GGUF path | Quant |
|---|-------------|-----------|-------|
| 1 | `llama-3.3-70b-instruct-q4` | `llms/Llama-3.3-70B-Instruct-Q4_K_M.gguf` | Q4_K_M |
| 2 | `gemma-4-31b-it-q5` | `llms/gemma-4-31B-it-Q5_K_M.gguf` | Q5_K_M |

Each model: 100 runs × 1000 steps × all scenarios. Results land in `experiments_with_llama_cpp/`.

### Current Status (last updated 2026-06-25 12:24 EDT)

**Model 1 (`llama-3.3-70b-instruct-q4`): ✅ COMPLETE**

| Scenario | Status | Notes |
|----------|--------|-------|
| baseline | ✅ Complete | 100/100 runs |
| race_white_black | ✅ Complete | 100/100 runs |
| ethnic_asian_hispanic | ✅ Complete | 100/100 runs |
| income_high_low | ✅ Complete | 100/100 runs |
| political_liberal_conservative | ✅ Complete | 100/100 runs |
| green_yellow | ✅ Complete | 100/100 runs |

Results in: `experiments_with_llama_cpp/run_20260605_141404_llama-3.3-70b-instruct-q4/`

**Model 2 (`gemma-4-31b-it-q5`): 🔄 PRODUCTION IN PROGRESS**

Smoke test PASSED 15:10 EDT Jun 11. Production running since then.

| Scenario | Status | Notes |
|----------|--------|-------|
| baseline | ✅ Complete | 100/100 runs |
| race_white_black | ✅ Complete | 100/100 runs |
| ethnic_asian_hispanic | ✅ Complete | 100/100 runs |
| income_high_low | 🔄 In progress | 70/100 runs (run 69, step ~710 as of 12:24 Jun 25) |
| political_liberal_conservative | ⏳ Pending | — |
| green_yellow | ⏳ Pending | — |

- Note: `income_high_low` runs are not converging early — many run near max steps (~2–8h/run vs ~12 min for converging scenarios)
- Server screen: `llama_server` (port 8080)
- Run screen: `llama_run` (PID 3990920)
- Transition screen: `transition` (PID 3993644) — will auto-start Mixtral when gemma completes
- Log: `logs/run_gemma-4-31b-it-q5.log`
- Results: `experiments_with_llama_cpp/run_20260611_151002_gemma-4-31b-it-q5/`

**Model 3 (`mixtral-8x7b-q5`): ⏳ PENDING — auto-starts when gemma completes**

GGUF at `~/llms/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf` (30 GB). `transition_to_mixtral.sh` running in `transition` screen.

### Automated transition (`transition_to_gemma.sh`)

A script (`transition_to_gemma.sh`) is running in the `transition` screen and will handle the llama→gemma handoff automatically. It polls `logs/run_llama-3.3-70b-instruct-q4.log` every 5 minutes and when `Pipeline completed.` is detected, it will:

1. Send llama completion email
2. Kill llama server
3. Swap `configs/llama_cpp_server.yaml` model path to the gemma GGUF
4. Swap `configs/llama_cpp_simulation_run.yaml` `llm_model:` label to `gemma-4-31b-it-q5`
5. Start gemma server in a new `llama_server` screen
6. Wait for server ready (up to 30 min)
7. Send gemma start email
8. Run smoke test
9. If smoke test passes: launch gemma production in `llama_run` screen → send launch email
10. If smoke test fails: send error email and exit (manual intervention required)

Monitor the transition: `tail -f logs/transition.log`

### When a new Claude session picks up

1. **Check screens are alive:**
   ```bash
   screen -list
   tail -5 logs/transition.log
   ```

2. **If `transition` screen is still running:** the automated handoff hasn't fired yet — check `tail -20 logs/run_llama-3.3-70b-instruct-q4.log` for progress and wait.

3. **If `transition` screen is gone and gemma run is live:** check `tail -20 logs/run_gemma-4-31b-it-q5.log` for progress. When gemma completes (`Pipeline completed.` in that log), send completion email:
   ```bash
   .venv/bin/python notify.py "Schelling run complete: gemma-4-31b-it-q5" "Production run finished. Results in experiments_with_llama_cpp/."
   ```

4. **If `transition` screen is gone but gemma run never started** (check `screen -list` for `llama_run` and `logs/transition.log` for errors): manual intervention — re-run from the step that failed.

### Step-by-step (for Claude Code to execute on "go")

**For each model in turn:**

1. **Edit `configs/llama_cpp_server.yaml`** — set the `model:` field to the GGUF's absolute path.

2. **Edit `configs/llama_cpp_simulation_run.yaml`** — set top-level `llm_model:` to the model label (e.g. `llama-3.3-70b-instruct-q4`).

3. **Start the server in a screen session:**
   ```bash
   screen -dmS llama_server bash -c "cd /srv/shared/schelling/PancsVriend && .venv/bin/python -m llama_cpp.server --config_file configs/llama_cpp_server.yaml 2>&1 | tee logs/server.log"
   ```

4. **Wait for the server to be ready** (poll until `http://localhost:8080/v1/models` responds).

5. **Send start email:**
   ```bash
   python notify.py "Schelling run started: <model label>" "Production run started: 100 runs x 1000 steps x all scenarios."
   ```

6. **Run smoke test first:**
   ```bash
   .venv/bin/python run_llm_probability_simulation_analysis.py \
     --config-yaml configs/llama_cpp_simulation_run.yaml \
     --config-profile smoke_test
   ```
   If smoke test fails, stop and report. Do not proceed to production.

7. **Run production in a screen session:**
   ```bash
   screen -dmS llama_run bash -c "cd /srv/shared/schelling/PancsVriend && .venv/bin/python run_llm_probability_simulation_analysis.py --config-yaml configs/llama_cpp_simulation_run.yaml --config-profile production 2>&1 | tee logs/run_<model_label>.log"
   ```

8. **Wait for completion**, then **send completion email:**
   ```bash
   python notify.py "Schelling run complete: <model label>" "Production run finished. Results in experiments_with_llama_cpp/."
   ```

9. **Kill the server**, then repeat from step 1 for the next model.

### Email Updates

Progress updates sent via `notify.py` (Gmail SMTP from `dr.duus@gmail.com`):

```bash
python notify.py "Subject" "Body"
```

Recipients: `dr.duus@gmail.com`, `siyer5@binghamton.edu`

## 🔬 The Bias Paradox Revealed

Our research uncovers a fundamental paradox in AI systems:

### The Paradox
- **No Explicit Bias Programming**: LLMs have no coded rules for discrimination
- **Yet Reproduce Human Prejudices**: They segregate based on race, politics, and ethnicity
- **Context-Dependent Biases**: Same LLM shows 12.3× different segregation levels based on framing
- **Emergent from Training Data**: Biases absorbed from human text, not programmed rules

### Why This Matters
1. **For AI Safety**: LLMs perpetuate hidden biases that vary by context
2. **For Social Science**: These biases make LLMs superior for modeling realistic human behavior  
3. **For Policy**: Understanding bias patterns enables better intervention design
4. **For Society**: Reveals how AI systems can amplify societal prejudices

### 🕒 NEW: Temporal Dynamics of Bias
Our analysis reveals that biases don't just vary in magnitude but in how they evolve:
- **Political contexts crystallize rapidly** (within 20 steps) - reflecting polarization dynamics
- **Economic contexts never stabilize** - showing perpetual residential fluidity
- **Racial/ethnic patterns develop slowly** (50-80 steps) - mirroring historical segregation

### 🏆 Key Research Findings

#### Agent Architecture Comparison (Baseline Red/Blue)
- **⚡ LLM agents converge 2.2× faster** than mechanical agents (84 vs 187 steps)
- **🏘️ Memory reduces extreme segregation** by 53.8% ("ghetto" formation)
- **📊 Similar final segregation levels** (~55% vs 58%) but different dynamics
- **🎯 100% convergence rate** for LLM agents vs 50% for mechanical

#### The Bias Paradox in Action
- **🔴 Political contexts show EXTREME segregation**: Ghetto rate 61.6 with rapid lock-in (1.95× early volatility)
- **💰 Economic contexts show MINIMAL segregation**: Ghetto rate 5.0 but never stabilizes (continuous churn)
- **🏘️ Racial/Ethnic contexts mirror real-world patterns**: ~40 ghetto rate with gradual historical development
- **🎭 Same LLM, different biases**: 12.3× difference based solely on social framing
- **🚨 No explicit bias rules**: These patterns emerge from implicit associations in training data

#### NEW: Temporal Dynamics Discovered
- **⚡ Political**: Rapid crystallization in first 20 steps (1.95× early volatility) - reflects polarization dynamics
- **🔄 Economic**: Perpetual fluidity (0.91× early/late volatility) - ongoing mobility
- **📈 Racial**: Slow burn over 50-80 steps (1.47× early volatility) - matches historical segregation patterns
- **🎯 Intervention Windows**: Different contexts require different timing strategies
- **📊 Stability Rankings**: Ethnic (most stable) > Baseline > Political > Race > Income (least stable)

## 📄 Scientific Papers

### Paper 1: Agent Architecture Comparison
**"Human-like Decision Making in Agent-Based Models: A Comparative Study of Large Language Model Agents versus Traditional Utility Maximization in the Schelling Segregation Model"**

- **Focus**: Comparing mechanical vs standard LLM vs memory-enhanced LLM agents
- **Key Finding**: LLM agents converge 2.2× faster with memory reducing extreme segregation
- **Status**: Original version prepared for submission
- **File**: [`schelling_llm_paper.qmd`](schelling_llm_paper.qmd)

### Paper 2: Social Context Effects (NEW)
**"Social Context Matters: How Large Language Model Agents Reproduce Real-World Segregation Patterns in the Schelling Model"**

- **Focus**: How different social framings (political, racial, economic) affect segregation
- **Key Finding**: Political contexts produce 12.3× more segregation than economic contexts
- **NEW Finding**: Temporal dynamics reveal context-specific evolution patterns
- **Status**: Analysis complete with dynamics, paper draft available
- **File**: [`schelling_llm_paper_updated.qmd`](schelling_llm_paper_updated.qmd)

### Paper 3: The Bias Paradox Study (FEATURED)
**"The Bias Paradox: How Large Language Models Reveal Human Prejudices While Advancing Agent-Based Social Science"**

- **Focus**: How LLMs reproduce human biases without explicit programming
- **Key Finding**: LLMs' greatest flaw (absorbing biases) is also their greatest strength for social science
- **NEW Analysis**: Temporal dynamics show biases evolve differently - political lock-in vs economic fluidity
- **Implications**: Both a warning for AI deployment and opportunity for research
- **Status**: Comprehensive analysis with AI ethics focus and policy recommendations
- **File**: [`schelling_llm_paper_comprehensive.qmd`](schelling_llm_paper_comprehensive.qmd)

- **Authors**: Andreas Pape, Carl Lipo, et al.
- **Institution**: Binghamton University
- **Render Instructions**: See [`paper_README.md`](paper_README.md)

## ✨ New Features (2024)

### 🔄 Easy LLM Model Switching
- **8 predefined LLM configurations** (Mixtral, Qwen, GPT-4, Claude, etc.)
- **Simple preset commands**: `--preset mixtral`, `--preset qwen`, `--preset gpt4`
- **Automatic validation** of API keys and configurations
- **Consistent arguments** across all scripts

### ⏱️ Real-Time Progress Monitoring
- **Live progress dashboard** showing "Run X of Y, Step Z of 1000"
- **Auto-refreshing monitoring** every 10 seconds
- **Progress files** updated every 10 simulation steps
- **No more guessing** if experiments are stuck or progressing

### 📊 Enhanced Dashboard System
- **Auto-detection** of active experiments
- **Quick launch** for progress monitoring
- **Multiple dashboard options** (analysis vs monitoring)
- **Smart experiment selection** based on current activity

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/clipo/PancsVriend.git
cd PancsVriend
pip install -r requirements.txt
```

### Easy LLM Model Switching
```bash
# List available LLM models
python switch_llm.py --list

# Test connectivity with different models
python switch_llm.py --preset mixtral --test
python switch_llm.py --preset qwen --test

# Run experiments with different models
python switch_llm.py --preset mixtral --quick-test
python switch_llm.py --preset gpt4 --run-experiments  # (requires OpenAI API key)
```

### Basic Usage
```bash
# Test LLM connectivity (uses Mixtral by default)
python check_llm.py

# Run comprehensive study with real-time progress monitoring
python comprehensive_comparison_study.py --quick-test

# Launch interactive dashboard with progress monitoring
python launch_dashboard_menu.py

# Full experiment suite
python run_experiments.py
```

## 📋 Available LLM Models

| Preset | Model | Provider | Status |
|--------|-------|----------|---------|
| `mixtral` | Mixtral 8x22B | Binghamton Uni | ✅ **Default** - Ready to use |
| `qwen` | Qwen 2.5 Coder 32B | Binghamton Uni | ✅ Ready to use |
| `gpt4` | GPT-4 | OpenAI | Requires API key |
| `gpt4o` | GPT-4o | OpenAI | Requires API key |
| `claude-sonnet` | Claude 3 Sonnet | Anthropic | Requires API key |
| `local-llama` | Llama2 | Local Ollama | Requires local setup |

## 🎯 Usage Examples

### Quick Testing
```bash
# Quick test with default model (Mixtral)
python comprehensive_comparison_study.py --quick-test

# Test different models
python switch_llm.py --preset qwen --quick-test
python switch_llm.py --preset mixtral --test
```

### Production Experiments
```bash
# Full comprehensive study
python comprehensive_comparison_study.py --preset mixtral

# Specific scenario testing
python switch_llm.py --preset qwen --llm race_white_black 30

# Custom configuration
python run_experiments.py --llm-model "gpt-4" --llm-url "https://api.openai.com/v1/chat/completions" --llm-api-key "your-key"
```

### Real-Time Monitoring
```bash
# Launch progress dashboard
python launch_dashboard_menu.py

# Direct progress monitoring
streamlit run dashboard_with_progress.py
```

## 🖥️ Running with a Local GGUF Model (llama.cpp)

Run the full simulation pipeline against a locally-hosted quantized model — no external API required.

### Overview

You serve a GGUF file via `llama-cpp-python`'s built-in OpenAI-compatible HTTP server, then point the simulation at `localhost:8080`. The pipeline skips the token-probability stage and runs simulation + analysis directly. Two scale profiles are available:

| Profile | Runs | Steps | Scenarios |
|---------|------|-------|-----------|
| `smoke_test` | 5 | 200 | baseline only |
| `production` | 100 | 1000 | all scenarios |

Always run `smoke_test` first to confirm end-to-end connectivity before committing to the long production run.

### Step 1 — Install llama-cpp-python server

```bash
pip install "llama-cpp-python[server]"
```

For GPU acceleration, install the CUDA build instead — see the
[llama-cpp-python installation docs](https://github.com/abetlen/llama-cpp-python#installation).
CPU-only works (slower) and is fine for `smoke_test`.

### Step 2 — Configure and start the server

Edit **one line** in `configs/llama_cpp_server.yaml` — set `model:` to the absolute path of your GGUF file:

```yaml
models:
  - model: "/absolute/path/to/your-model.gguf"
```

Then start the server (leave it running in a separate terminal or under `screen`/`tmux`):

```bash
python -m llama_cpp.server --config_file configs/llama_cpp_server.yaml
```

Ready when the terminal prints `Uvicorn running on http://0.0.0.0:8080`.

> **GPU OOM?** Lower `n_gpu_layers` in `configs/llama_cpp_server.yaml` from `-1` to a positive number (e.g. `28`) until the model loads.

### Step 3 — Label the run

In `configs/llama_cpp_simulation_run.yaml`, set `llm_model:` to a short label for the GGUF you loaded (e.g. `gemma-3-4b-it-q4`). This names the output folders — nothing else needs editing.

### Step 4 — Run the pipeline

```bash
# Validate setup first (fast — ~5 runs × 200 steps)
python run_llm_probability_simulation_analysis.py \
  --config-yaml configs/llama_cpp_simulation_run.yaml \
  --config-profile smoke_test

# Full production run (slow — run under screen/tmux)
screen -S schelling
python run_llm_probability_simulation_analysis.py \
  --config-yaml configs/llama_cpp_simulation_run.yaml \
  --config-profile production
```

### Output

Results land in a timestamped directory under `experiments_with_llama_cpp/`:

```
experiments_with_llama_cpp/run_<ts>_<model>/
├── run_config_effective.yaml
├── experiments/llm_<scenario>_<ts>/
│   ├── metrics_history.csv        # one row per (run, step)
│   ├── convergence_summary.csv
│   └── move_logs/  states/
├── analysis/                      # ANOVA, rankings, combined metrics
└── plots/                         # segregation plots (PNG)
```

Monitor a running job:

```bash
wc -l experiments_with_llama_cpp/run_*/experiments/*/metrics_history.csv
tail -f experiments_with_llama_cpp/run_*/experiments/*/convergence_summary.csv
nvidia-smi -l 5    # GPU utilization
```

### Throughput note

The llama.cpp server is single-stream (one request at a time). `processes` is pinned to `1` in both profiles — raising it only queues requests. For faster production runs, start several server instances on different ports behind a round-robin proxy, point `llm_url` at the proxy, and raise `processes` accordingly.

### Full guide

See [`LLAMA_CPP_SIMULATION_RUN_GUIDE.md`](LLAMA_CPP_SIMULATION_RUN_GUIDE.md) for troubleshooting and additional notes.

---

## 📊 Key Features

### Agent Types
- **Mechanical Agents**: Traditional Schelling model with utility functions
- **Standard LLM Agents**: Context-aware decisions using current neighborhood state
- **Memory LLM Agents**: Human-like agents with persistent memory and history

### Social Context Scenarios
- **baseline**: Red vs blue teams (control group)
- **race_white_black**: White middle class vs Black families
- **ethnic_asian_hispanic**: Asian American vs Hispanic/Latino families
- **economic_high_working**: High-income vs working-class households
- **political_liberal_conservative**: Liberal vs conservative households

### Segregation Metrics
- **Clustering Index**: Spatial concentration measurement
- **Switch Rate**: Frequency of agent relocations
- **Distance Metrics**: Average separation between groups
- **Mix Deviation**: Departure from random mixing
- **Share Metrics**: Proportional representation analysis
- **Ghetto Rate**: Extreme segregation detection

## 🛠 Core Components

### Experiment Scripts
- `comprehensive_comparison_study.py` - **Main entry point** for three-way comparisons
- `run_experiments.py` - Master orchestrator for complete experiment suites
- `baseline_runner.py` - Runs mechanical agent simulations
- `llm_runner.py` - Runs LLM agent simulations with social contexts

### LLM Management
- `switch_llm.py` - Easy model switching and testing
- `llm_presets.py` - Predefined LLM configurations
- `update_default_llm.py` - Change default model in config.py
- `check_llm.py` - Connectivity testing

### Real-Time Monitoring
- `dashboard_with_progress.py` - Live progress monitoring dashboard
- `launch_dashboard_menu.py` - Smart dashboard launcher
- `dashboard.py` - Comprehensive analysis dashboard

### Analysis Tools
- `statistical_analysis.py` - ANOVA, effect sizes, multivariate analysis
- `visualization.py` - Comprehensive PDF reports with time series and comparisons
- `plateau_detection.py` - Convergence detection and segregation speed calculation

## 🔧 Configuration

### Default Configuration
The system uses **Mixtral 8x22B** by default via Binghamton University's endpoint:
```python
# config.py
OLLAMA_MODEL = "mixtral:8x22b-instruct"
OLLAMA_URL = "https://chat.binghamton.edu/api/chat/completions"  
OLLAMA_API_KEY = "sk-571df6eec7f5495faef553ab5cb2c67a"
```

### Easy Model Switching
```bash
# Change default model
python update_default_llm.py --set qwen
python update_default_llm.py --set mixtral

# Show current default
python update_default_llm.py --show
```

### Command Line Overrides
All scripts support consistent LLM configuration:
```bash
# Use presets (recommended)
--preset mixtral
--preset qwen
--preset gpt4

# Custom configuration
--llm-model "model-name"
--llm-url "https://api.provider.com/v1/chat/completions"
--llm-api-key "your-api-key"
```

## 📊 Analysis & Visualization Tools

### Statistical Analysis
```bash
# Generate comprehensive statistical report
python statistical_analysis.py

# Detailed pairwise comparisons
python pairwise_comparison_analysis.py

# Convergence analysis with speed metrics
python convergence_analysis.py
```

### Comprehensive Visualization
```bash
# Generate publication-quality PDF report with all analyses
python comprehensive_visualization_report.py

# Creates: comprehensive_comparison_report.pdf with:
# - Executive summary
# - Convergence analysis 
# - Time series evolution
# - Final state comparisons
# - Statistical tables
# - Pairwise comparisons
```

### Academic Paper
```bash
# Render the scientific paper (requires Quarto + R)
quarto render schelling_llm_paper.qmd --to pdf

# See paper_README.md for setup instructions
```

## 📈 Experiment Workflows

### Comprehensive Comparison
```bash
# Quick validation
python comprehensive_comparison_study.py --quick-test

# Full three-way study (mechanical vs standard LLM vs memory LLM)
python comprehensive_comparison_study.py --preset mixtral

# Monitor progress in real-time
python launch_dashboard_menu.py  # Choose progress dashboard
```

### Social Context Studies
```bash
# Compare specific scenarios
python switch_llm.py --preset mixtral --llm race_white_black 30
python switch_llm.py --preset qwen --llm economic_high_working 30

# Multi-scenario comparison
python run_experiments.py --scenarios baseline race_white_black economic_high_working
```

### Model Comparisons
```bash
# Test different models on same scenario
python switch_llm.py --preset mixtral --quick-test
python switch_llm.py --preset qwen --quick-test
python switch_llm.py --preset gpt4 --quick-test  # (requires OpenAI key)
```

## 📊 Output Structure

Experiments generate structured outputs with real-time progress files:
```
experiments/
├── baseline_[timestamp]/              # Mechanical agent results
├── llm_[scenario]_[timestamp]/        # LLM scenario results
└── comprehensive_study_[timestamp]/   # Multi-agent comparisons
    ├── llm_results/
    │   ├── progress_realtime.json     # ← Real-time progress monitoring
    │   ├── experiments/exp_0001/
    │   └── logs/
    ├── comprehensive_analysis/
    └── three_way_comparison_report.md

reports/
├── comprehensive_report_[timestamp].pdf
├── statistical_analysis_[timestamp].txt
└── experiment_summary_[timestamp].json
```

## 🔍 Progress Monitoring

### Real-Time Dashboard Features
- **Live progress counters**: "Run 15 of 30, Step 450 of 1000"
- **Progress bars** for both run and step completion
- **Status indicators** (running/completed/failed)
- **Auto-refresh** every 10 seconds
- **Multi-experiment monitoring** for parallel runs

### Dashboard Access
```bash
# Smart launcher (detects active experiments)
python launch_dashboard_menu.py

# Direct progress dashboard
streamlit run dashboard_with_progress.py

# Analysis dashboard
streamlit run dashboard.py
```

## 🚨 AI Ethics and Bias Detection

### The Dual Nature of LLM Biases

Our research reveals that LLMs are not neutral tools but carriers of human cultural biases:

#### As a Problem
- **Hidden Biases**: LLMs perpetuate prejudices they were never explicitly taught
- **Context-Dependent**: Same model shows different biases based on framing
- **Unpredictable**: Biases may emerge in unexpected ways in applications
- **Amplification Risk**: Could reinforce societal prejudices at scale

#### As an Opportunity  
- **Bias Detection**: Use our framework to measure implicit prejudices in AI systems
- **Social Mirror**: LLMs reveal hidden biases in society through their outputs
- **Research Tool**: Study prejudice without survey response bias
- **Policy Testing**: Pre-test interventions across different bias scenarios

### Recommendations for AI Deployment

1. **Context-Aware Testing**: Test LLMs across multiple social framings before deployment
2. **Bias Monitoring**: Implement continuous monitoring for emergent biases
3. **Ensemble Approaches**: Use multiple LLMs to identify consistent vs variable biases
4. **Transparent Reporting**: Document bias profiles for different contexts

## 🎯 Research Applications

### Urban Planning
- Study segregation patterns under different policy scenarios
- Test integration policies virtually before implementation
- Understand how biases affect housing decisions

### Social Science  
- Use LLMs as mirrors to reveal societal biases
- Model realistic human behavior without explicit bias programming
- Study emergence of prejudice and discrimination

### AI Safety
- Develop comprehensive bias testing frameworks
- Create context-sensitive bias detection tools
- Design bias-aware AI governance structures

## 📈 Recent Analysis Tools Added

### Temporal Dynamics Analysis
- **analyze_rate_of_change.py** - Calculates how quickly segregation patterns evolve
- **analyze_stability_patterns.py** - Measures consistency and volatility across contexts  
- **analyze_convergence_patterns.py** - Identifies when patterns stabilize (or don't)
- **dynamics_analysis_summary.md** - Key findings from temporal analysis

### Key Dynamics Findings
1. **Political contexts show rapid crystallization** - 1.95× more volatile early then lock-in
2. **Economic contexts never stabilize** - Equal volatility throughout (0.91× ratio)
3. **Racial/ethnic show historical patterns** - Gradual development over 50-80 steps
4. **Different intervention windows** - Political needs immediate action, economic needs continuous management

## 🔍 Key Findings: The Bias Paradox Demonstrated

### Without Any Explicit Programming for Discrimination:

#### LLMs Reproduce Real-World Biases
- **Political**: 61.6 ghetto formation rate (extreme segregation)
- **Racial**: ~40 ghetto formation rate (matches empirical data)
- **Economic**: 5.0 ghetto formation rate (minimal segregation)
- **12.3× Variation**: Based solely on how groups are framed

### Emergent vs Programmed Behavior
Our research reveals the fundamental difference:

- **Mechanical agents**: Follow explicit rules (threshold = 0.5)
- **LLM agents**: No bias rules, yet produce human-like prejudices
- **Emergence**: Biases arise from training data, not programming
- **Variability**: Different contexts activate different implicit biases

### Why This Makes LLMs Superior for Social Science

#### Political Segregation (Most Extreme)
- **Ghetto formation rate**: 61.6 ± 9.3 (highest across all contexts)
- **Segregation share**: 0.928 ± 0.042 (near-complete segregation)
- **Switch rate**: 0.076 ± 0.036 (agents rarely move once settled)
- **Interpretation**: Reflects contemporary political polarization

#### Economic Integration (Most Integrated)
- **Ghetto formation rate**: 5.0 ± 3.1 (12.3× lower than political)
- **Number of clusters**: 25.0 ± 3.1 (most fragmented/mixed)
- **Convergence speed**: ~7 steps (10× faster than other contexts)
- **Interpretation**: Economic diversity more tolerable than other differences

#### Racial/Ethnic Patterns (Real-World Alignment)
- **Race (White/Black)**: Ghetto rate 40.8 ± 9.6, share 0.823 ± 0.060
- **Ethnic (Asian/Hispanic)**: Ghetto rate 38.9 ± 11.2, share 0.821 ± 0.076
- **Interpretation**: Matches empirical segregation indices from urban studies

#### Statistical Significance
- All contexts differ significantly from baseline (ANOVA p < 0.001)
- Large effect sizes (η² > 0.5) for all metrics
- Political > Racial/Ethnic > Baseline > Economic segregation levels

## 📚 Documentation

- **[LLM Switching Guide](README_LLM_SWITCHING.md)** - Complete guide to model switching
- **[CLAUDE.md](CLAUDE.md)** - Project instructions and command reference
- **Code Documentation** - Inline documentation throughout codebase

## 🔧 Advanced Features

### Supported LLM Providers
- **Local models**: Ollama, LM Studio, vLLM
- **OpenAI**: GPT-4, GPT-4o, GPT-3.5-turbo
- **Anthropic**: Claude-3-sonnet, Claude-3-haiku (via proxy)
- **Cloud providers**: Azure OpenAI, AWS Bedrock (via proxy)

### Performance Optimization
- **Parallel processing** with circuit breakers and automatic fallback
- **Error handling** with graceful degradation when LLM services fail
- **Scalability** for 100+ simulation runs with configurable parameters
- **Cost estimation** and resource planning tools

## 🤝 Contributing

We welcome contributions! Please:
- Report bugs or suggest features via GitHub issues
- Submit pull requests for improvements
- Share your research findings using this framework
- Test with different LLM models and report results

## 📁 Repository Structure

### Core Simulation Files
- `Agent.py` - Traditional utility-maximizing agents
- `LLMAgent.py` - LLM-powered agents with human-like decisions
- `Metrics.py` - Six segregation metrics (clusters, distance, share, etc.)
- `SchellingSim.py` - Interactive GUI simulation
- `config.py` - Central configuration for all parameters

### Experiment Runners
- `baseline_runner.py` - Mechanical agent experiments
- `llm_runner.py` - LLM agent experiments with social scenarios
- `comprehensive_comparison_study.py` - Full 3-way comparison
- `run_pure_comparison.py` - Pure agent type comparison
- `experiment_explorer.py` - Design space exploration

### Analysis & Visualization
- `statistical_analysis.py` - ANOVA, effect sizes, statistical tests
- `pairwise_comparison_analysis.py` - Detailed pairwise comparisons
- `convergence_analysis.py` - Convergence speed and rate analysis
- `comprehensive_visualization_report.py` - Complete PDF report generator
- `visualization.py` - Individual visualization tools
- **NEW**: `analyze_experiment_results.py` - Extract and compare final metrics across scenarios
- **NEW**: `analyze_convergence_patterns.py` - Time series analysis of segregation evolution
- **NEW**: `visualize_experiment_comparison.py` - Generate comparison plots and heatmaps
- **NEW**: `analyze_rate_of_change.py` - Temporal dynamics and phase transition analysis
- **NEW**: `analyze_stability_patterns.py` - Stability and volatility measurements

### LLM Configuration
- `switch_llm.py` - Interactive model switching
- `update_default_llm.py` - Update default LLM configuration
- `llm_presets.py` - Predefined LLM configurations
- `check_llm.py` - Connectivity and performance testing

### Dashboard & Monitoring
- `launch_dashboard_menu.py` - Dashboard launcher with options
- `dashboard_with_progress.py` - Real-time progress monitoring
- `cleanup_experiments.py` - Experiment cleanup utility

### Scientific Paper
- `schelling_llm_paper.qmd` - Quarto scientific paper
- `references.bib` - Bibliography
- `paper_README.md` - Paper rendering instructions
- `journal_style_guide.md` - Journal submission guide
- `verify_paper_data.R` - Data verification script

## 📄 Citation

If you use this framework in your research, please cite:
```bibtex
@software{pancs_vriend_llm,
  title={PancsVriend: LLM-Enhanced Schelling Segregation Model},
  author={[Research Team]},
  year={2024},
  url={https://github.com/clipo/PancsVriend},
  note={Research framework comparing mechanical and LLM agents in segregation dynamics}
}
```

## 📧 Contact

For questions about the research or technical issues:
- Create an issue on GitHub
- Contact the research team via [institutional contact]

## 🔮 The Bigger Picture: What This Means

### For AI Development
Our findings challenge the notion of "neutral" AI. LLMs trained on human text inevitably absorb human biases, creating systems that mirror our prejudices in complex, context-dependent ways. This isn't a bug to be fixed but a fundamental characteristic that must be understood and managed.

### For Social Science
The bias paradox offers unprecedented opportunities. Rather than programming our assumptions about human behavior, we can use LLMs to discover emergent patterns we might not have thought to look for. They serve as computational mirrors of society.

### For Society
As LLMs become integrated into decision-making systems (housing, hiring, lending), understanding their implicit biases becomes crucial. Our framework provides a method to detect and measure these biases before they cause harm.

### The Paradox Embraced
We propose embracing rather than eliminating the bias paradox:
- Use it as a tool to understand ourselves
- Leverage it for more realistic social modeling
- Monitor it carefully in applications
- Learn from it to build better societies

## 📜 License

[License information to be added]

---

*"The question is not whether LLMs have biases, but how we can use this mirror wisely." - This framework reveals the bias paradox at the heart of modern AI, providing tools to understand, measure, and leverage it for both research and responsible AI deployment.*